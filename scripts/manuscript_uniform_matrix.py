#!/usr/bin/env python3
"""One global k<0.5/PSIS sample per system, shared by six pooling comparisons."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import socket
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.manuscript_postprocess import build_plan, parse_args as postprocess_args, digest, write_json

PROTOCOL = {
    'protocol_id': 'jctc2026', 'manuscript_diagnostics_only': True,
    'qc_policy': 'exclude-frames', 'qc_method': 'gpd', 'qc_exclusion_scope': 'global',
    'qc_exclusion_strategy': 'global-gpd-k', 'qc_k_threshold': .5,
    'qc_reporting_scope': 'global', 'weight_treatment': 'gpd-psis',
    'prior_mode': 'raw_counts', 'psis_scope': 'state', 'psis_reff': 1.,
    'minimum_retained_frames': 20, 'selection': 'first passing stable ascending-score deletion prefix',
    'local_qc': 'diagnostic flags; no further state-dependent deletion or masking',
    'solver': 'blend', 'methods': ['independent', 'neighbor_pool'],
    'comparisons': ['cardinal_tau1', 'cardinal_tau100', 'cardinal_lcurve',
                    'cheb3_tau1', 'cheb3_tau100', 'cheb3_lcurve'],
}


def now():
    return datetime.now(timezone.utc).isoformat()


def code_identity():
    paths = sorted((REPO / 'felp').rglob('*.py')) + [Path(__file__), REPO / 'scripts/manuscript_postprocess.py', REPO / 'scripts/manuscript_postprocess_analysis.py']
    files = {str(p.relative_to(REPO)): digest(p) for p in paths}
    return {'sha256': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(), 'files': files}


def prepare(base, project, workers, *, manifest=None, systems=(), python=None, hosts=()):
    argv = ['--project-root', str(project), '--output-root', str(base / 'results'),
            '--tag', 'uniform', '--workers', str(workers)]
    if manifest is not None:
        argv += ['--manifest', str(manifest)]
    for system in systems:
        argv += ['--system', system]
    args = postprocess_args(argv)
    resolved = build_plan(args)
    if any(r['missing'] for r in resolved['runs']):
        raise ValueError({r['id']: r['missing'] for r in resolved['runs'] if r['missing']})
    code = code_identity()
    tasks = []
    for i, run in enumerate(resolved['runs']):
        run.pop('commands')
        run['output_dir'] = str(base / 'results' / run['id'])
        spec = {'run': run, 'protocol': PROTOCOL, 'code': code, 'workers': workers}
        path = base / 'control/specs' / f'{run["id"]}.json'
        write_json(path, spec)
        tasks.append({'id': run['id'], 'host': hosts[i % len(hosts)] if hosts else 'local',
                      'cwd': str(REPO), 'log': str(base / 'logs' / f'{run["id"]}.log'),
                      'command': [str(python or sys.executable), str(Path(__file__)), '--spec', str(path), '--resume']})
    write_json(base / 'control/matrix_plan.json', {'created_utc': now(), 'protocol': PROTOCOL, 'code': code,
        'tasks': tasks, 'systems': resolved['runs'], 'total_comparisons': len(tasks) * 6,
        'environment': {'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
                        'MPLCONFIGDIR': '/tmp/felp-uniform-mpl', 'NUMBA_CACHE_DIR': '/tmp/felp-uniform-numba',
                        'PYTHONDONTWRITEBYTECODE': '1'}})
    print(f'{len(tasks)} systems, {len(tasks) * 6} pooled comparisons', flush=True)


def write_outcome(outcome, path, frame_count):
    from felp.core import OutputConfig
    from felp.io.results import ResultsWriter
    path.mkdir(parents=True, exist_ok=False)
    writer = ResultsWriter(OutputConfig(output_dir=str(path)))
    writer.write_microstate_results(outcome.microstate_results)
    writer.write_summary({'frame_count': frame_count, 'reweighting_method': outcome.method_id,
                          **outcome.shared_metadata})
    # Per-frame projection weights are reproducible from the common prep; do
    # not write six copies of them or duplicate the original frame CSV.
    outcome.auxiliary.clear()


def without_psis(prep):
    import math
    import numpy as np
    from felp.core import frame_reweighting_delta
    from felp.reweighting.aggregator import _effective_sample_size, _psis_not_requested_metadata
    states = {}
    for key, state in prep.state_data.items():
        if state.frames:
            delta = np.array([frame_reweighting_delta(f) for f in state.frames])
            weights = np.exp(-prep.beta * (delta - delta.min()))
            ess = _effective_sample_size(weights)
            weights *= math.exp(-prep.beta * (delta.min() - prep.global_delta_shift))
        else:
            weights, ess = np.array([]), 0.
        states[key] = replace(state, treated_weights=weights, local_factor=float(weights.mean()) if len(weights) else 0.,
                              local_ess=ess, mean_boltzmann_factor=float(weights.mean()) if len(weights) else None,
                              std_boltzmann_factor=float(weights.std()) if len(weights) else None,
                              psis_weight_metadata=_psis_not_requested_metadata())
    return replace(prep, state_data=states, qc_config=replace(prep.qc_config, weight_treatment='raw'))


def run_system(spec, *, resume=False):
    import gc
    import numpy as np
    import pandas as pd
    from felp.app import _repartition_frame_results
    from felp.core import MicrostateConfig, ReweightingConfig, ReweightingQCConfig, NeighborPoolConfig
    from felp.io.results import load_frame_results_csv
    from felp.reweighting.atlas import MicrostateAtlas
    from felp.reweighting.prep import build_reweighting_prep
    from felp.reweighting.topology import build_grid_topology
    from felp.reweighting.engines import get_reweighting_engine
    from scripts.manuscript_postprocess_analysis import analyze_run, plot_run, KB

    run = spec['run']
    root = Path(run['output_dir'])
    paths = [run['input_csv'], run['states'].get('file'), run.get('target_probability_csv')]
    configuration = {'run': run, 'protocol': spec['protocol'], 'code_sha256': code_identity()['sha256'],
                     'inputs_sha256': {p: digest(Path(p)) for p in paths if p},
                     'packages': {n: importlib.metadata.version(n) for n in ['numpy', 'scipy', 'pandas', 'arviz', 'matplotlib', 'xarray']},
                     'python': sys.version, 'workers': spec['workers']}
    if configuration['code_sha256'] != spec['code']['sha256']:
        raise ValueError('Scientific source changed after the task plan was frozen')
    record = root / 'provenance.json'
    if root.exists():
        old = json.loads(record.read_text()) if record.exists() else {}
        if resume and old.get('configuration') == configuration and old.get('status') == 'complete':
            for name in PROTOCOL['comparisons']:
                if not (root / 'comparisons' / name / 'diagnostics.csv').is_file():
                    raise ValueError('Completed record is missing a comparison artifact')
            print(f'{run["id"]}: verified completed configuration', flush=True)
            return old
        raise ValueError(f'Preserve incomplete or changed run {root}; choose a new output root')
    root.mkdir(parents=True)
    status = {'configuration': configuration, 'host': socket.gethostname(), 'started_utc': now(),
              'status': 'running', 'stage': 'loading', 'comparisons': {}}
    write_json(record, status)

    def checkpoint(stage):
        status['stage'] = stage
        status['updated_utc'] = now()
        write_json(record, status)
        print(f'{run["id"]}: {stage}', flush=True)

    try:
        frames = load_frame_results_csv(run['input_csv'])
        if run.get('expected_frames') is not None and len(frames) != run['expected_frames']:
            raise ValueError(f'Expected {run["expected_frames"]} frames, got {len(frames)}')
        if len({f.frame_index for f in frames}) != len(frames):
            raise ValueError('Production score file has duplicate frame IDs')
        state = run['states']
        micro = MicrostateConfig(source=state['source'], definition_file=state.get('file'),
                                 phi_bins=state.get('phi_bins', 45), psi_bins=state.get('psi_bins', 45),
                                 reference_probability_mode='raw_counts')
        atlas = MicrostateAtlas.from_config(micro)
        frames = _repartition_frame_results(frames, atlas,
            grid_bins=(micro.phi_bins, micro.psi_bins) if state['source'] == 'grid' else None)
        beta = 1 / (KB * run['temperature_kelvin'])
        common = dict(beta=beta, temperature_kelvin=run['temperature_kelvin'], workers=spec['workers'],
                      periodic_cv_names=run.get('periodic_cvs', []))
        config = ReweightingConfig(methods=('independent',))
        independent = get_reweighting_engine('independent')
        stage_rows = []

        def emit(prep, name):
            outcome = independent.aggregate(prep, config=config)
            write_outcome(outcome, root / name, len(frames))
            table = pd.read_csv(root / name / 'microstate_summary.csv')
            p, q, ess = [table[column].to_numpy(float) for column in ['reference_probability', 'target_probability', 'effective_sample_size']]
            assert np.isclose(p.sum(), 1) and np.isclose(q.sum(), 1)
            if name != 'raw':
                np.testing.assert_allclose(p, raw_p, atol=1e-13)
            positive = q > 0
            implied = float(1 / np.sum(q[positive]**2 / ess[positive])) if np.all(ess[positive] > 0) else None
            gqc = prep.global_qc
            stage_rows.append({'stage': name, 'retained_frames': prep.filtered_eligible_frame_count,
                               'excluded_frames': prep.eligible_frame_count - prep.filtered_eligible_frame_count,
                               'global_raw_k': gqc.final_pareto_k, 'global_raw_ess': gqc.global_ess,
                               'global_treated_weight_ess': outcome.shared_metadata.get('global_effective_sample_size'),
                               'implied_frame_ess_from_state_probabilities': implied,
                               'emptied_reference_mass': float(p[table.frame_count.to_numpy() == 0].sum())})
            stage_rows[-1].update(outcome.shared_metadata['manuscript_diagnostics'])
            return p

        checkpoint('raw_baseline')
        raw_qc = ReweightingQCConfig(policy='none', method='gpd', exclusion_scope='global', pareto_k_threshold=.5)
        raw_prep = build_reweighting_prep(frames, atlas, qc_config=raw_qc, **common)
        raw_p = emit(raw_prep, 'raw')
        del raw_prep
        gc.collect()
        checkpoint('unfiltered_psis')
        unfiltered = build_reweighting_prep(frames, atlas, qc_config=replace(raw_qc, policy='flag', weight_treatment='gpd-psis'), **common)
        emit(unfiltered, 'raw_psis')
        del unfiltered
        gc.collect()
        checkpoint('global_gpd_filter_and_psis')
        qc = ReweightingQCConfig(policy='exclude-frames', method='gpd', weight_treatment='gpd-psis',
                                exclusion_scope='global', exclusion_strategy='global-gpd-k', pareto_k_threshold=.5)
        prep = build_reweighting_prep(frames, atlas, qc_config=qc, **common)
        meta = prep.global_qc.to_metadata()
        write_json(root / 'quality_control.json', meta)
        pd.DataFrame(meta['exclusion_history']).to_csv(root / 'global_k_cut_curve.csv', index=False)
        excluded = np.asarray(prep.global_qc.excluded_frame_indices, dtype='<i8')
        status['excluded_ids_sha256'] = hashlib.sha256(excluded.tobytes()).hexdigest()
        status['global_k'] = prep.global_qc.final_pareto_k
        status['global_k_pass'] = bool(status['global_k'] is not None and status['global_k'] < .5)
        status['excluded_frames'] = len(excluded)
        unsmoothed = without_psis(prep)
        emit(unsmoothed, 'filtered_raw')
        del unsmoothed
        emit(prep, 'filtered_psis')
        pd.DataFrame(stage_rows).to_csv(root / 'processing_stages.csv', index=False)
        pool_engine = get_reweighting_engine('neighbor_pool')
        for stencil in ['cardinal', 'cheb3']:
            connectivity, radius = ('cardinal', 1) if stencil == 'cardinal' else ('chebyshev', 3)
            topology = build_grid_topology(atlas, grid_connectivity=connectivity, chebyshev_radius=radius,
                                           periodic_cv_names=run.get('periodic_cvs', []))
            if topology is None or topology.connectivity != connectivity:
                raise ValueError(f'{run["id"]}: requested {connectivity} topology was not constructed')
            graph_prep = replace(prep, topology=topology)
            for mode, tau in [('tau1', 1.), ('tau100', 100.), ('lcurve', None)]:
                name = f'{stencil}_{mode}'
                checkpoint(name)
                pool = NeighborPoolConfig(tau=tau, tau_mode='lcurve', grid_connectivity=connectivity,
                                          chebyshev_radius=radius, periodic_cv_names=tuple(run.get('periodic_cvs', [])))
                outcome = pool_engine.aggregate(graph_prep, config=ReweightingConfig(methods=('neighbor_pool',), neighbor_pool=pool))
                directory = root / 'comparisons' / name
                directory.mkdir(parents=True)
                (directory / 'raw').symlink_to(root / 'raw', target_is_directory=True)
                (directory / 'treated').mkdir()
                (directory / 'treated/reweighting_independent').symlink_to(root / 'filtered_psis', target_is_directory=True)
                write_outcome(outcome, directory / 'treated/reweighting_neighbor_pool', len(frames))
                settings = {**PROTOCOL, 'connectivity': connectivity, 'radius': radius, 'tau': tau, 'tau_mode': 'lcurve'}
                analysis_run = {**run, 'output_dir': str(directory), 'label': f'{run.get("label", run["id"])} · {name}',
                                'plot_methods': ['raw', 'independent', 'neighbor_pool'] if name == 'cardinal_tau1' else ['neighbor_pool']}
                analyze_run(analysis_run, settings)
                plot_run(analysis_run, settings)
                status['comparisons'][name] = {'status': 'complete', 'output_dir': str(directory),
                    'selected_tau': outcome.shared_metadata.get('neighbor_pool', {}).get('tau'),
                    'global_k': status['global_k'], 'global_k_pass': status['global_k_pass'],
                    'excluded_ids_sha256': status['excluded_ids_sha256']}
                del outcome
                gc.collect()
        status['status'] = 'complete'
        status['completed_utc'] = now()
        checkpoint('complete')
        return status
    except Exception as exc:
        status.update(status='failed', error=str(exc), failed_utc=now())
        write_json(record, status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--prepare', type=Path)
    choice.add_argument('--spec', type=Path)
    parser.add_argument('--project-root', type=Path, help='Root for relative manifest input paths; required for --prepare')
    parser.add_argument('--manifest', type=Path, help='Optional portable system manifest')
    parser.add_argument('--system', action='append', default=[], help='Select a system ID; repeat as needed')
    parser.add_argument('--python', type=Path, help='Interpreter for generated tasks; defaults to this interpreter')
    parser.add_argument('--host', action='append', default=[], help='Optional round-robin task label; does not submit jobs')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        if args.project_root is None:
            parser.error('--prepare requires --project-root')
        prepare(args.prepare.resolve(), args.project_root.resolve(), args.workers,
                manifest=args.manifest, systems=args.system, python=args.python, hosts=args.host)
    else:
        run_system(json.loads(args.spec.read_text()), resume=args.resume)


if __name__ == '__main__':
    main()
