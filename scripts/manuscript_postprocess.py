#!/usr/bin/env python3
"""Rebuild paper summaries, diagnostics, and plots from existing FELP scores.

Never invokes run, mutation preparation, energy evaluation, or MD. Inputs and
state definitions live in a portable JSON manifest; all options are recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DEFAULT_MANIFEST = REPO/'scripts/manuscript_systems.json'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    p.add_argument('--project-root', type=Path, required=True, help='Root for manifest paths; original score files are read only.')
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--tag', required=True, help='Name such as cardinal_psis or radius2_tau100; a settings hash is appended.')
    p.add_argument('--system', action='append', default=[], help='Select run ID; repeat to select several.')
    p.add_argument('--group', action='append', default=[], help='Select dipeptide, chignolin, t4l, egfr, or a custom manifest group.')
    p.add_argument('--input', action='append', default=[], metavar='ID=CSV', help='Override a run input path.')
    p.add_argument('--atlas', action='append', default=[], metavar='ID=JSON', help='Override a run state file; reassign existing measured CVs.')
    p.add_argument('--methods', nargs='+', choices=['independent', 'neighbor_pool'], default=['independent', 'neighbor_pool'])
    p.add_argument('--weight-treatment', choices=['raw', 'gpd-psis'], default='gpd-psis', help="Within-state weight treatment; gpd-psis uses ArviZ 0.22.0.")
    p.add_argument('--qc-policy', choices=['none', 'flag', 'exclude-frames'], default='exclude-frames')
    p.add_argument('--qc-method', choices=['gpd'], default='gpd', help='ArviZ generalized-Pareto diagnostic.')
    p.add_argument('--qc-k-threshold', type=float, default=.5, help='Strict global Pareto-k reporting cutoff.')
    p.add_argument('--qc-exclusion-scope', choices=['global'], default='global', help='Global score-tail filtering precedes statewise PSIS.')
    p.add_argument('--qc-min-frames', type=int, default=20)
    p.add_argument('--tau', type=float, help='Fixed pooling strength; omit for automatic selection.')
    p.add_argument('--tau-mode', choices=['lcurve'], default='lcurve')
    p.add_argument('--solver', choices=['blend'], default='blend')
    p.add_argument('--connectivity', choices=['cardinal', 'chebyshev'], default=None)
    p.add_argument('--radius', type=int, default=3)
    p.add_argument('--max-neighbors', type=int, default=4)
    p.add_argument('--temperature', type=float, help='Override the temperature declared per run.')
    p.add_argument('--prior-mode', choices=['raw_counts','loaded'], help='Override manifest prior_mode; loaded requires probabilities in the saved atlas.')
    p.add_argument('--phi-bins', type=int, help='Override the dipeptide state partition; creates a separate configuration.')
    p.add_argument('--psi-bins', type=int)
    p.add_argument('--workers', type=int, default=1, help='Summarize threads; systems are processed sequentially to bound memory.')
    p.add_argument('--from-stage', choices=['summarize', 'analyze', 'plot'], default='summarize')
    p.add_argument('--resume', action='store_true', help='Reuse completed runs only when inputs, code, and settings match.')
    p.add_argument('--dry-run', action='store_true', help='Print resolved inputs and exact summarize commands without executing.')
    p.add_argument('--skip-missing', action='store_true', help='Explicitly skip missing inputs; record them in the batch report.')
    p.add_argument('--plan-file', type=Path, help='Optionally save the reviewable plan as JSON.')
    args = p.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.tag):
        p.error('--tag must be a filename-safe name beginning with a letter or number')
    if args.workers < 1 or args.radius < 1 or args.max_neighbors < 1:
        p.error('workers, radius, and max-neighbors must be positive')
    if args.qc_min_frames < 20:
        p.error('qc-min-frames must be at least 20')
    if args.tau is not None and not 0 <= args.tau < float('inf'):
        p.error('tau must be finite and nonnegative')
    if args.temperature is not None and args.temperature <= 0:
        p.error('temperature must be positive')
    if args.qc_k_threshold is not None and not 0 < args.qc_k_threshold < float('inf'):
        p.error('QC k threshold must be finite and positive')
    if any(v is not None and v < 1 for v in [args.phi_bins, args.psi_bins]):
        p.error('bin counts must be positive')
    return args


def overrides(items):
    result = {}
    for value in items:
        name, sep, path = value.partition('=')
        if not sep or not name or not path or name in result:
            raise ValueError(f'Expected a unique ID=PATH override, got {value!r}')
        result[name] = str(Path(path).expanduser().resolve())
    return result


def resolved_path(value, root):
    if not value:
        return None
    p = Path(str(value).format(project_root=str(root))).expanduser()
    return str((p if p.is_absolute() else root/p).resolve())


def build_plan(args):
    manifest = json.loads(args.manifest.read_text())
    if manifest.get('schema_version') != 1:
        raise ValueError('Expected manifest schema_version=1')
    inputs, atlases = overrides(args.input), overrides(args.atlas)
    ids = [r['id'] for r in manifest['runs']]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', i) for i in ids):
        raise ValueError('Manifest run IDs must be unique filename-safe names')
    unknown = (set(args.system) | set(inputs) | set(atlases)) - set(ids)
    if unknown:
        raise ValueError(f'Unknown run IDs: {sorted(unknown)}')
    unknown_groups = set(args.group) - {r['group'] for r in manifest['runs']}
    if unknown_groups:
        raise ValueError(f'Unknown groups: {sorted(unknown_groups)}')
    settings_keys = ['methods','weight_treatment','qc_policy','qc_method','qc_k_threshold','qc_exclusion_scope','qc_min_frames','tau','tau_mode','solver','connectivity','radius','max_neighbors','temperature','phi_bins','psi_bins','prior_mode']
    settings = {k:getattr(args,k) for k in settings_keys}
    settings["qc_reporting_scope"] = "global"
    runs = []
    root = args.project_root.expanduser().resolve()
    for original in manifest['runs']:
        if args.system and original['id'] not in args.system or args.group and original['group'] not in args.group:
            continue
        r = json.loads(json.dumps(original))
        r['input_csv'] = inputs.get(r['id']) or resolved_path(r.get('input_csv'), root)
        state = r['states']
        if r['id'] in atlases:
            state = r['states'] = {'source':'file', 'file':atlases[r['id']]}
        if state['source'] == 'file':
            state['file'] = resolved_path(state.get('file'), root)
        elif state['source'] == 'grid':
            state['phi_bins'] = args.phi_bins or state['phi_bins']
            state['psi_bins'] = args.psi_bins or state['psi_bins']
        else:
            raise ValueError(f"{r['id']}: use a declared grid or saved state file; no automatic repartition across mutations")
        r['temperature_kelvin'] = args.temperature or r.get('temperature_kelvin', 310)
        r['prior_mode'] = args.prior_mode or r.get('prior_mode','raw_counts')
        if r['prior_mode'] == 'loaded' and state['source'] != 'file':
            raise ValueError('Loaded priors require a saved state file with probabilities')
        r['target_probability_csv'] = resolved_path(r.get('target_probability_csv'), root)
        required = {'original frame results':r['input_csv']}
        if state['source'] == 'file':
            required['state definitions'] = state.get('file')
        if r['target_probability_csv']:
            required['target probability table'] = r['target_probability_csv']
        r['missing'] = [f'{kind}: {path or "UNCONFIGURED"}' for kind,path in required.items() if not path or not Path(path).is_file()]
        runs.append(r)
    if not runs:
        raise ValueError('No runs selected')
    fingerprint = hashlib.sha256(json.dumps({'settings':settings,'runs':runs},sort_keys=True).encode()).hexdigest()[:12]
    directory = (args.output_root.expanduser().resolve()/f'{args.tag}_{fingerprint}')
    for r in runs:
        dest = directory/r['id']
        # Summarize's overwrite/resume behavior must never touch original inputs.
        for source in [r['input_csv'], r['states'].get('file'), r['target_probability_csv']]:
            if source and (dest == Path(source) or dest in Path(source).parents):
                raise ValueError('Output directory would contain a source input')
        r['output_dir'] = str(dest)
        r['commands'] = [summarize_command(args, r, dest/'raw', raw=True), summarize_command(args, r, dest/'treated', raw=False)]
    return {'schema_version':1, 'settings':settings, 'output_dir':str(directory), 'runs':runs}


def pool_connectivity(run, override):
    """Choose the paper's stencil from the atlas dimensions."""
    if override is not None:
        return override
    from felp.core import MicrostateConfig
    from felp.diagnostics.harmonic import grid_from_atlas
    from felp.reweighting import MicrostateAtlas

    state = run['states']
    if state['source'] == 'file' and not Path(state['file']).is_file():
        return 'UNCONFIGURED'  # Missing inputs remain visible in dry-run plans.
    atlas = MicrostateAtlas.from_config(MicrostateConfig(
        source=state['source'], definition_file=state.get('file'),
        phi_bins=state.get('phi_bins', 45), psi_bins=state.get('psi_bins', 45),
        reference_probability_mode=run['prior_mode'],
    ))
    return 'chebyshev' if len(grid_from_atlas(atlas).shape) == 2 else 'cardinal'


def summarize_command(args, run, dest, *, raw):
    cmd = [sys.executable, str(REPO/'felp.py'), 'summarize', '--raw-frame-results', run['input_csv'] or 'UNCONFIGURED',
           '--output-dir', str(dest), '--temperature-kelvin', str(run['temperature_kelvin']),
           '--prior-probability-mode', run['prior_mode'], '--repartition', '--no-write-repartitioned-frames',
           '--workers', str(args.workers)]
    state = run['states']
    cmd += ['--microstate-source', state['source']]
    if state['source'] == 'grid':
        cmd += ['--phi-bins', str(state['phi_bins']), '--psi-bins', str(state['psi_bins'])]
    else:
        cmd += ['--microstate-file', state.get('file') or 'UNCONFIGURED']
    cmd += ['--reweighting-methods', *(['independent'] if raw else args.methods),
            '--reweighting-weight-treatment', 'raw' if raw else args.weight_treatment,
            '--reweighting-qc-policy', 'none' if raw else args.qc_policy,
            '--reweighting-qc-method', args.qc_method,
            '--reweighting-qc-exclusion-scope', args.qc_exclusion_scope,
            '--reweighting-qc-exclusion-strategy', 'global-gpd-k',
            '--reweighting-qc-min-frame-count', str(args.qc_min_frames),
            '--neighbor-pool-solver', args.solver, '--neighbor-pool-tau-mode', args.tau_mode,
            '--neighbor-pool-grid-connectivity', pool_connectivity(run, args.connectivity),
            '--neighbor-pool-chebyshev-radius', str(args.radius),
            '--neighbor-pool-max-neighbors', str(args.max_neighbors)]
    if args.tau is not None:
        cmd += ['--neighbor-pool-tau', str(args.tau)]
    if args.qc_k_threshold is not None:
        cmd += ['--reweighting-qc-k-threshold', str(args.qc_k_threshold)]
    if run.get('periodic_cvs'):
        cmd += ['--neighbor-pool-periodic-cvs', *run['periodic_cvs']]
    return cmd


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def execute(plan, args):
    # Check optional/binary dependencies before reading large input datasets.
    import matplotlib
    if args.weight_treatment == 'gpd-psis' or args.qc_method == 'gpd':
        import arviz
    from scripts.manuscript_postprocess_analysis import analyze_run, plot_run, collect_results
    missing = [r['id'] for r in plan['runs'] if r['missing']]
    if missing and not args.skip_missing:
        raise ValueError('Missing inputs for '+', '.join(missing)+'. Inspect --dry-run; configure paths or explicitly --skip-missing.')
    if not any(not r['missing'] for r in plan['runs']):
        raise ValueError('No runnable systems; configure original input paths first')
    output = Path(plan['output_dir'])
    code_files = sorted((REPO/'felp').rglob('*.py')) + [Path(__file__), REPO/'scripts/manuscript_postprocess_analysis.py']
    code_hash = hashlib.sha256(''.join(digest(p) for p in code_files).encode()).hexdigest()
    write_json(output/'plan.json', plan)
    batch = {'started_utc':datetime.now(timezone.utc).isoformat(), 'code_sha256':code_hash, 'runs':[]}
    for run in plan['runs']:
        if run['missing']:
            batch['runs'].append({'id':run['id'], 'status':'missing', 'details':run['missing']})
            continue
        dest = Path(run['output_dir'])
        paths = [run['input_csv'],run['states'].get('file'),run.get('target_probability_csv')]
        signatures = {p:digest(Path(p)) for p in paths if p}
        packages = {}
        for package in ['numpy','scipy','pandas','matplotlib','arviz','xarray']:
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                packages[package] = None
        expected = {'inputs_sha256':signatures, 'code_sha256':code_hash, 'packages':packages,
                    'environment':{k:os.environ.get(k) for k in ['LD_LIBRARY_PATH','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_DISABLE_JIT']},
                    'python':sys.version, 'settings':plan['settings'], 'run':run}
        record = dest/'provenance.json'
        if dest.exists():
            if not (args.resume or args.from_stage != 'summarize') or not record.exists():
                raise ValueError(f'{dest} already exists; use a new tag or --resume for matching completed outputs')
            old = json.loads(record.read_text())
            if old['configuration'] != expected:
                raise ValueError(f'{dest}: inputs, code, or settings changed; use a new tag')
        else:
            if args.from_stage != 'summarize':
                raise ValueError(f'{dest}: no completed summaries for --from-stage')
            dest.mkdir(parents=True)
            old = {'configuration':expected, 'completed_commands':[], 'status':'started'}
            write_json(record, old)
        print(f"[{run['id']}] {dest}", flush=True)
        if args.from_stage == 'summarize':
            for index, cmd in enumerate(run['commands']):
                if index in old['completed_commands']:
                    continue
                stage_dir = dest/('raw' if index == 0 else 'treated')
                if stage_dir.exists():
                    raise ValueError(f'Incomplete stage at {stage_dir}; preserve it and use a new tag')
                with (dest/f'summarize_{index}.log').open('w') as log:
                    proc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
                if proc.returncode:
                    old['status']='failed';write_json(record,old)
                    raise RuntimeError(f"{run['id']} summarize failed; see {dest/f'summarize_{index}.log'}")
                old['completed_commands'].append(index);write_json(record,old)
                if index == 0 and run.get('expected_frames') is not None:
                    summary = json.loads((stage_dir/'run_summary.json').read_text())
                    if summary['frame_count'] != run['expected_frames']:
                        raise ValueError(f"{run['id']}: expected {run['expected_frames']} original frames, found {summary['frame_count']}; verify the dataset version")
        if len(old['completed_commands']) != 2:
            raise ValueError(f'{dest}: both raw and treated summaries must be complete')
        if run.get('expected_frames') is not None:
            actual = json.loads((dest/'raw/run_summary.json').read_text())['frame_count']
            if actual != run['expected_frames']:
                raise ValueError(f"{run['id']}: expected {run['expected_frames']} frames, found {actual}")
        if args.from_stage != 'plot':
            analyze_run(run, plan['settings'])
        plot_run(run, plan['settings'])
        old['status']='complete';old['completed_utc']=datetime.now(timezone.utc).isoformat();write_json(record,old)
        batch['runs'].append({'id':run['id'],'status':'complete'})
        write_json(output/'batch.json',batch)
    collect_results(plan)
    write_json(output/'batch.json',batch)
    print(f'Completed outputs: {output}',flush=True)


def main(argv=None):
    args = parse_args(argv)
    try:
        plan = build_plan(args)
        if args.plan_file:
            write_json(args.plan_file, plan)
        if args.dry_run:
            print(f"Output: {plan['output_dir']}")
            for run in plan['runs']:
                print(f"{run['id']}: {'MISSING '+ '; '.join(run['missing']) if run['missing'] else 'ready'}")
                for cmd in run['commands']:
                    print('  '+shlex.join(cmd))
            return 0
        execute(plan,args)
    except (ValueError, OSError, RuntimeError, KeyError, ImportError) as exc:
        print(f'Postprocessing error: {exc}',file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
