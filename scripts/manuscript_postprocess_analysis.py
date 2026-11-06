"""Common diagnostics, basin tables, and state-probability figures for the paper.

Consumes summarize outputs, never structures or new energy evaluations. Plots
use the exact tabulated state probabilities rather than rebuilding weights.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

KB = 0.00831446261815324  # kJ mol^-1 K^-1


def load_table(path):
    d = pd.read_csv(path, dtype={'state_id':str}).set_index('state_id')
    if d.index.has_duplicates:
        raise ValueError(f'Duplicate state IDs: {path}')
    return d


def method_paths(run, settings):
    root = Path(run['output_dir'])
    result = [('raw', root/'raw/microstate_summary.csv')]
    for method in settings['methods']:
        base = root/'treated'
        if len(settings['methods']) > 1:
            base = base/f'reweighting_{method}'
        result.append((method,base/'microstate_summary.csv'))
    return result




def reporting_mask(table, settings):
    """Gate estimates using retained raw-score k; unavailable fits never pass.

    This masks reporting only. Full probabilities retain their normalization
    and are kept in the summary and diagnostic tables for audit.
    """
    cutoff = settings['qc_k_threshold']
    scope = settings.get('qc_reporting_scope', 'global')
    if scope == 'none':
        return np.ones(len(table), dtype=bool)
    if scope == 'global':
        values = [((json.loads(value).get('reweighting_qc') or {}).get('global_qc') or {})
                  for value in table.metadata]
        k = next((v.get('final_pareto_k') for v in values if v.get('final_pareto_k') is not None), None)
        passed = k is not None and np.isfinite(k) and k < cutoff
        return np.full(len(table), passed, dtype=bool)
    raise ValueError("qc_reporting_scope must be global or none; local k does not mask states.")


def analyze_run(run, settings):
    root = Path(run['output_dir'])
    raw = load_table(root/'raw/microstate_summary.csv')
    p = raw.reference_probability.to_numpy(float)
    original_metadata = [json.loads(value) for value in raw.metadata]
    n = int(raw.frame_count.sum())
    beta = 1/(KB*run['temperature_kelvin'])
    np.testing.assert_allclose(p.sum(),1,atol=1e-10)
    if run['prior_mode'] == 'raw_counts':
        np.testing.assert_allclose(p,raw.frame_count.to_numpy()/n,atol=1e-10)
    results, basins, state_rows = [], [], []
    target = None
    if run.get('target_probability_csv'):
        target = load_table(run['target_probability_csv']).target_probability
        if not np.isfinite(target).all() or (target < 0).any() or not np.isclose(target.sum(),1):
            raise ValueError('Target probability table must retain full normalization on the same declared state partition')
    for method,path in method_paths(run,settings):
        table = load_table(path)
        if set(table.index) != set(raw.index):
            raise ValueError('Raw and treated summaries use different state IDs')
        table = table.reindex(raw.index)
        q = table.target_probability.to_numpy(float)
        if not np.isfinite(q).all() or (q < 0).any() or not np.isclose(q.sum(),1):
            raise ValueError('Invalid normalized state probabilities')
        if np.any(q[p == 0] > 0):
            raise ValueError('Prediction assigns mass outside original state support')
        np.testing.assert_allclose(table.reference_probability,p,atol=1e-10)
        ess = table.effective_sample_size.to_numpy(float)
        metadata = [json.loads(value) for value in table.metadata]
        mask_settings = {**settings, 'qc_reporting_scope': 'none'} if method == 'raw' and settings.get('qc_reporting_scope') == 'global' else settings
        reportable = reporting_mask(table, mask_settings)
        remaining_flags = np.array([bool(m.get('state_qc_reasons', [])) for m in metadata])
        unavailable_k = np.array([m.get('pareto_k') is None for m in metadata])
        meaningful = q > 0
        implied_ess = (float(1/np.sum(q[meaningful]**2/ess[meaningful]))
                       if np.all(ess[meaningful] > 0) else None)
        retained = table.frame_count.to_numpy(float)
        surviving = retained > 0
        row = dict(system=run['id'], group=run['group'], method=method,
                   temperature_kelvin=run['temperature_kelvin'], original_frames=n,
                   reference_prior_mode=run['prior_mode'],
                   retained_frames=int(retained.sum()), excluded_frames=n-int(retained.sum()),
                   implied_frame_ess=implied_ess, predicted_mass_local_ess_lt10=float(q[ess < 10].sum()),
                   unresolved_qc_states=int(remaining_flags.sum()),
                   predicted_mass_unresolved_qc=float(q[remaining_flags].sum()),
                   predicted_mass_k_unavailable=float(q[unavailable_k].sum()),
                   reporting_k_max=settings['qc_k_threshold'],
                   reporting_k_scope=mask_settings.get('qc_reporting_scope', 'global'),
                   reportable_states=int(reportable.sum()),
                   reported_target_mass=float(q[reportable].sum()),
                   withheld_target_mass=float(q[~reportable].sum()),
                   emptied_original_reference_mass=float(p[~surviving].sum()),
                   tv_change_from_raw=float(np.abs(q-raw.target_probability.to_numpy()).sum()/2))
        if target is not None:
            predicted = pd.Series(q,index=raw.index)
            union = predicted.index.union(target.index)
            row['target_md_tv'] = float(np.abs(predicted.reindex(union,fill_value=0)-target.reindex(union,fill_value=0)).sum()/2)
            row['target_md_missing_mass'] = float(target[~target.index.isin(raw.index[p > 0])].sum())
        for index, (state_id, m) in enumerate(zip(table.index, metadata)):
            # State QC inside a globally filtered run has already lost the
            # excluded scores. The separate raw baseline is the original fit.
            original = original_metadata[index]
            state_rows.append(dict(system=run['id'], method=method, state_id=state_id,
                reference_probability=p[index], target_probability=q[index],
                raw_target_probability=raw.target_probability.iloc[index],
                original_frames=int(raw.frame_count.iloc[index]), retained_frames=int(retained[index]),
                raw_local_ess=raw.effective_sample_size.iloc[index], processed_local_ess=ess[index],
                original_k=original.get('pareto_k'), retained_raw_k=m.get('pareto_k'),
                k_threshold=m.get('pareto_k_threshold'), k_estimator=m.get('pareto_estimator'),
                reportable=bool(reportable[index]),
                raw_retained_max_weight=m.get('state_max_normalized_weight'),
                unresolved_qc=bool(remaining_flags[index]),
                qc_reasons='; '.join(m.get('state_qc_reasons', [])),
                psis_fit_status=m.get('psis_weight_smoothing', {}).get('psis_fit_status',
                    m.get('psis_weight_smoothing', {}).get('psis_pareto_fit_status'))))
        meta = metadata[0]
        row.update(meta.get('manuscript_diagnostics', {}))
        pool = meta.get('neighbor_pool',{}) or {}
        row.update(weight_treatment=meta.get('weight_treatment', 'raw' if method=='raw' else settings['weight_treatment']),
                   selected_tau=pool.get('tau'), tau_mode=pool.get('tau_mode'),
                   qc_method=settings['qc_method'],
                   qc_policy='none' if method == 'raw' else settings['qc_policy'])
        results.append(row)
        for basin in run.get('basins',[]):
            inside = np.ones(len(table),dtype=bool)
            for column, bounds in basin['ranges'].items():
                if column not in {'plot_x','plot_y'}:
                    raise ValueError('Basin ranges must use plot_x/plot_y from the declared state table')
                values = table[column].to_numpy(float)
                lo,hi = bounds
                inside &= (values >= lo) & (values < hi) if lo <= hi else (values >= lo) | (values < hi)
            probability = float(q[inside].sum())
            dg = -np.log(probability/(1-probability))/beta/4.184 if 0 < probability < 1 else None
            basins.append(dict(system=run['id'],method=method,basin=basin['name'],
                               reference_probability=float(p[inside].sum()),target_probability=probability,
                               delta_g_basin_vs_rest_kcal_mol=dg, membership='state centers; lower-inclusive upper-exclusive'))
    pd.DataFrame(results).to_csv(root/'diagnostics.csv',index=False)
    pd.DataFrame(state_rows).to_csv(root/'state_diagnostics.csv',index=False)
    pd.DataFrame(basins,columns=['system','method','basin','reference_probability','target_probability','delta_g_basin_vs_rest_kcal_mol','membership']).to_csv(root/'basin_populations.csv',index=False)


def plot_run(run, settings):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
                         'axes.linewidth':.7,'pdf.fonttype':42,'svg.fonttype':'none'})
    root = Path(run['output_dir'])
    raw = load_table(root/'raw/microstate_summary.csv')
    directory = root/'figures';directory.mkdir(exist_ok=True)
    labels = run.get('axis_labels',['State coordinate'])
    kbt = KB*run['temperature_kelvin']/4.184
    def free_energy(prob):
        out = np.full(len(prob),np.nan)
        positive = prob > 0
        out[positive] = -kbt*np.log(prob[positive]);out -= np.nanmin(out)
        return out
    for method,path in method_paths(run,settings):
        if run.get('plot_methods') is not None and method not in run['plot_methods']:
            continue
        table = load_table(path).reindex(raw.index)
        p,q = raw.reference_probability.to_numpy(float),table.target_probability.to_numpy(float)
        mask_settings = {**settings, 'qc_reporting_scope': 'none'} if method == 'raw' and settings.get('qc_reporting_scope') == 'global' else settings
        reportable = reporting_mask(table, mask_settings)
        reported_mass = float(q[reportable].sum())
        # Mask after normalization. Keep gaps and the full-distribution energy
        # zero, rather than inflating the displayed mass by renormalizing it.
        full_q_free_energy = free_energy(q)
        q = np.where(reportable, q, np.nan)
        x = table.plot_x.to_numpy(float)
        if not np.isfinite(x).all():
            raise ValueError(f"{run['id']}: declared state plot_x is unavailable")
        fig,axes = plt.subplots(1,3,figsize=(7.15,2.75),layout='constrained')
        if len(labels) == 1:
            order = np.argsort(x)
            for v,color,label in [(p,'#7050A0','Reference'),(q,'#C05050','FELP')]:
                axes[0].plot(x[order],v[order],color=color,lw=1,label=label)
                energy = free_energy(v) if label == 'Reference' else np.where(reportable, full_q_free_energy, np.nan)
                axes[1].plot(x[order],energy[order],color=color,lw=1)
            axes[0].set_ylabel('State probability');axes[0].legend(frameon=False,fontsize=7)
            axes[1].set_ylabel(r'Relative free energy (kcal mol$^{-1}$)')
            if 'fel_max_kcal_mol' in run or run.get('group') == 'egfr':
                axes[1].set_ylim(0, run.get('fel_max_kcal_mol', 10))
            axes[2].plot(x[order],(q-p)[order],color='#537F80',lw=1)
            axes[2].axhline(0,color='.5',lw=.6);axes[2].set_ylabel('Probability change')
            titles=['State populations','Free-energy profiles','FELP − reference']
        else:
            y = table.plot_y.to_numpy(float)
            if not np.isfinite(y).all():
                raise ValueError(f"{run['id']}: declared state plot_y is unavailable")
            bound = max(float(np.nanmax(np.abs(q-p))) if reportable.any() else 0.,1e-12)
            grid_x, grid_y = np.unique(x), np.unique(y)
            rectangular = len(grid_x)*len(grid_y) == len(x) and len(set(zip(x,y))) == len(x)
            for j,v in enumerate([free_energy(p),np.where(reportable, full_q_free_energy, np.nan),q-p]):
                kwargs = dict(cmap='gist_earth',vmin=0,vmax=run.get('fel_max_kcal_mol',6)) if j < 2 else dict(cmap='RdBu_r',vmin=-bound,vmax=bound)
                if rectangular and len(grid_x) > 1 and len(grid_y) > 1:
                    values = pd.DataFrame({'x':x,'y':y,'value':v}).pivot(index='y',columns='x',values='value')
                    scatter = axes[j].pcolormesh(grid_x,grid_y,values.to_numpy(),shading='nearest',**kwargs)
                else:
                    scatter = axes[j].scatter(x,y,c=v,s=8,marker='s',linewidths=0,**kwargs)
                fig.colorbar(scatter,ax=axes[j],fraction=.045,pad=.025,label=r'kcal mol$^{-1}$' if j < 2 else r'$\Delta p$')
                axes[j].set_ylabel(labels[1])
            titles=['Reference landscape','FELP landscape','FELP − reference']
        for index,ax in enumerate(axes):
            ax.set_xlabel(labels[0]);ax.set_title(titles[index],fontsize=8)
            ax.text(-.16,1.12,chr(65+index),transform=ax.transAxes,fontsize=11,weight='bold')
        scope = mask_settings.get('qc_reporting_scope', 'global')
        gate_label = ('Unfiltered raw diagnostic' if scope == 'none' else
                      f"{scope.capitalize()} retained raw-score k < {settings['qc_k_threshold']:g}")
        fig.suptitle(f"{run.get('label',run['id'])} · {method.replace('_',' ')}\n"
                     f"{gate_label}; displayed target mass {reported_mass:.1%}",fontsize=9)
        for ext in ['pdf','svg','png']:
            fig.savefig(directory/f"{run['id']}_{method}.{ext}",dpi=300)
        plt.close(fig)
        if method != 'raw':
            plot_quality_comparison(run, method, root, directory, plt)
        meta = json.loads(table.metadata.iloc[0])
        pool = meta.get('neighbor_pool',{}) or {}
        scan = pool.get('lcurve_scan',[])
        if method == 'neighbor_pool' and scan:
            scan = pd.DataFrame(scan)
            scan.to_csv(root/'neighbor_pool_lcurve.csv',index=False)
            f,ax = plt.subplots(figsize=(3.5,2.8),layout='constrained')
            ax.loglog(scan.misfit,scan.roughness,'o-',color='#537F80',ms=3,lw=.8)
            corner = pool.get('lcurve_corner_index')
            if corner is not None:
                row = scan.iloc[int(corner)]
                ax.plot(row.misfit,row.roughness,'o',color='#C05050',ms=5)
            ax.set(xlabel='Log-factor misfit',ylabel='Graph roughness',title=f"Selected τ = {pool.get('tau',float('nan')):.3g}")
            for ext in ['pdf','svg','png']:
                f.savefig(directory/f"{run['id']}_neighbor_pool_lcurve.{ext}",dpi=300)
            plt.close(f)


def plot_quality_comparison(run, method, root, directory, plt):
    """Compare diagnostics without refitting the deliberately smoothed tail."""
    states = pd.read_csv(root/'state_diagnostics.csv')
    states = states[(states.method == method) & (states.original_frames > 0)]
    removed = states.retained_frames < states.original_frames
    colors = np.where(removed, '#C05050', '#537F80')
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.75), layout='constrained')
    panels = [('raw_local_ess', 'processed_local_ess', 'Raw local ESS', 'Processed local ESS', True),
              ('original_k', 'retained_raw_k', r'Original score tail $\hat{k}$', r'Retained score tail $\hat{k}$', False),
              ('raw_target_probability', 'target_probability', 'Raw FELP probability', 'Processed FELP probability', True)]
    for i, (xcol, ycol, xlabel, ylabel, logarithmic) in enumerate(panels):
        ax = axes[i]
        x, y = states[xcol].to_numpy(float), states[ycol].to_numpy(float)
        valid = np.isfinite(x) & np.isfinite(y)
        if logarithmic:
            valid &= (x > 0) & (y > 0)
        if valid.any():
            ax.scatter(x[valid], y[valid], c=colors[valid], s=9, linewidths=0, alpha=.75)
            lo, hi = min(x[valid].min(), y[valid].min()), max(x[valid].max(), y[valid].max())
            if lo == hi:
                lo, hi = (lo/2, hi*2) if logarithmic else (lo-.1, hi+.1)
            ax.plot([lo, hi], [lo, hi], color='.5', lw=.65, ls='--', zorder=0)
            if logarithmic:
                ax.set_xscale('log'); ax.set_yscale('log')
        else:
            ax.text(.5, .5, 'No comparable finite values', ha='center', va='center', transform=ax.transAxes, fontsize=7)
        omitted = len(states) - int(valid.sum())
        if omitted:
            ax.text(.03, .97, f'{omitted} states omitted', transform=ax.transAxes, va='top', fontsize=6)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.text(-.16, 1.08, chr(65+i), transform=ax.transAxes, weight='bold', fontsize=11)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=c, marker='o', ls='', ms=4, label=t)
                        for c,t in [('#537F80','No frames removed'),('#C05050','Frames removed')]],
               loc='outside lower center', ncol=2, frameon=False, fontsize=7)
    fig.suptitle(f"{run.get('label',run['id'])} · {method.replace('_',' ')} · quality comparison", fontsize=9)
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(directory/f"{run['id']}_{method}_quality.{ext}", dpi=300)
    plt.close(fig)


def collect_results(plan):
    output = Path(plan['output_dir'])
    tables = [];basins=[]
    for run in plan['runs']:
        root = Path(run['output_dir'])
        if run['missing'] or not (root/'diagnostics.csv').exists():
            continue
        tables.append(pd.read_csv(root/'diagnostics.csv'))
        basins.append(pd.read_csv(root/'basin_populations.csv'))
    if tables:
        pd.concat(tables,ignore_index=True).to_csv(output/'all_system_diagnostics.csv',index=False)
        pd.concat(basins,ignore_index=True).to_csv(output/'all_basin_populations.csv',index=False)
