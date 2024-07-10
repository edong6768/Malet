import os
import re
import yaml
from functools import partial
from itertools import product

from absl import app, flags

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style, lines, colors, cm
import seaborn as sns

from .experiment import Experiment, ExperimentLog
from .utils import str2value, df2richtable

from rich import print
from rich.panel import Panel
from rich.columns import Columns
from rich.align import Align

from .plot_utils.data_processor import avgbest_df, select_df, homogenize_df
from .plot_utils.plot_drawer import ax_draw_curve, ax_draw_best_stared_curve, ax_draw_bar, ax_draw_heatmap, ax_draw_scatter, ax_draw_scatter_heat
from .plot_utils.utils import merge_dict, default_style, ax_styler, save_figure

FLAGS = flags.FLAGS


def get_plot_config(plot_config: dict, plot_args: dict):
    assert plot_args['mode'] in plot_config, f'Mode: {plot_args["mode"]} does not exist.'
    alias_mode = ('-' not in plot_args['mode'])
    
    p_cfg = plot_config[plot_args['mode']]
    if alias_mode:
        p_cfg_base = plot_config.get(p_cfg['mode'], dict())
        p_cfg_base = merge_dict(p_cfg_base, plot_args)
        p_cfg_base = merge_dict(p_cfg_base, plot_config['default_style'])
        return merge_dict(p_cfg, p_cfg_base)
    else:
        return {**plot_args, **p_cfg}
    
def __save_name_builder(pflt, pmlf, pcrf, pcfg, save_name=''):
    sn = []
    if pmlf: sn.append(f"mlf({'-'.join(pmlf)})")
    if pcrf: sn.append(f"crf({'-'.join(pcrf)})")
    if pflt: sn.append(f"flt({'-'.join(['_'.join([k, *v]) for k, v in pflt.items()])})")
    if any([pcfg[f'best_ref_{k}'] for k in ['x_fields', 'metric_field', 'ml_fields']]):
        sn.append(f"bref({pcfg['best_ref_x_fields']}, {pcfg['best_ref_metric_field']}, {pcfg['best_ref_ml_fields']})")
    sn.append("-max" if pcfg['best_at_max'] else "-min")
    return save_name + '-'.join(sn)


def draw_metric(tsv_file, plot_config, save_name='', preprcs_df=lambda *x: x):
        '''
        Some rules
        step:
            - Step in [filter] and [best_ref_x_fields] (when step is in x_fields) can have special values {'last', 'best'}
            - if step is not in x_fields and filter, the last step is chosen
            - if step is in filter, best_ref_x_fields is automatically set to the last step
        metric:
            - (if metric is in multi_line_fields, best_ref_metric_field is automatically set to the best metric)
        '''
        pcfg = plot_config
        
        ############################# Preprocess plot mode #############################
        # Parse mode string
        mode, x_fields, metrics = pcfg['mode'].split('-') # ex) {sam}-{epoch}-{train_loss}
        x_fields = xs if ['']!=(xs:=x_fields.split(' ')) else []
        metrics = metrics.split(' ')
        
        pflt, pcrf, pmlf = map(pcfg.get, ['filter', 'col_row_fields', 'multi_line_fields'])
        pflt = {fk: fvs for fk, *fvs in map(lambda flt: re.split('(?<!,) ', flt.strip()), pflt.split('/')) if fk} # split ' ' except ', '
        
        #---Set default pmlf
        if not pmlf:
            pcfg['ax_style'].pop('legend', None)
        
        #---preprocess best_ref_x_fields and automatically set best_ref_x_field of step 
        pcfg['best_ref_x_fields'] = [*map(str2value, pcfg['best_ref_x_fields'])]
        if 'step' in x_fields and not pcfg['best_ref_x_fields']:
            pcfg['best_ref_x_fields'] = ['' for _ in x_fields]
            pcfg['best_ref_x_fields'][x_fields.index('step')] = 'last'
        
        # choose plot mode
        if mode in {'curve', 'curve_best', 'bar'}:
            assert len(x_fields)==1 , f'Number of x_fields shoud be {1} when using curve mode, but you passed {len(x_fields)}.'
            assert len(metrics)==1  , f'Number of metric shoud be {1} when using curve mode, but you passed {metrics}.'
            assert len(pmlf)<=3     , f'Number of multi_line_fields should be less than {3}, but you passed {len(pmlf)}'
            ax_draw = {'curve':      ax_draw_curve,
                       'curve_best': ax_draw_best_stared_curve,
                       'bar':        ax_draw_bar}[mode]
            ax_draw = partial(ax_draw,
                              annotate=pcfg['annotate'], 
                              annotate_field=pcfg['annotate_field'],
                              best_at_max=pcfg['best_at_max'])
            x_label, y_label = (f.replace('_', ' ').capitalize() for f in (x_fields[0],  metrics[0]))
            
        elif mode=='heatmap':
            assert len(x_fields)==2, f'Number of x_fields shoud be {2} when using heatmap mode, but you passed {len(x_fields)}.'
            assert len(metrics)==1, f'Number of metric shoud be {1} when using heatmap mode, but you passed {metrics}.'
            assert not pmlf, f'No multi_line_fields are allowed in heatmap mode, but you passed {pmlf}.'
            ax_draw = partial(ax_draw_heatmap,
                              annotate=pcfg['annotate'], 
                              annotate_field=pcfg['annotate_field'],
                              best_at_max=pcfg['best_at_max'])
            x_label, y_label = (f.replace('_', ' ').capitalize() for f in x_fields)
            
        elif mode=='scatter':
            assert not x_fields, f'No x_fields are allowed in scatter mode, but you passed {x_fields}.'
            assert len(metrics)==2, f'Number of metric shoud be {2} when using scatter mode, but you passed {metrics}.'
            assert len(pmlf)<=2, f'Number of multi_line_fields should be less than {2}, but you passed {len(pmlf)}'
            ax_draw = partial(ax_draw_scatter, y_fields=metrics)
            x_label, y_label = (f.replace('_', ' ').capitalize() for f in metrics)
            
        elif mode=='scatter_heat':
            assert not x_fields, f'No x_fields are allowed in scatter_heat mode, but you passed {x_fields}.'
            assert len(metrics)==3, f'Number of metric shoud be {3} when using scatter_heat mode, but you passed {metrics}.'
            assert len(pmlf)<=1, f'Number of multi_line_fields should be less than {1}, but you passed {len(pmlf)}'
            ax_draw = partial(ax_draw_scatter_heat, y_fields=metrics)
            x_label, y_label = (f.replace('_', ' ').capitalize() for f in metrics[:-1])
            
        # build save name
        save_name = __save_name_builder(pflt, pmlf, pcrf, pcfg, save_name=save_name)
        
        ############################# Get and filter dataframe #############################
        
        # get dataframe, drop unused metrics for efficient process
        log = ExperimentLog.from_tsv(tsv_file)
        
        post_melt_k = {'step', 'total_steps', 'metric'}
        assert all(x in log.df.index.names for x in x_fields if x not in post_melt_k), \
            f'X-field {[x for x in x_fields if x not in log.df.index.names]} not in log. Choose between {list(log.df.index.names)}'
        assert all(kk in log.df.index.names for k in pflt.keys() if (kk:=k[:-1] if '!' in k else k) not in post_melt_k), \
            f'Filter keys {[k for k in pflt.keys() if k not in log.df.index.names]} not in log. Choose between {list(log.df.index.names)}'
        assert all(k in log.df.index.names for k in pcrf if k not in post_melt_k), \
            f'Column-row fields {[k for k in pcrf if k not in log.df.index.names]} not in log. Choose between {list(log.df.index.names)}'
        assert all(k in log.df.index.names for k in pmlf if k not in post_melt_k), \
            f'Multi-line (style) fields {[k for k in pmlf if k not in log.df.index.names]} not in log. Choose between {list(log.df.index.names)}'
        assert all(m in log.df for m in metrics), \
            f'Metric {[m for m in metrics if m not in log.df]} not in log. Choose between {list(log.df)}'
        
        #--- initial filter for df according to FLAGS.filter (except step and metric)
        if pflt:
            log.df = select_df(log.df, {fk     :[*map(str2value, fvs)] for fk, fvs in pflt.items() if fk[-1]!='!' and fk      not in post_melt_k})
            log.df = select_df(log.df, {fk[:-1]:[*map(str2value, fvs)] for fk, fvs in pflt.items() if fk[-1]=='!' and fk[:-1] not in post_melt_k}, equal=False)
        
        #--- melt and explode metric in log.df
        if 'metric' not in x_fields+pmlf+pcrf:
            log.df = log.df.drop(list(set(log.df)-{*metrics, pcfg['best_ref_metric_field']}), axis=1)
        df = log.melt_and_explode_metric(step=-1 if (pflt.get('step', None if ('step' in x_fields) else 'last')=='last') else None)
        
        assert not df.empty, f'Metrics {metrics}' +\
            (f' and best_ref_metric_field {pcfg["best_ref_metric_field"]} are' if pcfg["best_ref_metric_field"] else ' is') +\
                f' NaN in given dataframe: \n{log.df}'
        
        #---filter df according to FLAGS.filter step and metrics
        if pflt:
            pflt = {k: v for k, v in pflt.items() if (k, v)!=('step', ['best'])}  # Let `avgbest_df` handle 'best' step, remove from pflt
            e_rng = lambda fvs: [*range(*map(int, fvs[0].split(':')))] if (len(fvs)==1 and ':' in fvs[0]) else fvs # CNG 'a:b' step filter later
            df = select_df(df, {fk     :[*map(str2value, e_rng(fvs))] for fk, fvs in pflt.items() if fk[-1]!='!' and fk      in post_melt_k})
            df = select_df(df, {fk[:-1]:[*map(str2value, e_rng(fvs))] for fk, fvs in pflt.items() if fk[-1]=='!' and fk[:-1] in post_melt_k}, equal=False)
    
    
        ############################# Prepare dataframe #############################
        
        specified_field = {k for k in {*df.index.names} if len({*df.index.get_level_values(k)})==1}
        if mode in {'scatter', 'scatter_heat'}:
            key_field       = {*df.index.names} - specified_field
            avg_field = optimized_field = set()
        else:
            key_field       = {*x_fields, *pmlf, *pcrf, 'metric'}  - specified_field
            avg_field       = {'seed'} - specified_field - key_field
            optimized_field = {*df.index.names} - specified_field - key_field - avg_field
            
        # Report selected plot configs and field handling statistics
        print('\n\n',
            Align(
                Columns(
                    [Panel('\n'.join([f'- {k}: {pcfg[k]}' 
                                            for k in ('mode', 'multi_line_fields', 
                                                        'filter', 'best_at_max', 
                                                        'best_ref_x_fields', 'best_ref_metric_field', 
                                                        'best_ref_ml_fields') if pcfg[k]]),
                                    title='Plot configuration', padding=(1, 3)),
                     Panel(f"- Key field (has multiple values): {[*key_field]} ({len(key_field)})\n" + \
                           f"- Specified field: {[*specified_field]} ({len(specified_field)})\n"+ \
                           f"- Averaged field: {[*avg_field]} ({len(avg_field)})\n" + \
                           f"- Optimized field: {[*optimized_field]} ({len(optimized_field)})",
                           title='Field handling statistics', padding=(1, 3))]
                    ), align='center'
                )
            )
        
        
        if mode in {'scatter', 'scatter_heat'}: # no processing
            best_df = df
        else:               # change field name and avg over seed and get best result over best_over
            best_of = {}
            if pcfg['best_ref_x_fields']:       # same hyperparameter over all points in line
                best_of.update(dict(zip(x_fields, pcfg['best_ref_x_fields'])))

            if pcfg['best_ref_metric_field']:   # Optimize in terms of reference metric, and apply those hyperparameters to original
                best_of['metric'] = pcfg['best_ref_metric_field']
            
            if pcfg['best_ref_ml_fields']:      # same hyperparameter over all line in multi_line_fields
                best_of.update(dict(zip(pmlf, pcfg['best_ref_ml_fields'])))
                
            # do best_of operation on 'step' and 'metric' seperatly after avgbest_df
            sm_bestof = {}
            if best_of.get('step') in {'last', 'best'}:
                sm_bestof['step'] = best_of.pop('step')
                if sm_bestof['step']=='best': # add step to best_over to compute optimal step
                    optimized_field |= {'step'}
            if 'metric' in best_of:
                sm_bestof['metric'] = best_of.pop('metric')
            
            # avgbest without 'step' and 'metric' in best_of
            best_df = avgbest_df(df, 'metric_value',
                                 avg_over=avg_field, 
                                 best_over=optimized_field, 
                                 best_of=best_of,
                                 best_at_max=pcfg['best_at_max'])
            
            # process 'step' and 'metric'
            if sm_bestof:
                avg_df = avgbest_df(df, 'metric_value', avg_over=avg_field)
                sm_df = best_df
                
                if 'step' in sm_bestof:
                    if sm_bestof['step']=='last':
                        sm_df = sm_df.loc[sm_df.index.get_level_values('step')==sm_df.index.get_level_values('total_steps')]
                        # remove duplicates over total_step (pick best performing, might change later)
                        # e.g., step 50 best for config with total_step 50, and likewise for step 100, pick step 100 if better than 50
                        sm_df = avgbest_df(sm_df, 'metric_value', best_over=optimized_field|{'step'})
                    sm_df = sm_df.reset_index(['step'], drop=True)
                if 'metric' in sm_bestof:
                    sm_df = select_df(sm_df, {'metric': sm_bestof['metric']}, drop=True)
                    
                # homogenize best_df with sm_df, exclude 'total_steps'
                best_df = homogenize_df(avg_df, sm_df, {}, 'total_steps')
        
        # check if there is any duplicate key_field configs in the best_df
        assert not best_df.reset_index(list({*best_df.index.names}-key_field)).index.duplicated().any(), \
            f"Duplicate values in found in dataframe: \n" + str(
                best_df.reset_index(best_df.index.names)
                       .groupby([*key_field])
                       .size()
                       .loc[lambda x: x>1]
            )
        
        ############################# Print best_df #############################
        
        show_field_order = (['metric'] if 'metric' not in specified_field else []) + \
                           sorted(key_field-{'metric', 'step', 'seed'}) + \
                           sorted(optimized_field-{'metric', 'step', 'seed'}) + \
                           (['seed'] if not avg_field else []) + \
                           (['step'] if 'step' not in specified_field else [])
        show_df = (
                best_df.reset_index()
                       .reindex(show_field_order+['metric_value'], axis=1)
                       .sort_values(by=show_field_order, ignore_index=True)
            )
        
        print(
            '\n', Align('Metric Summary Table', align='center'),
            Align(
                Columns(
                    [Panel('\n'.join([f'- {k:{max(map(len, specified_field))}s} : {best_df.index.get_level_values(k)[0]}' for k in sorted(specified_field)]), padding=(1, 3)),
                     df2richtable(show_df)]
                ), align='center'
            )
        )
        
        if 'metric' not in x_fields+pmlf+pcrf:
            best_df = select_df(best_df, {'metric': metrics})
        
        ############################# Plot #############################
        
        pford = pcfg['field_orders']
        pford = {fk: fvs for fk, *fvs in map(lambda od: re.split('(?<!,) ', od.strip()), pford.split('/')) if fk} # split ' ' except ', '
        assert not (nmtch:={k: (set(v), set(best_df.index.get_level_values(k))) for k, v in pford.items() if set(v)!=set(best_df.index.get_level_values(k))}), \
            f'Field order does not match with the dataframe: {nmtch} (field_order, dataframe)'
        
        get_field_values = lambda f: (
            ['']            if not f else
            pford[f]        if f in pford else
            sorted(set(best_df.index.get_level_values(f)), key=str2value)
        )
        
        # col-row and multi_line fields
        col_vs, row_vs = map(get_field_values, pcrf+['']*(2-len(pcrf)))
        mlines = [*product(*map(get_field_values, pmlf))]
        
        # scale axis size according to number of col and row plots
        if isinstance(fig_size:=pcfg['ax_style']['fig_size'], (float, int)): fig_size = [fig_size]*2
        pcfg['ax_style']['fig_size'] = [p*l for p, l in zip([len(col_vs), len(row_vs)], fig_size)]
        legend_style = pcfg['ax_style'].pop('legend', [{}])[0]
        
        # set style types per mode
        has_cbar = False
        if mode in {'curve', 'curve_best'}:
            style_types = ['color', 'linestyle', 'marker']
            # set unif_xticks for curve
            if pcfg['xscale']=='unif':
                pcfg['ax_style'].pop('xscale', None)
                pcfg['line_style']['unif_xticks'] = True
        elif mode=='bar':
            style_types = ['color']
        elif mode=='heatmap':
            style_types = []
            has_cbar = True
        elif mode=='scatter':
            style_types = ['color', 'marker']
        elif mode=='scatter_heat':
            style_types = ['marker']
            has_cbar = True
            
        if has_cbar:
            norm_df = best_df[best_df.index.get_level_values('metric')==metrics[-1]]
            if pcfg['ax_style'].pop('zscale', [{}])[0]=='log':
                pcfg['line_style']['norm'] = colors.LogNorm(norm_df['metric_value'].min(), norm_df['metric_value'].max())
            else:
                pcfg['line_style']['norm'] = colors.Normalize(norm_df['metric_value'].min(), norm_df['metric_value'].max())    
            pcfg['line_style']['cmap'] = 'magma' if pcfg['colors'][0]=='default' else pcfg['colors'][0]
            
        
        rep, skp, sft = map(int, pcfg['colors_rep_skip_shift'])
        style_dict = {'color': [c for i, c in enumerate(sum(map(sns.color_palette, [None if c=='default' else c for c in pcfg['colors']]), [])*rep) if not (i-sft)%(skp+1)],
                      'marker': ['D', 'o', '>', 'X', 's', 'v', '^', '<', 'p', 'P', '*', '+', 'x', 'h', 'H', '|', '_'],
                      'linestyle': ['-', ':', '-.', '--']*3}
        styles = [*product(*[[*style_dict[s]][:len(set(k))] for s, k in zip(style_types, map(df.index.get_level_values, pmlf))])]
        
        
        # prepare plot, set figure and axes for multiple plots, and leave place for cmap
        fig, axs = plt.subplots(len(row_vs), len(col_vs)+int(has_cbar), 
                                sharex=True, sharey=True,
                                gridspec_kw={'width_ratios': [1]*len(col_vs)+([0.1] if has_cbar else [])})
        
        if isinstance(axs, plt.Axes):
            axs = np.array([[axs]])
        for _ in range(2-len(axs.shape)):
            axs = axs[None]
        
        for ci, col_v in enumerate(col_vs):
            for ri, row_v in enumerate(row_vs[::-1]):
                ax = axs[ri, ci]
                
                for mlvs, st in zip(mlines, styles):
                    try:
                        p_df = select_df(best_df, dict(zip(pmlf+pcrf, [*mlvs, col_v, row_v])), *x_fields)
                    except: # for log with incomplete grid
                        continue
                    
                    legend = ','.join([(v if isinstance(v, str) else f'{f} {v}').replace('_', ' ') for f, v in zip(pmlf, mlvs)])
                    p_df, legend, mlvs = preprcs_df(p_df, legend, mlvs)
                    
                    # remove unnessacery fields
                    if mode not in {'scatter', 'scatter_heat'}:
                        p_df = p_df.reset_index([*(set(p_df.index.names) - set(x_fields))], drop=False)
                    if len(x_fields)>1:
                        p_df = p_df.reorder_levels(x_fields)
                    p_df = (
                        p_df.sort_index(key=lambda s: [*map(str2value, s)])
                            .reindex(['metric_value', *(set(p_df)-{'metric_value'})], axis=1)
                    )
                    
                    # set line style
                    for stp, s in zip(style_types, st):
                        pcfg['line_style'][stp] = s
                        
                    ax = ax_draw(ax, p_df, 
                                label=legend,
                                **pcfg['line_style'])

                ax_styler(ax, **pcfg['ax_style'])
                
        if has_cbar:
            c_cmap, c_norm = (pcfg['line_style'].pop(k) for k in ['cmap', 'norm'])
            
            for ax in axs[:, -1]: ax.remove()
            cax = fig.add_subplot(axs[0, 0].get_gridspec()[:, -1])
            cbar = fig.colorbar(cm.ScalarMappable(norm=c_norm, cmap=c_cmap), cax=cax)
            cbar.ax.tick_params(labelsize=pcfg['font_size'])
            
            z_label = pcfg['zlabel'] or metrics[-1].replace('_', ' ').capitalize()
            cbar.ax.set_ylabel(z_label, fontsize=pcfg['font_size'])
        
        # add a big axis, hide frame, tick, and tick label
        big_ax = fig.add_subplot(111, frameon=False)
        big_ax.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        
        # set title and x, y axis labels
        if (title:=pcfg.get('title')):
            big_ax.set_title(title, fontsize=pcfg['font_size'])
        
        x_label= pcfg['xlabel'] or x_label
        y_label= pcfg['ylabel'] or y_label
        fig.supxlabel(x_label, fontsize=pcfg['font_size'])
        fig.supylabel(y_label, fontsize=pcfg['font_size'])
        
        if len(pcrf)>0:
            for ci, col_v in enumerate(col_vs):
                ax = axs[-1, ci]
                ax.set_xlabel(f'{pcrf[0]}={col_v}', size=pcfg['font_size'])
        if len(pcrf)>1:
            for ri, row_v in enumerate(row_vs[::-1]):
                axs[ri, 0].set_ylabel(f'{pcrf[1]}={row_v}', size=pcfg['font_size'])
        
        # set legend, improve later
        if pmlf:
            ax = axs[0, 0]
            base_styles = {'color': 'gray', 'marker': '', 'linestyle': '-'}
            first_styles = {k:v[0] for k, v in style_dict.items() if k in (style_types[len(pmlf):] if len(pmlf)<3 else [])} # when plmf is not full, legend are style with the first values of unused style_types
            
            is_wide = len(pmlf)<3
            max_row = max([len(set(df.index.get_level_values(k))) for k in pmlf])
            
            legendlines, legendlabels = [], []
            for s, k in zip(style_types, pmlf):
                extra = {'linewidth': 0} if s=='marker' else {}
                    
                vs = sorted(set(df.index.get_level_values(k)))
                legendlines += [lines.Line2D([], [], alpha=0)] + \
                                [lines.Line2D([], [], **{**pcfg['line_style'], **base_styles, **first_styles, **extra, **{s: ss}}) for ss in [*style_dict[s]][:len(vs)]] + \
                                ([lines.Line2D([], [], alpha=0) for _ in range(max_row-len(vs))] if is_wide else [])
                legendlabels += [f"[{k.replace('_', ' ').capitalize()}]", *vs] + (['' for _ in range(max_row-len(vs))] if is_wide else [])
            
            ax.legend(handles=legendlines, labels=legendlabels, **legend_style, #**pcfg['ax_style'].pop('legend', [{}])[0], 
                      ncol=len(pmlf) if is_wide else 1, columnspacing=0.8, handlelength=None if len(pmlf)==1 else 1.5)
                    
        return best_df, fig, save_name
    

def run(argv, preprcs_df):
    if len(argv)>2:
        raise app.UsageError('Too many command-line arguments.')
    
    # Preprocess plot_config
    flag_dict = FLAGS.flag_values_dict()
    
    plot_config = {**default_style, **flag_dict}
    if FLAGS.plot_config!='':
        with open(FLAGS.plot_config) as f:
            plot_config = yaml.safe_load(f.read())
            plot_config = get_plot_config(plot_config, flag_dict)
    
    # set ax_style related arguments
    ax_st = plot_config['ax_style']
    
    if (fig_size:=flag_dict.pop('fig_size')):
        if len(fig_size)==1:
            fig_size = fig_size*2
        fig_size = [*map(float, fig_size)]
        ax_st['fig_size'] = fig_size
    if (xscale:=flag_dict.pop('xscale')):
        ax_st['xscale'] = [xscale, {}]
    if (yscale:=flag_dict.pop('yscale')):
        ax_st['yscale'] = [yscale, {}]
    if (zscale:=flag_dict.pop('zscale')):
        ax_st['zscale'] = [zscale, {}]
        
    # set ax_style related arguments
    l_st = plot_config['line_style']
    
    if (msize:=flag_dict.pop('marker_size')):
        l_st['markersize'] = msize
     
    # set style
    style.use(plot_config['style'])
    
    # get paths
    _, tsv_file, fig_dir = Experiment.get_paths(plot_config['exp_folder'])
    save_dir = os.path.join(fig_dir, plot_config['mode'])
    
    assert plot_config['mode'].split('-')[0] in {'curve', 'curve_best', 'bar', 'heatmap', 'scatter', 'scatter_heat'}, f'Mode: {plot_config["mode"]} does not exist.'
    df, fig, save_name = draw_metric(tsv_file, plot_config, preprcs_df=preprcs_df)

    save_figure(fig, save_dir, save_name)
    df.to_csv(os.path.join(save_dir, f'{save_name}.tsv'), sep='\t')
    
    print('\n', Align(Panel(f'save {{plot, table}} at: {fig_dir}/[bold blue_violet]{plot_config["mode"]}[/bold blue_violet]/[bold spring_green1]{save_name}[/bold spring_green1].{{pdf, tsv}}', 
                      title='Plot complete', padding=(1, 3), expand=False), align='center'), '\n')

    
def main(preprcs_df = lambda *x: x):
    
    flags.DEFINE_string(        'exp_folder'            , ''        , "Experiment folder path.")
    flags.DEFINE_string(        'mode'                  , 'curve-step-val_loss', "Plot mode.")
    # flags.DEFINE_bool(          'animate_over_step'       , 'False'   , "Whether to animate over steps.")
    
    # data processing
    flags.DEFINE_string(        'filter'                , ''        , "Filter values. (e.g., 'step 0:100 / lr 0.01 0.1 / wd! 0.0')")
    flags.DEFINE_spaceseplist(  'multi_line_fields'     , ''        , "List of fields to plot multiple lines over.")
    flags.DEFINE_spaceseplist(  'col_row_fields'        , ''        , "column and row fields for multiple figures.")
    flags.DEFINE_string(        'field_orders'          , ''        , "Order of string fields (e.g., 'lr_schedule constant linear cosine')")
    
    flags.DEFINE_spaceseplist(  'best_ref_x_fields'     , ''        , "Reference x_field-values to evaluate optimal hyperparameters.")
    flags.DEFINE_string(        'best_ref_metric_field' , ''        , "Reference metric_field-values to evaluate optimal hyperparameters.")
    flags.DEFINE_spaceseplist(  'best_ref_ml_fields'    , ''        , "Reference multi_line_fields-value to evaluate optimal hyperparameters.")
    flags.DEFINE_bool(          'best_at_max'           , False     , 'Whether the bese metric value is the maximum value.')
    
    # plot estehtics
    flags.DEFINE_string(        'plot_config'           , ''        , "Yaml file path for various plot setups.")
    flags.DEFINE_string(        'style'                 , 'default' , "Matplotlib style.")
    flags.DEFINE_spaceseplist(  'colors'                ,['default'], "color type (e.g., default, light:#9467bd, Blues, rocket, crest, magma).")
    flags.DEFINE_spaceseplist(  'colors_rep_skip_shift' ,[1,0,0]    ,  "Skip n colors and shift color list ([c for i, c in enumerate(cs) if not i%(skp+1)+sft]).")
    
    flags.DEFINE_spaceseplist(  'fig_size'              , ''        , 'Figure size.')
    flags.DEFINE_string(        'xscale'                , ''        , "Scale of x-axis (linear, log).")
    flags.DEFINE_string(        'yscale'                , ''        , "Scale of y-axis (linear, log).")
    flags.DEFINE_string(        'zscale'                , ''        , "Scale of z-axis or colorbar (linear, log).")
    
    flags.DEFINE_string(        'title'                 , ''        , "Title.")
    flags.DEFINE_bool(          'annotate'              , True      , 'Run multiple plot according to given config.')
    flags.DEFINE_spaceseplist(  'annotate_field'        , ''        , 'List of fields to include in annotation.')
    flags.DEFINE_string(        'xlabel'                , ''        , "Label of x-axis.")
    flags.DEFINE_string(        'ylabel'                , ''        , "Label of y-axis.")
    flags.DEFINE_string(        'zlabel'                , ''        , "Label of z-axis or colorbar.")
    flags.DEFINE_integer(       'font_size'             , 22        , "Font size of title and label.")

    flags.DEFINE_float(         'marker_size'           , 10        , "Size of marker.")

    app.run(partial(run, preprcs_df=preprcs_df))
    
    
if __name__=='__main__':
    main()