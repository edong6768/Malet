import os
import re
import yaml
from functools import partial
from itertools import product

from absl import app, flags

import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.lines as lines
import seaborn as sns

from .experiment import Experiment, ExperimentLog
from .utils import str2value, df2richtable

from rich import print
from rich.panel import Panel
from rich.columns import Columns
from rich.align import Align

from .plot_utils.data_processor import avgbest_df, select_df, homogenize_df
from .plot_utils.plot_drawer import ax_draw_curve, ax_draw_best_stared_curve, ax_draw_bar, ax_draw_heatmap, ax_prcs_draw_scatter
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
    if pflt:
        save_name += pflt.replace(' / ', '-').replace(' ', '_')
    if pmlf:
        save_name = '-'.join([*pmlf, *pcrf, save_name])
    if any([pcfg[f'best_ref_{k}'] for k in ['x_fields', 'metric_field', 'ml_fields']]):
        save_name +=  f"-({pcfg['best_ref_x_fields']}, {pcfg['best_ref_metric_field']}, {pcfg['best_ref_ml_fields']})"
    save_name += "-max" if pcfg['best_at_max'] else "-min"
    return save_name


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
        
        pflt, pmlf = map(pcfg.get, ['filter', 'multi_line_fields'])
        pcrf = pcfg['col_row_fields']
        
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
            assert len(x_fields)==1 , f'Number of x_fields shoud be 1 when using curve mode, but you passed {len(x_fields)}.'
            assert len(metrics)==1  , f'Number of metric shoud be 1 when using heatmap mode, but you passed {metrics}.'
            assert len(pmlf)<=3     , f'Number of multi_line_fields should be less than 3, but you passed {len(pmlf)}'
            ax_draw = {'curve':      ax_draw_curve,
                       'curve_best': ax_draw_best_stared_curve,
                       'bar':        ax_draw_bar}[mode]
            ax_draw = partial(ax_draw,
                              annotate=pcfg['annotate'], 
                              annotate_field=pcfg['annotate_field'],
                              best_at_max=pcfg['best_at_max'])
            x_label, y_label = (f.replace('_', ' ').capitalize() for f in (x_fields[0],  metrics[0]))
            
        elif mode=='heatmap':
            assert len(x_fields)==2, f'Number of x_fields shoud be 2 when using heatmap mode, but you passed {len(x_fields)}.'
            assert len(metrics)==1, f'Number of metric shoud be 1 when using heatmap mode, but you passed {metrics}.'
            assert not pmlf, f'No multi_line_fields are allowed in heatmap mode, but you passed {pmlf}.'
            ax_draw = partial(ax_draw_heatmap,
                              annotate=pcfg['annotate'], 
                              annotate_field=pcfg['annotate_field'],
                              best_at_max=pcfg['best_at_max'])
            x_label, y_label = (f.replace('_', ' ').capitalize() for f in x_fields)
            
        elif mode=='scatter':
            assert not x_fields, f'No x_fields are allowed in scatter mode, but you passed {x_fields}.'
            assert len(metrics)==2, f'Number of metric shoud be 2 when using scatter mode, but you passed {metrics}.'
            assert len(pmlf)<=2, f'Number of multi_line_fields should be less than 2, but you passed {len(pmlf)}'
            ax_draw = partial(ax_prcs_draw_scatter, y_fields=metrics)
            x_label, y_label = (f.replace('_', ' ').capitalize() for f in metrics)
            
        # build save name
        save_name = __save_name_builder(pflt, pmlf, pcrf, pcfg, save_name=save_name)
        
        ############################# Get and filter dataframe #############################
        
        # get dataframe, drop unused metrics for efficient process
        log = ExperimentLog.from_tsv(tsv_file)
        assert all(m in log.df for m in metrics), f'Metric {[m for m in metrics if m not in log.df]} not in log. Choose between {list(log.df)}'
        
        #--- initial filter for df according to FLAGS.filter (except step and metric)
        filt_dict = {}
        if pflt:
            after_filt = {'step', 'total_steps', 'metric'}
            filt_dict = [(fk, fvs) for fk, *fvs in map(lambda flt: re.split('(?<!,) ', flt.strip()), pflt.split('/'))] # split ' ' except ', '
            log.df = select_df(log.df, {fk     :[*map(str2value, fvs)] for fk, fvs in filt_dict if fk[-1]!='!' and fk      not in after_filt})
            log.df = select_df(log.df, {fk[:-1]:[*map(str2value, fvs)] for fk, fvs in filt_dict if fk[-1]=='!' and fk[:-1] not in after_filt}, equal=False)
        
        #--- melt and explode metric in log.df
        if 'metric' not in x_fields+pmlf+pcrf:
            log.df = log.df.drop(list(set(log.df)-{*metrics, pcfg['best_ref_metric_field']}), axis=1)
        df = log.melt_and_explode_metric(step=-1 if (dict(filt_dict).get('step', None if ('step' in x_fields) else 'last')=='last') else None)
        
        assert not df.empty, f'Metrics {metrics}' +\
            (f' and best_ref_metric_field {pcfg["best_ref_metric_field"]} are' if pcfg["best_ref_metric_field"] else ' is') +\
                f' NaN in given dataframe: \n{log.df}'
        
        #---filter df according to FLAGS.filter step and metrics
        if pflt:
            filt_dict = [flt for flt in filt_dict if tuple(flt)!=('step', ['best'])]  # Let `avgbest_df` handle 'best' step, remove from filt_dict
            e_rng = lambda fvs: [*range(*map(int, fvs[0].split(':')))] if (len(fvs)==1 and ':' in fvs[0]) else fvs # CNG 'a:b' step filter later
            df = select_df(df, {fk     :[*map(str2value, e_rng(fvs))] for fk, fvs in filt_dict if fk[-1]!='!' and fk      in after_filt})
            df = select_df(df, {fk[:-1]:[*map(str2value, e_rng(fvs))] for fk, fvs in filt_dict if fk[-1]=='!' and fk[:-1] in after_filt}, equal=False)
    
    
        ############################# Prepare dataframe #############################
        
        # Report selected plot configs and field handling statistics
        if mode=='scatter':
            key_field       = {*df.index.names}
            specified_field = avg_field = optimized_field = set()
        else:
            key_field       = {*x_fields, *pmlf, *pcrf, 'metric'}
            avg_field       = {'seed'}
            specified_field = {k for k in {*df.index.names} if len({*df.index.get_level_values(k)})==1}
            optimized_field = {*df.index.names} - key_field - avg_field - specified_field
            
        print('\n\n',
            Align(
                Columns(
                    [Panel('\n'.join([f'- {k}: {pcfg[k]}' 
                                            for k in ('mode', 'multi_line_fields', 
                                                        'filter', 'best_at_max', 
                                                        'best_ref_x_fields', 'best_ref_metric_field', 
                                                        'best_ref_ml_fields') if pcfg[k]]),
                                    title='Plot configuration', padding=(1, 3)),
                     Panel(f"- Key field (has multiple values): {key_field} ({len(key_field)})\n" + \
                           f"- Specified field: {specified_field} ({len(specified_field)})\n"+ \
                           f"- Averaged field: {avg_field} ({len(avg_field)})\n" + \
                           f"- Optimized field: {optimized_field} ({len(optimized_field)})",
                           title='Field handling statistics', padding=(1, 3))]
                    ), align='center'
                )
            )
        
        # change field name and avg over seed and get best result over best_over
        if mode=='scatter':
            best_df = df
        else:
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
                                 avg_over='seed', 
                                 best_over=optimized_field, 
                                 best_of=best_of,
                                 best_at_max=pcfg['best_at_max'])
            
            # process 'step' and 'metric'
            if sm_bestof:
                avg_df = avgbest_df(df, 'metric_value', avg_over='seed')
                sm_df = best_df
                
                if 'step' in sm_bestof:
                    if sm_bestof['step']=='last':
                        sm_df = best_df.loc[best_df.index.get_level_values('step')==best_df.index.get_level_values('total_steps')]
                    sm_df = sm_df.reset_index(['step'], drop=True)
                if 'metric' in sm_bestof:
                    sm_df = select_df(sm_df, {'metric': sm_bestof['metric']}, drop=True)
                    
                # homogenize best_df with sm_df, exclude 'total_steps'
                best_df = homogenize_df(avg_df, sm_df, {}, 'total_steps')
        
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
        
        # col-row and multi_line fields
        get_field_values = lambda f: sorted(set(best_df.index.get_level_values(f)), key=str2value) if f else ['']
        col_vs, row_vs = map(get_field_values, pcrf+['']*(2-len(pcrf)))
        mlines = [*product(*map(get_field_values, pmlf))]
        
        # scale axis size according to number of col and row plots
        fig_size = pcfg['ax_style']['fig_size']
        if isinstance(fig_size, (float, int)): fig_size = [fig_size]*2
        pcfg['ax_style']['fig_size'] = [p*l for p, l in zip([len(row_vs), len(col_vs)], fig_size)]
        legend_style = pcfg['ax_style'].pop('legend', [{}])[0]
        
        # set unif_xticks for curve
        if mode in {'curve', 'curve_best'} and pcfg['xscale']=='unif':
            pcfg['ax_style'].pop('xscale', None)
            pcfg['line_style']['unif_xticks'] = True
        
        # prepare plot
        fig, axs = plt.subplots(len(col_vs), len(row_vs), sharex=True, sharey=True)
        
        # set style types per mode
        if mode in {'curve', 'curve_best'}:
            style_types = ['color', 'linestyle', 'marker']
        elif mode=='bar':
            style_types = ['color']
        elif mode=='heatmap':
            style_types = []
        elif mode=='scatter':
            style_types = ['color', 'marker']
        
        rep, skp, sft = map(int, pcfg['colors_rep_skip_shift'])
        style_dict = {'color': [c for i, c in enumerate(sum(map(sns.color_palette, [None if c=='default' else c for c in pcfg['colors']]), [])*rep) if not (i-sft)%(skp+1)],
                      'marker': ['D', 'o', '>', 'X', 's', 'v', '^', '<', 'p', 'P', '*', '+', 'x', 'h', 'H', '|', '_'],
                      'linestyle': ['-', ':', '-.', '--']*3}
        styles = [*product(*[[*style_dict[s]][:len(set(k))] for s, k in zip(style_types, map(df.index.get_level_values, pmlf))])]
        
        for ci, col_v in enumerate(col_vs):
            for ri, row_v in enumerate(row_vs):
                ax = axs[ci, ri] if pcrf else axs
                
                for mlvs, st in zip(mlines, styles):
                    try:
                        p_df = select_df(best_df, dict(zip(pmlf+pcrf, [*mlvs, col_v, row_v])), *x_fields)
                    except: # for log with incomplete grid
                        continue
                    legend = ','.join([(v if isinstance(v, str) else f'{f} {v}').replace('_', ' ') for f, v in zip(pmlf, mlvs)])
                    
                    p_df, legend, mlvs = preprcs_df(p_df, legend, mlvs)
                    
                    # remove unnessacery fields
                    if mode!='scatter':
                        p_df = p_df.reset_index([*(set(p_df.index.names) - set(x_fields))], drop=False)
                    if len(x_fields)>1:
                        p_df = p_df.reorder_levels(x_fields)
                    p_df = p_df.sort_index(key=lambda s: [*map(str2value, s)])
                    p_df = p_df[['metric_value', *(set(p_df)-{'metric_value'})]]
                    
                    # set line style
                    for stp, s in zip(style_types, st):
                        pcfg['line_style'][stp] = s
                        
                    ax = ax_draw(ax, p_df, 
                                label=legend,
                                **pcfg['line_style'])

                ax_styler(ax, **pcfg['ax_style'])
                
        
        # fig.colorbar(ax.collections[0], ax=axs, orientation='horizontal' if len(col_vs)>1 else 'vertical', pad=0.1)
        
        # add a big axis, hide frame
        big_ax = fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        big_ax.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        if (title:=pcfg.get('title')):
            big_ax.set_title(title, fontsize=pcfg['font_size'])
        
        # set x, y axis labels
        fig.supxlabel(x_label, fontsize=pcfg['font_size'])
        fig.supylabel(y_label, fontsize=pcfg['font_size'])
        
        if pcrf:
            for ci, col_v in enumerate(col_vs):
                axs[ci, 0].set_ylabel(f'{pcrf[0]}={col_v}', size=pcfg['font_size'])
            for ri, row_v in enumerate(row_vs):
                axs[-1, ri].set_xlabel(f'{pcrf[1]}={row_v}', size=pcfg['font_size'])
        
        # set legend, improve later
        
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
                
        return best_df, fig, ax, x_label, y_label, save_name.strip('-')
    

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
    # if (title:=flag_dict.pop('title')):
    #     ax_st['title']  = [title,  {'size': flag_dict['font_size']}]
    # if (xlabel:=flag_dict.pop('xlabel')):
    #     ax_st['xlabel'] = [xlabel, {'size': flag_dict['font_size']}]
    # if (ylabel:=flag_dict.pop('ylabel')):
    #     ax_st['ylabel'] = [ylabel, {'size': flag_dict['font_size']}]
        
    # set style
    style.use(plot_config['style'])
    
    # get paths
    _, tsv_file, fig_dir = Experiment.get_paths(plot_config['exp_folder'])
    save_dir = os.path.join(fig_dir, plot_config['mode'])
    
    
    assert plot_config['mode'].split('-')[0] in {'curve', 'curve_best', 'bar', 'heatmap', 'scatter'}, f'Mode: {plot_config["mode"]} does not exist.'
    df, fig, _, _, _, save_name = draw_metric(tsv_file, plot_config, preprcs_df=preprcs_df)

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
    
    flags.DEFINE_spaceseplist(  'best_ref_x_fields'     , ''        , "Reference x_field-values to evaluate optimal hyperparameters.")
    flags.DEFINE_string(        'best_ref_metric_field' , ''        , "Reference metric_field-values to evaluate optimal hyperparameters.")
    flags.DEFINE_spaceseplist(  'best_ref_ml_fields'    , ''        , "Reference multi_line_fields-value to evaluate optimal hyperparameters.")
    flags.DEFINE_bool(          'best_at_max'           , False     , 'Whether the bese metric value is the maximum value.')
    
    # plot estehtics
    flags.DEFINE_string(        'plot_config'           , ''        , "Yaml file path for various plot setups.")
    flags.DEFINE_string(        'style'                 , 'default' , "Matplotlib style.")
    flags.DEFINE_spaceseplist(  'colors'                ,['default'], "color type (e.g., default, light:#9467bd, Blues, rocket, crest, magma).")
    flags.DEFINE_spaceseplist(  'colors_rep_skip_shift'   ,['1', '0', '0'],  "Skip n colors and shift color list ([c for i, c in enumerate(cs) if not i%(skp+1)+sft]).")
    
    flags.DEFINE_spaceseplist(  'fig_size'              , ''        , 'Figure size.')
    flags.DEFINE_string(        'xscale'                , ''        , "Scale of x-axis (linear, log).")
    flags.DEFINE_string(        'yscale'                , ''        , "Scale of y-axis (linear, log).")
    
    flags.DEFINE_string(        'title'                 , ''        , "Title.")
    flags.DEFINE_bool(          'annotate'              , True      , 'Run multiple plot according to given config.')
    flags.DEFINE_spaceseplist(  'annotate_field'        , ''        , 'List of fields to include in annotation.')
    flags.DEFINE_string(        'xlabel'                , ''        , "Label of x-axis.")
    flags.DEFINE_string(        'ylabel'                , ''        , "Label of y-axis.")
    flags.DEFINE_integer(       'font_size'             , 22        , "Font size of title and label.")

    app.run(partial(run, preprcs_df=preprcs_df))
    
    
if __name__=='__main__':
    main()