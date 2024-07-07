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

from .plot_utils.data_processor import avgbest_df, select_df
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
    
def __save_name_builder(pflt, pmlf, pcfg, save_name=''):
    if pflt:
        save_name += pflt.replace(' / ', '-').replace(' ', '_')
    if pmlf:
        save_name = '-'.join([*pmlf, save_name])
    if any([pcfg[f'best_ref_{k}'] for k in ['x_fields', 'metric_field', 'ml_fields']]):
        save_name +=  f"-({pcfg['best_ref_x_fields']}, {pcfg['best_ref_metric_field']}, {pcfg['best_ref_ml_fields']})"
    save_name += "-max" if pcfg['best_at_max'] else "-min"
    return save_name


def draw_metric(tsv_file, plot_config, save_name='', preprcs_df=lambda *x: x):
        pcfg = plot_config
        
        # parse mode string
        mode, x_fields, metrics = pcfg['mode'].split('-') # ex) {sam}-{epoch}-{train_loss}
        x_fields = xs if ['']!=(xs:=x_fields.split(' ')) else []
        metrics = metrics.split(' ')
        
        pflt, pmlf = map(pcfg.get, ['filter', 'multi_line_fields'])
        
        # choose plot mode
        if mode=='curve':
            assert len(x_fields)==1, f'Number of x_fields shoud be 1 when using curve mode, but you passed {len(x_fields)}.'
            assert len(metrics)==1, f'Number of metric shoud be 1 when using heatmap mode, but you passed {metrics}.'
            assert len(pmlf)<=3, f'Number of multi_line_fields should be less than 3, but you passed {len(pmlf)}'
            ax_draw = ax_draw_curve
            x_label = x_fields[0].replace('_', ' ').capitalize()
            y_label = metrics[0].replace('_', ' ').capitalize()
        elif mode=='curve_best':
            assert len(x_fields)==1, f'Number of x_fields shoud be 1 when using curve mode, but you passed {len(x_fields)}.'
            assert len(metrics)==1, f'Number of metric shoud be 1 when using heatmap mode, but you passed {metrics}.'
            assert len(pmlf)<=3, f'Number of multi_line_fields should be less than 3, but you passed {len(pmlf)}'
            ax_draw = ax_draw_best_stared_curve
            x_label = x_fields[0].replace('_', ' ').capitalize()
            y_label = metrics[0].replace('_', ' ').capitalize()
        elif mode=='bar':
            assert len(x_fields)==1, f'Number of x_fields shoud be 1 when using bar mode, but you passed {len(x_fields)}.'
            assert len(metrics)==1, f'Number of metric shoud be 1 when using heatmap mode, but you passed {metrics}.'
            assert len(pmlf)<=3, f'Number of multi_line_fields should be less than 3, but you passed {len(pmlf)}'
            ax_draw = ax_draw_bar
            x_label = x_fields[0].replace('_', ' ').capitalize()
            y_label = metrics[0].replace('_', ' ').capitalize()
        elif mode=='heatmap':
            assert len(x_fields)==2, f'Number of x_fields shoud be 2 when using heatmap mode, but you passed {len(x_fields)}.'
            assert len(metrics)==1, f'Number of metric shoud be 1 when using heatmap mode, but you passed {metrics}.'
            assert not pmlf, f'No multi_line_fields are allowed in heatmap mode, but you passed {pmlf}.'
            ax_draw = ax_draw_heatmap
            x_label = x_fields[0].replace('_', ' ').capitalize()
            y_label = x_fields[1].replace('_', ' ').capitalize()
        elif mode=='scatter':
            assert not x_fields, f'No x_fields are allowed in scatter mode, but you passed {x_fields}.'
            assert len(metrics)==2, f'Number of metric shoud be 2 when using scatter mode, but you passed {metrics}.'
            assert len(pmlf)<=2, f'Number of multi_line_fields should be less than 2, but you passed {len(pmlf)}'
            ax_draw = ax_prcs_draw_scatter
            x_label = metrics[0].replace('_', ' ').capitalize()
            y_label = metrics[1].replace('_', ' ').capitalize()
        
        # get dataframe, drop unused metrics for efficient process
        log = ExperimentLog.from_tsv(tsv_file)
        assert all(m in log.df for m in metrics), f'Metric {[m for m in metrics if m not in log.df]} not in log. Choose between {list(log.df)}'
        
        #--- initial filter for df according to FLAGS.filter (except epoch and metric)
        if pflt:
            filt_dict = [*map(lambda flt: re.split('(?<!,) ', flt.strip()), pflt.split('/'))] # split ' ' except ', '
            log.df = select_df(log.df, {fk:[*map(str2value, fvs)] for fk, *fvs in filt_dict if fk[-1]!='!' and fk not in {'step', 'total_steps', 'metric'}})
            log.df = select_df(log.df, {fk[:-1]:[*map(str2value, fvs)] for fk, *fvs in filt_dict if fk[-1]=='!' in fk and fk not in {'step', 'total_steps', 'metric'}}, equal=False)
        
        #--- melt and explode metric in log.df
        if 'metric' not in pmlf and 'metric' not in x_fields:
            log.df = log.df.drop(list(set(log.df)-{*metrics, pcfg['best_ref_metric_field']}), axis=1)
        df = log.melt_and_explode_metric(step=None if 'step' in x_fields or 'step' in pflt else -1)
        
        assert not df.empty, f'Metrics {metrics}' +\
            (f' and best_ref_metric_field {pcfg["best_ref_metric_field"]} are' if pcfg["best_ref_metric_field"] else ' is') +\
                f' NaN in given dataframe: \n{log.df}'
        
        #---filter df according to FLAGS.filter step and metrics
        if pflt:
            e_rng = lambda fvs: [*range(*map(int, fvs[0].split(':')))] if (len(fvs)==1 and ':' in fvs[0]) else fvs # CNG 'a:b' step filter later
            df = select_df(df, {fk:[*map(str2value, e_rng(fvs))] for fk, *fvs in filt_dict if fk[-1]!='!' and  fk in {'step', 'total_steps', 'metric'}})
            df = select_df(df, {fk[:-1]:[*map(str2value, e_rng(fvs))] for fk, *fvs in filt_dict if fk[-1]=='!' and  fk in {'step', 'total_steps', 'metric'}}, equal=False)
        
        
        #---set mlines according to FLAGS.multi_line_fields
        if pmlf:
            mlines = product(*[sorted(set(df.index.get_level_values(f)), key=str2value) for f in pmlf])
        else:
            pmlf, mlines = ['metric'], [metrics]
            pcfg['ax_style'].pop('legend', None)
        
        #---preprocess best_ref_x_fields
        pcfg['best_ref_x_fields'] = [*map(str2value, pcfg['best_ref_x_fields'])]
        
        best_over = set(df.index.names) - {*x_fields, 'metric', 'seed', *pmlf}
        best_at_max = pcfg['best_at_max']
       
        #---automatically set best_ref_metric_field of step 
        if 'step' in x_fields:
            if not pcfg['best_ref_x_fields']:
                pcfg['best_ref_x_fields'] = ['']*len(x_fields)
            i = x_fields.index('step')
            if pflt and 'step' in dict(filt_dict):
                st = dict(filt_dict)['step']
                pcfg['best_ref_x_fields'][i] = int(st.split(':')[1])-1 if ':' in st else int(st) # CNG 'a:b' step filter later
            else:
                pcfg['best_ref_x_fields'][i]=min(*df.index.get_level_values('total_steps'))
                
        if mode=='scatter':
            x_fields = [*set(df.index.names) - {*pmlf}]
                
        # build save name
        save_name = __save_name_builder(pflt, pmlf, pcfg, save_name=save_name)
        
        
        ############################# Prepare dataframe #############################
        
        # Notify selected plot configs and field handling statistics
        
        if mode=='scatter':
            key_field = [*set(df.index.names)]
            specified_field = avg_field = optimized_field = []
        else:
            key_field = [*x_fields, *pmlf]
            specified_field = [*{k for k in best_over if len(set(df.index.get_level_values(k)))==1}, 'metric']
            avg_field = ['seed']
            optimized_field = list(best_over-set(specified_field))
            
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
                ))
        
        best_of = {}
        if pcfg['best_ref_x_fields']: # same hyperparameter over all points in line
            best_of.update(dict([*zip(x_fields, pcfg['best_ref_x_fields'])]))

        if pcfg['best_ref_metric_field']: # Optimize in terms of reference metric, and apply those hyperparameters to original
            best_of['metric'] = pcfg['best_ref_metric_field']
        
        if pcfg['best_ref_ml_fields']: # same hyperparameter over all line in multi_line_fields
            best_of.update(dict([*zip(pmlf, pcfg['best_ref_ml_fields'])]))
            
        # change field name and avg over seed and get best result over best_over
        if mode=='scatter':
            best_df = df
        else:
            best_df = avgbest_df(df, 'metric_value',
                                avg_over='seed', 
                                best_over=best_over, 
                                best_of=best_of,
                                best_at_max=best_at_max)
        
        print('\n', Align(df2richtable(best_df), align='center'))
        
        if 'metric' not in pmlf and 'metric' not in x_fields and mode!='scatter':
            best_df = select_df(best_df, {'metric': metrics})
        
        ############################# Plot #############################
        
        # prepare plot
        fig, ax = plt.subplots()
        
        style_dict = {}
        if pcfg['colors']=='':
            style_dict['color'] = sns.color_palette()*10
        elif pcfg['colors']=='cont':
            style_dict['color'] = [c for i, c in enumerate(sum(map(sns.color_palette, ["light:#9467bd", "Blues", "rocket", "crest", "magma"]*3), [])[1:])]# if i%2]

        style_dict.update({'marker': ['o', 's', 'D', 'v', '^', '<', '>', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', '|', '_'],
                           'linestyle': ['-', '--', '-.', ':']*3})
        styles = product(*[[*style_dict[s]][:len(set(k))] for s, k in zip(['color', 'marker', 'linestyle'], map(df.index.get_level_values, pmlf))])
        
        # select specified metric if multi_line_fields isn't metric
        if 'metric' not in pmlf:
            best_df = select_df(best_df, {'metric': metrics}, *x_fields)
            
        for mlvs, st in zip(mlines, styles):
            try:
                p_df = select_df(best_df, {f: v for f, v in zip(pmlf, mlvs)}, *x_fields)
            except:
                continue
            legend = ','.join([(v if isinstance(v, str) else f'{f} {v}').replace('_', ' ') for f, v in zip(pmlf, mlvs)])
            
            p_df, legend, mlvs = preprcs_df(p_df, legend, mlvs)
            
            # remove unnessacery fields
            p_df = p_df.reset_index([*(set(p_df.index.names) - set(x_fields))], drop=False)
            if len(x_fields)>1:
                p_df = p_df.reorder_levels(x_fields)
            p_df = p_df.sort_index(key=lambda s: [*map(str2value, s)])
            p_df = p_df[['metric_value', *(set(p_df)-{'metric_value'})]]
            
            pcfg['line_style']['color'] = st[0]
            if len(st)>1: pcfg['line_style']['marker'] = st[1]
            if len(st)>2: pcfg['line_style']['linestyle'] = st[2]
            ax = ax_draw(ax, p_df, 
                         label=legend,
                         annotate=pcfg['annotate'],
                         annotate_field=pcfg['annotate_field'],
                         std_plot=pcfg['std_plot'],
                         best_at_max=best_at_max,
                         y_fields=metrics, # for scatter plot
                         **pcfg['line_style'])

        # set legend
        legendlines, legendlabels = [], []
        base_styles = {'color': 'gray', 'marker': '', 'linestyle': '-'}
        max_row = max([len(set(df.index.get_level_values(k))) for k in pmlf])
        for s, k in zip(['color', 'marker', 'linestyle'], pmlf):
            vs = sorted(set(df.index.get_level_values(k)))
            legendlines += [lines.Line2D([], [], alpha=0)] + \
                           [lines.Line2D([], [], **{**pcfg['line_style'], **base_styles, **{s: ss}}) for ss in [*style_dict[s]][:len(vs)]] + \
                           [lines.Line2D([], [], alpha=0) for _ in range(max_row-len(vs))]
            legendlabels += [k, *vs] + ['' for _ in range(max_row-len(vs))]
        ax.legend(handles=legendlines, labels=legendlabels, **pcfg['ax_style'].pop('legend')[0], ncol=len(pmlf))
        
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
    fig_size = flag_dict.pop('fig_size')
    if fig_size:
        if len(fig_size)==1:
            fig_size = fig_size*2
        fig_size = [*map(float, fig_size)]
        plot_config['ax_style']['fig_size'] = fig_size
    
    xscale = flag_dict.pop('xscale')
    if xscale:
        plot_config['ax_style']['xscale'] = [xscale, {}]
    
    yscale = flag_dict.pop('yscale')
    if yscale:
        plot_config['ax_style']['yscale'] = [yscale, {}]
    
    title = flag_dict.pop('title')
    if title:
        plot_config['ax_style']['title'] = [title, {'size': flag_dict['font_size']}]
        
    xlabel = flag_dict.pop('xlabel')
    if xlabel:
        plot_config['ax_style']['xlabel'] = [xlabel, {'size': flag_dict['font_size']}]
        
    ylabel = flag_dict.pop('ylabel')
    if ylabel:
        plot_config['ax_style']['ylabel'] = [ylabel, {'size': flag_dict['font_size']}]
        
    
    # set style
    style.use(plot_config['style'])
    
    # get paths
    _, tsv_file, fig_dir = Experiment.get_paths(plot_config['exp_folder'])
    save_dir = os.path.join(fig_dir, plot_config['mode'])
    
    if plot_config['mode'].split('-')[0] in {'curve', 'curve_best', 'bar', 'heatmap', 'scatter'}:
        df, fig, ax, x_label, y_label, save_name = draw_metric(tsv_file, plot_config, preprcs_df=preprcs_df)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
    else:
        assert False, f'Mode: {plot_config["mode"]} does not exist.'
    
    ax_styler(ax, **plot_config['ax_style'])
    save_figure(fig, save_dir, save_name)
    df.to_csv(os.path.join(save_dir, f'{save_name}.tsv'), sep='\t')
    
    print('\n', Align(Panel(f'save {{plot, table}} at: {fig_dir}/[bold blue_violet]{plot_config["mode"]}[/bold blue_violet]/[bold spring_green1]{save_name}[/bold spring_green1].{{pdf, tsv}}', 
                      title='Plot complete', padding=(1, 3), expand=False), align='center'), '\n')

    
def main(preprcs_df = lambda *x: x):
    flags.DEFINE_string('exp_folder', '', "Experiment folder path.")
    
    flags.DEFINE_string('mode', 'curve-epoch-val_loss', "Plot mode.")
    flags.DEFINE_string('filter', '', "filter values.")
    flags.DEFINE_spaceseplist('multi_line_fields', '', "List of fields to plot multiple lines over.")
    flags.DEFINE_spaceseplist('col_row_fields', '', "column and row fields for multiple figures.")
    flags.DEFINE_spaceseplist('best_ref_x_fields', '', "Reference x_field-values to evaluate optimal hyperparameters.")
    flags.DEFINE_string('best_ref_metric_field', '', "Reference metric_field-values to evaluate optimal hyperparameters.")
    flags.DEFINE_spaceseplist('best_ref_ml_fields', '', "Reference multi_line_fields-value to evaluate optimal hyperparameters.")
    flags.DEFINE_bool('best_at_max', False, 'Whether the bese metric value is the maximum value.')
    
    flags.DEFINE_string('plot_config', '', "Yaml file path for various plot setups.")
    flags.DEFINE_string('colors', '', "color scheme ('', 'cont').")
    flags.DEFINE_string('style', 'default', "Matplotlib style.")
    flags.DEFINE_bool('annotate', True, 'Run multiple plot according to given config.')
    flags.DEFINE_spaceseplist('annotate_field', '', 'List of fields to include in annotation.')
    flags.DEFINE_spaceseplist('fig_size', '', 'Figure size.')
    flags.DEFINE_string('xscale', '', "Scale of x-axis (linear, log).")
    flags.DEFINE_string('yscale', '', "Scale of y-axis (linear, log).")
    flags.DEFINE_string('title', '', "Title.")
    flags.DEFINE_string('xlabel', '', "Label of x-axis.")
    flags.DEFINE_string('ylabel', '', "Label of y-axis.")
    flags.DEFINE_integer('font_size', 22, "Font size of title and label.")

    app.run(partial(run, preprcs_df=preprcs_df))
    
    
if __name__=='__main__':
    main()