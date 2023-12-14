import os
import re
import yaml

from absl import app, flags, logging
from ml_collections import ConfigDict

import matplotlib.pyplot as plt
import seaborn as sns

from .experiment import Experiment, ExperimentLog
from .utils import str2value, box_str

from .plot_utils.metric_drawer import *
from .plot_utils.utils import *

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


def draw_metric(tsv_file, plot_config, save_name='', preprcs_df=lambda *x: x):
        pcfg = plot_config
        
        # parse mode string
        mode, x_field, metric = pcfg['mode'].split('-') # ex) {sam}-{epoch}-{train_loss}
        pflt, pmlf = map(pcfg.get, ['filter', 'multi_line_field'])
            
        # get dataframe
        pai_history = ExperimentLog.from_tsv(tsv_file)
        df = pai_history.explode_and_melt_metric(epoch=None if x_field=='epoch' else -1)
        base_config = ConfigDict(pai_history.static_configs)
        
        
        #---filter df according to FLAGS.filter
        if pflt:
            save_name += pflt.replace(' / ', '-').replace(' ', '_')
            filt_dict = map(lambda flt: re.split('(?<!,) ', flt.strip()), pflt.split('/')) # split ' ' except ', '
            df = select_df(df, {fk:[*map(str2value, fvs)] for fk, *fvs in filt_dict}) 
        
        #---set mlines according to FLAGS.multi_line_field
        if pmlf:
            save_name = pmlf + (f'-{save_name}' if save_name else '')
            mlines = sorted(set(df.index.get_level_values(pmlf)), key=str2value)
        else:
            pmlf, mlines = 'metric', [metric]
        
        #---enter other configs in save name
        if any([pcfg[f'best_ref_{k}'] for k in ['x_field', 'metric_field', 'ml_field']]):
            save_name +=  f"-({pcfg['best_ref_x_field']}, {pcfg['best_ref_metric_field']}, {pcfg['best_ref_ml_field']})"
        
        save_name += "-max" if pcfg['best_at_max'] else "-min"
        
        best_over = set(df.index.names) - {x_field, 'metric', 'seed', pmlf}
        best_at_max = pcfg['best_at_max']
        if x_field=='epoch':
            pcfg['best_ref_x_field']=base_config.num_epochs-1
        
        # Notify selected plot configs and field handling statistics
        specified_field = {k for k in best_over if len(set(df.index.get_level_values(k)))==1}
        logging.info('\n\n' + box_str('Plot configuration', 
                                      '\n'.join([f'- {k}: {pcfg[k]}' 
                                                 for k in ('mode', 'multi_line_field', 
                                                           'filter', 'best_at_max', 
                                                           'best_ref_x_field', 'best_ref_metric_field', 
                                                           'best_ref_ml_field') if pcfg[k]]),
                                      box_width=150-9, indent=9, skip=0) +
                     '\n\n' + box_str("Field handling statistics",
                                      f'''- Key field (has multiple values): {[x_field, pmlf]} (2)
                                          - Specified field: {(spf:=[*specified_field, 'metric'])} ({len(spf)})
                                          - Averaged field: {['seed']} (1)
                                          - Optimized field: {(opf:=list(best_over-specified_field))} ({len(opf)})''',
                                      box_width=150-9, indent=9, skip=0)
                     )
        ############################# Prepare dataframe #############################
        
        best_of = {}
        if pcfg['best_ref_x_field']: # same hyperparameter over all points in line
            best_of[x_field] = pcfg['best_ref_x_field']

        if pcfg['best_ref_metric_field']: # Optimize in terms of reference metric, and apply those hyperparameters to original
            best_of['metric'] = pcfg['best_ref_metric_field']
        
        if pcfg['best_ref_ml_field']: # same hyperparameter over all line in multi_line_field
            best_of[pmlf] = pcfg['best_ref_ml_field']
            
        # change field name and avg over seed and get best result over best_over
        best_df = avgbest_df(df, 'metric_value',
                             avg_over='seed', 
                             best_over=best_over, 
                             best_of=best_of,
                             best_at_max=best_at_max)
        
        ############################# Plot #############################
        
        # prepare plot
        fig, ax = plt.subplots()
        if pcfg['colors']=='':
            colors = iter(sns.color_palette()*10)
        elif pcfg['colors']=='cont':
            colors = iter([c for i, c in enumerate(sum(map(sns.color_palette, ["light:#9467bd", "Blues", "rocket", "crest", "magma"]*3), [])[1:]) if i%2])
        
        # select specified metric if multi_line_field isn't metric
        if pmlf!='metric':
            best_df = select_df(best_df, {'metric': metric}, x_field)
            
        for mlv in mlines:
            p_df = select_df(best_df, {pmlf: mlv}, x_field)
            legend = str(mlv).replace('_', ' ')
            
            p_df, legend = preprcs_df(p_df, legend) 
            
            # remove unnessacery fields
            p_df = p_df.reset_index([*(set(p_df.index.names) - {x_field})], drop=True)
            p_df = p_df.sort_index(level=x_field, key=lambda s: [*map(str2value, s)])
            
            pcfg['line_style']['color'] = next(colors)
            ax = ax_draw(ax, p_df, legend,
                         annotate=pcfg['annotate'],
                         std_plot=pcfg['std_plot'],
                         plot_config=pcfg['line_style'])
        
        y_label = metric.replace('_', ' ').capitalize()
        
        return fig, ax, y_label, save_name.strip('-')
        
def run(argv):
    if len(argv)>2:
        raise app.UsageError('Too many command-line arguments.')
    
    # Preprocess plot_config
    flag_dict = FLAGS.flag_values_dict()
    plot_config = {**default_style, **flag_dict}
    if FLAGS.plot_config!='':
        with open(FLAGS.plot_config) as f:
            plot_config = yaml.safe_load(f.read())
            plot_config = get_plot_config(plot_config, flag_dict)
    
    # get paths
    _, tsv_file, fig_dir = Experiment.get_paths(plot_config['exp_folder'])
    save_dir = os.path.join(fig_dir, plot_config['mode'])
    
    if 'curve' in plot_config['mode']:
        fig, ax, y_label, save_name = draw_metric(tsv_file, plot_config)
        ax.set_ylabel(y_label)
    else:
        assert False, f'Mode: {plot_config["mode"]} does not exist.'
        
    ax_styler(ax, **plot_config['ax_style'])
    save_figure(fig, save_dir, save_name)
    logging.info('\n\n' + \
                  box_str('Plot complete', 
                         f'save plot at: {save_dir}/{save_name}.pdf',
                         box_width=150-9, indent=9, skip=0))
    
def main():
    flags.DEFINE_string('exp_folder', '', "Experiment folder path.")
    
    flags.DEFINE_string('mode', 'curve-epoch-val_loss', "Plot mode.")
    flags.DEFINE_string('filter', '', "filter values.")
    flags.DEFINE_string('multi_line_field', '', "Field to plot multiple lines over.")
    flags.DEFINE_string('best_ref_x_field', '', "Reference x_field-value to evaluate optimal hyperparameters.")
    flags.DEFINE_string('best_ref_metric_field', '', "Reference metric_field-value to evaluate optimal hyperparameters.")
    flags.DEFINE_string('best_ref_ml_field', '', "Reference multi_line_field-value to evaluate optimal hyperparameters.")
    flags.DEFINE_bool('best_at_max', False, 'Whether the bese metric value is the maximum value.')
    
    flags.DEFINE_string('plot_config', '', "Yaml file path for various plot setups.")
    flags.DEFINE_string('colors', '', "color scheme ('', 'cont').")
    flags.DEFINE_bool('annotate', True, 'Run multiple plot according to given config.')

    app.run(run)
    
    
if __name__=='__main__':
    main()