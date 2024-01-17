import os, shutil
from operator import attrgetter

from matplotlib.axes import Axes

def create_dir(dir):
  if os.path.exists(dir):
    for f in os.listdir(dir):
      if os.path.isdir(os.path.join(dir, f)):
        shutil.rmtree(os.path.join(dir, f))
      else:
        os.remove(os.path.join(dir, f))
  else:
    os.makedirs(dir)

def save_figure(fig, save_dir, save_name, png=False):
    fig.tight_layout()
    if not os.path.exists(save_dir):
        create_dir(save_dir)
    save_path = os.path.join(save_dir, save_name + '.pdf')
    fig.savefig(save_path, format='pdf')

    if png:
        png_save_dir = save_path.replace('/pdf', '/png')
        if not os.path.exists(png_save_dir):
            create_dir(png_save_dir)
        fig.savefig(os.path.join(png_save_dir, save_name + '.png'), bbox_inches='tight', format='png')
      
def merge_dict(base: dict, other: dict):
    """Merge plot_config dict (priority: ``base``)"""
    for k in (set(base) & set(other)):
        if isinstance(base[k], list):
            if base[k] and isinstance(base[k][-1], dict):
              base[k] = base[k][:-1] + other[k][:-1] \
                        + [merge_dict(base[k][-1], other[k][-1])]
        elif isinstance(base[k], dict):
            base[k] = merge_dict(base[k], other[k])
    for k in (set(other) - set(base)):
        base[k] = other[k]
    return base

default_style = {
    'annotate': False,
    'std_plot': 'fill',
    'ax_style':{
        'frame_width': 2.5,
        'fig_size': 7,
        'legend': [{'fontsize': 20}],
        'grid': [True, {'linestyle': '--'}],
        'tick_params':[{'axis': 'both',
                        'which': 'major',
                        'labelsize': 25,
                        'direction': 'in',
                        'length': 5}]
    },
    'line_style': {
        'linewidth': 4,
        'marker': 'D',
        'markersize': 10,
        'markevery': 1,
    }
}

def ax_styler(ax: Axes, **style_dict):
  if (n:='fig_size') in style_dict:
    dim = style_dict.pop(n)
    if isinstance(dim, int):
      w = h = dim
    else:
      w, h = dim
    ax.figure.set_figwidth(w)
    ax.figure.set_figheight(h)
  
  if (n:='frame_width') in style_dict:
    fw = style_dict.pop(n)
    for axis in ['top', 'bottom', 'left', 'right']:
      ax.spines[axis].set_linewidth(fw)
  
  non_set = ['tick_params', 'legend', 'grid']
  for name, (*arg_pos, arg_kw) in style_dict.items():
    attr_name = name if name in non_set else f'set_{name}'
    attrgetter(attr_name)(ax)(*arg_pos, **arg_kw)
    
  


