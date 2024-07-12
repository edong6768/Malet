from typing import Literal

import numpy as np
import pandas as pd
from matplotlib.axes import Axes


def ax_draw_curve(ax: Axes,
                  df: pd.DataFrame,
                  label: str,
                  annotate = True,
                  annotate_field = [],
                  std_plot: Literal['none','fill','bar'] = 'fill',
                  unif_xticks = False,
                  color = 'orange', 
                  linewidth = 4, 
                  marker = 'D', 
                  markersize = 10, 
                  markevery = 20,
                  linestyle = '-',
                  **_
    ) -> Axes:
    """
    Draws curve of y_field over arbitrary x_field setted as index of the dataframe.
    If there is column 'y_field_sdv' is in dataframe, draws in errorbar or fill_between depending on ``sdv_bar_plot``
    """
    assert std_plot in {'bar', 'fill', 'none'}, 'std_plot should be one of {"bar","fill","none"}'
    y_field = list(df)[0]
    
    x_values, metric_values = map(np.array, zip(*dict(df[y_field]).items()))
    assert not isinstance(metric_values[0], pd.Series), f'y_field should have only {1} values for each index.'
    
    if len(x_values)>100:
        marker = None
        if unif_xticks: ax.locator_params(tight=True, nbins=5)
    elif len(x_values)>20:
        markevery = 20
        if unif_xticks: ax.locator_params(tight=True, nbins=5)
    else:
        markevery = 1
    
    tick_values = x_values
    
    artists = []
    
    if len(tick_values)==1:
        artists.append(
            ax.axhline(metric_values, linewidth=linewidth, color=color, label=label)
        )
        if f'{y_field}_std' in df:
            metric_std = float(df[f'{y_field}_std'])
            artists.append(
                ax.axhspan(metric_values[0] + metric_std, metric_values[0] - metric_std, alpha=0.3, color=color)
            )

    else:
        if unif_xticks:
            tick_values = np.arange(len(x_values))
            ax.set_xticks(tick_values, x_values, fontsize=10, rotation=45)
            
        artists += ax.plot(tick_values, metric_values, label=label, color=color, linewidth=linewidth, 
                            marker=marker, markersize=markersize, markevery=markevery, linestyle=linestyle)
        
        if len(x_values)%markevery!=0:
            artists += ax.plot(tick_values[-1], metric_values[-1], color=color, marker=marker, markersize=markersize)
        
        if f'{y_field}_std' in df:
            x_values, metric_std = map(np.array, zip(*dict(df[f'{y_field}_std']).items()))
            
            if std_plot=='bar':
                artists.append(
                    ax.errorbar(tick_values, metric_values, yerr=metric_std, color=color, elinewidth=3)
                )
            elif std_plot=='fill':
                artists.append(
                    ax.fill_between(tick_values, metric_values + metric_std, metric_values - metric_std, alpha=0.3, color=color)
                )
                
        if annotate:
            assert not (f:=set(annotate_field) - (a:=set(df) - {'total_steps', 'step', y_field, f'{y_field}_std'})), f'Annotation field: {f} are not in dataframe field: {a}'
            annotate_field = set(annotate_field) & a
            abv = lambda s: ''.join([i[0] for i in s.split('_')] if '_' in s else \
                                      [s[0]] + [i for i in s[1:] if i not in 'aeiou']) if len(s)>3 else s
            abv_annot = [*map(abv, annotate_field)]
            for i, (x,y,t) in enumerate(zip(x_values, metric_values, tick_values)):
                if i%markevery and i!=len(x_values)-1: continue
                txt = '\n'.join([f'{y:.5f}'+(f'$\pm${metric_std[i]:.5f}' if (f'{y_field}_std' in df and pd.notna(metric_std[i])) else ''), str(x)]
                                +[f'{i}={df.loc[x][j]}' for i, j in zip(abv_annot, annotate_field)])
                artists.append(
                    ax.annotate(txt, (t,y), textcoords="offset points", xytext=(0,10), ha='center')
                )
    
    ax.tick_params(axis='both', which='major', labelsize=17, direction='in', length=5)

    return artists

def ax_draw_best_stared_curve(ax: Axes,
                  df: pd.DataFrame,
                  label: str,
                  annotate = True,
                  annotate_field = [],
                  std_plot: Literal['none','fill','bar'] = 'fill',
                  best_at_max=True,
                  unif_xticks = False,
                  color = 'orange', 
                  linewidth = 4, 
                  marker = 'D', 
                  markersize = 10, 
                  markevery = 20,
                  linestyle = '-',
                  **_
    ) -> Axes:
    """
    Draws curve of y_field over arbitrary x_field setted as index of the dataframe.
    If there is column 'y_field_sdv' is in dataframe, draws in errorbar or fill_between depending on ``sdv_bar_plot``
    """
    assert std_plot in {'bar', 'fill', 'none'}, 'std_plot should be one of {"bar","fill","none"}'
    y_field = list(df)[0]
    
    x_values, metric_values = map(np.array, zip(*dict(df[y_field]).items()))
    assert not isinstance(metric_values[0], pd.Series), f'y_field should have only {1} values for each index.'
    
    if len(x_values)>100:
        marker = None
        if unif_xticks: ax.locator_params(tight=True, nbins=5)
    elif len(x_values)>20:
        markevery = 20
        if unif_xticks: ax.locator_params(tight=True, nbins=5)
    else:
        markevery = 1
    
    tick_values = x_values
    
    artists = []
    
    if len(tick_values)==1:
        artists.append(
            ax.axhline(metric_values, linewidth=linewidth, color=color, label=label)
        )
        if f'{y_field}_std' in df:
            metric_std = float(df[f'{y_field}_std'])
            artists.append(
                ax.axhspan(metric_values[0] + metric_std, metric_values[0] - metric_std, alpha=0.3, color=color)
            )

    else:
        if unif_xticks:
            tick_values = np.arange(len(x_values))
            ax.set_xticks(tick_values, x_values, fontsize=10, rotation=45)
            
        ax.plot(tick_values, metric_values, color=color, linewidth=linewidth)
        if f'{y_field}_std' in df:
            x_values, metric_std = map(np.array, zip(*dict(df[f'{y_field}_std']).items()))
            
            if std_plot=='bar':
                artists.append(
                    ax.errorbar(tick_values, metric_values, yerr=metric_std, color=color, elinewidth=3)
                )
            elif std_plot=='fill':
                artists.append(
                    ax.fill_between(tick_values, metric_values + metric_std, metric_values - metric_std, alpha=0.3, color=color)
                )
                
        best_idx = list(metric_values).index((max if best_at_max else min)(metric_values))
        for i, (_,y,t) in enumerate(zip(x_values, metric_values, tick_values)):
            if i%markevery: continue
            if i==best_idx:
                artists += ax.plot(tick_values[i], metric_values[i], color='green', 
                                   marker='*', markersize=markersize+10)
            else:
                artists += ax.plot(tick_values[i], metric_values[i], color=color, 
                                   marker=marker, markersize=markersize, markevery=markevery, linestyle=linestyle)
                
                
        if annotate:
            assert not (f:=set(annotate_field) - (a:=set(df) - {'total_steps', 'step', y_field, f'{y_field}_std'})), f'Annotation field: {f} are not in dataframe field: {a}'
            annotate_field = set(annotate_field) & a
            abv = lambda s: ''.join([i[0] for i in s.split('_')] if '_' in s else \
                                      [s[0]] + [i for i in s[1:] if i not in 'aeiou']) if len(s)>3 else s
            abv_annot = [*map(abv, annotate_field)]
            for i, (x,y,t) in enumerate(zip(x_values, metric_values, tick_values)):
                if i%markevery: continue
                txt = '\n'.join([f'{y:.5f}'+(f'$\pm${metric_std[i]:.5f}' if (f'{y_field}_std' in df and pd.notna(metric_std[i])) else ''), str(x)]
                                +[f'{i}={df.loc[x][j]}' for i, j in zip(abv_annot, annotate_field)])
                artists.append(
                    ax.annotate(txt, (t,y), textcoords="offset points", xytext=(0,10), ha='center')
                )
    
    ax.tick_params(axis='both', which='major', labelsize=17, direction='in', length=5)

    return artists


def ax_draw_bar(ax: Axes,
                df: pd.DataFrame,
                label: str,
                annotate = True,
                annotate_field = [],
                std_plot = True,
                unif_xticks = False,
                color = 'orange',
                **_
    ) -> Axes:
    """
    Draws bar graph of y_field over arbitrary x_field setted as index of the dataframe.
    If there is column 'y_field_sdv' is in dataframe, draws in errorbar or fill_between depending on ``sdv_bar_plot``
    """
    y_field = list(df)[0]
    
    x_values, metric_values = map(np.array, zip(*dict(df[y_field]).items()))
    assert not isinstance(metric_values[0], pd.Series), f'y_field should have only {1} values for each index.'
    
    tick_values = np.arange(len(x_values))
    ax.set_xticks(tick_values, x_values, fontsize=10, rotation=45)

    artists = []
    
    if std_plot and f'{y_field}_std' in df:
        x_values, metric_std = map(np.array, zip(*dict(df[f'{y_field}_std']).items()))
        artists.append(
            ax.bar(tick_values, metric_values, yerr=metric_std, label=label, color=color)
        )
    else:
        artists.append(
            ax.bar(tick_values, metric_values, label=label, color=color)
        )
        
    if annotate:
        assert not (f:=set(annotate_field) - (a:=set(df) - {'total_steps', 'step', y_field, f'{y_field}_std'})), f'Annotation field: {f} are not in dataframe field: {a}'
        annotate_field = set(annotate_field) & a
        abv = lambda s: ''.join([i[0] for i in s.split('_')] if '_' in s else \
                                [s[0]] + [i for i in s[1:] if i not in 'aeiou']) if len(s)>3 else s
        abv_annot = [*map(abv, annotate_field)]
        for x,y,t in zip(x_values, metric_values, tick_values):
            txt = '\n'.join([f'{y:.5f}'+(f'$\pm${metric_std[i]:.5f}' if (f'{y_field}_std' in df and pd.notna(metric_std[i])) else '')]
                            +['' if unif_xticks else str(x)]+[f'{i}={df.loc[x][j]}' for i, j in zip(abv_annot, annotate_field)])
            artists.append(
                ax.annotate(txt, (t,y), textcoords="offset points", xytext=(0,10), ha='center')
            )
    
    ax.tick_params(axis='both', which='major', labelsize=17, direction='in', length=5)

    return artists


def ax_draw_heatmap(ax: Axes,
                    df: pd.DataFrame,
                    cmap = 'magma', 
                    annotate=True,
                    annotate_field=[],
                    norm = None,
                    **_
        ) -> Axes:
    """
    Draws heatmap of y_field over two arbitrary x_fields setted as multi-index of the dataframe.
    """
    
    y_field = list(df)[0]
    y_field_df = df.drop(columns=list(df)[1:])
    
    x_fields = y_field_df.index.names
    grid_df = (
        y_field_df.reset_index()
                  .pivot(index=x_fields[1], columns=x_fields[0])
    )
    
    artists = [
        ax.pcolor(grid_df, cmap=cmap, edgecolors='w', norm=norm)
    ]
    
    *x_values, = map(lambda l: sorted(set(y_field_df.index.get_level_values(l))), x_fields)
    ax.set_xticks(np.arange(0.5, len(x_values[0]), 1), x_values[0], fontsize=10, rotation=45)
    ax.set_yticks(np.arange(0.5, len(x_values[1]), 1), x_values[1], fontsize=10)
        
    if annotate:
        assert not (f:=set(annotate_field) - (a:=set(df) - {'total_steps', 'step', y_field, f'{y_field}_std'})), f'Annotation field: {f} are not in dataframe field: {a}'
        annotate_field = set(annotate_field) & a
        abv = lambda s: ''.join([i[0] for i in s.split('_')] if '_' in s else \
                                [s[0]] + [i for i in s[1:] if i not in 'aeiou']) if len(s)>3 else s
        abv_annot = [*map(abv, annotate_field)]
        
        if f'{y_field}_std' in df:
            y_std_df = df.drop(columns=list(set(df)-{f'{y_field}_std'}))
            std_grid_df = (
                y_std_df.reset_index()
                        .pivot(index=x_fields[1], columns=x_fields[0])
            )
            
        for i, (mtc, x) in enumerate([*grid_df]):
            for j, y in enumerate([*grid_df.index.get_level_values(0)]):
                txt = '\n'.join([f'{grid_df.loc[y, (mtc, x)]:.5f}'
                                 +(f'\n$\pm${std_grid_df.loc[y, (f"{mtc}_std", x)]:.5f}' if (f'{y_field}_std' in df and pd.notna(std_grid_df.loc[y, (f'{mtc}_std', x)])) else '')]
                                 +[f'{i}={df.loc[(x, y), j]}' for i, j in zip(abv_annot, annotate_field) if df.index.isin([(x, y)]).any()])
                artists.append(
                    ax.text(i+0.5, j+0.5, txt, c='dimgrey', ha='center', va='center', weight='bold')
                )
    
    # ax.tick_params(axis='both', which='major', labelsize=17, direction='in', length=5)
    return artists



def ax_draw_scatter(ax: Axes,
                    df: pd.DataFrame,
                    y_fields: list,
                    color = 'orange',
                    marker = 'D', 
                    markersize = 30,
                    **_
    ) -> Axes:
    """
    Draws scatter of two arbitrary y_fields.
    """
    assert len(set(df.index.get_level_values('metric')))==2, f'There should be {2} metrics in the dataframe, got {set(df.index.get_level_values("metric"))}.'
    assert set(df.index.get_level_values('metric'))==set(y_fields), 'y_fields should be the same as metrics in the dataframe.' 
    
    df = df.reset_index(['total_steps', 'step'], drop=True)
    
    # revert back melted metrics into original column form
    prcs = lambda y: (df.loc[df.index.get_level_values('metric')==y]
                        .reset_index('metric', drop=True)
                        .rename(columns={'metric_value': y}))
    
    df = pd.concat([*map(prcs, y_fields)], axis=1)
    
    y1, y2 = map(lambda y: list(df[y]), y_fields)
    
    artists = [
        ax.scatter(y1, y2, color=color, marker=marker, s=markersize*20, edgecolors='black')
    ]
    
    return artists

    


def ax_draw_scatter_heat(ax: Axes,
                         df: pd.DataFrame,
                         y_fields: list,
                         cmap = 'magma', 
                         marker = 'D', 
                         markersize = 30,
                         norm=None,
                         **_
    ) -> Axes:
    """
    Draws scatter with colors of three arbitrary y_fields.
    """
    assert len(set(df.index.get_level_values('metric')))==3, f'There should be {3} metrics in the dataframe, got {set(df.index.get_level_values("metric"))}.'
    assert set(df.index.get_level_values('metric'))==set(y_fields), 'y_fields should be the same as metrics in the dataframe.' 
    
    df = df.reset_index(['total_steps', 'step'], drop=True)
    
    # revert back melted metrics into original column form
    prcs = lambda y: (
                df.loc[df.index.get_level_values('metric')==y]
                  .reset_index('metric', drop=True)
                  .rename(columns={'metric_value': y})
            )
    
    df = pd.concat([*map(prcs, y_fields)], axis=1)
    
    y1, y2, y3 = map(lambda y: list(df[y]), y_fields)
    
    artists = [
        ax.scatter(y1, y2, c=y3, marker=marker, s=markersize*20, norm=norm, cmap=cmap)
    ]
    
    return ax