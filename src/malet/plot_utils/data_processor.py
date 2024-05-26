
import pandas as pd

def select_df(df, filt_dict, *exclude_fields, equal=True, drop=False):
    """Select df rows with matching from given filt_dict except ``exclude_fields``"""
    assert not df.empty, 'Given dataframe is empty.'
    assert not (k:=set(filt_dict.keys()) - set(df.index.names)), f'filt_dict keys {k} is not in df.'
    if not filt_dict: return df
    
    filt_keys = set(filt_dict.keys()) - set(exclude_fields)    # filter out exclude field
    
    nest = lambda vs: vs if isinstance(vs, list) else [vs]
    for i, k in enumerate(filt_keys):
        values = nest(filt_dict[k])
        assert not (v:=set(values)-(vs:=set(df.index.get_level_values(k)))), f"Values {v} are not in field '{k}': {sorted(vs)}"
        fltr = df.index.get_level_values(k).isin(values)
        df = df.loc[fltr if equal else ~fltr]
        assert not df.empty, f"Filter {k}:{values} return empty dataframe. Inspect {dict((k, filt_dict[k]) for k in filt_keys[:i+1])}" 
    
    if drop:
        df = df.reset_index([*filt_keys], drop=True)
        
    return df

    # for i, k in enumerate(filt_keys):
    #     has_none = any(map(pd.isnull, nest(filt_dict[k])))
    #     values = [*filter(pd.notnull, nest(filt_dict[k]))]
        
    #     assert not (v:=set(values)-(vs:=set(df.index.get_level_values(k).dropna()))), f"Values {v} are not in field '{k}': {sorted(vs)}"
    #     fltr = df.index.get_level_values(k).isin(values)
    #     sel_df = df.loc[fltr if equal else ~fltr]
        
    #     if has_none:
    #         assert df.reset_index(k).isnull().any().any(), f"Field '{k}' has no None value."
    #         sel_df = pd.concat([sel_df, df.loc[df.index.get_level_values(k).isnull()]])
            
    #     assert not sel_df.empty, f"Filter {k}:{values} return empty dataframe. Inspect {dict((k, filt_dict[k]) for k in filt_keys[:i+1])}" 
    
    # if drop:
    #     sel_df = sel_df.reset_index([*filt_keys], drop=True)
        
    # return sel_df


def homogenize_df(df, ref_df, filt_dict, *exclude_fields):
    """Homogenize index values of ``df`` with reference to ``select_df(ref_df, filt_dict)``."""
    ref_idx = select_df(ref_df, filt_dict, drop=True).index
    slcted_dfs = [select_df(df, dict(zip(ref_idx.names, d))) for d in ref_idx.values]
    df = pd.concat(slcted_dfs)
    return df

def avgbest_df(df, metric_field,
               avg_over=None, 
               best_over=tuple(), 
               best_of=dict(), 
               best_at_max=True):
    """Average over ``avg_over`` and get best result over ``best_over``
    
    Args:
        df (pandas.DataFrame): Base dataframe to operate over. All hyperparameters should be set as `MultiIndex`.
        metric_field (str): Column name of the metric. Used to evaluate best hyperparameter.
        avg_over (str): `MultiIndex` level name to average over.
        best_over (List[str]): List of `MultiIndex` level names to find value yielding best values of `metric_field`.
        best_of (Dict[str, Any]): Dictionary of pair `{MultiIndex name}: {value in MultiIndex}` to find best hyperparameter of. The other values in `{MultiIndex name}` will follow the best hyperparamter found for these values.
        best_at_max (bool): `True` when larger metric is better, and `False` otherwise.
        
    Returns: 
        pandas.DataFrame: Processed DataFrame
    """
    '''
    - aggregate index : avg_over, best_over
    - key index : best_of, others
    '''
    df_fields = set(df.index.names)
    
    # avg over avg_over
    if avg_over is not None:
        df_fields -= {avg_over}
        avg_over_group = df.groupby([*df_fields], dropna=True)
        df = avg_over_group.mean(numeric_only=True)
        df[metric_field+'_std'] = avg_over_group.sem(numeric_only=True)[metric_field]  # add std column
    
    # best result over best_over
    if best_over:
        # find best result over best_over for best_of
        df_fields -= set(best_over)
        best_df = select_df(df, best_of)
        if df_fields:
            best_df = best_df.loc[best_df.groupby([*df_fields])[metric_field]
                                         .aggregate(('idxmin', 'idxmax')[best_at_max])]
        else: # need since groupby returns series and causes error when df_fields is empty
            best_df = best_df.loc[best_df[metric_field].aggregate(('idxmin', 'idxmax')[best_at_max])]
            

        # match best_over values for non-best_of-key-index with best_of-key-index
        df_fields -= set(best_of)
        df = homogenize_df(df, best_df, best_of, *df_fields)
    
    return df
