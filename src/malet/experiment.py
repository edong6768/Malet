import os, glob, shutil, io
import copy
import yaml
import re
from functools import reduce, partial
from typing import Optional, ClassVar, Callable, Union, TypeVar
from dataclasses import dataclass
from itertools import product, chain
from datetime import datetime
from datetime import timedelta
from git import Repo

import pandas as pd
import numpy as np
from rich.progress import track

from absl import logging
from ml_collections.config_dict import ConfigDict

from .utils import list2tuple, str2value, QueuedFileLock

Self = TypeVar("Self", bound="Experiment")
ExpFunc = Union[Callable[[ConfigDict], dict], Callable[[ConfigDict, Self], dict]]

class ConfigIter:
  '''
  Iterator of configs generated from yaml config generator file
  Generates grid of configs from given experiment plans in yaml.
    
  [yaml exp_file example]
   model: ResNet32
   dataset: cifar10
   ...
   grid:
     - optimizer: [sgd]
       group:
         pai_type: [[random], [snip]]
         pai_scope: [[local], [global]]
       rho: [0.05]
       seed: [1, 2, 3]
     - optimizer: [sam]
       pai_type: [random, snip, lth]
       pai_sparsity: [0, 0.9, 0.95, 0.98, 0.99]
       rho: [0.05, 0.1, 0.2, 0.3]
       seed : [1, 2, 3]
  '''
  
  def __init__(self, exp_config_path):
    with open(exp_config_path) as f:
      cnfg_str = self.__sub_cmd(f.read())
      self.grid_fields = self.__extract_grid_order(cnfg_str)
      self.raw_config = yaml.safe_load(cnfg_str)
    
    self.name = os.path.split(exp_config_path)[0].split('/')[-1]
    self.grid = self.raw_config.get('grid', {})
    self.static_configs = {k:self.raw_config[k] for k in set(self.raw_config)-{'grid_fields', 'grid'}}
    
    assert not (f:={k for k in self.static_configs.keys() if k in self.grid_fields}), f'Overlapping fields {f} in Static configs and grid fields.'
  
    self.grid_iter = self.__get_iter()
    
  
  def filter_iter(self, filt_fn):
    """filters ConfigIter with ``filt_fn`` which has (idx, dict) as arguments"""
    self.grid_iter = [d for i, d in enumerate(self.grid_iter) if filt_fn(i, d)]
    
  @property
  def grid_dict(self):
    """dictionary of all grid values"""
    if not self.grid_fields: return dict()
    st, *sts = self.grid
    castl2t = lambda vs: map(lambda v: (tuple(v) if isinstance(v, list) else v), vs)
    acc = lambda a, d: {k: [*{*castl2t(va), *castl2t(vd)}] 
                        for (k, va), (_, vd) in zip(a.items(), d.items())}
    grid_dict = {k: [*map(self.field_type(k), vs)] for k, vs in reduce(acc, sts, st).items()}
    return grid_dict
  
  def field_type(self, field):
    """Returns type of a field"""
    return type(self[0][field])
  
  @staticmethod
  def __sub_cmd(cfg_str):
    '''compile special commands in experiment plan (study) of cfg_str'''
    
    # \[__;0:2:10--11:5:20] -> [__ for i in range(0, 10, 2)] + [__ for i in range(11, 20, 5)]
    for entry in re.finditer(p:='\[[^;\n]+;(\d+:\d+:\d+(--)?)+\]', cfg_str):
      f, rngs = entry.group()[1:-1].split(';')
      sdes = [map(int, rng.split(':')) for rng in rngs.split('--')]
      assert 'i' in f, f"Variable i should be in the expression '{entry}'"
      assert re.sub('[^\+\-\*/\[\]i\d\(\), ]', '', f)==f, f"Cannot use alphabet other than 'i' in expression '{entry}'"
      
      rep = sum([eval(f'[{f} for i in {range(s, e, d)}]') for s, d, e in sdes], start=[])
      cfg_str=re.sub(p, str(rep), cfg_str, 1)
    
    return cfg_str
    
  @staticmethod
  def __extract_grid_order(cfg_str):
    """parse grid order from raw config string"""
    if 'grid' not in cfg_str: return []
    
    grid = re.split('grid ?:', cfg_str)[1]
    names = re.findall('[\w_]+(?= ?:)', grid)
    
    dupless_names = []
    for n in names:
      if n in dupless_names or n=='group': continue
      dupless_names.append(n)
      
    return dupless_names
    
  @staticmethod
  def __ravel_group(grid):
    # Return list of grid if there is no 'group'
    if 'group' not in grid: return [grid]
    group = grid['group']
    
    # Ravel grouped fields into list of experiment plans.
    def grid_g(g):
      g_len = [*map(len, g.values())]
      assert all([l==g_len[0] for l in g_len]), f'Grouped fields should have same length, got fields with length {dict(zip(g.keys(), g_len))}'
      return ([*zip(g.keys(), value)] for value in zip(*g.values()))
    
    if isinstance(group, dict):
      r_groups = grid_g(group)
    elif isinstance(group, list):
      r_groups = (chain(*gs) for gs in product(*map(grid_g, group)))
    grid.pop('group')
    
    raveled_study = ({**grid, **dict(g)} for g in r_groups)
    return raveled_study

  def __get_iter(self):
    if self.grid_fields is None: return [dict()]
    
    # Prepare Experiment, create experiment plan (grid)
    if type(self.grid)==dict:
      self.grid = [*self.__ravel_group(self.grid)]
    elif type(self.grid)==list:
      self.grid = [*chain(*map(self.__ravel_group, self.grid))]
      
    grid_s = lambda s: product(*map(s.get, self.grid_fields))
    grid_iter = chain(*map(grid_s, self.grid))
    grid_iter = [dict(zip(self.grid_fields, i)) for i in grid_iter]
    
    return grid_iter
  
  def __getitem__(self, idx):
    if isinstance(idx, int):
      return ConfigDict({**self.static_configs, **self.grid_iter[idx]})
    elif isinstance(idx, slice):
      new_ci = copy.deepcopy(self)
      new_ci.grid_iter = new_ci.grid_iter[idx]
      return new_ci
      
  def __len__(self):
    return len(self.grid_iter)
    
  
# Only a temporary measure for empty grid_fields
pd.DataFrame.old_set_index = pd.DataFrame.set_index
pd.DataFrame.old_reset_index = pd.DataFrame.reset_index
pd.DataFrame.old_drop = pd.DataFrame.drop
pd.DataFrame.set_index = lambda self, idx, *__, **_: self if not idx else self.old_set_index(idx, *__, **_)
pd.DataFrame.reset_index = lambda self, *__, **_: self if self.index.names==[None] else self.old_reset_index(*__, **_)
pd.DataFrame.drop = lambda self, *_, axis=0, **__: pd.DataFrame(columns=self.columns) if len(self)<2 and axis==0 else self.old_drop(*_, axis=axis, **__)

@dataclass
class ExperimentLog:
  static_configs: dict
  grid_fields: list
  logs_file: str
  
  metric_fields: Optional[list] = None
  df: Optional[pd.DataFrame]=None
  auto_update_tsv: bool = False
  
  __sep: ClassVar[str] = '-'*45 + '\n'
  
  def __post_init__(self):
    if self.df is None:
      assert self.metric_fields is not None, 'Specify the metric fields of the experiment.'
      assert not (f:=set(self.grid_fields) & set(self.metric_fields)), f'Overlapping field names {f} in grid_fields and metric_fields. Remove one of them.'
      self.df = pd.DataFrame(columns=self.grid_fields+self.metric_fields).set_index(self.grid_fields)

    else:
      self.metric_fields = list(self.df)
    
    self.filelock = QueuedFileLock(self.logs_file+'.lock', timeout=3*60*60)
  
  # Constructors.
  # -----------------------------------------------------------------------------  
  @classmethod
  def from_exp_config(cls, exp_config, logs_file: str, metric_fields: Optional[list]=None, auto_update_tsv: bool=False):
    return cls(*(exp_config[k] for k in ['static_configs', 'grid_fields']), logs_file=logs_file,
               metric_fields=metric_fields, auto_update_tsv = auto_update_tsv)

  @classmethod
  def from_tsv(cls, logs_file: str, parse_str=True, auto_update_tsv: bool=False):
    '''open tsv with yaml header'''
    with QueuedFileLock(logs_file+'.lock', timeout=3*60*60):
      log = cls(**cls.parse_tsv(logs_file, parse_str=parse_str), logs_file=logs_file, auto_update_tsv=auto_update_tsv)
    return log
  
  
  # tsv handlers.
  # -----------------------------------------------------------------------------
  @classmethod
  def parse_tsv(cls, log_file: str, parse_str=True):
    '''parses tsv file into usable datas'''
    assert os.path.exists(log_file), f'File path "{log_file}" does not exists.'

    with open(log_file, 'r') as fd:
      # process yaml config header
      def header():
        next(fd)
        header = ''
        for s in fd:
          if s==cls.__sep: break
          header += s
        return header
    
      # get workload data from yaml header
      static_configs = yaml.safe_load(header())

      # get dataframe from csv body
      tsv_str = fd.read()
    
    tsv_col, tsv_idx, *tsv_body = tsv_str.split('\n')
    col = tsv_col.strip().split('\t')
    idx = tsv_idx.strip().split('\t')
    tsv_head = '\t'.join(idx+col)
    tsv_str = '\n'.join([tsv_head, *tsv_body])
    
    df = pd.read_csv(io.StringIO(tsv_str), sep='\t')
    df = df.drop(['id'], axis=1)
    
    if parse_str:
        df = df.applymap(str2value)
        
    # set grid_fields to multiindex
    df = df.set_index(idx[1:])
      
    return {'static_configs': static_configs,
            'grid_fields': idx[1:],
            'metric_fields': col,
            'df': df}
  
  
  def lock_file(func):
    '''Decorator for filelock acquire/release before/after given function call'''
    def wrapped(self, *args, **kwargs):
      with self.filelock:
        ret = func(self, *args, **kwargs)
      return ret
    return wrapped
  
  @lock_file
  def load_tsv(self, logs_file=None, parse_str=True):
    '''load tsv with yaml header'''
    if logs_file is not None:
      self.logs_file=logs_file
    
    for k, v in self.parse_tsv(self.logs_file, parse_str=parse_str).items():
      self.__dict__[k] = v

  @lock_file
  def to_tsv(self, logs_file=None):
    logs_file = self.logs_file if logs_file==None else logs_file
    
    logs_path, _ = os.path.split(logs_file)
    if not os.path.exists(logs_path):
        os.makedirs(logs_path) 

    # pandas dataframe to tsv string
    df = self.df.reset_index()
    df['id'] = [*range(len(df))]
    df = df.set_index(['id', *self.grid_fields])
    tsv_str = df.to_csv(sep='\t')
    
    tsv_head, *tsv_body = tsv_str.split('\n')
    tsv_head = tsv_head.split('\t')
    col = '\t'.join([' '*len(i) if i in df.index.names else i for i in tsv_head])
    idx = '\t'.join([i if i in df.index.names else ' '*len(i) for i in tsv_head])
    tsv_str = '\n'.join([col, idx, *tsv_body])
    
    # write static_configs and table of results
    with open(logs_file, 'w') as fd:
      fd.write('[Static Configs]\n')
      yaml.dump(self.static_configs, fd)
      fd.write(self.__sep)
      fd.write(tsv_str)
      
  
  def update_tsv(func, mode='rw'):
    '''Decorator for read/write tsv before/after given function call'''
    def wrapped(self, *args, **kwargs):
      if self.auto_update_tsv and 'r' in mode: self.load_tsv(self.logs_file)
      ret = func(self, *args, **kwargs)
      if self.auto_update_tsv and 'w' in mode: self.to_tsv()
      return ret
    return wrapped
  
  # Add results.
  # -----------------------------------------------------------------------------
  
  @partial(update_tsv, mode='r')
  def add_result(self, configs, **metrics):
    '''Add experiment run result to dataframe'''
    if configs in self:
      cur_gridval = list2tuple([configs[k] for k in self.grid_fields])
      self.df = self.df.drop(cur_gridval)
    
    configs = {k:list2tuple(configs[k]) for k in self.grid_fields}
    metrics = {k:metrics.get(k) for k in self.metric_fields}
    result_dict = {k:[v] for k, v in {**configs, **metrics}.items()}
    result_df = pd.DataFrame(result_dict).set_index(self.grid_fields)
    self.df = pd.concat([self.df, result_df])[self.metric_fields]
    
    # Dataframe.loc based code
    # problem: pandas unable to distinguish between 
    #              ``.loc[(col_idx, row_idx)]`` and ``.loc[(multi_idx_1, multi_idx_2)]``
    #          for 2-level multiindex dataframe
    """
    cur_gridval = list2tuple([configs[k] for k in self.grid_fields])
    
    df_row = [metrics.get(k) for k in self.field_order]
      
    # Write over metric results if there is a config saved
    if configs in self:
      self.df = self.df.drop(cur_gridval)
    
    self.df.loc[cur_gridval] = df_row
    """
    
  @staticmethod
  def __add_column(df, new_column_name, fn, *fn_arg_fields):
    '''Add new column field computed from existing fields in self.df'''
    def mapper(*args):
      if all(isinstance(i, (int, float, str, tuple, list)) for i in args):
        return fn(*args)
      return None
    df[new_column_name] = df.apply(lambda df: mapper(*[df[c] for c in fn_arg_fields]), axis=1)
    return df

  def add_computed_metric(self, new_metric_name, fn, *fn_arg_fields):
    '''Add new metric computed from existing metrics in self.df'''
    self.df = self.__add_column(self.df, new_metric_name, fn, *fn_arg_fields)
    self.metric_fields.append(new_metric_name)
    
  def add_derived_index(self, new_index_name, fn, *fn_arg_fields):
    '''Add new index field computed from existing fields in self.df'''
    df = self.df.reset_index(self.grid_fields)
    df = self.__add_column(df, new_index_name, fn, *fn_arg_fields)
    self.grid_fields.append(new_index_name)
    self.df = df.set_index(self.grid_fields)
    
  def remove_metric(self, *metric_names):
    self.df = self.df.drop(columns=[*metric_names])
    self.metric_fields = [m for m in self.grid_fields if m not in metric_names]
    
  def remove_index(self, *field_names):
    self.df = self.df.reset_index([*field_names], drop=True)
    self.grid_fields = [f for f in self.grid_fields if f not in field_names]

  # Merge ExperimentLogs.
  # -----------------------------------------------------------------------------
  def __merge_one(self, other, same=True):
    '''
    Merge two logs into self.
    - The order of grid_fields follows self.
    - Difference between static_configs are moved to grid_fields.
    - If grid_fields are different between self & other
       - If it exists in static_configs, they are moved to grid_fields.
       - else it is filled with np.nan
    '''
    if same:
      assert self==other, 'Different experiments cannot be merged by default.'

    # find different fixed configs
    def same_diff(dictl, dictr):
      keys = set(dictl.keys()) & set(dictr.keys())
      same, diff = dict(), []
      for k in keys:
        if dictl[k]==dictr[k]: same[k]=dictl[k]
        else: diff.append(k)
      return same, diff
    
    new_sttc, diff_sttc = same_diff(self.static_configs, other.static_configs)

    # find new grid_fields
    new_to_self_sf = [sf for sf in other.grid_fields if sf not in self.grid_fields] + diff_sttc
    new_to_othr_sf = [sf for sf in self.grid_fields if sf not in other.grid_fields] + diff_sttc

    # fill in new grid_fields in each df from static_configs and configs
    # change list configs to tuple for hashablilty
    for sf in new_to_self_sf:
      self.df[sf] = [list2tuple(self.static_configs.get(sf, np.nan))]*len(self)

    for sf in new_to_othr_sf:
      other.df[sf] = [list2tuple(other.static_configs.get(sf, np.nan))]*len(other)

    self.static_configs = new_sttc
    self.grid_fields += new_to_self_sf
    
    self.df, other.df = (obj.df.reset_index() for obj in (self, other))
    self.df = pd.concat([self.df, other.df]) \
                .set_index(self.grid_fields)[self.metric_fields]
    
    return self

  def merge(self, *others, same=True):
    '''Merge multiple logs into self'''
    for other in others:
      self.__merge_one(other, same=same)

  @staticmethod
  def merge_tsv(*names, logs_path, save_path=None, same=True):
    if save_path is None:
      save_path = os.path.join(logs_path, 'log_merged.tsv')
    base, *logs = [ExperimentLog.from_tsv(os.path.join(logs_path, n+'.tsv'), parse_str=False) for n in names]
    base.merge(*logs, same=same)
    base.to_tsv(save_path)

  @staticmethod
  def merge_folder(logs_path, save_path=None, same=True):
    """change later if we start saving tsvs to other directories"""
    os.chdir(logs_path)
    logs = [f[:-4] for f in glob.glob("*.tsv")]
    ExperimentLog.merge_tsv(*logs, logs_path=logs_path, save_path=save_path, same=same)
    
  
  # Utilities.
  # -----------------------------------------------------------------------------

  def __cfg_match_row(self, config):
    if not self.grid_fields: return self.df
    
    grid_filt = reduce(lambda l, r: l & r, 
                       (self.df.index.get_level_values(k)==(str(config[k]) if isinstance(config[k], list) else config[k]) 
                        for k in self.grid_fields))
    return self.df[grid_filt]
  
  
  @partial(update_tsv, mode='r')
  def isin(self, config):
    '''Check if specific experiment config was already executed in log.'''
    if self.df.empty: return False

    cfg_same_in_static = all([config[k]==v for k, v in self.static_configs.items() if k in config])
    cfg_matched_df = self.__cfg_match_row(config)
    
    return cfg_same_in_static and not cfg_matched_df.empty


  def get_metric(self, config):
    '''Search matching log with given config dict and return metric_dict, info_dict'''
    assert config in self, 'config should be in self when using get_metric_dict.'
    
    cfg_matched_df = self.__cfg_match_row(config)
    metric_dict = {k:(v.iloc[0] if not (v:=cfg_matched_df[k]).empty else None) for k in self.metric_fields}
    return metric_dict

  def is_same_exp(self, other):
    '''Check if both logs have same config fields.'''
    fields = lambda log: set(log.static_configs.keys()) | set(log.grid_fields)
    return fields(self)==fields(other)
    
    
  def melt_and_explode_metric(self, df=None, step=None):
    if df is None: 
      df = self.df
    mov_to_index = lambda *fields: df.reset_index().set_index([*df.index.names, *fields])
    
    # melt
    df = df.melt(value_vars=list(df), var_name='metric', value_name='metric_value', ignore_index=False)
    df = mov_to_index('metric')
    
    # Create step field and explode
    pseudo_len = lambda x: len(x) if isinstance(x, list) else 1
    
    df['total_steps'] = df['metric_value'].map(pseudo_len)
    
    if step is None:
        df['step'] = df['metric_value'].map(lambda x: range(1, pseudo_len(x)+1))
        df = df.explode('step')  # explode metric list so each step gets its own row
    else:
        df['step'] = df['metric_value'].map(lambda x: step + (pseudo_len(x) if step<0 else 0))
    
    df['metric_value'] = df.apply(lambda df: df['metric_value'][df.step-1] if isinstance(df['metric_value'], list) else df['metric_value'], axis=1) # list[epoch] for all fields
    
    df = mov_to_index('step', 'total_steps')
    
    # delete string and NaN valued rows
    df = df[pd.to_numeric(df['metric_value'], errors='coerce').notnull()]\
              .dropna()\
              .astype('float')
    
    return df

    
  def __contains__(self, config):
    return self.isin(config)

  def __eq__(self, other):
    return self.is_same_exp(other)

  def __len__(self):
    return len(self.df)

  def __str__(self):
    return '[Static Configs]\n' + \
           '\n'.join([f'{k}: {v}' for k,v in self.static_configs.items()]) + '\n' + \
           self.__sep + \
           str(self.df)


class RunInfo:
  infos: ClassVar[list] = ['datetime', 'duration', 'commit_hash']

  def __init__(self, prev_duration: timedelta=timedelta(0)):
    self.__datetime = datetime.now()
    self.__duration = prev_duration
    
    try:
      self.__commit_hash = Repo.init().head.commit.hexsha
    except:
      self.__commit_hash = None
      logging.info('No git exist in current directory.')
     
  def get(self):
    return {
      'datetime':     self.__datetime,
      'duration':     self.__duration,
      'commit_hash':  self.__commit_hash
    }
  
  def update_and_get(self):
    curr_t = datetime.now()
    self.__duration += curr_t - self.__datetime
    self.__datetime = curr_t 
    return self.get()


class Experiment:
  '''
  Executes experiments according to experiment configs
  
  Following is supported
  - Provides 2 methods parallel friedly experiments scheduling (can choose with bash arguments).
    - (plan splitting) Splits experiment plans evenly.
    - (current run checking) Save configs of currently running experiments to tsv so other running code can know.
  - Saves experiment logs, automatically resumes experiment using saved log.
  '''
  __RUNNING: ClassVar[str] = 'R'
  __FAILED: ClassVar[str] = 'F'
  __COMPLETED: ClassVar[str] = 'C'
  
  infos: ClassVar[list] = [*RunInfo.infos, 'status']
  
  def __init__(self, 
               exp_folder_path: str,
               exp_function: ExpFunc,
               exp_metrics: Optional[list] = None,
               total_splits: Union[int, str] = 1, 
               curr_split: Union[int, str] = 0,
               auto_update_tsv: bool = False,
               configs_save: bool = False,
               checkpoint: bool = False
    ):
    
    if checkpoint:
      assert auto_update_tsv, "argument 'auto_update_tsv' should be set to True when checkpointing."
    
    self.exp_func = exp_function

    self.configs_save = configs_save
    self.checkpoint = checkpoint
    
    do_split = isinstance(total_splits, int) and total_splits>1 or isinstance(total_splits, str)
    cfg_file, tsv_file, _ = self.get_paths(exp_folder_path, split=curr_split if do_split else None)
    
    self.configs = self.__get_and_split_configs(cfg_file, total_splits, curr_split)
    self.log = self.__get_log(tsv_file, self.infos+exp_metrics, auto_update_tsv)
    
    self.__check_matching_static_configs()
    
  @staticmethod
  def __get_and_split_configs(cfg_file, exp_bs, exp_bi):
    
    configiter = ConfigIter(cfg_file)
    
    assert isinstance(exp_bs, int) or (exp_bs in configiter.grid_fields), f'Enter valid splits (int | Literal{configiter.grid_fields}).'
    
    # if total exp split is given as integer : uniformly split
    if isinstance(exp_bs, int):
      assert exp_bs > 0, 'Total number of experiment splits should be larger than 0'
      assert exp_bs > exp_bi, 'Experiment split index should be smaller than the total number of experiment splits'
      if exp_bs>1:
        configiter.filter_iter(lambda i, _: i%exp_bs==exp_bi)
      
      logging.info(f'Experiment : {configiter.name} (split : {exp_bi+1}/{exp_bs})')
      
    # else split across certain study field
    elif exp_bs in configiter.grid_fields:
      exp_bi = [*map(str2value, exp_bi.split())]
      configiter.filter_iter(lambda _, d: d[exp_bs] in exp_bi)
      
      logging.info(f'Experiment : {configiter.name} (split : {exp_bi}/{configiter.grid_dict[exp_bs]})')
    
    return configiter
      
  def __get_log(self, logs_file, metric_fields=None, auto_update_tsv=False):
    # Configure experiment log
    if os.path.exists(logs_file): # Check if there already is a file
      log = ExperimentLog.from_tsv(logs_file, auto_update_tsv=auto_update_tsv) # resumes automatically
      
    else: # Create new log
      logs_path, _ = os.path.split(logs_file)
      if not os.path.exists(logs_path):
        os.makedirs(logs_path)
      log = ExperimentLog.from_exp_config(self.configs.__dict__, logs_file,
                                          metric_fields=metric_fields, auto_update_tsv=auto_update_tsv)
      log.to_tsv()
      
    return log
  
  
  def __check_matching_static_configs(self):
    iter_statics = self.configs.static_configs
    log_statics = self.log.static_configs
    # check matching keys
    ist, lst = {*iter_statics.keys()}, {*log_statics.keys()}
    assert not (k:=ist^lst), f"Found non-matching keys {k} in static config of configiter and experiement log."
    
    # check matching values
    non_match = {k:(v1, v2) for k in ist if (v1:=iter_statics[k])!=(v2:=log_statics[k])}
    assert not non_match, f"Found non-matching values {non_match} in static config of configiter and experiement log."
  
  
  @staticmethod
  def get_paths(exp_folder, split=None):
    cfg_file = os.path.join(exp_folder, 'exp_config.yaml')
    
    if split==None:
      tsv_file = os.path.join(exp_folder, 'log.tsv')
    else:
      tsv_file = os.path.join(exp_folder, 'log_splits', f'split_{split}.tsv')
      
    fig_dir = os.path.join(exp_folder, 'figure')
    return cfg_file, tsv_file, fig_dir
  
  
  def get_metric_info(self, config):
    if config not in self.log: 
      logging.info("Log of matching config is not found. Returning empty dictionaries.")
      return {}, {} # return empty dictionaries if no log is found
    
    metric_dict = self.log.get_metric(config)
    info_dict = {k:v for k in self.infos if (k in metric_dict and pd.notna(v:=metric_dict.pop(k)))}
    metric_dict = {k:v for k, v in metric_dict.items() if not (np.isscalar(v) and pd.isna(v))}
    return metric_dict, info_dict
  
    
  def update_log(self, config, status=None, **metric_dict):
    if status==None: 
      status = self.__RUNNING
    self.log.add_result(config, **metric_dict,
                        **self.__curr_runinfo.update_and_get(),
                        status=status)
    self.log.to_tsv()
    
    
  def run(self):
    logging.info('Start running experiments.')
    
    logging.info(self.log.filelock.is_locked)
    logging.info(self.log.filelock.id)
    
    self.log.filelock.acquire()
    self.log.load_tsv()
    
    # run experiment plans 
    for i, config in enumerate(self.configs):
      
      self.log.filelock.acquire() ##################################################################
      
      metric_dict, info_dict = self.get_metric_info(config)
      
      # skip already executed runs
      if info_dict.get('status') in {self.__RUNNING, self.__COMPLETED}: continue
      
      self.__curr_runinfo = RunInfo(prev_duration=pd.to_timedelta(info_dict.get('duration', '0')))
      
      # if config not in self.log or status==self.__FAILED
      if self.configs_save:
        self.update_log(config, **metric_dict, status=self.__RUNNING)

      self.log.filelock.release(force=True) ##################################################################
      
      logging.info('###################################')
      logging.info(f'   Experiment count : {i+1}/{len(self.configs)}')
      logging.info('###################################') 


      try:
        if self.checkpoint:
          metric_dict = self.exp_func(config, self)
        else:
          metric_dict = self.exp_func(config)
        status = self.__COMPLETED
        
      except:
        metric_dict, _ = self.get_metric_info(config)
        status = self.__FAILED
        logging.error("Experiment failure occured.")
        raise
      
      finally:
        self.update_log(config, **metric_dict, status=status)
        logging.info("Saved experiment data to log.")
      
    logging.info('Complete experiments.')
      
      
  @staticmethod
  def resplit_logs(exp_folder_path: str, target_split: int=1, save_backup: bool=True):
    """Resplit splitted logs into ``target_split`` number of splits."""
    assert target_split > 0, 'Target split should be larger than 0'
    
    cfg_file, logs_file, _ = Experiment.get_paths(exp_folder_path)
    logs_folder = os.path.join(exp_folder_path, 'log_splits')
    
    # merge original log_splits
    if os.path.exists(logs_folder): # if log is splitted
      os.chdir(logs_folder)
      base, *logs = [ExperimentLog.from_tsv(os.path.join(logs_folder, sp_n), parse_str=False) for sp_n in glob.glob("*.tsv")]
      base.merge(*logs)
      shutil.rmtree(logs_folder)
    elif os.path.exists(logs_file): # if only single log file exists 
      base = ExperimentLog.from_tsv(logs_file, parse_str=False)
      shutil.rmtree(logs_file)
    
    # save backup
    if save_backup:
      base.to_tsv(os.path.join(exp_folder_path, 'logs_backup.tsv'))
    
    # resplit merged logs based on target_split
    if target_split==1:
      base.to_tsv(logs_file)
    
    elif target_split>1:
      # get configs
      configs = ConfigIter(cfg_file)
      
      for n in range(target_split):
        # empty log
        logs = ExperimentLog.from_exp_config(configs.__dict__, 
                                            os.path.join(logs_folder, f'split_{n}.tsv',),
                                            base.metric_fields)
        
        # resplitting nth split
        cfgs_temp = copy.deepcopy(configs)
        cfgs_temp.filter_iter(lambda i, _: i%target_split==n)
        for cfg in track(cfgs_temp, description=f'split: {n}/{target_split}'):
          if cfg in base:
            metric_dict = base.get_metric(cfg)
            logs.add_result(cfg, **metric_dict)
          
        logs.to_tsv()
        
        
  @classmethod 
  def set_log_status_as_failed(cls, exp_folder_path: str):
    _, logs_file, _ = Experiment.get_paths(exp_folder_path)
    logs_folder = os.path.join(exp_folder_path, 'log_splits')
    
    # merge original log_splits
    if os.path.exists(logs_folder): # if log is splitted
      os.chdir(logs_folder)
      paths = [os.path.join(logs_folder, sp_n) for sp_n in glob.glob("*.tsv")]
    elif os.path.exists(logs_file): # if only single log file exists 
      paths = [logs_file]

    for p in paths:
      log = ExperimentLog.from_tsv(p, parse_str=False)
      log.df['status'] = log.df['status'].map(lambda x: cls.__FAILED if x==cls.__RUNNING else x)
      log.to_tsv()
