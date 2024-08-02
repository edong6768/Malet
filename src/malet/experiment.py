import os, glob, shutil, io
import copy
import yaml, re
import traceback
from functools import reduce, partial
from typing import ClassVar, Any, Mapping, Callable, Optional, Union, Tuple, List, Dict
from dataclasses import dataclass
from itertools import product, chain
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from git import Repo

from rich import print
from rich.progress import track
from rich.table import Table
from rich.panel import Panel
from rich.align import Align

from absl import logging
from ml_collections.config_dict import ConfigDict

from .utils import list2tuple, str2value, QueuedFileLock, FuncTimeoutError, settimeout_func, path_common_decomposition

ExpFunc = Union[Callable[[ConfigDict], dict], Callable[[ConfigDict, 'Experiment'], dict]]

class ConfigIter:
  '''Iterator of ConfigDict generated from yaml config grid plan file.
  
  Usage example:
    ```python
    for config in ConfigIter('exp_file.yaml'):
      train_func(config)
    ```
    
  yaml exp_file example:
  
    ```yaml
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
    ```
    
  Attributes:
      static_configs (dict): Dictionary of static configs of the experiment.
      grid_fields (list): list of grid fields indicating order of griding.
      grid (dict): list of grid values.
      grid_iter (list): list of ConfigDicts generated from grid values.
  '''
  
  def __init__(self, exp_config_path: str):
    with open(exp_config_path) as f:
      cnfg_str = self.__sub_cmd(f.read())
      
    self.static_configs = yaml.safe_load(cnfg_str)
    
    self.grid_fields = self.__extract_grid_order(cnfg_str)
    self.grid = self.static_configs.pop('grid', {})
    
    assert not (f:={k for k in self.static_configs.keys() if k in self.grid_fields}), f'Overlapping fields {f} in Static configs and grid fields.'
  
    self.grid_iter = self.__get_iter()
    
  
  def filter_iter(self, filt_fn: Callable[[int, dict], bool]):
    """filters ConfigIter with ``filt_fn`` which has (idx, dict) as arguments.

    Args:
        filt_fn (Callable[[int, dict], bool]): Filter function to filter ConfigIter.
    """
    self.grid_iter = [d for i, d in enumerate(self.grid_iter) if filt_fn(i, d)]
    
  @property
  def grid_dict(self) -> dict:
    """Dictionary of all grid values"""
    if not self.grid_fields: return dict()
    st, *sts = self.grid
    castl2t = lambda vs: map(lambda v: (tuple(v) if isinstance(v, list) else v), vs)
    acc = lambda a, d: {k: [*{*castl2t(va), *castl2t(vd)}] 
                        for (k, va), (_, vd) in zip(a.items(), d.items())}
    grid_dict = {k: [*map(self.field_type(k), vs)] for k, vs in reduce(acc, sts, st).items()}
    return grid_dict
  
  def field_type(self, field: str):
    """Returns type of a field in grid.

    Args:
        field (str): Name of the field.

    Returns:
        Any: Type of the field.
    """
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
  def __extract_grid_order(cfg_str) -> List[str]:
    """Parse grid order from raw config string"""
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
pd.DataFrame.drop = lambda self, *_, axis=0, **__: pd.DataFrame(columns=self.columns) if self.index.names==[None] and len(self)<2 and axis==0 else self.old_drop(*_, axis=axis, **__)

@dataclass
class ExperimentLog:
  """Logging class for experiment results.
  
  Logs all configs for reproduction, and resulting pre-defined metrics from experiment run as DataFrame.
  Changing configs are stored as multiindex and metrics are stored as columns.
  Other static configs are passed in and stored as dictionary.
  
  These can be written to tsv file with yaml header, and loaded back from it.
  Filelocks are used for multiple experiment runs to safely write to the same log file.

  Attributes:
      df (pd.DataFrame): DataFrame of experiment results.
      static_configs (dict): Dictionary of static configs of the experiment.
      logs_file (str): File path to tsv file.
      use_filelock (bool, optional): Whether to use file lock for reading/writing log file. Defaults to False.
  """
  df: pd.DataFrame
  static_configs: dict
  logs_file: str
  use_filelock: bool = False
  
  __sep: ClassVar[str] = '-'*45 + '\n'
  
  def __post_init__(self):
    if self.use_filelock:
      self.filelock = QueuedFileLock(self.logs_file+'.lock', timeout=3*60*60)
      
  
  @property
  def grid_fields(self): return list(self.df.index.names) if self.df.index.names!=[None] else []  

  @property
  def metric_fields(self): return list(self.df)

  def grid_dict(self) -> Dict[str, Any]:
    return {k: sorted(set(self.df.index.get_level_values(k))) for k in self.grid_fields}
  
  # Constructors.
  # -----------------------------------------------------------------------------  
  
  @classmethod
  def from_fields(
        cls, 
        grid_fields: list, 
        metric_fields: list, 
        static_configs: dict, 
        logs_file: str,
        use_filelock: bool = False
    ) -> 'ExperimentLog':
    """Create ExperimentLog from grid and metric fields.

    Args:
        grid_fields (list): Field names of configs to be grid-searched.
        metric_fields (list): Field names of metrics to be logged from experiment results.
        static_configs (dict): Other static configs of the experiment.
        logs_file (str): File path to tsv file.
        use_filelock (bool, optional): Whether to use file lock for reading/writing log file. Defaults to False.

    Returns:
        ExperimentLog: New experiment log object.
    """
    assert metric_fields is not None, 'Specify the metric fields of the experiment.'
    assert not (f:=set(grid_fields) & set(metric_fields)), f'Overlapping field names {f} in grid_fields and metric_fields. Remove one of them.'
    return cls(pd.DataFrame(columns=grid_fields+metric_fields).set_index(grid_fields), 
               static_configs, 
               logs_file=logs_file, 
               use_filelock=use_filelock)
  
  @classmethod
  def from_config_iter(
        cls, 
        config_iter: ConfigIter, 
        metric_fields: list,
        logs_file: str, 
        use_filelock: bool = False
    ) -> 'ExperimentLog':
    """Create ExperimentLog from ConfigIter object.

    Args:
        config_iter (ConfigIter): ConfigIter object to reference static_configs and grid_fields.
        metric_fields (list): list of metric fields.
        logs_file (str): File path to tsv file.
        use_filelock (bool, optional): Whether to use file lock for reading/writing log file. Defaults to False.

    Returns:
        ExperimentLog: New experiment log object.
    """
    return cls.from_fields(config_iter.grid_fields, 
                           metric_fields, 
                           config_iter.static_configs, 
                           logs_file, 
                           use_filelock=use_filelock)

  @classmethod
  def from_tsv(
        cls, 
        logs_file: str, 
        use_filelock: bool = False,
        parse_str = True
    ) -> 'ExperimentLog':
    """Create ExperimentLog from tsv file with yaml header.

    Args:
        logs_file (str): File path to tsv file.
        use_filelock (bool, optional): Whether to use file lock for reading/writing log file. Defaults to False.
        parse_str (bool, optional): Whether to parse and cast string into speculated type. Defaults to True.

    Returns:
        ExperimentLog: New experiment log object.
    """
    if use_filelock:
      with QueuedFileLock(logs_file+'.lock', timeout=3*60*60):
        logs = cls.parse_tsv(logs_file, parse_str=parse_str)
    else:
      logs = cls.parse_tsv(logs_file, parse_str=parse_str)
      
    return cls(logs['df'], 
               logs['static_configs'],
               logs_file, 
               use_filelock=use_filelock)
  
  
  # tsv handlers.
  # -----------------------------------------------------------------------------
  @classmethod
  def parse_tsv(cls, log_file: str, parse_str=True)->dict:
    """Parse tsv file into usable datas.
    
    Parse tsv file generated by ExperimentLog.to_tsv method.
    Has static_config as yaml header, and DataFrame as tsv body where multiindices is set as different line with column names.

    Args:
        log_file (str): File path to tsv file.
        parse_str (bool, optional): Whether to parse and cast string into speculated type. Defaults to True.

    Raises:
        Exception: Error while reading log file.

    Returns:
        dict: Dictionary of pandas.DataFrame, grid_fields, metric_fields, and static_configs.
    """
    assert os.path.exists(log_file), f'File path "{log_file}" does not exists.'

    try:
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

        # get dataframe from tsv body
        tsv_str = fd.read()
        
    except:
      raise Exception(f'Error while reading log file: {log_file}')
      
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
      
    return {'df': df,
            'grid_fields': idx[1:],
            'metric_fields': col,
            'static_configs': static_configs}
  
  
  def lock_file(func):
    '''Decorator for filelock acquire/release before/after given function call'''
    def wrapped(self, *args, **kwargs):
      if self.use_filelock:
        with self.filelock:
          return func(self, *args, **kwargs)
      else:
        return func(self, *args, **kwargs)
    return wrapped
  
  @lock_file
  def load_tsv(self, logs_file: Optional[str]=None, parse_str: bool=True):
    """load tsv with yaml header into ExperimentLog object.

    Args:
        logs_file (Optional[str], optional): Specify other file path to tsv file. Defaults to None.
        parse_str (bool, optional): Whether to parse and cast string into speculated type. Defaults to True.
    """
    if logs_file is not None:
      self.logs_file=logs_file
    
    logs = self.parse_tsv(self.logs_file, parse_str=parse_str)
    self.df = logs['df']
    self.static_configs = logs['static_configs']

  @lock_file
  def to_tsv(self, logs_file: Optional[str]=None):
    """Write ExperimentLog object to tsv file with yaml header.

    Args:
        logs_file (Optional[str], optional): Specify other file path to tsv file. Defaults to None.
    """
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
  
  # Add results.
  # -----------------------------------------------------------------------------
  
  def add_result(self, configs: Mapping[str, Any], **metrics):
    """Add experiment run result to dataframe.
    
    Args:
        configs (Mapping[str, Any]): Dictionary or Mapping of configurations of the result of the experiment instance to add.
        **metrics (Any): Metrics of the result of the experiment instance to add.
    """
    if configs in self:
      cur_gridval = list2tuple([configs[k] for k in self.grid_fields])
      self.df = self.df.drop(cur_gridval)
    
    configs = {k:list2tuple(configs[k]) for k in self.grid_fields}
    metrics = {k:metrics.get(k) for k in self.metric_fields}
    result_dict = {k:[v] for k, v in {**configs, **metrics}.items()}
    result_df = pd.DataFrame(result_dict).set_index(self.grid_fields)
    self.df = pd.concat([self.df, result_df])[self.metric_fields]
    

  # Field manipulations.
  # -----------------------------------------------------------------------------

  @staticmethod
  def __add_column(df, new_column_name: str, fn: Callable, *fn_arg_fields: str) -> pd.DataFrame:
    '''Add new column field computed from existing fields in self.df'''
    def mapper(*args):
      if all(isinstance(i, (int, float, str, tuple, list)) for i in args):
        return fn(*args)
      return None
    df[new_column_name] = df.apply(lambda df: mapper(*[df[c] for c in fn_arg_fields]), axis=1)
    return df
    
  def derive_field(self, new_field_name: str, fn: Callable, *fn_arg_fields: str, is_index: bool = False):
    """Add new field computed from existing fields in self.df.

    Args:
        new_field_name (str): Name of the new field.
        fn (Callable): Function to compute new field.
        *fn_arg_fields (str): Field names to be used as arguments for the function.
        is_index (bool, optional): Whether to add field as index. Defaults to False.
    """
    df = self.df.reset_index(self.grid_fields)
    df = self.__add_column(df, new_field_name, fn, *fn_arg_fields)
    new_grid_fields = [*self.grid_fields, new_field_name] if is_index else self.grid_fields
    self.df = df.set_index(new_grid_fields)
    
  def drop_fields(self, field_names: List[str]):
    """Drop fields from the log.

    Args:
        field_names (List[str]): list of field names to drop.
    """
    assert not (ns:=set(field_names)-set(list(self.static_configs)+self.grid_fields+self.metric_fields)), \
      f'Field names {ns} not in any of static {list(self.static_configs)}, grid {self.grid_fields}, or metric {self.metric_fields} field names.'
    
    grid_ns, metric_ns = [], [] 
    for fn in field_names:
      if fn in self.static_configs:
        del(self.static_configs[fn])                    # remove static fields
      elif fn in self.grid_fields:
        grid_ns.append(fn)
      elif fn in self.metric_fields:
        metric_ns.append(fn)
    
    self.df = self.df.reset_index(grid_ns, drop=True)   # remove grid field
    self.df = self.df.drop(columns=metric_ns)           # remove metric field

  def rename_fields(self, name_map: Dict[str, str]):
    """Rename fields in the log.

    Args:
        name_map (Dict[str, str]): Mapping of old field names to new field names.
    """
    assert not (ns:=set(name_map)-set(list(self.static_configs)+self.grid_fields+self.metric_fields)), \
      f'Field names {ns} not in any of static {list(self.static_configs)}, grid {self.grid_fields}, or metric {self.metric_fields} field names.'
    
    grid_l, metric_d = self.grid_fields, {}
    for on, nn in name_map.items():
      if on in self.static_configs:
        self.static_configs[nn] = self.static_configs.pop(on)   # update static field name
      elif on in self.grid_fields:
        grid_l[grid_l.index(on)] = nn
      elif on in self.metric_fields:
        metric_d[on] = nn

    self.df.index.rename(grid_l, inplace=True)                  # update grid field names
    self.df.rename(columns=metric_d, inplace=True)              # update metric field names


  # Merge ExperimentLogs.
  # -----------------------------------------------------------------------------
  
  def log_conflict_resolver(self, other: 'ExperimentLog') -> Tuple['ExperimentLog', 'ExperimentLog']:
    """Summarize conflicts and accept user input for resolution.

    Args:
        other (ExperimentLog): Target log to merge with self.
        
    Returns:
        Tuple[ExperimentLog, ExperimentLog]: Resolved logs (self, other).
    """
    if self==other: return self, other
    print(f'\nConflict detected between logs: ')
    print(f'  - Self :{self.logs_file}')
    print(f'  - Other:{other.logs_file}')
    print('Start resolving conflict...')
    
    self_d, other_d = {}, {}
    
    for log, d in [(self, self_d), (other, other_d)]:
      d['sttc_d'] = log.static_configs
      d['grid_d'] = {k: sorted(set(log.df.index.get_level_values(k))) for k in log.grid_fields}
      d['dict'] = {**d['sttc_d'], **d['grid_d']}
      d['fields'] = list(log.static_configs.keys()) + list(log.grid_fields)
      
    sfs, sfo = map(lambda d: set(d['fields']), (self_d, other_d))
    same_fields = sorted(sfs & sfo)
    new_to_self = sorted(sfo - sfs)
    new_to_othr = sorted(sfs - sfo)
    
    ln_k = max([len(k) for k in same_fields+new_to_self+new_to_othr])
    ln_s = max([len(str(self_d['dict'].get(k, ""))) for k in same_fields+new_to_self+new_to_othr])
    ln_o = max([len(str(other_d['dict'].get(k, ""))) for k in same_fields+new_to_self+new_to_othr])
    
    ############################# Print conflict summary #############################
    
    _, (self_post, othr_post) = path_common_decomposition([self.logs_file, other.logs_file])
    
    summary_tab = Table(title='Log field conflict summary')
    
    summary_tab.add_column('Field', style='bold')
    summary_tab.add_column(f'[blue]Self[/blue] ({self_post[:-4]})')
    summary_tab.add_column(f'[green]Other[/green] ({othr_post[:-4]})')
      
    for i, k in enumerate(same_fields):
      summary_tab.add_row(f'{k:{ln_k}s}', 
                          f'{str(self_d["dict"].get(k, "")):{ln_s}s}', 
                          f'{str(other_d["dict"].get(k, "")):{ln_o}s}', 
                          style="on bright_black" if i%2 else "", 
                          end_section=(i==len(same_fields)-1))

    rd = lambda s, i: f'[on {"red" if i%2 else "dark_red"}]{s} [/on {"red" if i%2 else "dark_red"}]'
    for i, k in enumerate(new_to_self):
      i += len(same_fields)
      summary_tab.add_row(f'{k:{ln_k}s}', 
                       rd(f'{str(self_d["dict"].get(k, "")):{ln_s}s}', i), 
                          f'{str(other_d["dict"].get(k, "")):{ln_o}s}', 
                          style="on bright_black" if i%2 else "", 
                          end_section=(i==len(same_fields)+len(new_to_self)-1))
      
    for i, k in enumerate(new_to_othr):
      i += len(same_fields) + len(new_to_self)
      summary_tab.add_row(f'{k:{ln_k}s}', 
                          f'{str(self_d["dict"].get(k, "")):{ln_s}s}', 
                       rd(f'{str(other_d["dict"].get(k, "")):{ln_o}s}', i), 
                          style="on bright_black" if i%2 else "")
      
    print(Align(summary_tab, align='center'))
    print(Align(Panel(f'Detected [bold red]{len(new_to_self+new_to_othr)}[/bold red] conflicts to resolve.', padding=(1, 3)), align='center'))
    
    
    ############################# Resolve conflicts #############################
    
    i_cfl, n_cfl = 0, len(new_to_self+new_to_othr)
    logs = [
      (self,  f'[blue]{self_post[:-4]}[/blue]',  self_d,  new_to_self),
      (other, f'[green]{othr_post[:-4]}[/green]', other_d, new_to_othr)
    ]
    
    # resolve conflict for each log
    for i in (False, True):
      tlog, ts, td, ntt = logs[i]
      flog, fs, fd, ntf = logs[not i]

      if ntt:
        print(f'\n[bold][Handle missing fields in {ts}][/bold] (Default: same/first value of {fs})')
        for k in ntt:
          i_cfl += 1
          print(f'│\n├─[{i_cfl}/{n_cfl}] [bold]({k})[/bold]')
          # set default value
          dflt = False
          dflt_val = fd['dict'].get(k, "")
          if k in fd['grid_d']: # set list to single value if it is in grid_fields
            dflt_val = dflt_val[0] if len(dflt_val)>0 else None
          
          # choose mode
          modes = ['Add new value']
          if ntf               : modes.append('merge with existing field')
          if k in fd['sttc_d'] : modes.append('remove')
          
          mode = 0
          if len(modes)>1:
            print(f"│  Choose process mode ({' / '.join([f'{i}: {md}' for i, md in enumerate(modes)])} / else: set value to {dflt_val})")
            mode = str2value(input("│  ↳ "))

          # process for each modes
          if isinstance(mode, int):
            if modes[mode]=='Add new value':
              print(f"│   ({mode}) Add new value ({fs} {'=' if k in fd['sttc_d'] else 'in'} {fd['dict'].get(k, '')})")
              new_val = str2value(input(f"│   ↳ "))
              if new_val:
                tlog.static_configs[k] = new_val
                print(f'│   - Set to {new_val}')
              else: dflt = True
            elif modes[mode]=='merge with existing field':
              print(f'│   ({mode}) Merge with existing field in {ts}: {ntf}')
              while True:
                new_field = input(f"│   ↳ ")
                if new_field in ntf+['']: break
                print(f"│   There is no field:{new_field} to merge with. Choose from {ntf}")
              if new_field:
                flog.rename_fields({k: new_field})
                ntf.remove(new_field)
                n_cfl -= 1
                print(f'│   - Merged with {new_field}')
              else: dflt=True
            elif modes[mode]=='remove':
              print(f'│   ({mode}) Remove field')
              del(flog.static_configs[k])
            else: dflt = True
          else: dflt = True

          if dflt:
            print(f'│   - Set to {dflt_val}')
            tlog.static_configs[k] = str2value(dflt_val)
            
        print(f'│\n└─[[bold cyan]Done[/bold cyan]]')
        

    return self, other
    
  
  def __merge_one(self, other: 'ExperimentLog', same=True) -> 'ExperimentLog':
    '''
    Merge two logs into self.
    - The order of grid_fields follows self
    - Static fields stays only if they are same for both logs.
    - else move to grid_fields if not in grid_fields
    '''
    if same:
      assert self==other, 'Different experiments cannot be merged by default.'
 
    # new static_field: field in both log.static_field and have same value
    sc1, sc2 = (log.static_configs for log in (self, other))
    new_sttc = {k: sc1[k] for k in set(sc1)&set(sc2) if sc1[k]==sc2[k]}
    new_gridf = self.grid_fields + list(set(sc1)-set(new_sttc))
    new_mtrcf = self.metric_fields + [k for k in other.metric_fields if k not in self.metric_fields]

    # field static->grid: if not in new static_field and not in grid
    dfs = []
    for log in (self, other):
      dfs.append(log.df.reset_index())
      for k in set(log.static_configs)-set(new_sttc):
        if k in log.grid_fields: continue
        dfs[-1][k] = [list2tuple(log.static_configs.get(k, np.nan))]*len(log)

    # merge and update self
    self.static_configs = new_sttc
    self.df = (
        pd.concat(dfs)
          .set_index(new_gridf)[new_mtrcf]
    )

    return self
    
  def merge(self, *others: 'ExperimentLog', same: bool=True):
    """Merge multiple logs into self.

    Args:
        *others (ExperimentLog): Logs to merge with self.
        same (bool, optional): Whether to raise error when logs are not of matching experiments. Defaults to True.
    """
    for other in others:
      self, other = self.log_conflict_resolver(other)
      self.__merge_one(other, same=same)

  @staticmethod
  def merge_tsv(*log_files: str, save_path: Optional[str]=None, same: bool=True) -> 'ExperimentLog':
    """Merge multiple logs into one from tsv file paths.

    Args:
        *logs_path (str): Path to logs.
        save_path (Optional[str]): Path to save merged log.
        same (bool, optional): Whether to raise error when logs are not of matching experiments. Defaults to True.
    """
    base, *logs = [ExperimentLog.from_tsv(f, parse_str=False) for f in log_files]
    base.merge(*logs, same=same)
    if save_path:
      base.to_tsv(save_path)
    return base

  @staticmethod
  def merge_folder(logs_path: str, save_path: Optional[str]=None, same: bool=True) -> 'ExperimentLog':
    """Merge multiple logs into one from tsv files in folder.

    Args:
        logs_path (str): Folder path to logs.
        save_path (Optional[str], optional): Path to save merged log. Defaults to None.
        same (bool, optional): Whether to raise error when logs are not of matching experiments. Defaults to True.
    """
    log_files = glob.glob(os.path.join(logs_path, "*.tsv"))
    assert log_files, f'No tsv files found in {logs_path}'
    
    return ExperimentLog.merge_tsv(*log_files, save_path=save_path, same=same)
    
  
  # Utilities.
  # -----------------------------------------------------------------------------

  def __cfg_match_row(self, config):
    if not self.grid_fields: return self.df
    
    grid_filt = reduce(lambda l, r: l & r, 
                       (self.df.index.get_level_values(k)==(str(config[k]) if isinstance(config[k], list) else config[k]) 
                        for k in self.grid_fields))
    return self.df[grid_filt]
  
  
  def isin(self, config: Mapping[str, Any]) -> bool:
    """Check if specific experiment config was already executed in log.

    Args:
        config (Mapping[str, Any]): Configuration instance to check if it is in the log.

    Returns:
        bool: Whether the config is in the log.
    """
    if self.df.empty: return False

    cfg_same_in_static = all([config[k]==v for k, v in self.static_configs.items() if k in config])
    cfg_matched_df = self.__cfg_match_row(config)
    
    return cfg_same_in_static and not cfg_matched_df.empty


  def get_metric(self, config: Mapping[str, Any])->dict:
    """Search matching log with given config dict and return metric_dict, info_dict.

    Args:
        config (Mapping[str, Any]): Configuration instance to search in the log.

    Returns:
        dict: Found metric dictionary of the given config.
    """
    assert config in self, 'config should be in self when using get_metric_dict.'
    
    cfg_matched_df = self.__cfg_match_row(config)
    metric_dict = {k:(v.iloc[0] if not (v:=cfg_matched_df[k]).empty else None) for k in self.metric_fields}
    return metric_dict

  def is_same_exp(self, other: 'ExperimentLog')->bool:
    """Check if both logs have same config fields.

    Args:
        other (ExperimentLog): Log to compare with.

    Returns:
        bool: Whether both logs have same config fields.
    """
    fields = lambda log: set(log.static_configs.keys()) | set(log.grid_fields)
    return fields(self)==fields(other)
    
    
  def melt_and_explode_metric(self, df: Optional[pd.DataFrame]=None, step: Optional[int]=None, dropna: bool=True)->pd.DataFrame:
    """Melt and explode metric values in DataFrame.
    
    Melt column (metric) names into 'metric' field (multi-index) and their values into 'metric_value' columns.
    Explode metric with list of values into multiple rows with new 'step' and 'total_steps' field.
    If step is specified, only that step is selected, otherwise all steps are exploded.
    
    Args:
        df (Optional[pd.DataFrame], optional): Base DataFrame to operate over. Defaults to None.
        step (Optional[int], optional): Specific step to select. Defaults to None.
        dropna (bool, optional): Whether to drop rows with NaN metric values. Defaults to True.

    Returns:
        pd.DataFrame: Melted and exploded DataFrame.
    """
    if df is None: 
      df = self.df
    mov_to_index = lambda *fields: df.reset_index().set_index((dn if (dn:=df.index.names)!=[None] else [])+[*fields])
    
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
        df['step'] = df['metric_value'].map(lambda x: step + (pseudo_len(x)+1 if step<0 else 0))
    
    df['metric_value'] = df.apply(lambda df: df['metric_value'][df.step-1] if isinstance(df['metric_value'], list) else df['metric_value'], axis=1) # list[epoch] for all fields
    
    df = mov_to_index('step', 'total_steps')
    
    # delete string and NaN valued rows
    if dropna:
      df = (
        df[pd.to_numeric(df['metric_value'], errors='coerce').notnull()]
          .dropna()
          .astype('float')
      )
    return df

    
  def __contains__(self, config: Mapping[str, Any]) -> bool:
    return self.isin(config)
  
  def __getitem__(self, config: Mapping[str, Any]) -> dict:
    return self.get_metric(config)

  def __eq__(self, other: 'ExperimentLog') -> bool:
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
               configs_save: bool = False,
               checkpoint: bool = False,
               filelock: bool = False,
               timeout: Optional[float] = None
    ):
    
    if checkpoint:
      assert filelock, "argument 'filelock' should be set to True when checkpointing."
    
    self.exp_func = exp_function

    self.configs_save = configs_save
    self.checkpoint = checkpoint
    self.filelock = filelock
    self.timeout = timeout
    
    do_split = isinstance(total_splits, int) and total_splits>1 or isinstance(total_splits, str)
    cfg_file, tsv_file, _ = self.get_paths(exp_folder_path, split=curr_split if do_split else None)
    
    self.configs = self.__get_and_split_configs(cfg_file, total_splits, curr_split)
    self.log = self.__get_log(tsv_file, self.infos+exp_metrics, filelock)
    
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
      
  def __get_log(self, logs_file, metric_fields=None, filelock=False):
    # Configure experiment log
    if os.path.exists(logs_file): # Check if there already is a file
      log = ExperimentLog.from_tsv(logs_file, use_filelock=filelock) # resumes automatically
      
    else: # Create new log
      logs_path, _ = os.path.split(logs_file)
      if not os.path.exists(logs_path):
        os.makedirs(logs_path)
      log = ExperimentLog.from_config_iter(self.configs, metric_fields, logs_file, use_filelock=filelock)
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
    
    metric_dict = self.log[config]
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
    
    start_t = datetime.now()
    
    if self.filelock:
      logging.info(self.log.filelock.is_locked)
      logging.info(self.log.filelock.id)
      
      self.log.filelock.acquire()
      self.log.load_tsv()
      
    # check for left-over experiments (10 times for now)
    for _ in range(10):
      # run experiment plans 
      for i, config in enumerate(self.configs):
        
        if self.filelock: self.log.filelock.acquire() ##################################################################
        
        metric_dict, info_dict = self.get_metric_info(config)
        
        # skip already executed runs
        if info_dict.get('status') in {self.__RUNNING, self.__COMPLETED}: continue
        
        self.__curr_runinfo = RunInfo(prev_duration=pd.to_timedelta(info_dict.get('duration', '0')))
        
        # if config not in self.log or status==self.__FAILED
        if self.configs_save:
          self.update_log(config, **metric_dict, status=self.__RUNNING)

        if self.filelock: self.log.filelock.release(force=True) ##################################################################
        
        logging.info('###################################')
        logging.info(f'   Experiment count : {i+1}/{len(self.configs)}')
        logging.info('###################################') 


        try:
          exp_func = self.exp_func
          if self.timeout:
            exp_func = settimeout_func(exp_func, timeout = self.timeout - (datetime.now()-start_t).total_seconds())
          if self.checkpoint:
            metric_dict = exp_func(config, self)
          else:
            metric_dict = exp_func(config)
          status = self.__COMPLETED
        
        except Exception as exc:
          metric_dict, _ = self.get_metric_info(config)
          status = self.__FAILED
          
          if isinstance(exc, FuncTimeoutError):
            logging.error(f"Experiment timeout ({self.timeout}s) occured:")
            raise exc
          else:
            logging.error(f"Experiment failure occured:\n{traceback.format_exc()}{exc}")
            
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
            metric_dict = base[cfg]
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
