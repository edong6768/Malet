import os, shutil, time, uuid
from typing import Optional
from ast import literal_eval
from contextlib import ContextDecorator

from absl import logging

from rich.table import Table

def create_dir(dir):
  if os.path.exists(dir):
    for f in os.listdir(dir):
      if os.path.isdir(os.path.join(dir, f)):
        shutil.rmtree(os.path.join(dir, f))
      else:
        os.remove(os.path.join(dir, f))
  else:
    os.makedirs(dir)
    

def df2richtable(df, title=None, max=None):
  df = df.reset_index()
  
  if max:
    df_tail = str(len(df)), *map(str, df.tail(1).iloc[0].values)
    df = df.head(max)
  
  table = Table(title=title)
  table.add_column('id')
  for f in list(df):       
    table.add_column(f)
  
  for row in df.itertuples(name=None):
    table.add_row(*map(str, row))
    
  if max:
    table.add_row('...', *(['...']*len(df.columns)))
    table.add_row(*df_tail)
    
  return table


def list2tuple(l):
  if isinstance(l, list):
    return tuple(map(list2tuple, l))
  if isinstance(l, dict):
    return {k: list2tuple(v) for k, v in l.items()}
  return l
    

def str2value(value_str):
    """Casts string back to standard python types"""
    if not isinstance(value_str, str): return value_str
    value_str = value_str.replace('inf', '2e+308')\
                         .replace('nan', 'None')
    try:
      return literal_eval(value_str)
    except:
      return value_str


def append_metrics(metric_log=None, **new_metrics):
    '''Add new metrics to metric_log'''
    if metric_log==None:
        metric_log = {}
    for k, v in new_metrics.items():
        assert type(v) in {int, float, bool, str}
        metric_log[k] = metric_log.get(k, [])
        metric_log[k].append(v)
    return metric_log


class QueuedFileLock(ContextDecorator):
  __delim = '\n'
  
  def __init__(self, lock_file: str, timeout: float = 10):
    self.lock_file = lock_file
    self.timeout = timeout
    self.id = uuid.uuid4().int
    
    self.acquire_count = 0
    
    if not os.path.exists(lock_file):
      with open(lock_file, 'w') as f:
        f.write('')
      
    self.__read_queue()
    
  def __read_queue(self):
    success=False
    for i in range(10):
      try:
        with open(self.lock_file, 'r') as f:
          s = f.read()
          parseint = lambda x: int(x.strip('\x00'))
          self.queue = [*map(parseint, filter(bool, s.split(self.__delim)))]
          success = True
        break
      except:
        logging.info(f'Failed to read queue from {self.lock_file} (Attempt: {i+1}/10). Retrying after 0.1s.')
        time.sleep(0.1)
        continue
    if not success:
      raise Exception(f'Failed to read queue from {self.lock_file}.')
      
  def __write_queue(self):
    with open(self.lock_file, 'w') as f:
      s = self.__delim.join(map(str, self.queue))
      f.write(s)
    
  def __append_write(self):
    with open(self.lock_file, 'a') as f:
      f.write(f'{self.__delim}{self.id}')
    self.__read_queue()
    
  @property
  def is_locked(self):
    self.__read_queue()
    return not self.queue or self.queue[0] != self.id
  
  def acquire(
    self, 
    timeout: Optional[float]=None, 
    poll_interval: float = 0.05
  ):
    self.acquire_count += 1
    
    if timeout is None:
      timeout = self.timeout
      
    self.__read_queue()
    if self.id not in self.queue:
      self.__append_write()
    
    logging.debug(f'Attempting to acquire filelock {self.id} on {self.lock_file}.')
    start_t = time.time()
    while self.is_locked:
      logging.debug(f'Failed to acquire filelock {self.id} on {self.lock_file}. Waiting for {poll_interval} seconds.')
      time.sleep(poll_interval)
      
      if time.time() - start_t > timeout:
        raise TimeoutError(f'Timeout while acquiring filelock {self.id} on {self.lock_file}.')
    
    logging.debug(f'Filelock {self.id} acquired on {self.lock_file}.')
      
  def release(self, force=False):
    if self.acquire_count == 0: return
    
    if self.acquire_count >= 1:
      self.acquire_count -= 1
    
    if self.acquire_count == 0 or force:
      self.acquire_count = 0
      self.__read_queue()
      self.queue.remove(self.id)
      self.__write_queue()
      
      logging.debug(f'Released filelock {self.id} on {self.lock_file}.')
  
  def __enter__(self):
    self.acquire()
    return self
    
  def __exit__(self, *args):
    self.release()
    
  def __del__(self):
    self.release(force=True)
