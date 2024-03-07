import os, shutil
from ast import literal_eval

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
    

def df2richtable(df):
  table = Table(title='Metric Summary Table')
  df = df.reset_index()
  
  table.add_column('id')
  for f in list(df):       
    table.add_column(f)
  
  for row in df.itertuples(name=None):
    table.add_row(*(str(i) for i in row))
    
  return table


def list2tuple(l):
  if isinstance(l, list):
    return tuple(map(list2tuple, l))
  if isinstance(l, dict):
    return {k: list2tuple(v) for k, v in l.items()}
  return l
    

def str2value(value_str):
    """Casts string back to standard python types"""
    return literal_eval(value_str) if isinstance(value_str, str) else value_str
