import os, shutil
import re

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
    """Casts string to corresponding field type"""
    if not isinstance(value_str, str): return value_str
    value_str = value_str.strip() \
                         .replace('\\', '') \
                         .replace('\'', '') \
                         .replace('"', '')
    match_unique = lambda p: (m:=re.findall(p, value_str)) and len(m)==1 and m[0]==value_str
    # list
    if '[' in value_str:
      return [str2value(v) for v in value_str[1:-1].split(',') if v!='']
    # tuple
    if '(' in value_str:
      return tuple(str2value(v) for v in value_str[1:-1].split(',') if v!='')
    # sci. notation
    elif match_unique('-?\d\.?\d*e[+-]\d+'):
      return float(value_str) 
    # float
    elif match_unique('-?\d*\.\d*'):
      return float(value_str)
    # int
    elif match_unique('-?\d+'):
      return int(value_str) 
    # NaN
    elif value_str.lower()=='nan':
      return None
    return value_str
