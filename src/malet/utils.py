import os, shutil
import re

def create_dir(dir):
  if os.path.exists(dir):
    for f in os.listdir(dir):
      if os.path.isdir(os.path.join(dir, f)):
        shutil.rmtree(os.path.join(dir, f))
      else:
        os.remove(os.path.join(dir, f))
  else:
    os.makedirs(dir)
    

def box_str(title: str, content: str, 
             boundaries="""┌─┐
                           │ │
                           └─┘""",
             box_width=100, indent=0, skip=0, align='<', strip=True):
    
    b = [s[-3:] for s in boundaries.split('\n')]
    line_b = lambda s='', l=1, a='<': ' '*indent + f'{b[l][0]}{str(s):{b[l][1]}{a}{box_width}s}{b[l][2]}\n'
    result = ''
    
    contents = content.split('\n')
    result += line_b(f' {title} ', 0, '^')
    result += line_b()
    for i, content in enumerate(contents):
        if strip: content = content.strip()
        for j in range(0, len(content), box_width-4):
            result += line_b(f"  {content[j:j+box_width-4]}", 1, align)
        if i==len(contents)-1: break
        for _ in range(skip):
            result += line_b()
    result += line_b()
    result += line_b(l=2, a='^')
    return result
  

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
      return [str2value(v) for v in value_str[1:-1].split(',')]
    # tuple
    if '(' in value_str:
      return tuple(str2value(v) for v in value_str[1:-1].split(','))
    # float
    elif match_unique('-?\d*\.\d*'):
      return float(value_str)
    # sci. notation
    elif match_unique('-?\d\.\d+e(\+|-)\d+'):
      return float(value_str) 
    # int
    elif match_unique('-?\d+'):
      return int(value_str) 
    # NaN
    elif value_str.lower()=='nan':
      return None
    return value_str
