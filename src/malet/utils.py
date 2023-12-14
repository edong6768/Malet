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
             box_width=100, indent=0, skip=0, align='<'):
    
    b = [s[-3:] for s in boundaries.split('\n')]
    line_b = lambda s='', l=1, a='<': ' '*indent + f'{b[l][0]}{str(s):{b[l][1]}{a}{box_width}s}{b[l][2]}\n'
    result = ''
    
    contents = content.split('\n')
    result += line_b(f' {title} ', 0, '^')
    result += line_b()
    for i, content in enumerate(contents):
        content = content.strip()
        for j in range(0, len(content), box_width-4):
            result += line_b(f"  {content[j:j+box_width-4]}", 1, align)
        if i==len(contents)-1: break
        for _ in range(skip):
            result += line_b()
    result += line_b()
    result += line_b(l=2, a='^')
    return result
    
    
def str2value(value_str):
    """Casts string to corresponding field type"""
    if not isinstance(value_str, str): return value_str
    
    value_str = value_str.strip()
    # list
    if '[' in value_str:
      return [str2value(v) for v in value_str[1:-1].split(',')]
    # tuple
    if '(' in value_str:
      return tuple(str2value(v) for v in value_str[1:-1].split(','))
    # float
    elif (m:=re.findall('-?\d*\.\d*', value_str)) and len(m)==1 and m[0]==value_str:
      return float(value_str)
    # int
    elif (m:=re.findall('-?\d+', value_str)) and len(m)==1 and m[0]==value_str:
      return int(value_str) 
    return value_str
