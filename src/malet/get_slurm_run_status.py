"""
Get Experiement run status from slurm outs
"""

import os, glob
import re
from datetime import datetime
import argparse
from .utils import box_str

parser = argparse.ArgumentParser(description='Arguemnts for plots')
parser.add_argument('-slurm_outs_folder', type=str, default='./slurm_outs', help='Slurm out folder path')
parser.add_argument('-box_width', type=int, default=100, help='width of the boxes')
parser.add_argument('-exp_name', type=str, default='all', help='Experiment name to search')
parser.add_argument('-save_path', type=str, default='', help='Save if specified.')


args = parser.parse_args()

files = glob.glob(args.slurm_outs_folder + '/*.out')

def title_str(title, box_width=None, indent=0):
    result = ''
    result += ' '*indent + f'{"─────────────────────────────────────────────────": ^{box_width}s}\n'
    result += ' '*indent + f'{title: ^{box_width}s}\n'
    result += ' '*indent + f'{"─────────────────────────────────────────────────": ^{box_width}s}\n'
    return result


#------------------ Parse slurm outs ------------------        
exp_status_dict = dict()

for fpath in files:
    with open(fpath, 'r') as f:
        sout = f.read()
        
        if not re.findall('Train: loss \d\.\d+ acc \d\.\d+; Val: loss \d\.\d+ acc \d\.\d+', sout): continue
        
        exp_infos = re.findall('Experiment : .*', sout)[0].split(' ')
        exp_name = exp_infos[2]
        
        if len(exp_infos)<4:
            i_splt, n_splt = 0, 1
        else:
            i_splt, n_splt = map(int, exp_infos[5][:-1].split('/'))
        
        if exp_name not in exp_status_dict:
            exp_status_dict[exp_name] = {
                'split_stats' : [{'alloc': 0, 
                                  'alive': 0, 
                                  'dead_list': [],
                                  'exp_counts': set(), 
                                  'tot_counts': 0,
                                  'elapsed_times': [],
                                  'last_epochs': [], 
                                  'n_epochs': 0,
                                  'train_accs': [],
                                  'val_accs': [], 
                                  } for _ in range(n_splt)],
                'total_stats' : {'alloc': 0,
                                 'alive': 0,
                                 'dead_list': [],
                                 'exp_counts': [set() for _ in range(n_splt)],
                                 'tot_counts': [*range(n_splt)],
                                },
            }
        split_dict = exp_status_dict[exp_name]['split_stats'][i_splt-1]
        tot_dict = exp_status_dict[exp_name]['total_stats']
        
        #------------------ current running job count ------------------
        ended = len(re.findall('##### END', sout))
        
        split_dict['alloc'] += 1
        split_dict['alive'] += 1-ended
        
        tot_dict['alloc'] += 1
        tot_dict['alive'] += 1-ended
        
        if ended:
            dead = os.path.split(fpath)[1][:-4]
            split_dict['dead_list'].append(dead)
            tot_dict['dead_list'].append(dead)
        
        #------------------ Experiment count statistics ------------------
        count_info = re.findall('Experiment count : \d+\/\d+', sout)
        get_counts = lambda s: (*map(int, s.split(' ')[3].split('/')),)
        
        split_dict['exp_counts'] |= {get_counts(s)[0]-1 for s in count_info}
        split_dict['tot_counts'] = get_counts(count_info[0])[1]
        
        tot_dict['exp_counts'][i_splt-1] |= {get_counts(s)[0]-1 for s in count_info}
        tot_dict['tot_counts'][i_splt-1] = get_counts(count_info[0])[1]

        #------------------ Elapsed time computation ------------------
        start_t, *_, end_t = re.findall('\d{4} \d\d:\d\d:\d\d', sout)
        s2t = lambda s: datetime.strptime(s, '%m%d %H:%M:%S')
        start_t, end_t = map(s2t, [start_t, end_t])
        elapsed_times = str(end_t - start_t)[:-3].replace(' day,', 'd')
        split_dict['elapsed_times'].append(elapsed_times)
        
        if 'start_t' not in tot_dict:
            tot_dict['start_t'] = start_t
            tot_dict['end_t'] = end_t
            
        tot_dict['start_t'] = min(start_t, tot_dict['start_t'])
        tot_dict['end_t'] = max(end_t, tot_dict['start_t'])
        
        #------------------ Latest experiment's epoch ------------------
        epoch_info = int(re.findall('Epoch \d+ \/ \d+', sout)[-1].split()[-3])
        n_epochs = int(re.findall('Epoch \d+ \/ \d+', sout)[-1].split()[-1])
        split_dict['last_epochs'].append(epoch_info)
        split_dict['n_epochs'] = n_epochs

        #------------------ Latest experiment's train / val accs ------------------
        metric_info = re.findall('Train: loss \d\.\d+ acc \d\.\d+; Val: loss \d\.\d+ acc \d\.\d+', sout)[-1]
        train_acc, val_acc = map(lambda x: str(float(x)*100)[:4]+'%', [metric_info.split()[4][:-1], metric_info.split()[-1]])
        split_dict['train_accs'].append(train_acc)
        split_dict['val_accs'].append(val_acc)
        
        exp_status_dict[exp_name]['split_stats'][i_splt-1] = split_dict
        exp_status_dict[exp_name]['total_stats'] = tot_dict

#------------------ Print stats ------------------       
 
summary = f'\nRun experiment status from Slurm outs (path : {args.slurm_outs_folder}):\n\n'

for i, (k, stats) in enumerate(exp_status_dict.items()):
    alive ,  alloc ,  dead_list ,  exp_counts ,  tot_counts ,  start_t ,  end_t = [stats['total_stats'][i] for i in \
  ['alive', 'alloc', 'dead_list', 'exp_counts', 'tot_counts', 'start_t', 'end_t']]
    
    exp_counts = sum(map(len, exp_counts))
    tot_counts = sum(tot_counts)

    summary += title_str(f"{i+1}. {k} summary", box_width=args.box_width, indent=5) + '\n'
    
    summary += box_str("",
                        f'''Experiment count status : {exp_counts} / {tot_counts} ({(p:=exp_counts/tot_counts)*100:.2f}%)
                            Alive / Allocated       : {alive} / {alloc} {f'(dead: {", ".join(dead_list)})' if dead_list else ''}
                            Max elapsed time        : {(t:=end_t-start_t)}
                            Expected time left      : {str(t*(1-p)/p)[:-7]}''',
                        boundaries='''   \n│  \n   ''',
                        box_width=args.box_width-9, indent=9, skip=1) + '\n'
    
    split_stats = stats['split_stats']
    for j, split_dict in enumerate(split_stats):
        alloc ,  alive ,  dead_list ,  exp_counts ,  tot_counts ,  elapsed_times ,  last_epochs ,  n_epochs ,  train_accs ,  val_accs  = [split_dict[i] for i in \
      ['alloc', 'alive', 'dead_list', 'exp_counts', 'tot_counts', 'elapsed_times', 'last_epochs', 'n_epochs', 'train_accs', 'val_accs']]
        
        exp_counts = sorted(exp_counts)
        all_exp_until = [*range(j, exp_counts[-1], len(split_stats))]
        skipover = [i for i in all_exp_until if i not in exp_counts]
        
        summary += box_str(f'Split ( {j+1} / {len(split_stats)} )', 
                            f'''Alive / Allocated : {alive} / {alloc} {f'(dead: {", ".join(dead_list)})' if dead_list else ''}
                                Experiments count status: {len(exp_counts)} / {tot_counts}
                                Elapsed times : {elapsed_times}
                                Skipped over experiments : {skipover} ({len(skipover)}/{len(all_exp_until)})
                                Current epochs : {last_epochs} (total epochs : {n_epochs})
                                Train accs : {train_accs}
                                Val accs : {val_accs}''',
                            box_width=args.box_width, indent=5) + '\n'
    summary += '\n'

if args.save_path:
    root, _ = os.path.split(args.save_path)
    if not os.path.exists(root):
        print(f'save_path does not exist : {root}')
    else:
        with open(args.save_path, 'w') as f:
            f.write(summary)
else:
    print(summary)
    