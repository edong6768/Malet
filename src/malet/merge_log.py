import os, glob

from absl import app, flags
from rich import print
from rich.panel import Panel
from rich.align import Align

from malet.experiment import ExperimentLog

FLAGS = flags.FLAGS

def merger(argv):
    if not FLAGS.folder and not FLAGS.files:
        raise ValueError('Either folder or files must be specified.')
    
    print()
    if FLAGS.files is None:
        print(
            Align(
                Panel(f'Merging all {len(glob.glob(os.path.join(FLAGS.folder, "*.tsv")))} files in folder {FLAGS.folder}.', padding=(1, 3)),
                align='center'
            )
        )
        FLAGS.save_path = FLAGS.save_path or os.path.join(FLAGS.folder, 'log_merged.tsv')
        ExperimentLog.merge_folder(FLAGS.folder, FLAGS.save_path)
    else:
        if FLAGS.folder:
            FLAGS.save_path = FLAGS.save_path or os.path.join(FLAGS.folder, 'log_merged.tsv')
            files = [os.path.join(FLAGS.folder, f) for f in FLAGS.files]
        else:
            FLAGS.save_path = FLAGS.save_path or os.path.join(os.path.dirname(FLAGS.files[0]), 'log_merged.tsv')
            files = FLAGS.files
        print(
            Align(
                Panel(f'Merging {len(files)} files: \n'+'\n'.join(files), padding=(1, 3)),
                align='center'
            )
        )
        ExperimentLog.merge_tsv(*files, FLAGS.save_path)
    
    print(
        Align(
            Panel(f'Merged log saved to {FLAGS.save_path}.', padding=(1, 3)),
            align='center'
        )
    )
    
def main():
    flags.DEFINE_string(        'folder'    , None, 'Folder containing the logs to merge. If files are specified, only the specified files in the folder will be merged.')
    flags.DEFINE_spaceseplist(  'files'     , None, 'Files to merge. If folder is specified, this should be relative to the folder.')
    flags.DEFINE_string(        'save_path' , None, 'Path to save the merged log file.')
    app.run(merger)

if __name__ == '__main__':
    main()