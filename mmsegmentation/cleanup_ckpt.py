import glob
import pdb
import os 


ckpt_dir = './work_dirs'

for ckpt_file in glob.glob(ckpt_dir + '/**/*.pth'):

    if 'best' in ckpt_file:
        print('keep : ', ckpt_file)
    else:
        print('remove: ', ckpt_file)
        os.remove(ckpt_file)





