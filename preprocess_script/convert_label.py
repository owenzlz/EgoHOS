"""
    This script converts label into different version, depending
    on the task (allhands, twohands, handobj1, etc.)
"""
from PIL import Image
import numpy as np 
import glob
import pdb
import os
from skimage.io import imsave
from tqdm import tqdm 



def cvt_lbl(src_lbl_dir, dst_lbl_dir, mode):

    os.makedirs(dst_lbl_dir, exist_ok = True)

    for file in tqdm(glob.glob(src_lbl_dir + '/*.png')):

        fname = os.path.basename(file)
        lbl = np.array(Image.open(file))

        if mode == 'allhands':
            lbl_out = np.zeros((lbl.shape))
            lbl_out[lbl == 1] = 1
            lbl_out[lbl == 2] = 1
        elif mode == 'twohands':
            lbl_out = lbl.copy()
            lbl_out[lbl_out > 2] = 0
        elif mode == 'handobj1':
            lbl_out = lbl.copy()
            lbl_out[lbl_out > 5] = 0      

        imsave(os.path.join(dst_lbl_dir, fname), lbl_out)



if __name__ == '__main__':

    lbl_type_list = ['train', 'val', 'test_indomain', 'test_outdomain']

    for lbl_type in lbl_type_list:
            
        lbl_dir = '../data/'+lbl_type+'/label'

        lbl_allhands_dir = '../data/'+lbl_type+'/label_allhands'
        cvt_lbl(lbl_dir, lbl_allhands_dir, 'allhands')

        lbl_twohands_dir = '../data/'+lbl_type+'/label_twohands'
        cvt_lbl(lbl_dir, lbl_twohands_dir, 'twohands')

        lbl_handobj1_dir = '../data/'+lbl_type+'/label_handobj1'
        cvt_lbl(lbl_dir, lbl_handobj1_dir, 'handobj1')





