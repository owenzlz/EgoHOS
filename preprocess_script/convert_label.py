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

    if os.path.exists(dst_lbl_dir):
        os.system('rm -rf ' + dst_lbl_dir)
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
        elif mode == 'obj1':
            lbl_out = lbl.copy()
            lbl_out[lbl_out > 5] = 0      
            lbl_out[lbl_out == 1] = 0
            lbl_out[lbl_out == 2] = 0
            lbl_out[lbl_out == 3] = 1
            lbl_out[lbl_out == 4] = 2
            lbl_out[lbl_out == 5] = 3
        elif mode == 'obj2':
            lbl_out = lbl.copy()   
            lbl_out[lbl_out == 1] = 0
            lbl_out[lbl_out == 2] = 0
            lbl_out[lbl_out == 3] = 1
            lbl_out[lbl_out == 4] = 2
            lbl_out[lbl_out == 5] = 3
            lbl_out[lbl_out == 6] = 4
            lbl_out[lbl_out == 7] = 5
            lbl_out[lbl_out == 8] = 6
        elif mode == 'foreground':
            lbl_out = lbl.copy()
            lbl_out[lbl_out > 0] = 1
            
            # print(np.unique(lbl_out))
            # pdb.set_trace()

            # lbl_out[lbl_out == 3] = 1
            # lbl_out[lbl_out == 4] = 2            

        imsave(os.path.join(dst_lbl_dir, fname), lbl_out)



if __name__ == '__main__':

    # lbl_type_list = ['train', 'val', 'test_indomain', 'test_outdomain']
    lbl_type_list = ['train']
    
    for lbl_type in lbl_type_list:
        
        lbl_dir = '../data/'+lbl_type+'/label_ccda'

        lbl_twohands_dir = '../data/'+lbl_type+'/label_twohands_ccda'
        cvt_lbl(lbl_dir, lbl_twohands_dir, 'twohands')

        lbl_handobj1_dir = '../data/'+lbl_type+'/label_handobj1_ccda'
        cvt_lbl(lbl_dir, lbl_handobj1_dir, 'handobj1')

        lbl_handobj1_dir = '../data/'+lbl_type+'/label_obj1_ccda'
        cvt_lbl(lbl_dir, lbl_handobj1_dir, 'obj1')

        lbl_handobj1_dir = '../data/'+lbl_type+'/label_obj2_ccda'
        cvt_lbl(lbl_dir, lbl_handobj1_dir, 'obj2')

        lbl_allhands_dir = '../data/'+lbl_type+'/label_allhands_ccda'
        cvt_lbl(lbl_dir, lbl_allhands_dir, 'allhands')
        
        # lbl_handobj1_dir = '../data/'+lbl_type+'/label_foreground'
        # cvt_lbl(lbl_dir, lbl_handobj1_dir, 'foreground')


        

