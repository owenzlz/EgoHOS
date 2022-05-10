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



if __name__ == '__main__':

    lbl_type_list = ['train', 'val', 'test_indomain', 'test_outdomain']

    for lbl_type in lbl_type_list:

        print('processing: ' + lbl_type)
            
        img_dir = '../data/'+lbl_type+'/image'
        lbl_dir = '../data/'+lbl_type+'/label'
        lbl_allhands_dir = '../data/'+lbl_type+'/label_allhands'
        lbl_twohands_dir = '../data/'+lbl_type+'/label_twohands'
        lbl_handobj1_dir = '../data/'+lbl_type+'/label_handobj1'

        dst_img_dir = img_dir + '_downsize'; os.makedirs(dst_img_dir, exist_ok = True)
        dst_lbl_dir = lbl_dir + '_downsize'; os.makedirs(dst_lbl_dir, exist_ok = True)
        dst_lbl_allhands_dir = lbl_allhands_dir + '_downsize'; os.makedirs(dst_lbl_allhands_dir, exist_ok = True)
        dst_lbl_twohands_dir = lbl_twohands_dir + '_downsize'; os.makedirs(dst_lbl_twohands_dir, exist_ok = True)
        dst_lbl_handobj1_dir = lbl_handobj1_dir + '_downsize'; os.makedirs(dst_lbl_handobj1_dir, exist_ok = True)

        for img_file in tqdm(glob.glob(img_dir + '/*')):
            
            fname = os.path.basename(img_file).split('.')[0]
            lbl_file = os.path.join(lbl_dir, fname + '.png')
            lbl_allhands_file = os.path.join(lbl_allhands_dir, fname + '.png')
            lbl_twohands_file = os.path.join(lbl_twohands_dir, fname + '.png')
            lbl_handobj1_file = os.path.join(lbl_handobj1_dir, fname + '.png')

            img = Image.open(img_file).resize((480, 360))
            lbl = Image.open(lbl_file).resize((480, 360))
            lbl_allhands = Image.open(lbl_allhands_file).resize((480, 360))
            lbl_twohands = Image.open(lbl_twohands_file).resize((480, 360))
            lbl_handobj1 = Image.open(lbl_handobj1_file).resize((480, 360))

            img.save(os.path.join(dst_img_dir, fname + '.jpg'))
            lbl.save(os.path.join(dst_lbl_dir, fname + '.png'))
            lbl_allhands.save(os.path.join(dst_lbl_allhands_dir, fname + '.png'))
            lbl_twohands.save(os.path.join(dst_lbl_twohands_dir, fname + '.png'))
            lbl_handobj1.save(os.path.join(dst_lbl_handobj1_dir, fname + '.png'))

            
            # pdb.set_trace()

    


