"""
    This script checks unique values in label
"""
from PIL import Image
import numpy as np 
import glob
import pdb
import os 
from tqdm import tqdm

lbl_type_list = ['train', 'val', 'test_indomain', 'test_outdomain']

for lbl_type in lbl_type_list:

    image_dir = '../data/'+lbl_type+'/image'
    label_dir = '../data/'+lbl_type+'/label'

    for lbl_file in tqdm(glob.glob(label_dir + '/*.png')):

        fname = os.path.basename(lbl_file).split('.')[0]
        lbl = np.array(Image.open(lbl_file))
        img_file = os.path.join(image_dir, fname + '.jpg')
        img = np.array(Image.open(img_file))
        unique_vals = np.unique(lbl)

        if min(unique_vals) < 0 or max(unique_vals) > 8:
            print('label out of bound!')

        if lbl.shape[0] != img.shape[0] or lbl.shape[1] != img.shape[1]:
            print('size mismatch!')

        if len(lbl.shape) != 2 or len(img.shape) != 3:
            print('chanel mismatch!')
            print(fname, lbl.shape, img.shape)

            os.remove(img_file)
            os.remove(lbl_file)




