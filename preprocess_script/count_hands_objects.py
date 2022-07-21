from PIL import Image
import numpy as np 
import glob
import pdb
import os
from tqdm import tqdm

train_dir = '../data/train/label'
val_dir = '../data/val/label'
test_dir = '../data/test_indomain/label'

dir_list = [train_dir, val_dir, test_dir]

hand_cnt = 0; obj_cnt = 0
for dir in dir_list:
    for file in tqdm(glob.glob(dir + '/*')):

        lbl = np.array(Image.open(file))
        unique_vals = np.unique(lbl)

        for unique_val in unique_vals:
            if unique_val == 1 or unique_val == 2:
                hand_cnt += 1
            elif unique_val > 2:
                obj_cnt += 1

print(hand_cnt, obj_cnt)
pdb.set_trace()

