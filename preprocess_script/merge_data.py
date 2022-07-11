from shutil import copyfile
import glob
import pdb
import os 
from tqdm import tqdm 

train_img_dir = '../data/train/image'
train_lbl_dir = '../data/train/label'
train_pred_twohands_dir = '../data/train/pred_twohands'
val_img_dir = '../data/val/image'
val_lbl_dir = '../data/val/label'
val_pred_twohands_dir = '../data/val/pred_twohands'
test_img_dir = '../data/test_indomain/image'
test_lbl_dir = '../data/test_indomain/label'
test_pred_twohands_dir = '../data/test_indomain/pred_twohands'

dst_img_dir = '../data/train/image_ccda'
dst_lbl_dir = '../data/train/label_ccda'
dst_pred_twohands_dir = '../data/train/pred_twohands_ccda'

os.system('rm -rf ' + dst_pred_twohands_dir); os.makedirs(dst_pred_twohands_dir, exist_ok = True)

'''

os.system('rm -rf ' + dst_img_dir); os.makedirs(dst_img_dir, exist_ok = True)
os.system('rm -rf ' + dst_lbl_dir); os.makedirs(dst_lbl_dir, exist_ok = True)

img_dir_list = [train_img_dir, val_img_dir, test_img_dir]
lbl_dir_list = [train_lbl_dir, val_lbl_dir, test_lbl_dir]

for img_dir in img_dir_list:
    for file in tqdm(glob.glob(img_dir + '/*')):
        fname = os.path.basename(file)
        dst_file = os.path.join(dst_img_dir, fname)
        copyfile(file, dst_file)

for lbl_dir in lbl_dir_list:
    for file in tqdm(glob.glob(lbl_dir + '/*')):
        fname = os.path.basename(file)
        dst_file = os.path.join(dst_lbl_dir, fname)
        copyfile(file, dst_file)
'''

pred_twohands_dir_list = [train_pred_twohands_dir, val_pred_twohands_dir, test_pred_twohands_dir]
for pred_twohands_dir in pred_twohands_dir_list:
    for file in tqdm(glob.glob(pred_twohands_dir + '/*')):
        fname = os.path.basename(file)
        dst_file = os.path.join(dst_pred_twohands_dir, fname)
        copyfile(file, dst_file)

        # pdb.set_trace()




