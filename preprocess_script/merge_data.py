from shutil import copyfile
import glob
import pdb
import os 
from tqdm import tqdm 

train_img_dir = '../data/train/image'
train_lbl_dir = '../data/train/label'
val_img_dir = '../data/val/image'
val_lbl_dir = '../data/val/label'
test_img_dir = '../data/test_indomain/image'
test_lbl_dir = '../data/test_indomain/label'

dst_img_dir = '../data/train/image_ccda'
dst_lbl_dir = '../data/train/label_ccda'
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








