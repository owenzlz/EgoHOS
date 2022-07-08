from shutil import copyfile
import glob
import pdb
import os 
import random
from tqdm import tqdm

img_dir = '/mnt/session_space/home/lingzzha/EgoHOS/data/train/image'
lbl_dir = '/mnt/session_space/home/lingzzha/EgoHOS/data/train/label'

sub_k_list = [200]
# ,500,1000]

fname_list = []
for file in glob.glob(img_dir + '/*'):
    fname = os.path.basename(file)
    fname_list.append(fname)

for sub_k in sub_k_list:

    sub_img_dir = '/mnt/session_space/home/lingzzha/EgoHOS/data/train/image_' + str(sub_k); os.makedirs(sub_img_dir, exist_ok = True)
    sub_lbl_dir = '/mnt/session_space/home/lingzzha/EgoHOS/data/train/label_' + str(sub_k); os.makedirs(sub_lbl_dir, exist_ok = True)

    print('Processing: ', sub_k)
    sub_fname_list = random.sample(fname_list, sub_k)
    for fname in tqdm(sub_fname_list):

        src_img_file = os.path.join(img_dir, fname)
        dst_img_file = os.path.join(sub_img_dir, fname)
        copyfile(src_img_file, dst_img_file)

        src_lbl_file = os.path.join(lbl_dir, fname.split('.')[0] + '.png')
        dst_lbl_file = os.path.join(sub_lbl_dir, fname.split('.')[0] + '.png')
        copyfile(src_lbl_file, dst_lbl_file)





