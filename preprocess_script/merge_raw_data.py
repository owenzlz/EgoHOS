"""
    This script helps merge raw maadaa labels into the 
    train/val/test (indomain/outdomain)
"""
from shutil import copyfile
import glob
import pdb
import os 
from tqdm import tqdm
import random
import numpy as np 
from PIL import Image
from skimage.io import imsave


# # Label Mapping
# LEFT_HAND = 61 -> 1
# RIGHT_HAND = 71 -> 2
# LEFT_OBJ1 = 1 -> 3
# LEFT_OBJ2 = 11 -> 6
# RIGHT_OBJ1 = 21 -> 4
# RIGHT_OBJ2 = 31 -> 7
# TWO_OBJ1 = 41 -> 5
# TWO_OBJ2 = 51 -> 8


def append_flist_dst_fname(p_dir, data_source, src_flist_list, dst_fname_list):
    for file in tqdm(glob.glob(p_dir + '/*.png')):
        fname = os.path.basename(file).split('.')[0]
        src_flist_list.append(file)
        dst_fname = data_source + '_' + fname
        dst_fname_list.append(dst_fname)
    
    return src_flist_list, dst_fname_list



def process_img_lbl(idx_list, dst_img_dir, dst_lbl_dir):

    for idx in tqdm(idx_list):
        src_file = src_flist_list[idx]
        src_fname = src_file.split('.')[0]
        dst_fname = dst_fname_list[idx]
        path_dir = os.path.dirname(src_file)

        src_img_file = os.path.join(path_dir, src_fname + '.jpg')
        dst_img_file = os.path.join(dst_img_dir, dst_fname + '.jpg')
        copyfile(src_img_file, dst_img_file)
        
        lbl = np.array(Image.open(src_file))
        lbl[lbl == 1] = 3 
        lbl[lbl == 61] = 1 
        lbl[lbl == 71] = 2
        lbl[lbl == 11] = 6
        lbl[lbl == 21] = 4
        lbl[lbl == 31] = 7
        lbl[lbl == 41] = 5
        lbl[lbl == 51] = 8

        imsave(os.path.join(dst_lbl_dir, dst_fname + '.png'), lbl)



if __name__ == '__main__':

    # Source directories
    p1_dir = '/mnt/session_space/home/lingzzha/EgoHOS/data/raw_mada/220003_part01_v3_2120_20220119/epic_extract_frames_move_use'
    p2a_dir = '/mnt/session_space/home/lingzzha/EgoHOS/data/raw_mada/220003_part02_v1_1355_20220214/escape_room_airbnb_use'
    p2b_dir = '/mnt/session_space/home/lingzzha/EgoHOS/data/raw_mada/220003_part02_v1_1355_20220214/escape_select_extract_frames'
    p2c_dir = '/mnt/session_space/home/lingzzha/EgoHOS/data/raw_mada/220003_part02_v1_1355_20220214/sample_thu_frames_use'
    p3_dir = '/mnt/session_space/home/lingzzha/EgoHOS/data/raw_mada/220003_part03_v1_500_20220216/selected_ofd_cooking'
    p4_dir = '/mnt/session_space/home/lingzzha/EgoHOS/data/raw_mada/220003_part04_v1_1253_20220222/ego4d_batch1'
    p5_dir = '/mnt/session_space/home/lingzzha/EgoHOS/data/raw_mada/220003_part05_v1_855_20220301/ego4d_batch2'
    p6_dir = '/mnt/session_space/home/lingzzha/EgoHOS/data/raw_mada/220003_part06_1_v1_560_20220311/ego4d_batch3'
    p7_dir = '/mnt/session_space/home/lingzzha/EgoHOS/data/raw_mada/220003_part07_v1_5100_20220402/ego4d_batch_final'


    # Save directories
    train_img_dir = '/mnt/session_space/home/lingzzha/EgoHOS/data/train/image'
    train_lbl_dir = '/mnt/session_space/home/lingzzha/EgoHOS/data/train/label'
    val_img_dir = '/mnt/session_space/home/lingzzha/EgoHOS/data/val/image'
    val_lbl_dir = '/mnt/session_space/home/lingzzha/EgoHOS/data/val/label'
    test_indomain_img_dir = '/mnt/session_space/home/lingzzha/EgoHOS/data/test_indomain/image'
    test_indomain_lbl_dir = '/mnt/session_space/home/lingzzha/EgoHOS/data/test_indomain/label'
    test_outdomain_img_dir = '/mnt/session_space/home/lingzzha/EgoHOS/data/test_outdomain/image'
    test_outdomain_lbl_dir = '/mnt/session_space/home/lingzzha/EgoHOS/data/test_outdomain/labal'
    
    dst_dir_list = [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir, \
                    test_indomain_img_dir, test_indomain_lbl_dir, test_outdomain_img_dir, test_outdomain_lbl_dir]
    for dst_dir in dst_dir_list:
        os.system('rm -rf ' + dst_dir)
    
    os.makedirs(train_img_dir, exist_ok = True)
    os.makedirs(train_lbl_dir, exist_ok = True)
    os.makedirs(val_img_dir, exist_ok = True)
    os.makedirs(val_lbl_dir, exist_ok = True)
    os.makedirs(test_indomain_img_dir, exist_ok = True)
    os.makedirs(test_indomain_lbl_dir, exist_ok = True)
    os.makedirs(test_outdomain_img_dir, exist_ok = True)
    os.makedirs(test_outdomain_lbl_dir, exist_ok = True)
    

    # Label values
    LEFT_HAND = 61
    RIGHT_HAND = 71
    LEFT_OBJ1 = 1
    LEFT_OBJ2 = 11
    RIGHT_OBJ1 = 21
    RIGHT_OBJ2 = 31
    TWO_OBJ1 = 41
    TWO_OBJ2 = 51

    
    # merge train / val / test (indomain)
    src_flist_list = []; dst_fname_list = []
    src_flist_list, dst_fname_list = append_flist_dst_fname(p1_dir, 'epic', src_flist_list, dst_fname_list)
    src_flist_list, dst_fname_list = append_flist_dst_fname(p2a_dir, 'escape', src_flist_list, dst_fname_list)
    src_flist_list, dst_fname_list = append_flist_dst_fname(p2b_dir, 'escape', src_flist_list, dst_fname_list)
    src_flist_list, dst_fname_list = append_flist_dst_fname(p2c_dir, 'thu', src_flist_list, dst_fname_list)
    src_flist_list, dst_fname_list = append_flist_dst_fname(p4_dir, 'ego4d', src_flist_list, dst_fname_list)
    src_flist_list, dst_fname_list = append_flist_dst_fname(p5_dir, 'ego4d', src_flist_list, dst_fname_list)
    src_flist_list, dst_fname_list = append_flist_dst_fname(p6_dir, 'ego4d', src_flist_list, dst_fname_list)
    src_flist_list, dst_fname_list = append_flist_dst_fname(p7_dir, 'ego4d', src_flist_list, dst_fname_list)

    leng = len(src_flist_list)
    train_leng = int(leng * 0.8)
    val_leng = int(leng * 0.1)
    idx_list = np.arange(0, leng, 1).tolist()
    train_idx_list = sorted(random.sample(idx_list, train_leng))
    valtest_idx_list = list(set(idx_list) - set(train_idx_list))
    val_idx_list = random.sample(valtest_idx_list, val_leng)
    test_idx_list = list(set(valtest_idx_list) - set(val_idx_list))

    print('train: ', len(train_idx_list))
    print('val: ', len(val_idx_list))
    print('test: ', len(test_idx_list))

    process_img_lbl(train_idx_list, train_img_dir, train_lbl_dir)
    process_img_lbl(val_idx_list, val_img_dir, val_lbl_dir)
    process_img_lbl(test_idx_list, test_indomain_img_dir, test_indomain_lbl_dir)

    src_flist_list = []; dst_fname_list = []
    src_flist_list, dst_fname_list = append_flist_dst_fname(p3_dir, 'youtube', src_flist_list, dst_fname_list)
    test_outdomain_idx_list = np.arange(0, len(src_flist_list), 1)
    process_img_lbl(test_outdomain_idx_list, test_outdomain_img_dir, test_outdomain_lbl_dir)


    

    