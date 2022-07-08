from PIL import Image
import numpy as np 
from skimage.io import imsave
import glob
import pdb
import os 
from tqdm import tqdm
# from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
import random
from shutil import copyfile
import argparse

parser = argparse.ArgumentParser(description="composite")
parser.add_argument("--aug_numbers", default=3, type=int)
parser.add_argument("--top_k", default=10, type=int)
parser.add_argument("--random_aug", default=True, type=bool)
parser.add_argument("--composite_hr", default=True, type=bool)
parser.add_argument("--hr_img_dir", default='/mnt/session_space/home/lingzzha/EgoHOS/data/train/image', type=str)
parser.add_argument("--hr_lbl_dir", default='/mnt/session_space/home/lingzzha/EgoHOS/data/train/label', type=str)
parser.add_argument("--img_dir", default='/mnt/session_space/home/lingzzha/EgoHOS/data/train/image_512', type=str)
parser.add_argument("--lbl_dir", default='/mnt/session_space/home/lingzzha/EgoHOS/data/train/label_512', type=str)
parser.add_argument("--lama_dir", default='/mnt/session_space/home/lingzzha/EgoHOS/data/train/lama_512', type=str)
parser.add_argument("--lama_feat_dir", default='/mnt/session_space/home/lingzzha/EgoHOS/data/train/lama_512_feature', type=str)
parser.add_argument("--aug_img_dir", default='/mnt/session_space/home/lingzzha/EgoHOS/data/train/image_ccda', type=str)
parser.add_argument("--aug_lbl_dir", default='/mnt/session_space/home/lingzzha/EgoHOS/data/train/label_ccda', type=str)
args = parser.parse_args()

os.system('rm -rf ' + args.aug_img_dir); os.makedirs(args.aug_img_dir, exist_ok = True)
os.system('rm -rf ' + args.aug_lbl_dir); os.makedirs(args.aug_lbl_dir, exist_ok = True)

# Load combine features into a list
fname_list = []
feature_list = []
for file in glob.glob(args.lama_feat_dir + '/*'):
    fname = os.path.basename(file).split('.')[0]
    feature = np.load(file)
    # print(feature.shape)
    feature_list.append(feature)    
    fname_list.append(fname)



# composite images and generate labels
for file in tqdm(glob.glob(args.lama_feat_dir + '/*')): 

    query_fname = os.path.basename(file).split('.')[0]
    query_feature = np.load(file)

    score_list = []
    for feature in feature_list:
        score = cosine(query_feature, feature)
        if score != 0.0:
            score_list.append(score)

    sorted_index_list = np.argsort(np.array(score_list))
    
    if args.random_aug:
        top_k_sorted_index_list = sorted_index_list
    else:
        top_k_sorted_index_list = sorted_index_list[:args.top_k]
    
    for aug_idx in range(args.aug_numbers):
        
        size_match = False
        while not size_match:
            sample_idx = random.randint(0, len(top_k_sorted_index_list)-1)
            index = top_k_sorted_index_list[sample_idx]
            select_fname = fname_list[index]

            query_img = np.array(Image.open(os.path.join(args.img_dir, query_fname + '.jpg')))
            query_lbl = np.array(Image.open(os.path.join(args.lbl_dir, query_fname + '.png')))
            query_msk = np.zeros((query_lbl.shape)); query_msk[query_lbl>0] = 1
            query_msk = np.repeat(np.expand_dims(query_msk, 2), 3, 2)
            select_img = np.array(Image.open(os.path.join(args.lama_dir, select_fname + '.jpg')))

            if query_img.shape == select_img.shape:
                size_match = True
        
        if args.composite_hr:
            hr_query_img = np.array(Image.open(os.path.join(args.hr_img_dir, query_fname + '.jpg')))
            hr_query_lbl = np.array(Image.open(os.path.join(args.hr_lbl_dir, query_fname + '.png')))
            hr_query_msk = np.zeros((hr_query_lbl.shape)); hr_query_msk[hr_query_lbl>0] = 1
            hr_query_msk = np.repeat(np.expand_dims(hr_query_msk, 2), 3, 2)
            hr_select_img = np.array(Image.open(os.path.join(args.lama_dir, select_fname + '.jpg')).resize((hr_query_img.shape[1], hr_query_img.shape[0])))
            new_img = hr_query_img * hr_query_msk + hr_select_img * (1 - hr_query_msk)
        else:
            new_img = query_img * query_msk + select_img * (1 - query_msk)
        
        imsave(os.path.join(args.aug_img_dir, query_fname + '_' + str(aug_idx) + '.jpg'), new_img)
        src_lbl_file = os.path.join(args.lbl_dir, query_fname + '.png')
        dst_lbl_file = os.path.join(args.aug_lbl_dir, query_fname + '_' + str(aug_idx) + '.png')
        copyfile(src_lbl_file, dst_lbl_file)
    
    src_ori_img_file = os.path.join(args.img_dir, query_fname + '.jpg')
    dst_ori_img_file = os.path.join(args.aug_img_dir, query_fname + '.jpg')
    copyfile(src_ori_img_file, dst_ori_img_file)
    
    src_ori_lbl_file = os.path.join(args.lbl_dir, query_fname + '.png')
    dst_ori_lbl_file = os.path.join(args.aug_lbl_dir, query_fname + '.png')
    copyfile(src_ori_lbl_file, dst_ori_lbl_file)



    
    # pdb.set_trace()