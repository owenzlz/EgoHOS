

from skimage.io import imsave
from tqdm import tqdm
from PIL import Image
import numpy as np 
import glob
import cv2
import pdb
import os 

dilate_iter = 3
msk_dir = '/mnt/session_space/home/lingzzha/EgoHOS/data/train/label_foreground_512_pad'
msk_dilate_dir = '/mnt/session_space/home/lingzzha/EgoHOS/data/train/label_foreground_512_pad_d' 

msk_dilate_dir = msk_dilate_dir + str(dilate_iter)
os.makedirs(msk_dilate_dir, exist_ok = True)

kernel = np.ones((5,5), np.uint8)

for file in tqdm(glob.glob(msk_dir + '/*')):
    fname = os.path.basename(file)
    mask = np.array(Image.open(file).convert('RGB')); mask[mask > 0] = 1
    mask_dilate = cv2.dilate(mask, kernel, iterations=dilate_iter)
    imsave(os.path.join(msk_dilate_dir, fname), mask_dilate)




