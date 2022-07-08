import numpy as np 
from PIL import Image
from tqdm import tqdm
import torch
import glob
import pdb
import os
from skimage.io import imsave
import argparse

parser = argparse.ArgumentParser(description="resize image using nearest-neighbor upsampling")
parser.add_argument("--img_dir", default='/mnt/session_space/home/lingzzha/EgoHOS/data/train/label', type=str)
parser.add_argument("--mask_tf", default=True, type=bool)
parser.add_argument("--max_scale", default=512, type=int)
args = parser.parse_args()

save_dir = args.img_dir+'_'+str(args.max_scale)
os.makedirs(save_dir, exist_ok = True)

for file in tqdm(glob.glob(args.img_dir + '/*')):
    fname = file.split('/')[-1]
    img = Image.open(file); w, h = img.size
    
    if w > h: 
        aspect_ratio = h/w
        if args.mask_tf:
            img_resize = Image.open(file).resize((args.max_scale, int(args.max_scale * aspect_ratio)), resample = Image.NEAREST)
        else:
            img_resize = Image.open(file).resize((args.max_scale, int(args.max_scale * aspect_ratio)), resample = Image.BICUBIC)
    else:
        aspect_ratio = w/h
        if args.mask_tf:
            img_resize = Image.open(file).resize((int(args.max_scale * aspect_ratio), args.max_scale), resample = Image.NEAREST)
        else:
            img_resize = Image.open(file).resize((int(args.max_scale * aspect_ratio), args.max_scale), resample = Image.BICUBIC)
    

    # if not fname.startswith('20210619'):
    #     img_resize = Image.fromarray(np.array(np.rot90(img_resize, 3)))

    img_resize.save(os.path.join(save_dir, fname))