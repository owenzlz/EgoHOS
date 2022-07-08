from PIL import Image
from tqdm import tqdm
import numpy as np  
import argparse
import glob
import pdb
import os

parser = argparse.ArgumentParser(description="resize image using bicubic upsampling")
parser.add_argument("--in_dir", default='/mnt/session_space/home/lingzzha/EgoHOS/data/train/label_foreground_512', type=str)
parser.add_argument("--out_dir", default='/mnt/session_space/home/lingzzha/EgoHOS/data/train/label_foreground_512_pad', type=str)
parser.add_argument("--pad_info_dir", default='/mnt/session_space/home/lingzzha/EgoHOS/data/train/label_foreground_512_pad_info', type=str)
parser.add_argument("--size", default=512, type=int)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
os.makedirs(args.pad_info_dir, exist_ok=True)

i = 0
for img_file in tqdm(glob.glob(args.in_dir+'/*')):

    fname = img_file.split('/')[-1]

    img = np.array(Image.open(img_file))
    w, h = img.shape[0], img.shape[1]
    
    if len(img.shape) == 3:
        img_pad = np.zeros((args.size, args.size, img.shape[2]))
    else:
        img_pad = np.zeros((args.size, args.size))

    pad_info = np.zeros((1,2))

    if w < h: 
        if len(img.shape) == 3:
            img_pad[args.size - w:, :, :] = img
        else:
            img_pad[args.size - w:, :] = img
        pad_info[0, 0] = 0
        pad_info[0, 1] = w
    else:
        if len(img.shape) == 3:
            img_pad[:, :h, :] = img
        else:
            img_pad[:, :h] = img
        pad_info[0, 0] = 1
        pad_info[0, 1] = h
    
    img_pad = Image.fromarray(img_pad.astype('uint8'))
    img_pad.save(os.path.join(args.out_dir, fname))
    
    np.save(os.path.join(args.pad_info_dir, str(fname.split('.')[0])+'.npy'), pad_info)

    i += 1

    # pdb.set_trace()



    