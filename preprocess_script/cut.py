from PIL import Image
from tqdm import tqdm
import numpy as np  
import argparse
import glob
import pdb
import os

parser = argparse.ArgumentParser(description="resize image using bicubic upsampling")
parser.add_argument("--in_dir", default='/mnt/session_space/home/lingzzha/EgoHOS/data/train/lama_512_pad', type=str)
parser.add_argument("--out_dir", default='/mnt/session_space/home/lingzzha/EgoHOS/data/train/lama_512', type=str)
parser.add_argument("--ict_tf", default=False, type=bool)
parser.add_argument("--medfe_tf", default=False, type=bool)
parser.add_argument("--pad_info_dir", default='/mnt/session_space/home/lingzzha/EgoHOS/data/train/image_512_pad_info', type=str)
parser.add_argument("--size", default=512, type=int)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

for img_file in tqdm(glob.glob(args.in_dir + '/*')):
    
    fname = img_file.split('/')[-1].split('.')[0]   
    img = np.array(Image.open(img_file))
    cut_info = np.load(os.path.join(args.pad_info_dir, fname+'.npy'))
    
    if args.ict_tf or args.medfe_tf: 
        cut_info = cut_info/2
    # pdb.set_trace()
    
    if cut_info[0,0] == 0:
        img_cut = img[args.size - int(cut_info[0,1]):,:,:]
    else:
        img_cut = img[:,:int(cut_info[0,1]),:]

    img_cut = Image.fromarray(img_cut.astype('uint8'))
    img_cut.save(os.path.join(args.out_dir, fname+'.jpg'))



