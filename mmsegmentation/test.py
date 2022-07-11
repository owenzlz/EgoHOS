from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import glob
import os
from tqdm import tqdm
import argparse
from PIL import Image
import numpy as np 
from skimage.io import imsave
import pdb

def visualize_segmentation(img, seg_result, alpha = 0.4):
    seg_color = np.zeros((img.shape))
    seg_color[seg_result == 0] = (0,    0,   0)     # background
    seg_color[seg_result == 1] = (255,  0,   0)     # left_hand
    seg_color[seg_result == 2] = (0,    0,   255)   # right_hand
    seg_color[seg_result == 3] = (255,  0,   255)   # left_object1
    seg_color[seg_result == 4] = (0,    255, 255)   # right_object1
    seg_color[seg_result == 5] = (0,    255, 0)     # two_object1
    seg_color[seg_result == 6] = (255,  204, 255)   # left_object2
    seg_color[seg_result == 7] = (204,  255, 255)   # right_object2
    seg_color[seg_result == 8] = (204,  255, 204)   # two_object2
    vis = img * (1 - alpha) + seg_color * alpha
    return vis
 


parser = argparse.ArgumentParser(description="")
parser.add_argument("--config_file", default='./work_dirs/upernet_swin_base_patch4_window12_512x512_160k_egohos_handobj2_pretrain_480x360_22K/upernet_swin_base_patch4_window12_512x512_160k_egohos_handobj2_pretrain_480x360_22K.py', type=str)
parser.add_argument("--checkpoint_file", default='./work_dirs/upernet_swin_base_patch4_window12_512x512_160k_egohos_handobj2_pretrain_480x360_22K/best_mIoU_iter_42000.pth', type=str)
parser.add_argument("--img_dir", default='../data/train/image', type=str)
parser.add_argument("--vis_dir", default='./work_dirs/upernet_swin_base_patch4_window12_512x512_160k_egohos_handobj2_pretrain_480x360_22K/outputs/train', type=str)
parser.add_argument("--seg_dir", default='./work_dirs/upernet_swin_base_patch4_window12_512x512_160k_egohos_handobj2_pretrain_480x360_22K/outputs/train_seg', type=str)
args = parser.parse_args()

# build the model from a config file and a checkpoint file
model = init_segmentor(args.config_file, args.checkpoint_file, device='cuda:0')

os.makedirs(args.vis_dir, exist_ok = True)
os.makedirs(args.seg_dir, exist_ok = True)

alpha = 0.3
for file in tqdm(glob.glob(args.img_dir + '/*')):
    fname = os.path.basename(file).split('.')[0]
    img = np.array(Image.open(os.path.join(args.img_dir, fname + '.jpg')))
    seg_result = inference_segmentor(model, file)[0]
    vis = visualize_segmentation(img, seg_result, alpha = alpha)
    imsave(os.path.join(args.vis_dir, fname + '.jpg'), vis)
    imsave(os.path.join(args.seg_dir, fname + '.png'), seg_result.astype(np.uint8))

    

    # pdb.set_trace()

