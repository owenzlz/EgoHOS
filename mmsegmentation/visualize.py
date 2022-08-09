from PIL import Image
import numpy as np 
import argparse
import glob
import pdb
import os 
from skimage.io import imsave

def visualize_twohands(img, seg_result, alpha = 0.4):
    seg_color = np.zeros((img.shape))
    seg_color[seg_result == 0] = (0,    0,   0)     # background
    seg_color[seg_result == 1] = (255,  0,   0)     # left_hand
    seg_color[seg_result == 2] = (0,    0,   255)   # right_hand
    vis = img * (1 - alpha) + seg_color * alpha
    return vis

def visualize_cb(img, seg_result, alpha = 0.4):
    seg_color = np.zeros((img.shape))
    seg_color[seg_result == 0] = (0,    0,   0)     # background
    seg_color[seg_result == 1] = (255,  0,   0)     # contact 
    vis = img * (1 - alpha) + seg_color * alpha
    return vis

def visualize_twohands_obj1(img, seg_result, alpha = 0.4):
    seg_color = np.zeros((img.shape))
    seg_color[seg_result == 0] = (0,    0,   0)     # background
    seg_color[seg_result == 1] = (255,  0,   0)     # left_hand
    seg_color[seg_result == 2] = (0,    0,   255)   # right_hand
    seg_color[seg_result == 3] = (255,  0,   255)   # left_object1
    seg_color[seg_result == 4] = (0,    255, 255)   # right_object1
    seg_color[seg_result == 5] = (0,    255, 0)     # two_object1
    vis = img * (1 - alpha) + seg_color * alpha
    return vis

def visualize_twohands_obj2(img, seg_result, alpha = 0.4):
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



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--mode", default='twohands_obj1', type=str, help='options: twohands, cb, twohands_obj1, twohands_obj2')
    parser.add_argument("--img_dir", default='/mnt/session_space/home/lingzzha/EgoHOS/testimages/images', type=str)
    parser.add_argument("--twohands_dir", default='/mnt/session_space/home/lingzzha/EgoHOS/testimages/pred_twohands', type=str)
    parser.add_argument("--cb_dir", default='/mnt/session_space/home/lingzzha/EgoHOS/testimages/pred_cb', type=str)
    parser.add_argument("--obj1_dir", default='/mnt/session_space/home/lingzzha/EgoHOS/testimages/pred_obj1', type=str)
    parser.add_argument("--obj2_dir", default='/mnt/session_space/home/lingzzha/EgoHOS/testimages/pred_obj2', type=str)
    parser.add_argument("--vis_dir", default='/mnt/session_space/home/lingzzha/EgoHOS/testimages/pred_obj1_vis', type=str)
    args = parser.parse_args()

    os.makedirs(args.vis_dir, exist_ok = True)
    
    for img_file in glob.glob(args.img_dir + '/*'):
        fname = os.path.basename(img_file).split('.')[0]
        img = np.array(Image.open(img_file))

        if args.mode == 'twohands':
            twohands = np.array(Image.open(os.path.join(args.twohands_dir, fname + '.png')))
            twohands_vis = visualize_twohands(img, twohands)
            imsave(os.path.join(args.vis_dir, fname + '.jpg'), twohands_vis)

        elif args.mode == 'cb':
            cb = np.array(Image.open(os.path.join(args.cb_dir, fname + '.png')))
            cb_vis = visualize_cb(img, cb)
            imsave(os.path.join(args.vis_dir, fname + '.jpg'), cb_vis)       

        elif args.mode == 'twohands_obj1':
            twohands = np.array(Image.open(os.path.join(args.twohands_dir, fname + '.png')))
            obj1 = np.array(Image.open(os.path.join(args.obj1_dir, fname + '.png')))
            twohands_obj1 = twohands.copy()
            twohands_obj1[obj1 == 1] = 3
            twohands_obj1[obj1 == 2] = 4
            twohands_obj1[obj1 == 3] = 5
            twohands_obj1_vis = visualize_twohands_obj1(img, twohands_obj1)
            imsave(os.path.join(args.vis_dir, fname + '.jpg'), twohands_obj1_vis)

        elif args.mode == 'twohands_obj2':
            twohands = np.array(Image.open(os.path.join(args.twohands_dir, fname + '.png')))
            obj2 = np.array(Image.open(os.path.join(args.obj2_dir, fname + '.png')))
            twohands_obj2 = twohands.copy()
            twohands_obj2[obj2 == 1] = 3
            twohands_obj2[obj2 == 2] = 4
            twohands_obj2[obj2 == 3] = 5
            twohands_obj2[obj2 == 4] = 6
            twohands_obj2[obj2 == 5] = 7
            twohands_obj2[obj2 == 6] = 8
            twohands_obj2_vis = visualize_twohands_obj2(img, twohands_obj2)
            imsave(os.path.join(args.vis_dir, fname + '.jpg'), twohands_obj2_vis)

        else:
            raise NotImplementedError


