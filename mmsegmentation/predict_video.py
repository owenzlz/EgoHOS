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
import imageio
from visualize import visualize_twohands_obj1, visualize_twohands_obj2



parser = argparse.ArgumentParser(description="")
parser.add_argument("--mode", default='obj1', type=str, help='options: obj1, obj2')
parser.add_argument("--input_video_file", default='../testvideos/testvideo1_short.mp4', type=str)
parser.add_argument("--output_video_file", default='../testvideos/testvideo1_short_result.mp4', type=str)
parser.add_argument("--twohands_config_file", default='./work_dirs/seg_twohands_ccda/seg_twohands_ccda.py', type=str)
parser.add_argument("--twohands_checkpoint_file", default='./work_dirs/seg_twohands_ccda/best_mIoU_iter_56000.pth', type=str)
parser.add_argument("--cb_config_file", default='./work_dirs/twohands_to_cb_ccda/twohands_to_cb_ccda.py', type=str)
parser.add_argument("--cb_checkpoint_file", default='./work_dirs/twohands_to_cb_ccda/best_mIoU_iter_76000.pth', type=str)
parser.add_argument("--obj1_config_file", default='./work_dirs/twohands_cb_to_obj1_ccda/twohands_cb_to_obj1_ccda.py', type=str)
parser.add_argument("--obj1_checkpoint_file", default='./work_dirs/twohands_cb_to_obj1_ccda/best_mIoU_iter_34000.pth', type=str)
parser.add_argument("--obj2_config_file", default='./work_dirs/twohands_cb_to_obj2_ccda/twohands_cb_to_obj2_ccda.py', type=str)
parser.add_argument("--obj2_checkpoint_file", default='./work_dirs/twohands_cb_to_obj2_ccda/best_mIoU_iter_32000.pth', type=str)
parser.add_argument("--remove_intermediate_images", default=False, type=bool)
args = parser.parse_args()

video_dir = args.input_video_file.split('.mp4')[0]; os.makedirs(video_dir, exist_ok = True)
video_image_dir = os.path.join(video_dir, 'images'); os.makedirs(video_image_dir, exist_ok = True)
video_twohands_dir = os.path.join(video_dir, 'pred_twohands'); os.makedirs(video_twohands_dir, exist_ok = True)
video_cb_dir = os.path.join(video_dir, 'pred_cb'); os.makedirs(video_cb_dir, exist_ok = True)
video_obj1_dir = os.path.join(video_dir, 'pred_obj1'); os.makedirs(video_obj1_dir, exist_ok = True)
video_obj2_dir = os.path.join(video_dir, 'pred_obj2'); os.makedirs(video_obj2_dir, exist_ok = True)


# # extract video frames and save them into a directory
print('Reading and extracting video frames......')
reader = imageio.get_reader(args.input_video_file, 'ffmpeg')
fps = reader.get_meta_data()['fps']
for num, image in enumerate(reader):
    save_img_file = os.path.join(video_image_dir, str(num).zfill(8)+'.jpg')
    imsave(save_img_file, image)

# predict twohands
cmd_pred_twohands = 'python predict_image.py \
                    --config_file %s \
                    --checkpoint_file %s \
                    --img_dir %s \
                    --pred_seg_dir %s' % (args.twohands_config_file, args.twohands_checkpoint_file, video_image_dir, video_twohands_dir)

print('Predicting twohands......')
print(cmd_pred_twohands)
os.system(cmd_pred_twohands)

# predict cb
cmd_pred_cb = 'python predict_image.py \
                    --config_file %s \
                    --checkpoint_file %s \
                    --img_dir %s \
                    --pred_seg_dir %s' % (args.cb_config_file, args.cb_checkpoint_file, video_image_dir, video_cb_dir)

print('Predicting contact boundary......')
print(cmd_pred_cb)
os.system(cmd_pred_cb)

if args.mode == 'obj1':
    # predict obj1
    cmd_pred_obj1 = 'python predict_image.py \
                        --config_file %s \
                        --checkpoint_file %s \
                        --img_dir %s \
                        --pred_seg_dir %s' % (args.obj1_config_file, args.obj1_checkpoint_file, video_image_dir, video_obj1_dir)

    print('Predicting 1st order interacting objects......')
    print(cmd_pred_obj1)
    os.system(cmd_pred_obj1)
elif args.mode == 'obj2':
    # predict obj2
    cmd_pred_obj2 = 'python predict_image.py \
                        --config_file %s \
                        --checkpoint_file %s \
                        --img_dir %s \
                        --pred_seg_dir %s' % (args.obj2_config_file, args.obj2_checkpoint_file, video_image_dir, video_obj2_dir)

    print('Predicting 2nd order interacting objects......')
    print(cmd_pred_obj2)
    os.system(cmd_pred_obj2)
else:
    raise notimplementederror

# stitch prediction into a video
print('stitch prediction into a video......')
writer = imageio.get_writer(args.output_video_file, fps = fps)
for img_file in tqdm(sorted(glob.glob(video_image_dir + '/*'))):
    fname = os.path.basename(img_file).split('.')[0]
    twohands_file = os.path.join(video_twohands_dir, fname + '.png')

    if args.mode == 'obj1':
        obj1_file = os.path.join(video_obj1_dir, fname + '.png')
        img = np.array(Image.open(img_file))
        twohands = np.array(Image.open(twohands_file))
        obj1 = np.array(Image.open(obj1_file))
        twohands_obj1 = twohands.copy()
        twohands_obj1[obj1 == 1] = 3
        twohands_obj1[obj1 == 2] = 4
        twohands_obj1[obj1 == 3] = 5
        twohands_obj1_vis = visualize_twohands_obj1(img, twohands_obj1)
        writer.append_data(twohands_obj1_vis)
    elif args.mode == 'obj2':
        obj2_file = os.path.join(video_obj2_dir, fname + '.png')
        img = np.array(Image.open(img_file))
        twohands = np.array(Image.open(twohands_file))
        obj2 = np.array(Image.open(obj2_file))
        twohands_obj2 = twohands.copy()
        twohands_obj2[obj2 == 1] = 3
        twohands_obj2[obj2 == 2] = 4
        twohands_obj2[obj2 == 3] = 5
        twohands_obj2[obj2 == 4] = 6
        twohands_obj2[obj2 == 5] = 7
        twohands_obj2[obj2 == 6] = 8
        twohands_obj2_vis = visualize_twohands_obj2(img, twohands_obj2)
        writer.append_data(twohands_obj2_vis)
    else:
        raise notimplementederror

writer.close()

# remove all folders
if args.remove_intermediate_images:
    os.system('rm -rf ' + video_dir)

