from pathlib import Path
from tkinter import RIGHT
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import shutil
import cv2
import tqdm

LEFT_HAND = 61
RIGHT_HAND = 71
LEFT_OBJ1 = 1
LEFT_OBJ2 = 11
RIGHT_OBJ1 = 21
RIGHT_OBJ2 = 31
TWO_OBJ1 = 41
TWO_OBJ2 = 51 # FIXME, if it is 31, it is overlapping with RIGHT_OBJ2
# all_classes = [LEFT_HAND, RIGHT_HAND, LEFT_OBJ1, LEFT_OBJ2, RIGHT_OBJ1, RIGHT_OBJ2, TWO_OBJ1, TWO_OBJ2]
# all_classes = [LEFT_HAND, RIGHT_HAND, LEFT_OBJ1, RIGHT_OBJ1, TWO_OBJ1]
all_classes = [LEFT_HAND, RIGHT_HAND, LEFT_OBJ1, RIGHT_OBJ1, TWO_OBJ1]

def load_ann(ann_path):
    ann = Image.open(ann_path).convert('P')
    return np.array(ann)

def load_img(img_path):
    img = Image.open(img_path)
    return np.array(img)

def split_data(all_img_lst, train_frac, test_frac):
    """Given the list of all images, randomly sample a portion of index as training, 
    validation and test set. The weight is specified as train_frac, val_frac, test_frac
    """
    leng = len(all_img_lst)
    permuted_img = np.array(all_img_lst)[np.random.permutation(leng)]
    train_idx = int(leng * train_frac)
    test_idx = int(leng * test_frac)
    return permuted_img[:train_idx], permuted_img[train_idx + test_idx:], permuted_img[train_idx:train_idx + test_idx]

def convert_ann(ann):
    new_ann = np.zeros_like(ann)
    new_ann[ann > 0] = 255 # for all objects
    for i, cls in enumerate(all_classes):
        new_ann[ann == cls] = i + 1 # (cls // 10) + 1
    # new_ann[new_ann == 255] = len(all_classes) + 1 # for all objects
    new_ann[new_ann == 255] = 0 # for all objects
    return new_ann

def cv2_imsave(path, img):
    if img.shape[-1] == 3:
        img = img[...,::-1]
    cv2.imwrite(str(path), img)

def convert_and_save(vis_paths, save_path):
    save_img_path = save_path.joinpath('img')
    save_img_path.mkdir(exist_ok=True, parents=True)
    save_anno_path = save_path.joinpath('anno')
    save_anno_path.mkdir(exist_ok=True)
    for vis_path in tqdm.tqdm(vis_paths):
        vis_path = str(vis_path)
        img_path = vis_path.replace('_draw', '')
        ann_path = img_path.replace('jpg', 'png')
        img = load_img(img_path)
        ann = load_ann(ann_path)
        new_ann = convert_ann(ann)
        cv2_imsave(str(save_anno_path.joinpath(Path(ann_path).name)), new_ann)
        cv2_imsave(str(save_img_path.joinpath(Path(img_path).name)), img)



if __name__ == "__main__":

    vis_paths = list(Path('all_raw').glob('*_draw.jpg'))
    train_paths, val_paths, test_paths = split_data(vis_paths, 0.8, 0.1)
    save_path = Path('/mnt/volume2/Data/zlz/EgoHandObject/data/epic/obj1_hand_2k')
    convert_and_save(train_paths, save_path.joinpath('train'))
    convert_and_save(val_paths, save_path.joinpath('val'))
    convert_and_save(test_paths, save_path.joinpath('test'))

