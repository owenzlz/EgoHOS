from PIL import Image
import numpy as np 
import glob
import pdb
import os 
from skimage.io import imsave
import cv2
from tqdm import tqdm

alpha = 0.9
kernel = np.ones((5,5), np.uint8)

lbl_type_list = ['train', 'val', 'test_indomain', 'test_outdomain']

for lbl_type in lbl_type_list:

    all_img_dir = '../data/'+lbl_type+'/image'
    all_label_dir = '../data/'+lbl_type+'/label'

    vis_contact_boundary_dir = '../data/'+lbl_type+'/vis_contact'
    contact_boundary_dir = '../data/'+lbl_type+'/label_contact'
    os.makedirs(vis_contact_boundary_dir, exist_ok = True)
    os.makedirs(contact_boundary_dir, exist_ok = True)

    for file in tqdm(glob.glob(all_label_dir + '/*')):

        fname = os.path.basename(file)

        img = np.array(Image.open(os.path.join(all_img_dir, fname.split('.')[0]+'.jpg')))
        label = np.array(Image.open(file))

        hand_msk = np.zeros((label.shape))
        obj_msk = np.zeros((label.shape))

        hand_msk[label==1] = 1
        hand_msk[label==2] = 1
        obj_msk[label==3] = 1
        obj_msk[label==4] = 1
        obj_msk[label==5] = 1

        hand_msk_dialte = cv2.dilate(hand_msk, kernel, iterations=int(img.shape[1]/456))
        obj_msk_dialte = cv2.dilate(obj_msk, kernel, iterations=int(img.shape[1]/456))
        merge_hand_obj = hand_msk_dialte + obj_msk_dialte
        contact_boundary = np.zeros((hand_msk.shape))
        contact_boundary[merge_hand_obj == 2] = 1

        imsave(os.path.join(contact_boundary_dir, fname), contact_boundary.astype(np.uint8))

        '''
        red = np.zeros((img.shape)); red[:,:,0] = 255
        blue = np.zeros((img.shape)); blue[:,:,2] = 255
        yellow = np.zeros((img.shape)); yellow[:,:,0] = 255; yellow[:,:,1] = 255; yellow[:,:,2] = 0
        contact_boundary_rgb = np.repeat(np.expand_dims(contact_boundary, 2), 3, 2)
        img_contact_boundary = img * (1 - contact_boundary_rgb) + \
                            yellow * contact_boundary_rgb * alpha + \
                            img * contact_boundary_rgb * (1 - alpha)

        hand_msk_rgb = np.repeat(np.expand_dims(hand_msk, 2), 3, 2)
        obj_msk_rgb = np.repeat(np.expand_dims(obj_msk, 2), 3, 2)
        merge_hand_obj_rgb = np.repeat(np.expand_dims(merge_hand_obj, 2), 3, 2)
        vis_merge_hand_obj = red * hand_msk_rgb + blue * obj_msk_rgb
        # imsave('vis_merge_hand_obj.png', vis_merge_hand_obj)
        
        vis_merge_hand_obj = vis_merge_hand_obj * (1 - contact_boundary_rgb) + yellow * contact_boundary_rgb
        imsave(os.path.join(vis_contact_boundary_dir, fname.split('.')[0]+'_lbl.jpg'), vis_merge_hand_obj)
        imsave(os.path.join(vis_contact_boundary_dir, fname.split('.')[0]+'.jpg'), img_contact_boundary)

        # pdb.set_trace()
        '''

