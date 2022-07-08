from PIL import Image
import numpy as np 
from skimage.io import imsave
import glob
import pdb
import os 
import torchvision
import torch
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms
from tqdm import tqdm
from shutil import copyfile

feat_extractor = torchvision.models.vgg16(pretrained=True).features.cuda()

# pdb.set_trace()

tsfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

lama_dir = '/mnt/session_space/home/lingzzha/EgoHOS/data/train/lama_512'
lama_feat_dir = '/mnt/session_space/home/lingzzha/EgoHOS/data/train/lama_512_feature'
os.makedirs(lama_feat_dir, exist_ok = True)

for file in tqdm(glob.glob(lama_dir + '/*')):
    fname = os.path.basename(file).split('.')[0]
    clean_back = tsfm(Image.open(file).convert('RGB').resize((480, 360))).cuda()
    clean_back_feat = feat_extractor(clean_back.unsqueeze(0)).reshape(-1)
    clean_back_feat_np = clean_back_feat.cpu().data.numpy()
    np.save(os.path.join(lama_feat_dir, fname + '.npy'), clean_back_feat_np)




