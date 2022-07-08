from PIL import Image
import glob
import pdb
import os 
from tqdm import tqdm

img_dir = '/mnt/session_space/home/lingzzha/EgoHOS/data/train/lama_512'

for file in tqdm(glob.glob(img_dir + '/*.png')):
    fname = os.path.basename(file).split('.')[0]
    img = Image.open(file)
    img.save(os.path.join(img_dir, fname + '.jpg'))
    os.remove(file)

