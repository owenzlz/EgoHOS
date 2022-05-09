"""
    This script checks unique values in label
"""
from PIL import Image
import numpy as np 
import glob
import pdb
import os 



label_dir = '../data/train/label'

for file in glob.glob(label_dir + '/*.png'):
    lbl = np.array(Image.open(file))
    unique_vals = np.unique(lbl)
    print(unique_vals)
    pdb.set_trace()

