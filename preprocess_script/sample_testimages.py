from shutil import copyfile
from tqdm import tqdm
import random
import glob
import pdb
import os 

indomain_test_dir = '../data/test_indomain/image'
outdomain_test_dir = '../data/test_outdomain/image'
sample_test_dir = '../testimages'
os.makedirs(sample_test_dir, exist_ok = True)

indomain_test_files = glob.glob(indomain_test_dir + '/*')
outdomain_test_files = glob.glob(outdomain_test_dir + '/*')

sample_test_files = random.sample(indomain_test_files, 10) + random.sample(outdomain_test_files, 10)

for sample_test_file in tqdm(sample_test_files):
    fname = os.path.basename(sample_test_file)
    dst_file = os.path.join(sample_test_dir, fname)
    copyfile(sample_test_file, dst_file)

    # pdb.set_trace()

