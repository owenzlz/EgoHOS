# EgoHOS
[Project Page](https://www.seas.upenn.edu/~shzhou2/projects/eos_dataset/) |  [Paper] | [Bibtex]

<img src="https://github.com/owenzlz/EgoHOS/blob/main/stitch.gif" style="width:800px;">

**Fine-Grained Egocentric Hand-Object Segmentation: Dataset, Model, and Applications**\
*European Conference on Computer Vision (ECCV), 2022*\
[Lingzhi Zhang*](https://owenzlz.github.io/), [Shenghao Zhou*](https://scholar.google.com/citations?user=kWdwbUYAAAAJ&hl=en), [Simon Stent](https://scholar.google.com/citations?user=f3aij5UAAAAJ&hl=en), [Jianbo Shi](https://www.cis.upenn.edu/~jshi/) (* indicates equal contribution)


## Prerequisites
- Linux
- Python 3
- NVIDIA GPU + CUDA CuDNN

**Table of Contents:**<br>
1. [Setup](#setup) - download pretrained models and resources
2. [Datasets](#datasets) - download our egocentric hand-object segmentation datasets
3. [Quick Usage](#pretrained) - quickstart with pretrained models<br>


## Setup
- Clone this repo:
```bash
git clone https://github.com/owenzlz/EgoHOS
```

- Install dependencies:
```bash
pip install -U openmim
mim install mmcv-full
cd mmsegmentation
pip install -v -e .
```
For more information, please refer to MMSegmentation: https://mmsegmentation.readthedocs.io/en/latest/

## Datasets
- Download our dataset using the following command line.
```bash
bash download_datasets.sh
```

After downloading, the dataset is structured as follows: 
```bash
- [egohos dataset root]
    |- train
        |- image
        |- label
        |- contact
    |- val 
        |- image
        |- label
        |- contact
    |- test_indomain
        |- image
        |- label
        |- contact
    |- test_outdomain
        |- image
        |- label
        |- contact
```

In each label image, the category ids are referred as below. In the contact labels, 'ones' indicate the dense contact region.  
```bash
0 -> background
1 -> left hand
2 -> right hand
3 -> 1st order interacting object by left hand
4 -> 1st order interacting object by right hand
5 -> 1st order interacting object by both hands
6 -> 2nd order interacting object by left hand
7 -> 2nd order interacting object by right hand
8 -> 2nd order interacting object by both hands
```

## Checkpoints

- Download resources:
	- we provide a script for downloading the pretrained checkpoints. 
```bash
bash download_checkpoints.sh
```

- Download test images:
	- we provide a script for downloading a few test images. 
```bash
bash download_testimages.sh
```

## Quick Inference on Images

Depending on the application scenarios, you may want to use one of these commands to generate the segmentation predictions. The backen segmentation model is Swin-L backbone and UPerNet head. 

- Predict left and right hands. Required input: RGB image
```bash
bash ...
```

- Predict dense contact boundary between and hands and interacting objects. Required input: RGB image, hand masks
```bash
bash ...
```

- Predict hands and (1st order) interacting objects. Required input: RGB image, hand masks, contact boundary
```bash
bash ...
```

- Predict hands and (both 1st and 2nd orders) interacting objects. Required input: RGB image, hand masks, contact boundary
```bash
bash ...
```

## Quick Inference on Videos

- Predict hands and (1st order) interacting objects. Required input: RGB video
```bash
bash ...
```




