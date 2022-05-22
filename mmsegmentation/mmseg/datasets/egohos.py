# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class EgoHOSDataset(CustomDataset):
    """EgoHOS dataset.

    Args:
        split (str): Split txt file for EgoHOS.
    """
    
    # CLASSES = ('background', 'aeroplane')

    # PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
    #            [128, 0, 128]]

    CLASSES = ('background', 'Left_Hand', 'Right_Hand', \
               'Left_Object1', 'Right_Object1', 'Two_Object1', \
               'Left_Object2', 'Right_Object2', 'Two_Object2')

    PALETTE = [[0, 0, 0], [255, 0, 0], [0, 0, 255], \
               [255, 0, 255], [0, 255, 255], [0, 255, 0], \
               [255, 204, 255], [204, 255, 255], [204, 255, 204]]

    def __init__(self, **kwargs):
        super(EgoHOSDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', **kwargs)



