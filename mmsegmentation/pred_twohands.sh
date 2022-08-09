# inference 
python predict_image.py \
       --config_file ./work_dirs/seg_twohands_ccda/seg_twohands_ccda.py \
       --checkpoint_file ./work_dirs/seg_twohands_ccda/best_mIoU_iter_56000.pth \
       --img_dir ../testimages/images \
       --pred_seg_dir ../testimages/pred_twohands \

# visualize
python visualize.py \
       --mode twohands \
       --img_dir ../testimages/images \
       --twohands_dir ../testimages/pred_twohands \
       --vis_dir ../testimages/pred_twohands_vis

