# inference 
python predict_image.py \
       --config_file ./work_dirs/twohands_cb_to_obj1_ccda/twohands_cb_to_obj1_ccda.py \
       --checkpoint_file ./work_dirs/twohands_cb_to_obj1_ccda/best_mIoU_iter_34000.pth \
       --img_dir ../testimages/images \
       --pred_seg_dir ../testimages/pred_obj1

# visualize
python visualize.py \
       --mode twohands_obj1 \
       --img_dir ../testimages/images \
       --twohands_dir ../testimages/pred_twohands \
       --obj1_dir ../testimages/pred_obj1 \
       --vis_dir ../testimages/pred_obj1_vis
