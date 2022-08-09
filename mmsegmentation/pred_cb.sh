# inference 
python predict_image.py \
       --config_file ./work_dirs/twohands_to_cb_ccda/twohands_to_cb_ccda.py \
       --checkpoint_file ./work_dirs/twohands_to_cb_ccda/best_mIoU_iter_76000.pth \
       --img_dir ../testimages/images \
       --pred_seg_dir ../testimages/pred_cb

# visualize
python visualize.py \
       --mode cb \
       --img_dir ../testimages/images \
       --cb_dir ../testimages/pred_cb \
       --vis_dir ../testimages/pred_cb_vis

 