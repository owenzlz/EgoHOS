# inference 
python predict.py \
       --config_file /mnt/session_space/home/lingzzha/EgoHOS/mmsegmentation/work_dirs/twohands_cb_to_obj1_ccda/twohands_cb_to_obj1_ccda.py \
       --checkpoint_file /mnt/session_space/home/lingzzha/EgoHOS/mmsegmentation/work_dirs/twohands_cb_to_obj1_ccda/best_mIoU_iter_34000.pth \
       --img_dir /mnt/session_space/home/lingzzha/EgoHOS/testimages/images \
       --pred_seg_dir /mnt/session_space/home/lingzzha/EgoHOS/testimages/pred_obj1

# visualize
python visualize.py \
       --mode twohands_obj1 \
       --img_dir /mnt/session_space/home/lingzzha/EgoHOS/testimages/images \
       --twohands_dir /mnt/session_space/home/lingzzha/EgoHOS/testimages/pred_twohands \
       --obj1_dir /mnt/session_space/home/lingzzha/EgoHOS/testimages/pred_obj1 \
       --vis_dir /mnt/session_space/home/lingzzha/EgoHOS/testimages/pred_obj1_vis
