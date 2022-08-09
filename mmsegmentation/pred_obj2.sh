# inference 
python predict_image.py \
       --config_file /mnt/session_space/home/lingzzha/EgoHOS/mmsegmentation/work_dirs/twohands_cb_to_obj2_ccda/twohands_cb_to_obj2_ccda.py \
       --checkpoint_file /mnt/session_space/home/lingzzha/EgoHOS/mmsegmentation/work_dirs/twohands_cb_to_obj2_ccda/best_mIoU_iter_32000.pth \
       --img_dir /mnt/session_space/home/lingzzha/EgoHOS/testimages/images \
       --pred_seg_dir /mnt/session_space/home/lingzzha/EgoHOS/testimages/pred_obj2

# visualize
python visualize.py \
       --mode twohands_obj2 \
       --img_dir /mnt/session_space/home/lingzzha/EgoHOS/testimages/images \
       --twohands_dir /mnt/session_space/home/lingzzha/EgoHOS/testimages/pred_twohands \
       --obj1_dir /mnt/session_space/home/lingzzha/EgoHOS/testimages/pred_obj2 \
       --vis_dir /mnt/session_space/home/lingzzha/EgoHOS/testimages/pred_obj2_vis
