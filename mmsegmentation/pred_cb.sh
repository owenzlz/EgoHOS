# inference 
python predict.py \
       --config_file /mnt/session_space/home/lingzzha/EgoHOS/mmsegmentation/work_dirs/twohands_to_cb_ccda/twohands_to_cb_ccda.py \
       --checkpoint_file /mnt/session_space/home/lingzzha/EgoHOS/mmsegmentation/work_dirs/twohands_to_cb_ccda/best_mIoU_iter_76000.pth \
       --img_dir /mnt/session_space/home/lingzzha/EgoHOS/testimages/images \
       --pred_seg_dir /mnt/session_space/home/lingzzha/EgoHOS/testimages/pred_cb

# visualize
python visualize.py \
       --mode cb \
       --img_dir /mnt/session_space/home/lingzzha/EgoHOS/testimages/images \
       --cb_dir /mnt/session_space/home/lingzzha/EgoHOS/testimages/pred_cb \
       --vis_dir /mnt/session_space/home/lingzzha/EgoHOS/testimages/pred_cb_vis

 