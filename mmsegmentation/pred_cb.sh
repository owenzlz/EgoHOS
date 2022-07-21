python predict.py \
       --config_file /mnt/session_space/home/lingzzha/EgoHOS/mmsegmentation/work_dirs/twohands_to_cb/twohands_to_cb.py \
       --checkpoint_file /mnt/session_space/home/lingzzha/EgoHOS/mmsegmentation/work_dirs/twohands_to_cb/best_mIoU_iter_26000.pth \
       --img_dir /mnt/session_space/home/lingzzha/EgoHOS/data/train/image \
       --pred_seg_dir /mnt/session_space/home/lingzzha/EgoHOS/data/train/pred_cb

python predict.py \
       --config_file /mnt/session_space/home/lingzzha/EgoHOS/mmsegmentation/work_dirs/twohands_to_cb/twohands_to_cb.py \
       --checkpoint_file /mnt/session_space/home/lingzzha/EgoHOS/mmsegmentation/work_dirs/twohands_to_cb/best_mIoU_iter_26000.pth \
       --img_dir /mnt/session_space/home/lingzzha/EgoHOS/data/val/image \
       --pred_seg_dir /mnt/session_space/home/lingzzha/EgoHOS/data/val/pred_cb

python predict.py \
       --config_file /mnt/session_space/home/lingzzha/EgoHOS/mmsegmentation/work_dirs/twohands_to_cb/twohands_to_cb.py \
       --checkpoint_file /mnt/session_space/home/lingzzha/EgoHOS/mmsegmentation/work_dirs/twohands_to_cb/best_mIoU_iter_26000.pth \
       --img_dir /mnt/session_space/home/lingzzha/EgoHOS/data/test_indomain/image \
       --pred_seg_dir /mnt/session_space/home/lingzzha/EgoHOS/data/test_indomain/pred_cb

python predict.py \
       --config_file /mnt/session_space/home/lingzzha/EgoHOS/mmsegmentation/work_dirs/twohands_to_cb/twohands_to_cb.py \
       --checkpoint_file /mnt/session_space/home/lingzzha/EgoHOS/mmsegmentation/work_dirs/twohands_to_cb/best_mIoU_iter_26000.pth \
       --img_dir /mnt/session_space/home/lingzzha/EgoHOS/data/test_outdomain/image \
       --pred_seg_dir /mnt/session_space/home/lingzzha/EgoHOS/data/test_outdomain/pred_cb

