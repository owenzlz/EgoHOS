python predict.py \
       --config_file /mnt/session_space/home/lingzzha/EgoHOS/mmsegmentation/work_dirs/seg_twohands_ccda/seg_twohands_ccda.py \
       --checkpoint_file /mnt/session_space/home/lingzzha/EgoHOS/mmsegmentation/work_dirs/seg_twohands_ccda/best_mIoU_iter_56000.pth \
       --img_dir /mnt/session_space/home/lingzzha/EgoHOS/data/train/image \
       --pred_seg_dir /mnt/session_space/home/lingzzha/EgoHOS/data/train/pred_twohands_ccda \

python predict.py \
       --config_file /mnt/session_space/home/lingzzha/EgoHOS/mmsegmentation/work_dirs/seg_twohands_ccda/seg_twohands_ccda.py \
       --checkpoint_file /mnt/session_space/home/lingzzha/EgoHOS/mmsegmentation/work_dirs/seg_twohands_ccda/best_mIoU_iter_56000.pth \
       --img_dir /mnt/session_space/home/lingzzha/EgoHOS/data/val/image \
       --pred_seg_dir /mnt/session_space/home/lingzzha/EgoHOS/data/val/pred_twohands_ccda \

python predict.py \
       --config_file /mnt/session_space/home/lingzzha/EgoHOS/mmsegmentation/work_dirs/seg_twohands_ccda/seg_twohands_ccda.py \
       --checkpoint_file /mnt/session_space/home/lingzzha/EgoHOS/mmsegmentation/work_dirs/seg_twohands_ccda/best_mIoU_iter_56000.pth \
       --img_dir /mnt/session_space/home/lingzzha/EgoHOS/data/test_indomain/image \
       --pred_seg_dir /mnt/session_space/home/lingzzha/EgoHOS/data/test_indomain/pred_twohands_ccda \

python predict.py \
       --config_file /mnt/session_space/home/lingzzha/EgoHOS/mmsegmentation/work_dirs/seg_twohands_ccda/seg_twohands_ccda.py \
       --checkpoint_file /mnt/session_space/home/lingzzha/EgoHOS/mmsegmentation/work_dirs/seg_twohands_ccda/best_mIoU_iter_56000.pth \
       --img_dir /mnt/session_space/home/lingzzha/EgoHOS/data/test_outdomain/image \
       --pred_seg_dir /mnt/session_space/home/lingzzha/EgoHOS/data/test_outdomain/pred_twohands_ccda \

