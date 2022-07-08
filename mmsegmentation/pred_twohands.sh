python predict.py \
       --config_file /mnt/session_space/home/lingzzha/EgoHOS/mmsegmentation/work_dirs/seg_twohands/upernet_swin_base_patch4_window12_512x512_160k_egohos_twohands_pretrain_480x360_22K.py \
       --checkpoint_file /mnt/session_space/home/lingzzha/EgoHOS/mmsegmentation/work_dirs/seg_twohands/best_mIoU_iter_84000.pth \
       --img_dir /mnt/session_space/home/lingzzha/EgoHOS/data/train/image \
       --pred_seg_dir /mnt/session_space/home/lingzzha/EgoHOS/data/train/pred_twohands \

python predict.py \
       --config_file /mnt/session_space/home/lingzzha/EgoHOS/mmsegmentation/work_dirs/seg_twohands/upernet_swin_base_patch4_window12_512x512_160k_egohos_twohands_pretrain_480x360_22K.py \
       --checkpoint_file /mnt/session_space/home/lingzzha/EgoHOS/mmsegmentation/work_dirs/seg_twohands/best_mIoU_iter_84000.pth \
       --img_dir /mnt/session_space/home/lingzzha/EgoHOS/data/val/image \
       --pred_seg_dir /mnt/session_space/home/lingzzha/EgoHOS/data/val/pred_twohands \

python predict.py \
       --config_file /mnt/session_space/home/lingzzha/EgoHOS/mmsegmentation/work_dirs/seg_twohands/upernet_swin_base_patch4_window12_512x512_160k_egohos_twohands_pretrain_480x360_22K.py \
       --checkpoint_file /mnt/session_space/home/lingzzha/EgoHOS/mmsegmentation/work_dirs/seg_twohands/best_mIoU_iter_84000.pth \
       --img_dir /mnt/session_space/home/lingzzha/EgoHOS/data/test_indomain/image \
       --pred_seg_dir /mnt/session_space/home/lingzzha/EgoHOS/data/test_indomain/pred_twohands \

python predict.py \
       --config_file /mnt/session_space/home/lingzzha/EgoHOS/mmsegmentation/work_dirs/seg_twohands/upernet_swin_base_patch4_window12_512x512_160k_egohos_twohands_pretrain_480x360_22K.py \
       --checkpoint_file /mnt/session_space/home/lingzzha/EgoHOS/mmsegmentation/work_dirs/seg_twohands/best_mIoU_iter_84000.pth \
       --img_dir /mnt/session_space/home/lingzzha/EgoHOS/data/test_outdomain/image \
       --pred_seg_dir /mnt/session_space/home/lingzzha/EgoHOS/data/test_outdomain/pred_twohands \

