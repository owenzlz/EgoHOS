# python test.py --img_dir ../data/val/image \
#                --vis_dir ./work_dirs/upernet_swin_base_patch4_window12_512x512_160k_egohos_handobj2_pretrain_480x360_22K/outputs/val \
#                --seg_dir ./work_dirs/upernet_swin_base_patch4_window12_512x512_160k_egohos_handobj2_pretrain_480x360_22K/outputs/val_seg

# python test.py --img_dir ../data/test_indomain/image \
#                --vis_dir ./work_dirs/upernet_swin_base_patch4_window12_512x512_160k_egohos_handobj2_pretrain_480x360_22K/outputs/test_indomain \
#                --seg_dir ./work_dirs/upernet_swin_base_patch4_window12_512x512_160k_egohos_handobj2_pretrain_480x360_22K/outputs/test_indomain_seg

# python test.py --img_dir ../data/test_outdomain/image \
#                --vis_dir ./work_dirs/upernet_swin_base_patch4_window12_512x512_160k_egohos_handobj2_pretrain_480x360_22K/outputs/test_outdomain \
#                --seg_dir ./work_dirs/upernet_swin_base_patch4_window12_512x512_160k_egohos_handobj2_pretrain_480x360_22K/outputs/test_outdomain_seg


python test.py --img_dir ../data/val/image \
               --checkpoint_file ./work_dirs/upernet_swin_base_patch4_window12_512x512_160k_egohos_handobj2_subset_pretrain_480x360_22K/best_mIoU_iter_4000.pth \
               --config_file ./work_dirs/upernet_swin_base_patch4_window12_512x512_160k_egohos_handobj2_subset_pretrain_480x360_22K/upernet_swin_base_patch4_window12_512x512_160k_egohos_handobj2_subset_pretrain_480x360_22K.py \
               --vis_dir ./work_dirs/upernet_swin_base_patch4_window12_512x512_160k_egohos_handobj2_subset_pretrain_480x360_22K/outputs/val \
               --seg_dir ./work_dirs/upernet_swin_base_patch4_window12_512x512_160k_egohos_handobj2_subset_pretrain_480x360_22K/outputs/val_seg

