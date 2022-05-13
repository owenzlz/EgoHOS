# CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh ./configs/pspnet/pspnet_r50-d8_480x360_80k_egohos_allhands.py 4

# CUDA_VISIBLE_DEVICES=4,5,6,7 ./tools/dist_train.sh ./configs/pspnet/pspnet_r50-d8_480x360_80k_egohos_allhands.py 4



# ./tools/dist_train.sh ./configs/pspnet/pspnet_r50-d8_480x360_80k_egohos_allhands.py 8

# python tools/train.py ./configs/pspnet/pspnet_r50-d8_480x360_80k_egohos_allhands.py


# sleep 2h && ./tools/dist_train.sh ./configs/pspnet/pspnet_r50-d8_480x360_80k_egohos_handobj2_reducelabel.py 8

# ./tools/dist_train.sh ./configs/swin/upernet_swin_base_patch4_window12_512x512_160k_egohos_handobj2_pretrain_480x360_22K.py 8



./tools/dist_train.sh ./configs/segformer/segformer_mit-b5_480x360_160k_egohos_handobj2.py 8




