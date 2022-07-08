####### Basic Experiments #######



### Parallel Model ###

# Test original
# ./tools/dist_train.sh ./configs/swin/seg_handobj2.py 8 --work-dir ./work_dirs/seg_handobj2_new

# Test train on 512 resolution
# ./tools/dist_train.sh ./configs/swin/seg_handobj2_512.py 8


# rgb -> handobj2
# python tools/train.py ./configs/swin/seg_handobj2_ccda.py # single gpu test
./tools/dist_train.sh ./configs/swin/seg_handobj2_ccda.py 8

# rgb -> handobj1
# python tools/train.py ./configs/swin/seg_handobj1_ccda.py # single gpu test
# ./tools/dist_train.sh ./configs/swin/seg_handobj1_ccda.py 8

# rgb -> twohands
# python tools/train.py ./configs/swin/seg_twohands_ccda.py # single gpu test
# ./tools/dist_train.sh ./configs/swin/seg_twohands_ccda.py 8



### Sequential Model ###

# rgb -> twohands [Finished]


# rgb + twohands -> obj2                                                          [Done]
# python tools/train.py ./configs/swin/twohands_to_obj2.py # single gpu test
# ./tools/dist_train.sh ./configs/swin/twohands_to_obj2.py 8

# rgb + twohands -> obj1                                                          [Done]
# python tools/train.py ./configs/swin/twohands_to_obj1.py # single gpu test
# ./tools/dist_train.sh ./configs/swin/twohands_to_obj1.py 8

# rgb + twohands -> cb                                                            [Done]
# python tools/train.py ./configs/swin/twohands_to_cb.py # single gpu test
# ./tools/dist_train.sh ./configs/swin/twohands_to_cb.py 8

# rgb + twohands + cb -> obj2



# rgb + twohands + cb -> obj1



# rgb + twohands -> obj2 (CCDA)                                                   [TODO]



# rgb + twohands -> obj1 (CCDA)                                                   [TODO]



# rgb + twohands -> cb (CCDA)                                                     [TODO]



# rgb + twohands + cb -> obj2 (CCDA)



# rgb + twohands + cb -> obj1 (CCDA)





