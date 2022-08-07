####### Basic Experiments #######



### Parallel Model ###

# Test original
# ./tools/dist_train.sh ./configs/swin/seg_handobj2.py 8 --work-dir ./work_dirs/seg_handobj2_new

# Test train on 512 resolution
# ./tools/dist_train.sh ./configs/swin/seg_handobj2_512.py 8


# rgb -> handobj2
# python tools/train.py ./configs/swin/seg_handobj2_ccda.py # single gpu test
# ./tools/dist_train.sh ./configs/swin/seg_handobj2_ccda.py 8


# rgb -> handobj1
# python tools/train.py ./configs/swin/seg_handobj1.py # single gpu test
# ./tools/dist_train.sh ./configs/swin/seg_handobj1.py 8


# rgb -> handobj1 (ccda)
# python tools/train.py ./configs/swin/seg_handobj1_ccda.py # single gpu test
# ./tools/dist_train.sh ./configs/swin/seg_handobj1_ccda.py 8


# rgb -> twohands
# python tools/train.py ./configs/swin/seg_twohands_ccda.py # single gpu test
# ./tools/dist_train.sh ./configs/swin/seg_twohands.py 8
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
# ./tools/dist_train.sh ./configs/swin/twohands_to_cb_ccda.py 8

# rgb + twohands + cb -> obj2                                                     [*]
# python tools/train.py ./configs/swin/twohands_cb_to_obj2.py # single gpu test

# rgb + twohands + cb -> obj2 (CCDA)                                              [*]
# python tools/train.py ./configs/swin/twohands_cb_to_obj2_ccda.py # single gpu test
./tools/dist_train.sh ./configs/swin/twohands_cb_to_obj2_ccda.py 8

# rgb + twohands + cb -> obj1                                                     [TODO]
# python tools/train.py ./configs/swin/twohands_cb_to_obj1.py # single gpu test
# ./tools/dist_train.sh ./configs/swin/twohands_cb_to_obj1.py 8

# rgb + twohands -> obj2 (CCDA)                                                   [*]
# python tools/train.py ./configs/swin/twohands_to_obj2_ccda.py # single gpu test
# ./tools/dist_train.sh ./configs/swin/twohands_to_obj2_ccda.py 8

# rgb + twohands -> obj1 (CCDA)                                                   [RUNING]
# python tools/train.py ./configs/swin/twohands_to_obj1_ccda.py # single gpu test
# ./tools/dist_train.sh ./configs/swin/twohands_cb_to_obj1_ccda.py 8

# rgb + twohands -> cb (CCDA)                                                     [RUNING]
# python tools/train.py ./configs/swin/twohands_to_cb_ccda.py # single gpu test
# ./tools/dist_train.sh ./configs/swin/twohands_to_cb_ccda.py 8

# rgb + twohands + cb -> obj2 (CCDA)                                              [*]


# rgb + twohands + cb -> obj1 (CCDA)                                              [TODO]





