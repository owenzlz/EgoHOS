_base_ = [
    '../_base_/models/fcn_hr18.py', '../_base_/datasets/egohos_handobj2.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=dict(
        num_classes=9, in_channels=[48, 96, 192, 384], channels=sum([48, 96, 192, 384])))

checkpoint_config = dict(by_epoch=False, interval=5000)
evaluation = dict(interval=2000, metric=['mIoU', 'mFscore'], pre_eval=True, save_best='mIoU')




