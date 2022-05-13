_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/egohos_allhands.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(
    decode_head=dict(num_classes=8), auxiliary_head=dict(num_classes=8))

checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=2000, metric=['mIoU', 'mFscore'], pre_eval=True, save_best='mIoU')

train_pipeline = [dict(type='LoadAnnotations', reduce_zero_label=True)]



