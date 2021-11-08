
import sys
sys.path.append('/data/gaoyan/training_code/masktrack_rcnn_gy/MaskTrackRCNN')
import pdb
from mmdet.datasets import OSSYTVOSDataset
from ossutils import OSSHelper
OSSHelper.endpoint = 'http://cn-hangzhou.oss.aliyun-inc.com'
import os

oss_root = 'data/youtubeVIS/'
data_root = 'data/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

imgs_per_gpu=8,
workers_per_gpu=2,
train=dict(
        ann_file=os.path.join(oss_root,'train.json'),
        img_prefix=os.path.join(oss_root,'train/JPEGImages'),
        img_scale=(640, 360),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=True,
        with_crowd=True,
        with_label=True,
        with_track=True)
val=dict(
        ann_file=os.path.join(oss_root,'valid.json'),
        img_prefix=os.path.join(oss_root,'valid/JPEGImages'),
        img_scale=(640, 360),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=True,
        with_crowd=True,
        with_label=True,
        with_track=True)


if __name__ == '__main__':
    val_dataset =  OSSYTVOSDataset(**val,test_mode=True)
    data = val_dataset[0]
    import pdb
    pdb.set_trace()

    