# CMaskTrack R-CNN for OVIS

This repo serves as the official code release of the CMaskTrack R-CNN model on the [**Occluded Video Instance Segmentation**](http://songbai.site/ovis/) dataset described in the tech report:

## [Occluded Video Instance Segmentation](https://arxiv.org/abs/2102.01558)

>Jiyang Qi<sup>1,2</sup>\*, Yan Gao<sup>2</sup>\*, [Yao Hu](https://scholar.google.com/citations?user=LIu7k7wAAAAJ)<sup>2</sup>, [Xinggang Wang](https://xinggangw.info/index_cn.htm)<sup>1</sup>, Xiaoyu Liu<sup>2</sup>,  
>[Xiang Bai](http://122.205.5.5:8071/~xbai/)<sup>1</sup>, [Serge Belongie](s.belongie@di.ku.dk)<sup>3</sup>, [Alan Yuille](http://www.cs.jhu.edu/~ayuille/)<sup>4</sup>, [Philip Torr](http://www.robots.ox.ac.uk/~phst/)<sup>5</sup>, [Song Bai](http://songbai.site)<sup>2,5 :email:</sup>  
><sup>1</sup>Huazhong University of Science and Technology  <sup>2</sup>Alibaba Group  <sup>3</sup>University of Copenhagen  
><sup>4</sup>Johns Hopkins University  <sup>5</sup>University of Oxford

In this work, we collect a large-scale dataset called **OVIS** for **O**ccluded **V**ideo **I**nstance **S**egmentation. OVIS consists of 296k high-quality instance masks from 25 semantic categories, where object occlusions usually occur. While our human vision systems can understand those occluded instances by contextual reasoning and association, our experiments suggest that current video understanding systems cannot, which reveals that we are still at a nascent stage for understanding objects, instances, and videos in a real-world scenario. 

We also present a simple plug-and-play module that performs temporal feature calibration to complement missing object cues caused by occlusion.

Some annotation examples can be seen below:

<table style="display:flex;justify-content:center;border:0" rules=none frame=void >
<tr>
<td><img src="http://songbai.site/ovis/data/webp/2592056.webp" alt="2592056" width="160" height="90" />
</td>
<td><img src="http://songbai.site/ovis/data/webp/2930398.webp" alt="2930398" width="160" height="90">
</td>
<td><img src="http://songbai.site/ovis/data/webp/2932104.webp" alt="2932104" width="160" height="90">
</td>
<td><img src="http://songbai.site/ovis/data/webp/3021160.webp" alt="3021160" width="160" height="90">
</td>
</tr>
<tr>
<td><img src="http://songbai.site/ovis/data/webp/2524877_0_170.webp" width="160" height="90" />
</td>
<td><img src="http://songbai.site/ovis/data/webp/2591274.webp" width="160" height="90">
</td>
<td><img src="http://songbai.site/ovis/data/webp/2592058.webp" width="160" height="90">
</td>
<td><img src="http://songbai.site/ovis/data/webp/2592138.webp" width="160" height="90">
</td>
</tr>
<tr>
<td><img src="http://songbai.site/ovis/data/webp/2932109.webp" width="160" height="90" />
</td>
<td><img src="http://songbai.site/ovis/data/webp/2932131.webp" width="160" height="90">
</td>
<td><img src="http://songbai.site/ovis/data/webp/2932134.webp" width="160" height="90">
</td>
<td><img src="http://songbai.site/ovis/data/webp/3163218.webp" width="160" height="90">
</td>
</tr>
<tr>
<td><img src="http://songbai.site/ovis/data/webp/3383476.webp" width="160" height="90" />
</td>
<td><img src="http://songbai.site/ovis/data/webp/3441792.webp" width="160" height="90">
</td>
<td><img src="http://songbai.site/ovis/data/webp/3441794.webp" width="160" height="90">
</td>
<td><img src="http://songbai.site/ovis/data/webp/3441797.webp" width="160" height="90">
</td>
</tr>
</table>
<center><i>Visualization of the annotations.</i></center>

For more details about the dataset, please refer to our [paper](https://arxiv.org/abs/2102.01558) or [website](http://songbai.site/ovis/).

## Model training and evaluation

### Installation

This repo is built based on [MaskTrackRCNN](https://github.com/youtubevos/MaskTrackRCNN).
You also need to install a customized [COCO API](https://github.com/youtubevos/cocoapi) for the OVIS dataset.
You can use following commands to create conda env with all dependencies.

```
conda create -n cmtrcnn python=3.6 -y
conda activate cmtrcnn

conda install -c pytorch pytorch=1.3.1 torchvision=0.2.2 cudatoolkit=10.1 -y
pip install -r requirements.txt
pip install git+https://github.com/qjy981010/cocoapi.git#"egg=pycocotools&subdirectory=PythonAPI"

bash compile.sh
```

### Data preparation
1. Download OVIS from [our website](http://songbai.site/ovis/).
2. Symlink the train/validation dataset to `data/OVIS/` folder. Put COCO-style annotations under `data/annotations`.

```
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── OVIS
│   │   ├── train_images
│   │   ├── valid_images
│   │   ├── annotations
│   │   │   ├── annotations_train.json
│   │   │   ├── annotations_valid.json
```

### Training

Our model is based on MaskRCNN-resnet50-FPN. The model is trained end-to-end on OVIS based on a MSCOCO pretrained checkpoint ([mmlab link](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/mask_rcnn_r50_fpn_1x_20181010-069fa190.pth) or [google drive](https://drive.google.com/file/d/1pPjjKrG9VDEyzZJt6psCiPVj5wL9w1_I/view?usp=sharing)).

Run the command below to train the model.
```CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py configs/cmasktrack_rcnn_r50_fpn_1x_ovis.py --work_dir ./workdir/cmasktrack_rcnn_r50_fpn_1x_ovis --gpus 4```
For reference to arguments such as learning rate and model parameters, please refer to `configs/cmasktrack_rcnn_r50_fpn_1x_ovis.py`.

### Evaluation

Our pretrained model is available for download at [Google Drive (comming soon)]().
Run the following command to evaluate the model on YouTubeVIS.
```
CUDA_VISIBLE_DEVICES=0 python test_video.py configs/cmasktrack_rcnn_r50_fpn_1x_ovis.py [MODEL_PATH] --out [OUTPUT_PATH.pkl] --eval segm
```
A json file containing the predicted result will be generated as `OUTPUT_PATH.pkl.json`. OVIS currently only allows evaluation on the codalab server. Please upload the generated result to [codalab server](https://competitions.codalab.org/competitions/32377) to see actual performances.

## License
This project is released under the [Apache 2.0 license](LICENSE), while the correlation ops is under [MIT license](mmdet/ops/correlation/LICENSE).

## Acknowledgement

This project is based on [mmdetection (commit hash f3a939f)](https://github.com/open-mmlab/mmdetection/tree/f3a939fa697ce23d8a6435b59529791002f64fdf), [mmcv](https://github.com/open-mmlab/mmcv), [MaskTrackRCNN](https://github.com/youtubevos/MaskTrackRCNN) and [Pytorch-Correlation-extension](https://github.com/ClementPinard/Pytorch-Correlation-extension). Thanks for their wonderful works.

## Citation
If you find our paper and code useful in your research, please consider giving a star ⭐ and citation 📝 :

```
@article{qi2021occluded,
    title={Occluded Video Instance Segmentation},
    author={Jiyang Qi and Yan Gao and Yao Hu and Xinggang Wang and Xiaoyu Liu and Xiang Bai and Serge Belongie and Alan Yuille and Philip Torr and Song Bai},
    journal={arXiv preprint arXiv:2102.01558},
    year={2021},
}
```
