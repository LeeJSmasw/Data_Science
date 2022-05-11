esNeSt
Split-Attention Network, A New ResNet Variant. It significantly boosts the performance of downstream models such as Mask R-CNN, Cascade R-CNN and DeepLabV3.



Table of Contents
Pretrained Models
Transfer Learning Models
Verify ImageNet Results
How to Train
Reference
Pypi / GitHub Install
Install this package repo, note that you only need to choose one of the options
# using github url
pip install git+https://github.com/zhanghang1989/ResNeSt

# using pypi
pip install resnest --pre
Pretrained Models
crop size	PyTorch	Gluon
ResNeSt-50	224	81.03	81.04
ResNeSt-101	256	82.83	82.81
ResNeSt-200	320	83.84	83.88
ResNeSt-269	416	84.54	84.53
3rd party implementations are available: Tensorflow, Caffe, JAX.

Extra ablation study models are available in link

PyTorch Models
Load using Torch Hub
import torch
# get list of models
torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)

# load pretrained models, using ResNeSt-50 as an example
net = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
Load using python package
# using ResNeSt-50 as an example
from resnest.torch import resnest50
net = resnest50(pretrained=True)
Gluon Models
Load pretrained model:
# using ResNeSt-50 as an example
from resnest.gluon import resnest50
net = resnest50(pretrained=True)
Transfer Learning Models
Detectron2
We provide a wrapper for training Detectron2 models with ResNeSt backbone at d2. Training configs and pretrained models are released. See details in d2.

MMDetection
The ResNeSt backbone has been adopted by MMDetection.

Semantic Segmentation
PyTorch models and training: Please visit PyTorch Encoding Toolkit.
Gluon models and training: Please visit GluonCV Toolkit.
Verify ImageNet Results:
Note: the inference speed reported in the paper are tested using Gluon implementation with RecordIO data.

Prepare ImageNet dataset:
Here we use raw image data format for simplicity, please follow GluonCV tutorial if you would like to use RecordIO format.

cd scripts/dataset/
# assuming you have downloaded the dataset in the current folder
python prepare_imagenet.py --download-dir ./
Torch Model
# use resnest50 as an example
cd scripts/torch/
python verify.py --model resnest50 --crop-size 224
Gluon Model
# use resnest50 as an example
cd scripts/gluon/
python verify.py --model resnest50 --crop-size 224
How to Train
ImageNet Models
Training with MXNet Gluon: Please visit Gluon folder.
Training with PyTorch: Please visit PyTorch Encoding Toolkit (slightly worse than Gluon implementation).
Detectron Models
For object detection and instance segmentation models, please visit our detectron2-ResNeSt fork.

Semantic Segmentation
Training with PyTorch: Encoding Toolkit.
Training with MXNet: GluonCV Toolkit.
Reference
ResNeSt: Split-Attention Networks [arXiv]

Hang Zhang, Chongruo Wu, Zhongyue Zhang, Yi Zhu, Zhi Zhang, Haibin Lin, Yue Sun, Tong He, Jonas Muller, R. Manmatha, Mu Li and Alex Smola

@article{zhang2020resnest,
title={ResNeSt: Split-Attention Networks},
author={Zhang, Hang and Wu, Chongruo and Zhang, Zhongyue and Zhu, Yi and Zhang, Zhi and Lin, Haibin and Sun, Yue and He, Tong and Muller, Jonas and Manmatha, R. and Li, Mu and Smola, Alexander},
journal={arXiv preprint arXiv:2004.08955},
year={2020}
}
Major Contributors
ResNeSt Backbone (Hang Zhang)
Detectron Models (Chongruo Wu, Zhongyue Zhang)
Semantic Segmentation (Yi Zhu)
Distributed Training (Haibin Lin)