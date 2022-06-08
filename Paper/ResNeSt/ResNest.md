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


3.2 The AnyNet Design Space
이 section에서는 표준이며 고정된 네트워크르 블럭을 가정한 네트워크의 구조를 탐구하는 것이다. 즉, 블럭의 수, 블럭의 너비 그리고 블럭의 다른 매개변수들에 대한 탐구다. 이러한 것들은 계산량, 매개변수 그리고 메모리의 분포를 결정하는 동시에 정확도와 효율성을 결정한다.

 

AnyNet의 디자인 공간은 간단하면서 쉬운데, 그림 3의 a와 같이 입력단의 stem, 그 뒤에 body 그리고 class를 예측하는 head 부분으로 나눠져있다. 여기서 stem과 head를 최대한 고정시키고 네트워크의 body를 변경하여 계산량과 정확도를 결정한다.


body는 총 4개의 stages로 구성되어 있으며, 계속해서 resolution을 줄여간다. (그림 3의 b) 각 stage는 개별적인 block들을 가지며, 각각의 stage는 block의 갯수 (di), block의 너비 (wi), 그리고 block 매개변수를 가진다. 이러한 AnyNet의 구조는 매우 방대하다.

 

그림 4에서 볼 수 있듯이 대부분의 실험에서는 residual bottleneck blcok을 사용하며, 이것을 x block 이라고 지칭한다. 그리고 이러한 block으로 이뤄진 네트워크를 AnyNetX라고 하며, 네트워크 구조가 최적화 되었을 때, 놀랍운 효율성을 보인다.

 


AnyNetX의 디자인 공간은 총 16단계로 정해지는데 4개의 stage와 각 stage에서 4개의 매개변수를 가진다. blocks의 수 (di), block의 너비 (wi), bottleneck ratio (bi), 그리고 group width (gi)이다. di≤16인 log-uniform sampling, wi≤1024인 8의 배수, bi∈1,2,4, gi∈1,2,...,32 이다. n=500, epoch 은 10, 그리고 목표 계산량은 300MF~400MF이다.

 

총 1018의 가능성이 나오며 이것이 AnyNetX의 디자인 공간이다. ~1018의 최고성능 모델을 찾는 것이 아닌 일반적인 디자인 원칙을 찾는데 집중했으며 디자인 공간을 이해하고 재정의 하는데 도움이 된다. 그리고 4가지의 접근법을 사용해 AnyNetX를 총5가지로 나눴다.

 

AnyNeyXA 초기의 AnyNetX의 디자인 공간이다.

 

AnyNetXB 여기서 제한하는 부분은 bottleneck ratio를 공유하는 것이다. 즉, 모든 stage에서 bottleneck ratio bi=b로 고정한다. 똑같이 500개의 모델을 만들어 AnyNetXA와 비교했으며, 그림 5의 좌측은 결과이다. 이것으로 보아 bottlenck ratio는 그렇게 큰 차이를 못내는 것 같으며 그림 5의 우측은 b에 따른 결과를 보여준다.