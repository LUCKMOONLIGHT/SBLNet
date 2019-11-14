# MMDetection For Remote Sensing
![demo image](demo/P0678.png)

## Introduction

The master branch works with **PyTorch 1.1** or higher.
mmdetection is an open source object detection toolbox based on PyTorch. It is
a part of the open-mmlab project developed by [Multimedia Laboratory, CUHK](http://mmlab.ie.cuhk.edu.hk/).

## Benchmark and model zoo

Supported methods and backbones are shown in the below table.
Results and models are available in the [Model zoo](docs/MODEL_ZOO.md).

|                    | ResNet   | ResNeXt  | SENet    | VGG      | HRNet |
|--------------------|:--------:|:--------:|:--------:|:--------:|:-----:|
| RPN                | ✓        | ✓        | ☐        | ✗        | ✓     |
| Fast R-CNN         | ✓        | ✓        | ☐        | ✗        | ✓     |
| Faster R-CNN       | ✓        | ✓        | ☐        | ✗        | ✓     |
| Mask R-CNN         | ✓        | ✓        | ☐        | ✗        | ✓     |
| Cascade R-CNN      | ✓        | ✓        | ☐        | ✗        | ✓     |
| Cascade Mask R-CNN | ✓        | ✓        | ☐        | ✗        | ✓     |
| SSD                | ✗        | ✗        | ✗        | ✓        | ✗     |
| RetinaNet          | ✓        | ✓        | ☐        | ✗        | ✓     |
| GHM                | ✓        | ✓        | ☐        | ✗        | ✓     |
| Mask Scoring R-CNN | ✓        | ✓        | ☐        | ✗        | ✓     |
| FCOS               | ✓        | ✓        | ☐        | ✗        | ✓     |
| Double-Head R-CNN  | ✓        | ✓        | ☐        | ✗        | ✓     |
| Grid R-CNN (Plus)  | ✓        | ✓        | ☐        | ✗        | ✓     |
| Hybrid Task Cascade| ✓        | ✓        | ☐        | ✗        | ✓     |
| Libra R-CNN        | ✓        | ✓        | ☐        | ✗        | ✓     |
| Guided Anchoring   | ✓        | ✓        | ☐        | ✗        | ✓     |

Other features
- [x] DCNv2
- [x] Group Normalization
- [x] Weight Standardization
- [x] OHEM
- [x] Soft-NMS
- [x] Generalized Attention
- [x] GCNet
- [x] Mixed Precision (FP16) Training


## Installation

1. Please refer to [INSTALL.md](docs/INSTALL.md) for installation and dataset preparation.
2. Before install, you should make sure the configuration is correct

```shell
vim ~/.bashrc
PATH="/mnt/lustre/share/gcc/gcc-5.3.0/bin:$PATH"
export CC="/mnt/lustre/share/gcc/gcc-5.3.0/bin/gcc"
export CXX="/mnt/lustre/share/gcc/gcc-5.3.0/bin/g++"
vim ~/.condarc
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
show_channel_urls: true
```

3. You can install directly from the script below

```shell
export INSTALL_DIR=$PWD
mkdir checkpoints
mkdir data
mkdir work_dirs
conda create -n mmlab python=3.7 -y
source activate mmlab
conda install pytorch torchvision==0.2.2 cuda90 cudatoolkit=9.0 -y
conda install cython -y
git clone git@gitlab.bj.sensetime.com:yanhongchang/mmdetection.git
cd mmdetection
git checkout horizontal
python setup.py build develop
ln -s ../data data
ln -s ../checkpoints checkpoints
ln -s ../work_dirs work_dirs
# rm -rf /mnt/lustre/yanhongchang/.conda/envs/open-mmlab/lib/python3.7/site-packages/torchvision-0.4.1-py3.7-linux-x86_64.egg/
unset INSTALL_DIR
```

## Get Started

0. Please see [GETTING_STARTED.md](docs/GETTING_STARTED.md) for the basic usage of MMDetection.
1. [Data processing](http://gitlab.bj.sensetime.com/yanhongchang/rs_devkit/blob/master/DOTA_devkit/DOTA2COCO.py)
2. Preare data and checkpoints.
3. [run scripts](http://gitlab.bj.sensetime.com/yanhongchang/mmdetection/blob/horizontal/train_faster_hrnet.sh)
