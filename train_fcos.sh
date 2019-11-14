#!/bin/bash
set -x
srun -p $1 -n$2 --gres=gpu:1 --ntasks-per-node=1 \
    --job-name=inference --kill-on-bad-exit=1 \
python -u demo/inference_faster.py \
     --config configs/myconfig/faster_rcnn_hrnetv2p_w32_1x_libra.py \
     --checkpoint checkpoints/faster_rcnn_hrnetv2p_w32_1x_libra_epoch_40.pth \
     --out work_dirs/faster_rcnn_hrnetv2p_w32_1x_libra_allaug \
     --cropsize 800 \
     --stride 400 \
     --testImgpath data/rpn15/test/ \
     --saveTxtpath work_dirs/faster_rcnn_hrnetv2p_w32_1x_libra_allaug/txt \
     --saveImgpath work_dirs/faster_rcnn_hrnetv2p_w32_1x_libra_allaug/img \
     --patchImgPath work_dirs/faster_rcnn_hrnetv2p_w32_1x_libra_allaug/patch \
