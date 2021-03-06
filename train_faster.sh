#!/bin/bash
set -x
srun -p $1 -n$2 --gres=gpu:8 --ntasks-per-node=8 \
    --job-name=fx101 --kill-on-bad-exit=1 \
python -u tools/train.py \
    configs/myconfig/faster_rcnn_x101_64x4d_fpn_1x.py \
    --work_dir=work_dirs/faster_rcnn_x101_64x4d_fpn_1x_trainval \
    --launcher="slurm" \
    --validate \
