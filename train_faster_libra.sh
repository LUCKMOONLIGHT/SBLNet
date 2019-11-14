#!/bin/bash
set -x
srun -p $1 -n$2 --gres=gpu:8 --ntasks-per-node=8 \
    --job-name=libra --kill-on-bad-exit=1 \
python -u tools/train.py \
    configs/myconfig/libra_faster_rcnn_r50_fpn_1x.py \
    --work_dir=work_dirs/libra_faster_rcnn_r50_fpn_1x_noclip_trainval \
    --launcher="slurm" \
    --validate \
