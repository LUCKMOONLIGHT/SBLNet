#!/bin/bash
set -x
srun -p $1 -n$2 --gres=gpu:8 --ntasks-per-node=8 \
    --job-name=fovea --kill-on-bad-exit=1 \
python -u tools/train.py \
    configs/myconfig/fovea_align_gn_ms_r101_fpn_4gpu_2x.py \
    --work_dir=work_dirs/fovea_align_gn_ms_r101_fpn_4gpu_2x \
    --launcher="slurm" \
    --validate \
