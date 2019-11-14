#!/bin/bash
set -x
srun -p $1 -n$2 --gres=gpu:1 --ntasks-per-node=1 \
    --job-name=f512 --kill-on-bad-exit=1 \
python -u inference_512.py \

