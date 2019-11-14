#!/bin/bash
set -x
srun -p $1 -n$2 --gres=gpu:1 --ntasks-per-node=1 \
    --job-name=inf --kill-on-bad-exit=1 \
python -u inference.py \

