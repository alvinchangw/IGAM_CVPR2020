#!/bin/sh

python train_tinyimagenet_only_advtrain.py -b 16 --model_type igamsource --do_advtrain --step_size_schedule 0,0.1 160000,0.01 240000,0.001 --train_steps 320000 