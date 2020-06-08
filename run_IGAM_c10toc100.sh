#!/bin/sh

python train_igam.py -b 32 --disc_update_steps 5 --step_size_schedule 0,0.1 160000,0.01 240000,0.001 --train_steps 320000 --steps_before_adv_opt 280000 --img_random_pert --beta 2 --gamma 10