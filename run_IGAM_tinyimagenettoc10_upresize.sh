#!/bin/sh

python train_igam_tinyimagenet2cifar_upresize.py -b 32 --disc_update_steps 5 --step_size_schedule 0,0.1 160000,0.01 240000,0.001 --train_steps 320000 --steps_before_adv_opt 280000 --no-img_random_pert --beta 5 --gamma 10 --source_model_dir models/model_AdvTrain-igamsource-IGAM-tinyimagenet_b16 --finetuned_source_model_dir models/adv_trained_tinyimagenet_finetuned_on_c10_upresize
