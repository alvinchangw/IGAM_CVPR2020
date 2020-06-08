#!/bin/sh

python train_igam_tinyimagenet2cifar_upresize.py --train_finetune_source_model -b 64 --source_model_dir models/model_AdvTrain-igamsource-IGAM-tinyimagenet_b16 --finetuned_source_model_dir models/adv_trained_tinyimagenet_finetuned_on_c10_upresize --only_finetune --finetune_train_steps 4700