#!/bin/sh



python pgd_attack.py -d cifar100 --data_path datasets/cifar100 --num_steps 5 --attack_name pgd5 --save_eval_log --model_dir models/modelIGAM-cifar10to100 
python run_attack.py -d cifar100 --data_path datasets/cifar100 --num_steps 5 --save_eval_log --model_dir models/modelIGAM-cifar10to100 --attack_name pgd5


python pgd_attack.py -d cifar100 --data_path datasets/cifar100 --num_steps 20 --step_size 2 --epsilon 8 --attack_name pgd20 --save_eval_log --model_dir models/modelIGAM-cifar10to100 
python run_attack.py -d cifar100 --data_path datasets/cifar100 --num_steps 20 --step_size 2 --epsilon 8 --save_eval_log --model_dir models/modelIGAM-cifar10to100 --attack_name pgd20


python pgd_attack.py -d cifar100 --data_path datasets/cifar100 --num_steps 100 --step_size 2 --epsilon 8 --attack_name pgd100 --save_eval_log --model_dir models/modelIGAM-cifar10to100 
python run_attack.py -d cifar100 --data_path datasets/cifar100 --num_steps 100 --step_size 2 --epsilon 8 --save_eval_log --model_dir models/modelIGAM-cifar10to100 --attack_name pgd100
