#!/bin/sh


python pgd_attack.py --attack_name fgsm --save_eval_log --num_steps 1 --no-random_start --step_size 8 --model_dir models/modelIGAM-tinyimagenettocifar10
python run_attack.py --attack_name fgsm --save_eval_log --model_dir models/modelIGAM-tinyimagenettocifar10

python pgd_attack.py --save_eval_log --model_dir models/modelIGAM-tinyimagenettocifar10
python run_attack.py --save_eval_log --model_dir models/modelIGAM-tinyimagenettocifar10 

python pgd_attack.py --attack_name pgds5 --save_eval_log --num_steps 5 --model_dir models/modelIGAM-tinyimagenettocifar10
python run_attack.py --attack_name pgds5 --save_eval_log --num_steps 5 --model_dir models/modelIGAM-tinyimagenettocifar10

python pgd_attack.py --attack_name pgds20 --save_eval_log --num_steps 20 --model_dir models/modelIGAM-tinyimagenettocifar10
python run_attack.py --attack_name pgds20 --save_eval_log --num_steps 20 --model_dir models/modelIGAM-tinyimagenettocifar10


python pgd_attack.py --attack_name pgds100 --save_eval_log --num_steps 100 --model_dir models/modelIGAM-tinyimagenettocifar10
python run_attack.py --attack_name pgds100 --save_eval_log --num_steps 100 --model_dir models/modelIGAM-tinyimagenettocifar10
