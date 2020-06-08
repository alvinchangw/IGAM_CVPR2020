# IGAM_CVPR2020
This is our Tensorflow implementation of Input Gradient Adversarial Matching (IGAM). 

**What it Thinks is Important is Important: Robustness Transfers through Input Gradients (CVPR 2020 Oral)**<br>
*Alvin Chan, Yi Tay, Yew Soon Ong*<br>
https://arxiv.org/abs/1912.05699

TL;DR: By regularizing for similar input gradients, we can transfer adversarial robustness from a teacher to a student classifier even with different training dataset and model architecture.


## Dependencies
1. Tensorflow 1.14.0
2. Python 3.7

## Datasets
CIFAR10 & 100: https://www.cs.toronto.edu/~kriz/cifar.html
TinyImagenet: http://cs231n.stanford.edu/tiny-imagenet-200.zip

## Usage (IGAM TinyImagenet to CIFAR10)
1. Install dependencies with `pip install -r requirements.txt`.
2. Download and save datasets in `datasets/` folder.
3. Run adversarial training on TinyImagenet with `sh run_advtrain_tinyimagenet.sh`. Adversarial trained model is saved in `models/`.
4. Finetune Tinyimagenet teacher model on CIFAR10 with `sh run_finetune_advtrainedtinyimagenet_on_cifar10_igamupresize.sh`. Finetuned model is saved in `models/`.
5. Transfer robustness from teacher to student model through IGAM with `sh run_IGAM_tinyimagenettoc10_upresize.sh`. Student model is saved in `models/`.
6. Run adversarial evaluation on IGAM models with `run_eval_igam_tinyimagenettoc10_upresize.sh`.


## Saved model weights
- Link: "https://drive.google.com/drive/folders/1UiIonctsE4rJHLpD8bPlXXxC-7TSfruP?usp=sharing"
- `adv_trained_finetuned_on_cifar100`: adversarial trained TinyImagenet model from (Step 3).
- `adv_trained_tinyimagenet_finetuned_on_c10_upresize`: Finetuned TinyImagenet teacher model (Step 4).
- `modelIGAM-tinyimagenettocifar10`: IGAM student model (Step 5).
- `adv_trained`: adversarial trained CIFAR10 model.
- `adv_trained_finetuned_on_cifar100`: Finetuned CIFAR10 teacher model.
- `modelIGAM-cifar10to100`: IGAM student model for CIFAR10 to CIFAR100 transfer.


## Code overview
- `train_tinyimagenet_only_advtrain.py`: adversarial training script for TinyImagenet teacher model.
- `train_igam_tinyimagenet2cifar_upresize.py`: trains the IGAM model for for IGAM upresize, Tinyimagenet to CIFAR10 robustness transfer.
- `train_igam.py`: trains the IGAM model for CIFAR10 to CIFAR100 robustness transfer.
- `config_igam.py`: training parameters for for IGAM upresize, CIFAR10 to CIFAR100.
- `config_igam_tinyimagenet2cifar10_upresize.py`: training parameters for IGAM upresize, Tinyimagenet to CIFAR10.
- `model_new.py`: contains code for IGAM model architectures.
- `cifar10_input.py` provides utility functions and classes for loading the CIFAR10 dataset.
- `cifar100_input.py` provides utility functions and classes for loading the CIFAR100 dataset.
- `pgd_attack.py`: generates adversarial examples and save them in `attacks/`.
- `run_attack.py`: evaluates model on adversarial examples from `attacks/`.
- `config_attack.py`: parameters for adversarial example evaluation.


## Citation
If you find our repository useful, please consider citing our paper:

```
@article{chan2019thinks,
  title={What it Thinks is Important is Important: Robustness Transfers through Input Gradients},
  author={Chan, Alvin and Tay, Yi and Ong, Yew-Soon},
  journal={arXiv preprint arXiv:1912.05699},
  year={2019}
}
```


## Acknowledgements

Useful code bases we used in our work:
- https://github.com/MadryLab/cifar10_challenge (for adversarial example generation and evaluation)
- https://github.com/ashafahi/free_adv_train (for model code)