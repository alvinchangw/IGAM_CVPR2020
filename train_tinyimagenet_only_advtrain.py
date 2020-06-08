"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import shutil
from timeit import default_timer as timer
import tensorflow as tf
import numpy as np
import sys
from model_new import Model, ModelTinyImagnet, ModelTinyImagenetSource, IgamConvDiscriminatorModel
from model_original_cifar_challenge import ModelExtendedLogits

import tinyimagenet_input
import pdb
from tqdm import tqdm
import subprocess
import time
from numba import cuda

import config_advtrain_tinyimagenet

# for adversarial training
from pgd_attack import LinfPGDAttack

def get_path_dir(data_dir, dataset, **_):
    path = os.path.join(data_dir, dataset)
    if os.path.islink(path):
        path = os.readlink(path)
    return path


def train(tf_seed, np_seed, train_steps, only_finetune, finetune_train_steps, out_steps, summary_steps, checkpoint_steps, step_size_schedule,
          weight_decay, momentum, train_batch_size, do_advtrain, do_advreg, epsilon, pgd_steps, step_size, random_start, loss_func, replay_m, model_dir, source_model_dir, dataset, data_dir, 
          beta, gamma, disc_update_steps, adv_update_steps_per_iter, disc_layers, disc_base_channels, steps_before_adv_opt, steps_before_adv_training, adv_encoder_type, enc_output_activation, 
          sep_opt_version, grad_image_ratio, final_grad_image_ratio, num_grad_image_ratios, normalize_zero_mean, eval_adv_attack, same_optimizer, only_fully_connected, disc_avg_pool_hw, 
          finetuned_source_model_dir, train_finetune_source_model, finetune_img_random_pert, img_random_pert, model_suffix, model_type, **kwargs):
    tf.set_random_seed(tf_seed)
    np.random.seed(np_seed)

    # Add pgd params to model name
    if do_advtrain:
        model_dir = model_dir + '_AdvTrain'
        if epsilon != 8:
            model_dir = model_dir + '_ep%d' % (epsilon)
        if random_start != True:
            model_dir = model_dir + '_norandstart'
        if pgd_steps != 7:
            model_dir = model_dir + '_%dsteps' % (pgd_steps)
        if step_size != 2:
            model_dir = model_dir + '_stepsize%d' % (step_size)
        model_dir = model_dir + '-{}-'.format(model_type)
    
    model_dir = model_dir + 'IGAM-%s_b%d' % (dataset, train_batch_size)  # TODO Replace with not defaults

    if tf_seed != 451760341:
        model_dir = model_dir + '_tf_seed%d' % (tf_seed)
    if np_seed != 216105420:
        model_dir = model_dir + '_np_seed%d' % (np_seed)

    model_dir = model_dir + model_suffix

    # Setting up the data and the model
    data_path = data_dir #"./datasets/tiny-imagenet/tiny-imagenet-200"
    raw_data = tinyimagenet_input.TinyImagenetData(data_path)
    global_step = tf.train.get_or_create_global_step()
    increment_global_step_op = tf.assign(global_step, global_step+1)
    reset_global_step_op = tf.assign(global_step, 0)

    if model_type == "igamsource":
        model = ModelTinyImagenetSource(mode='train', dataset='tinyimagenet', train_batch_size=train_batch_size, normalize_zero_mean=normalize_zero_mean)
    else:
        model = ModelTinyImagnet(mode='train', dataset='tinyimagenet', train_batch_size=train_batch_size, normalize_zero_mean=normalize_zero_mean)

    # Setting up the optimizers
    boundaries = [int(sss[0]) for sss in step_size_schedule][1:]
    values = [sss[1] for sss in step_size_schedule]
    learning_rate = tf.train.piecewise_constant(tf.cast(global_step, tf.int32), boundaries, values)
    c_optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)

    t_vars = tf.trainable_variables()
    C_vars = [var for var in t_vars if 'classifier' in var.name]
    
    classification_c_loss = model.mean_xent + weight_decay * model.weight_decay_loss
    total_loss = classification_c_loss

    classification_final_grads = c_optimizer.compute_gradients(classification_c_loss, var_list=t_vars)
    classification_no_pert_grad = [(tf.zeros_like(v), v) if 'perturbation' in v.name else (g, v) for g, v in classification_final_grads]
    c_classification_min_step = c_optimizer.apply_gradients(classification_no_pert_grad)


    # Setting up the Tensorboard and checkpoint outputs
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    saver = tf.train.Saver(max_to_keep=1)
    tf.summary.scalar('C accuracy', model.accuracy)
    tf.summary.scalar('C xent', model.xent / train_batch_size)
    merged_summaries = tf.summary.merge_all()

    # Set up adversary
    attack = LinfPGDAttack(model,
                            epsilon,
                            pgd_steps,
                            step_size,
                            random_start,
                            loss_func,
                            dataset=dataset)

    with tf.Session() as sess:
        print('important params >>> \n model dir: %s \n dataset: %s \n training batch size: %d \n' % (model_dir, dataset, train_batch_size))
        # initialize data augmentation
        data = tinyimagenet_input.AugmentedTinyImagenetData(raw_data, sess, model)

        # Initialize the summary writer, global variables, and our time counter.
        summary_writer = tf.summary.FileWriter(model_dir + '/train', sess.graph)
        eval_summary_writer = tf.summary.FileWriter(model_dir + '/eval')
        sess.run(tf.global_variables_initializer())

        # Main training loop
        for ii in tqdm(range(train_steps)):
            x_batch, y_batch = data.train_data.get_next_batch(train_batch_size, multiple_passes=True)            
            if img_random_pert and not (do_advtrain and random_start and ii >= steps_before_adv_training):
                x_batch = x_batch + np.random.uniform(-epsilon, epsilon, x_batch.shape)
                x_batch = np.clip(x_batch, 0, 255) # ensure valid pixel range

            labels_source_modelgrad_disc = np.ones_like( y_batch, dtype=np.int64)
            nat_dict = {model.x_input: x_batch, model.y_input: y_batch}

            # Generate adversarial training examples
            if do_advtrain and ii >= steps_before_adv_training:
                x_batch_adv = attack.perturb(x_batch, y_batch, sess)
                
                train_dict = {model.x_input: x_batch_adv, model.y_input: y_batch}
            else:
                train_dict = nat_dict

            # Output to stdout
            if ii % summary_steps == 0:
                train_acc, train_c_loss, summary = sess.run([model.accuracy, total_loss, merged_summaries], feed_dict=train_dict)
                summary_writer.add_summary(summary, global_step.eval(sess))
                
                x_eval_batch, y_eval_batch = data.eval_data.get_next_batch(train_batch_size, multiple_passes=True)
                if img_random_pert and not (do_advtrain and random_start):
                    x_eval_batch = x_eval_batch + np.random.uniform(-epsilon, epsilon, x_eval_batch.shape)
                    x_eval_batch = np.clip(x_eval_batch, 0, 255) # ensure valid pixel range

                labels_source_modelgrad_disc = np.ones_like( y_eval_batch, dtype=np.int64)
                eval_nat_dict = {model.x_input: x_eval_batch, model.y_input: y_eval_batch}
                if do_advtrain:
                    x_eval_batch_adv = attack.perturb(x_eval_batch, y_eval_batch, sess)
                    eval_dict = {model.x_input: x_eval_batch_adv, model.y_input: y_eval_batch}
                else:
                    eval_dict = eval_nat_dict

                val_acc, val_c_loss, summary = sess.run([model.accuracy, total_loss, merged_summaries], feed_dict=eval_dict)
                eval_summary_writer.add_summary(summary, global_step.eval(sess))
                print('Step {}:    ({})'.format(ii, datetime.now()))
                print('    training nat accuracy {:.4}% -- validation nat accuracy {:.4}%'.format(train_acc * 100,
                                                                                                  val_acc * 100))
                print('    training nat c loss: {}'.format( train_c_loss))
                print('    validation nat c loss: {}'.format( val_c_loss))

                sys.stdout.flush()
            # Tensorboard summaries
            elif ii % out_steps == 0:
                nat_acc, nat_c_loss = sess.run([model.accuracy, total_loss], feed_dict=train_dict)
                print('Step {}:    ({})'.format(ii, datetime.now()))
                print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
                print('    training nat c loss: {}'.format( nat_c_loss))

            # Write a checkpoint
            if (ii+1) % checkpoint_steps == 0:
                saver.save(sess, os.path.join(model_dir, 'checkpoint'), global_step=global_step)
            
            sess.run(c_classification_min_step, feed_dict=train_dict)
            sess.run(increment_global_step_op)

        # full test evaluation
        raw_data = tinyimagenet_input.TinyImagenetData(data_path)
        
        data_size = raw_data.eval_data.n
        if data_size % train_batch_size == 0:
            eval_steps = data_size // train_batch_size
        else:
            eval_steps = data_size // train_batch_size 
            # eval_steps = data_size // train_batch_size + 1
        total_num_correct = 0
        for ii in tqdm(range(eval_steps)):
            x_eval_batch, y_eval_batch = raw_data.eval_data.get_next_batch(train_batch_size, multiple_passes=False)
            eval_dict = {model.x_input: x_eval_batch, model.y_input: y_eval_batch}
            num_correct = sess.run(model.num_correct, feed_dict=eval_dict)
            total_num_correct += num_correct
        eval_acc = total_num_correct / data_size
        
        clean_eval_file_path = os.path.join(model_dir, 'full_clean_eval_acc.txt')
        with open(clean_eval_file_path, "a+") as f:
            f.write("Full clean eval_acc: {}%".format(eval_acc*100))
        print("Full clean eval_acc: {}%".format(eval_acc*100))


        devices = sess.list_devices()
        for d in devices:
            print("sess' device names:")
            print(d.name)

    return model_dir
            
if __name__ == '__main__':
    args = config_advtrain_tinyimagenet.get_args()
    # train(**vars(args))
    args_dict = vars(args)
    model_dir = train(**args_dict)
    if args_dict['eval_adv_attack']:
        cuda.select_device(0)
        cuda.close()

        print("{}: Evaluating on fgsm and pgd attacks".format(datetime.now()))
        print("model_dir: ", model_dir)
        subprocess.run("python pgd_attack_w_mnist_dcgan_w_svhn_w_tinyimagenet.py -d tinyimagenet --attack_name fgsm --save_eval_log --num_steps 1 --no-random_start --step_size 8 --model_dir {} ; python run_attack_w_mnist_dcgan_w_svhn_w_tinyimagenet.py -d tinyimagenet --attack_name fgsm --save_eval_log --model_dir {} ; python pgd_attack_w_mnist_dcgan_w_svhn_w_tinyimagenet.py -d tinyimagenet --save_eval_log --model_dir {} ; python run_attack_w_mnist_dcgan_w_svhn_w_tinyimagenet.py -d tinyimagenet --save_eval_log --model_dir {} ; python pgd_attack_w_mnist_dcgan_w_svhn_w_tinyimagenet.py -d tinyimagenet --attack_name pgds5 --save_eval_log --num_steps 5 --model_dir {} ; python run_attack_w_mnist_dcgan_w_svhn_w_tinyimagenet.py -d tinyimagenet --attack_name pgds5 --save_eval_log --num_steps 5 --model_dir {} ; python pgd_attack_w_mnist_dcgan_w_svhn_w_tinyimagenet.py -d tinyimagenet --attack_name pgds20 --save_eval_log --num_steps 20 --model_dir {} ; python run_attack_w_mnist_dcgan_w_svhn_w_tinyimagenet.py -d tinyimagenet --attack_name pgds20 --save_eval_log --num_steps 20 --model_dir {}".format(model_dir, model_dir, model_dir, model_dir, model_dir, model_dir, model_dir, model_dir), shell=True)
        print("{}: Ended evaluation on fgsm and pgd  attacks".format(datetime.now()))

