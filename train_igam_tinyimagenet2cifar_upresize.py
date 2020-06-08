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
from model_new import Model, ModelTinyImagnet, ModelTinyImagenetSourceExtendedLogits, IgamConvDiscriminatorModel
from model_original_cifar_challenge import ModelExtendedLogits
import cifar10_input
import cifar100_input

import pdb
from tqdm import tqdm
import subprocess
import time
from numba import cuda

import config_igam_tinyimagenet2cifar10_upresize

def get_path_dir(data_dir, dataset, **_):
    path = os.path.join(data_dir, dataset)
    if os.path.islink(path):
        path = os.readlink(path)
    return path


def train(tf_seed, np_seed, train_steps, only_finetune, finetune_train_steps, out_steps, summary_steps, checkpoint_steps, step_size_schedule,
          weight_decay, momentum, train_batch_size, epsilon, replay_m, model_dir, source_model_dir, dataset, 
          beta, gamma, disc_update_steps, adv_update_steps_per_iter, disc_layers, disc_base_channels, steps_before_adv_opt, adv_encoder_type, enc_output_activation, 
          sep_opt_version, grad_image_ratio, final_grad_image_ratio, num_grad_image_ratios, normalize_zero_mean, eval_adv_attack, same_optimizer, only_fully_connected, disc_avg_pool_hw, 
          finetuned_source_model_dir, train_finetune_source_model, finetune_img_random_pert, img_random_pert, model_suffix, source_task, **kwargs):
    tf.set_random_seed(tf_seed)
    np.random.seed(np_seed)

    model_dir = model_dir + 'IGAM-%sto%s_b%dupresize_beta_%.3f_gamma_%.3f_disc_update_steps%d_l%dbc%d' % (source_task, dataset, train_batch_size, beta, gamma, disc_update_steps, disc_layers, disc_base_channels)  # TODO Replace with not defaults

    if disc_avg_pool_hw:
        model_dir = model_dir + 'avgpool'

    if img_random_pert:
        model_dir = model_dir + '_imgpert'

    if steps_before_adv_opt != 0:
        model_dir = model_dir + '_advdelay%d' % (steps_before_adv_opt)

    if train_steps != 80000:
        model_dir = model_dir + '_%dsteps' % (train_steps)
    if same_optimizer == False:
        model_dir = model_dir + '_adamDopt'

    if tf_seed != 451760341:
        model_dir = model_dir + '_tf_seed%d' % (tf_seed)
    if np_seed != 216105420:
        model_dir = model_dir + '_np_seed%d' % (np_seed)

    model_dir = model_dir + model_suffix

    # Setting up the data and the model
    data_path = get_path_dir(dataset=dataset, **kwargs)
    if dataset == 'cifar10':
        raw_data = cifar10_input.CIFAR10Data(data_path)
    else:
        raw_data = cifar100_input.CIFAR100Data(data_path)


    global_step = tf.train.get_or_create_global_step()
    increment_global_step_op = tf.assign(global_step, global_step+1)
    reset_global_step_op = tf.assign(global_step, 0)

    full_source_model_x_input = tf.placeholder(tf.float32, shape = [None, 32, 32, 3])
    upresized_full_source_model_x_input = tf.image.resize_images(full_source_model_x_input, size=[64, 64])
    if dataset == 'cifar10':
        source_model = ModelTinyImagenetSourceExtendedLogits(mode='train', dataset=source_task, target_task_class_num=10, train_batch_size=train_batch_size, input_tensor=upresized_full_source_model_x_input)
    elif dataset == 'cifar100':
        source_model = ModelTinyImagenetSourceExtendedLogits(mode='train', dataset=source_task, target_task_class_num=100, train_batch_size=train_batch_size, input_tensor=upresized_full_source_model_x_input)
    model = Model(mode='train', dataset=dataset, train_batch_size=train_batch_size, normalize_zero_mean=normalize_zero_mean)

    # Setting up the optimizers
    boundaries = [int(sss[0]) for sss in step_size_schedule][1:]
    values = [sss[1] for sss in step_size_schedule]
    learning_rate = tf.train.piecewise_constant(tf.cast(global_step, tf.int32), boundaries, values)
    c_optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
    finetune_optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
    
    if same_optimizer:
        d_optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
    else:
        print("Using ADAM opt for DISC model")
        d_optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)

    # Compute input gradient (saliency map)
    input_grad = tf.gradients(model.target_softmax, model.x_input, name="gradients_ig")[0]
    source_model_input_grad = tf.gradients(source_model.target_softmax, full_source_model_x_input, name="gradients_ig_source_model")[0]

    # lp norm diff between input_grad & source_model_input_grad
    input_grad_l2_norm_diff = tf.reduce_mean(tf.reduce_sum(tf.pow(tf.subtract(input_grad, source_model_input_grad), 2.0), keepdims=True))

    # Setting up the discriminator model
    labels_input_grad = tf.zeros( tf.shape(input_grad)[0] , dtype=tf.int64)
    labels_source_model_input_grad = tf.ones( tf.shape(input_grad)[0] , dtype=tf.int64)
    disc_model = IgamConvDiscriminatorModel(mode='train', dataset=dataset, train_batch_size=train_batch_size, image_size=32, num_conv_layers=disc_layers, base_num_channels=disc_base_channels, normalize_zero_mean=normalize_zero_mean,
        x_modelgrad_input_tensor=input_grad, y_modelgrad_input_tensor=labels_input_grad, x_source_modelgrad_input_tensor=source_model_input_grad, y_source_modelgrad_input_tensor=labels_source_model_input_grad, only_fully_connected=only_fully_connected, avg_pool_hw=disc_avg_pool_hw)

    t_vars = tf.trainable_variables()
    C_vars = [var for var in t_vars if 'classifier' in var.name]
    D_vars = [var for var in t_vars if 'discriminator' in var.name]
    source_model_vars = [var for var in t_vars if ('discriminator' not in var.name and 'classifier' not in var.name and 'target_task_logit' not in var.name)]
    source_model_target_logit_vars = [var for var in t_vars if 'target_task_logit' in var.name]
    
    source_model_saver = tf.train.Saver(var_list=source_model_vars)
    finetuned_source_model_vars = source_model_vars + source_model_target_logit_vars
    finetuned_source_model_saver = tf.train.Saver(var_list=finetuned_source_model_vars)

    # Source model finetune optimization
    source_model_finetune_loss = source_model.target_task_mean_xent + weight_decay * source_model.weight_decay_loss
    # Classifier: Optimizing computation
    # total classifier loss: Add discriminator loss into total classifier loss
    total_loss = model.mean_xent + weight_decay * model.weight_decay_loss - beta * disc_model.mean_xent + gamma * input_grad_l2_norm_diff
    
    classification_c_loss = model.mean_xent + weight_decay * model.weight_decay_loss
    adv_c_loss = - beta * disc_model.mean_xent

    # Discriminator: Optimizating computation
    # discriminator loss
    total_d_loss = disc_model.mean_xent + weight_decay * disc_model.weight_decay_loss

    # Finetune source_model
    source_model_new_weights = source_model_target_logit_vars
    finetune_min_step = finetune_optimizer.minimize(source_model_finetune_loss, var_list=source_model_new_weights)

    # Train classifier
    final_grads = c_optimizer.compute_gradients(total_loss, var_list=C_vars)
    no_pert_grad = [(tf.zeros_like(v), v) if 'perturbation' in v.name else (g, v) for g, v in final_grads]
    c_min_step = c_optimizer.apply_gradients(no_pert_grad)

    classification_final_grads = c_optimizer.compute_gradients(classification_c_loss, var_list=C_vars)
    classification_no_pert_grad = [(tf.zeros_like(v), v) if 'perturbation' in v.name else (g, v) for g, v in classification_final_grads]
    c_classification_min_step = c_optimizer.apply_gradients(classification_no_pert_grad)

    # discriminator opt step
    d_min_step = d_optimizer.minimize(total_d_loss, var_list=D_vars)

    # Loss gradients to the model params
    logit_weights = tf.get_default_graph().get_tensor_by_name('classifier/logit/DW:0')
    last_conv_weights = tf.get_default_graph().get_tensor_by_name('classifier/unit_3_4/sub2/conv2/DW:0')
    first_conv_weights = tf.get_default_graph().get_tensor_by_name('classifier/input/init_conv/DW:0')

    model_xent_logit_grad_norm = tf.norm(tf.gradients(model.mean_xent, logit_weights)[0], ord='euclidean')
    
    disc_xent_logit_grad_norm = tf.norm(tf.gradients(disc_model.mean_xent, logit_weights)[0], ord='euclidean')
    
    input_grad_l2_norm_diff_logit_grad_norm = tf.norm(tf.gradients(input_grad_l2_norm_diff, logit_weights)[0], ord='euclidean')

    model_xent_last_conv_grad_norm = tf.norm(tf.gradients(model.mean_xent, last_conv_weights)[0], ord='euclidean')
    disc_xent_last_conv_grad_norm = tf.norm(tf.gradients(disc_model.mean_xent, last_conv_weights)[0], ord='euclidean')
    input_grad_l2_norm_diff_last_conv_grad_norm = tf.norm(tf.gradients(input_grad_l2_norm_diff, last_conv_weights)[0], ord='euclidean')
    model_xent_first_conv_grad_norm = tf.norm(tf.gradients(model.mean_xent, first_conv_weights)[0], ord='euclidean')
    disc_xent_first_conv_grad_norm = tf.norm(tf.gradients(disc_model.mean_xent, first_conv_weights)[0], ord='euclidean')
    input_grad_l2_norm_diff_first_conv_grad_norm = tf.norm(tf.gradients(input_grad_l2_norm_diff, first_conv_weights)[0], ord='euclidean')

    # Setting up the Tensorboard and checkpoint outputs
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    saver = tf.train.Saver(max_to_keep=1)
    tf.summary.scalar('C accuracy', model.accuracy)
    tf.summary.scalar('D accuracy', disc_model.accuracy)
    tf.summary.scalar('C xent', model.xent / train_batch_size)
    tf.summary.scalar('D xent', disc_model.xent / train_batch_size)
    tf.summary.scalar('total C loss', total_loss / train_batch_size)
    tf.summary.scalar('total D loss', total_d_loss / train_batch_size)
    tf.summary.scalar('adv C loss', adv_c_loss / train_batch_size)
    tf.summary.scalar('C cls xent loss', model.mean_xent)
    tf.summary.scalar('D xent loss', disc_model.mean_xent)
    # Loss gradients
    tf.summary.scalar('model_xent_logit_grad_norm', model_xent_logit_grad_norm)
    tf.summary.scalar('disc_xent_logit_grad_norm', disc_xent_logit_grad_norm)
    tf.summary.scalar('input_grad_l2_norm_diff_logit_grad_norm', input_grad_l2_norm_diff_logit_grad_norm)
    tf.summary.scalar('model_xent_last_conv_grad_norm', model_xent_last_conv_grad_norm)
    tf.summary.scalar('disc_xent_last_conv_grad_norm', disc_xent_last_conv_grad_norm)
    tf.summary.scalar('input_grad_l2_norm_diff_last_conv_grad_norm', input_grad_l2_norm_diff_last_conv_grad_norm)
    tf.summary.scalar('model_xent_first_conv_grad_norm', model_xent_first_conv_grad_norm)
    tf.summary.scalar('disc_xent_first_conv_grad_norm', disc_xent_first_conv_grad_norm)
    tf.summary.scalar('input_grad_l2_norm_diff_first_conv_grad_norm', input_grad_l2_norm_diff_first_conv_grad_norm)
    merged_summaries = tf.summary.merge_all()

    with tf.Session() as sess:
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        print('important params >>> \n model dir: %s \n dataset: %s \n training batch size: %d \n' % (model_dir, dataset, train_batch_size))
        # initialize data augmentation\
        if dataset == 'cifar10':
          data = cifar10_input.AugmentedCIFAR10Data(raw_data, sess, model)
        elif dataset == 'cifar100':
          data = cifar100_input.AugmentedCIFAR100Data(raw_data, sess, model)

        # Initialize the summary writer, global variables, and our time counter.
        summary_writer = tf.summary.FileWriter(model_dir + '/train', sess.graph)
        eval_summary_writer = tf.summary.FileWriter(model_dir + '/eval')
        sess.run(tf.global_variables_initializer())

        # Restore source model
        source_model_file = tf.train.latest_checkpoint(source_model_dir)
        source_model_saver.restore(sess, source_model_file)

        # Finetune source model here
        if train_finetune_source_model:
            for ii in tqdm(range(finetune_train_steps)):
                x_batch, y_batch = data.train_data.get_next_batch(train_batch_size, multiple_passes=True)
                if finetune_img_random_pert:
                    x_batch = x_batch + np.random.uniform(-epsilon, epsilon, x_batch.shape)
                    x_batch = np.clip(x_batch, 0, 255) # ensure valid pixel range
                
                nat_dict = {full_source_model_x_input: x_batch, source_model.y_input: y_batch}
                                        
                # Output to stdout
                if ii % summary_steps == 0:
                    train_finetune_acc, train_finetune_loss = sess.run([source_model.target_task_accuracy, source_model_finetune_loss], feed_dict=nat_dict)
                    
                    x_eval_batch, y_eval_batch = data.eval_data.get_next_batch(train_batch_size, multiple_passes=True)
                    if img_random_pert:
                        x_eval_batch = x_eval_batch + np.random.uniform(-epsilon, epsilon, x_eval_batch.shape)
                        x_eval_batch = np.clip(x_eval_batch, 0, 255) # ensure valid pixel range

                    eval_dict = {full_source_model_x_input: x_eval_batch, source_model.y_input: y_eval_batch}
                    val_finetune_acc, val_finetune_loss = sess.run([source_model.target_task_accuracy, source_model_finetune_loss], feed_dict=eval_dict)
                    print('Source Model Finetune Step {}:    ({})'.format(ii, datetime.now()))
                    print('    training nat accuracy {:.4}% -- validation nat accuracy {:.4}%'.format(train_finetune_acc * 100,
                                                                                                    val_finetune_acc * 100))
                    print('    training nat c loss: {}'.format( train_finetune_loss ))
                    print('    validation nat c loss: {}'.format( val_finetune_loss ))

                    sys.stdout.flush()

                sess.run(finetune_min_step, feed_dict=nat_dict)
                sess.run(increment_global_step_op)
            finetuned_source_model_saver.save(sess, os.path.join(finetuned_source_model_dir, 'checkpoint'), global_step=global_step)
            
            if only_finetune:
                # full test evaluation
                if dataset == 'cifar10':
                    raw_data = cifar10_input.CIFAR10Data(data_path, init_shuffle=False)
                else:
                    raw_data = cifar100_input.CIFAR100Data(data_path, init_shuffle=False)
                
                data_size = raw_data.eval_data.n
                if data_size % train_batch_size == 0:
                    eval_steps = data_size // train_batch_size
                else:
                    eval_steps = data_size // train_batch_size 
                total_num_correct = 0
                for ii in tqdm(range(eval_steps)):
                    x_eval_batch, y_eval_batch = raw_data.eval_data.get_next_batch(train_batch_size, multiple_passes=False)
                    eval_dict = {full_source_model_x_input: x_eval_batch, source_model.y_input: y_eval_batch}
                    val_finetune_acc, num_correct = sess.run([source_model.target_task_accuracy, source_model.target_task_num_correct], feed_dict=eval_dict)
                    total_num_correct += num_correct
                eval_acc = total_num_correct / data_size
                
                print('Evaluated finetuned source_model on full eval cifar')
                print("Full clean eval_acc: {}%".format(eval_acc*100))

                # generate input gradients for tinyimagenet train and eval set
                if dataset == 'cifar10':
                    raw_data = cifar10_input.CIFAR10Data(data_path, init_shuffle=False)
                else:
                    raw_data = cifar100_input.CIFAR100Data(data_path, init_shuffle=False)
                # Train set
                all_input_gradients = []
                iter_steps = raw_data.train_data.n // train_batch_size
                if raw_data.train_data.n % train_batch_size != 0:
                    iter_steps += 1
                for ii in tqdm(range(iter_steps)):
                    x_batch, y_batch = raw_data.train_data.get_next_batch(train_batch_size, multiple_passes=False)   
                    nat_dict = {full_source_model_x_input: x_batch, source_model.y_input: y_batch}
                    ig = sess.run(source_model_input_grad, feed_dict=nat_dict)
                    all_input_gradients.append(ig)
                path = os.path.join(finetuned_source_model_dir, "{}_train_ig.npy".format(dataset))
                all_input_gradients = np.concatenate(all_input_gradients, axis=0)
                np.save(path, all_input_gradients[:raw_data.train_data.n])
                
                # Eval set
                if dataset == 'cifar10':
                    raw_data = cifar10_input.CIFAR10Data(data_path, init_shuffle=False)
                else:
                    raw_data = cifar100_input.CIFAR100Data(data_path, init_shuffle=False)
                all_input_gradients = []
                iter_steps = raw_data.eval_data.n // train_batch_size
                for ii in tqdm(range(iter_steps)):
                    x_batch, y_batch = raw_data.eval_data.get_next_batch(train_batch_size, multiple_passes=False)   
                    nat_dict = {full_source_model_x_input: x_batch, source_model.y_input: y_batch}
                    ig = sess.run(source_model_input_grad, feed_dict=nat_dict)
                    all_input_gradients.append(ig)
                path = os.path.join(finetuned_source_model_dir, "{}_eval_ig.npy".format(dataset))
                all_input_gradients = np.concatenate(all_input_gradients, axis=0)
                np.save(path, all_input_gradients[:raw_data.eval_data.n])

                return

        else:
            finetuned_source_model_file = tf.train.latest_checkpoint(finetuned_source_model_dir)
            finetuned_source_model_saver.restore(sess, finetuned_source_model_file)


        # reset global step to 0 before running main training loop
        sess.run(reset_global_step_op)

        # Main training loop
        for ii in tqdm(range(train_steps)):
            x_batch, y_batch = data.train_data.get_next_batch(train_batch_size, multiple_passes=True)            
            if img_random_pert:
                x_batch = x_batch + np.random.uniform(-epsilon, epsilon, x_batch.shape)
                x_batch = np.clip(x_batch, 0, 255) # ensure valid pixel range

            # Sample randinit input grads
            nat_dict = {model.x_input: x_batch, model.y_input: y_batch, full_source_model_x_input: x_batch, source_model.y_input: y_batch}

            # Output to stdout
            if ii % summary_steps == 0:
                train_acc, train_disc_acc, train_c_loss, train_d_loss, train_adv_c_loss, summary = sess.run([model.accuracy, disc_model.accuracy, total_loss, total_d_loss, adv_c_loss, merged_summaries], feed_dict=nat_dict)
                summary_writer.add_summary(summary, global_step.eval(sess))
                
                x_eval_batch, y_eval_batch = data.eval_data.get_next_batch(train_batch_size, multiple_passes=True)
                if img_random_pert:
                    x_eval_batch = x_eval_batch + np.random.uniform(-epsilon, epsilon, x_eval_batch.shape)
                    x_eval_batch = np.clip(x_eval_batch, 0, 255) # ensure valid pixel range

                eval_dict = {model.x_input: x_eval_batch, model.y_input: y_eval_batch, full_source_model_x_input: x_eval_batch, source_model.y_input: y_eval_batch}
                val_acc, val_disc_acc, val_c_loss, val_d_loss, val_adv_c_loss, summary = sess.run([model.accuracy, disc_model.accuracy, total_loss, total_d_loss, adv_c_loss, merged_summaries], feed_dict=eval_dict)
                eval_summary_writer.add_summary(summary, global_step.eval(sess))
                print('Step {}:    ({})'.format(ii, datetime.now()))
                print('    training nat accuracy {:.4}% -- validation nat accuracy {:.4}%'.format(train_acc * 100,
                                                                                                  val_acc * 100))
                print('    training nat disc accuracy {:.4}% -- validation nat disc accuracy {:.4}%'.format(train_disc_acc * 100,
                                                                                                  val_disc_acc * 100))
                print('    training nat c loss: {},     d loss: {},     adv c loss: {}'.format( train_c_loss, train_d_loss, train_adv_c_loss))
                print('    validation nat c loss: {},     d loss: {},     adv c loss: {}'.format( val_c_loss, val_d_loss, val_adv_c_loss))

                sys.stdout.flush()
            # Tensorboard summaries
            elif ii % out_steps == 0:
                nat_acc, nat_disc_acc, nat_c_loss, nat_d_loss, nat_adv_c_loss = sess.run([model.accuracy, disc_model.accuracy, total_loss, total_d_loss, adv_c_loss], feed_dict=nat_dict)
                print('Step {}:    ({})'.format(ii, datetime.now()))
                print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
                print('    training nat disc accuracy {:.4}%'.format(nat_disc_acc * 100))
                print('    training nat c loss: {},     d loss: {},      adv c loss: {}'.format( nat_c_loss, nat_d_loss, nat_adv_c_loss))

            # Write a checkpoint
            if (ii+1) % checkpoint_steps == 0:
                saver.save(sess, os.path.join(model_dir, 'checkpoint'), global_step=global_step)
            
            if sep_opt_version == 1:
                if ii >= steps_before_adv_opt:
                    # Actual training step for Classifier
                    sess.run(c_min_step, feed_dict=nat_dict)
                    sess.run(increment_global_step_op)
                    
                    if ii % disc_update_steps == 0:
                        # Actual training step for Discriminator
                        sess.run(d_min_step, feed_dict=nat_dict)
                else:
                    # only train on classification loss
                    sess.run(c_classification_min_step, feed_dict=nat_dict)
                    sess.run(increment_global_step_op)

                # # Use this to optimize classifier and discriminator at the same step
            elif sep_opt_version == 2:
                # Actual training step for Classifier
                if ii >= steps_before_adv_opt:
                    if adv_update_steps_per_iter > 1:
                        sess.run(c_classification_min_step, feed_dict=nat_dict)
                        sess.run(increment_global_step_op)
                        for i in range(adv_update_steps_per_iter):
                            x_batch, y_batch = data.train_data.get_next_batch(train_batch_size, multiple_passes=True)
                            if img_random_pert:
                                x_batch = x_batch + np.random.uniform(-epsilon, epsilon, x_batch.shape)
                                x_batch = np.clip(x_batch, 0, 255) # ensure valid pixel range
                                
                            nat_dict = {model.x_input: x_batch, model.y_input: y_batch, full_source_model_x_input: x_batch, source_model.y_input: y_batch}

                            sess.run(c_adv_min_step, feed_dict=nat_dict)
                    else:
                        sess.run(c_min_step, feed_dict=nat_dict)
                        sess.run(increment_global_step_op)
                    
                    if ii % disc_update_steps == 0:
                        # Actual training step for Discriminator
                        sess.run(d_min_step, feed_dict=nat_dict)
                else:
                    # only train on classification loss
                    sess.run(c_classification_min_step, feed_dict=nat_dict)
                    sess.run(increment_global_step_op)
            elif sep_opt_version == 0:            
                if ii >= steps_before_adv_opt:
                    if ii % disc_update_steps == 0:
                        sess.run([c_min_step, d_min_step], feed_dict=nat_dict)
                        sess.run(increment_global_step_op)
                    else:
                        sess.run(c_min_step, feed_dict=nat_dict)
                        sess.run(increment_global_step_op)
                else:
                    sess.run(c_classification_min_step, feed_dict=nat_dict)
                    sess.run(increment_global_step_op)

        # full test evaluation
        if dataset == 'cifar10':
            raw_data = cifar10_input.CIFAR10Data(data_path, init_shuffle=False)
        else:
            raw_data = cifar100_input.CIFAR100Data(data_path, init_shuffle=False)
        
        data_size = raw_data.eval_data.n
        if data_size % train_batch_size == 0:
            eval_steps = data_size // train_batch_size
        else:
            eval_steps = data_size // train_batch_size 
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


        # generate input gradients for tinyimagenet train and eval set
        # Train set
        all_input_gradients = []
        iter_steps = raw_data.train_data.n // train_batch_size
        if raw_data.train_data.n % train_batch_size != 0:
            iter_steps += 1
        for ii in tqdm(range(iter_steps)):
            x_batch, y_batch = raw_data.train_data.get_next_batch(train_batch_size, multiple_passes=False)   
            nat_dict = {model.x_input: x_batch, model.y_input: y_batch}
            ig = sess.run(input_grad, feed_dict=nat_dict)
            all_input_gradients.append(ig)
        path = os.path.join(model_dir, "{}_train_ig.npy".format(dataset))
        all_input_gradients = np.concatenate(all_input_gradients, axis=0)
        np.save(path, all_input_gradients[:raw_data.train_data.n])
        
        # Eval set
        all_input_gradients = []
        if dataset == 'cifar10':
            raw_data = cifar10_input.CIFAR10Data(data_path, init_shuffle=False)
        else:
            raw_data = cifar100_input.CIFAR100Data(data_path, init_shuffle=False)
        iter_steps = raw_data.eval_data.n // train_batch_size
        for ii in tqdm(range(iter_steps)):
            x_batch, y_batch = raw_data.eval_data.get_next_batch(train_batch_size, multiple_passes=False)   
            nat_dict = {model.x_input: x_batch, model.y_input: y_batch}
            ig = sess.run(input_grad, feed_dict=nat_dict)
            all_input_gradients.append(ig)
        path = os.path.join(model_dir, "{}_eval_ig.npy".format(dataset))
        all_input_gradients = np.concatenate(all_input_gradients, axis=0)
        np.save(path, all_input_gradients[:raw_data.eval_data.n])


        devices = sess.list_devices()
        for d in devices:
            print("sess' device names:")
            print(d.name)

    return model_dir
            
if __name__ == '__main__':
    args = config_igam_tinyimagenet2cifar10_upresize.get_args()
    args_dict = vars(args)
    model_dir = train(**args_dict)
    if args_dict['eval_adv_attack']:
        cuda.select_device(0)
        cuda.close()

        print("{}: Evaluating on fgsm and pgd attacks".format(datetime.now()))
        print("model_dir: ", model_dir)
        subprocess.run("python pgd_attack.py --attack_name fgsm --save_eval_log --num_steps 1 --no-random_start --step_size 8 --model_dir {} ; python run_attack.py --attack_name fgsm --save_eval_log --model_dir {} ; python pgd_attack.py --save_eval_log --model_dir {} ; python run_attack.py --save_eval_log --model_dir {} ; python pgd_attack.py --attack_name pgds5 --save_eval_log --num_steps 5 --model_dir {} ; python run_attack.py --attack_name pgds5 --save_eval_log --num_steps 5 --model_dir {} ; python pgd_attack.py --attack_name pgds20 --save_eval_log --num_steps 20 --model_dir {} ; python run_attack.py --attack_name pgds20 --save_eval_log --num_steps 20 --model_dir {}".format(model_dir, model_dir, model_dir, model_dir, model_dir, model_dir, model_dir, model_dir), shell=True)
        print("{}: Ended evaluation on fgsm and pgd  attacks".format(datetime.now()))

