"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import tensorflow as tf
import numpy as np

import cifar10_input
import cifar100_input

import config_attack

class LinfPGDAttack:
  def __init__(self, model, epsilon, num_steps, step_size, random_start, loss_func, dataset='cifar10'):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point."""
    self.model = model
    self.epsilon = epsilon
    self.num_steps = num_steps
    self.step_size = step_size
    self.rand = random_start

    if loss_func == 'xent':
      loss = model.xent
    elif loss_func == 'target_task_xent':
      loss = model.target_task_mean_xent
      correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
      wrong_logit = tf.reduce_max((1-label_mask) * model.pre_softmax - 1e4*label_mask, axis=1)
      loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
    else:
      print('Unknown loss function. Defaulting to cross-entropy')
      loss = model.xent

    self.grad = tf.gradients(loss, model.x_input)[0]

  def perturb(self, x_nat, y, sess):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
    if self.rand:
      x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
      x = np.clip(x, 0, 255) # ensure valid pixel range
    else:
      x = np.copy(x_nat)

    for i in range(self.num_steps):
      grad = sess.run(self.grad, feed_dict={self.model.x_input: x,
                                            self.model.y_input: y})

      x = np.add(x, self.step_size * np.sign(grad), out=x, casting='unsafe')

      x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
      x = np.clip(x, 0, 255) # ensure valid pixel range

    return x

  def perturb_l2(self, x_nat, y, sess):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_2 norm."""
    if self.rand:
      pert = np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
      pert_norm = np.linalg.norm(pert)
      pert = pert / max(1, pert_norm)
    else:
      pert = np.zeros(x_nat.shape)

    for i in range(self.num_steps):
      x = x_nat + pert
      # x = np.clip(x, 0, 255)
      grad = sess.run(self.grad, feed_dict={self.model.x_input: x,
                                            self.model.y_input: y})

      normalized_grad = grad / np.linalg.norm(grad)
      pert = np.add(pert, self.step_size * normalized_grad, out=pert, casting='unsafe')
      
      # project pert to norm ball
      pert_norm = np.linalg.norm(pert)
      rescale_factor = pert_norm / self.epsilon
      pert = pert / max(1, rescale_factor)

    x = x_nat + pert
    x = np.clip(x, 0, 255)
    
    return x


if __name__ == '__main__':
  import json
  import sys
  import math

  config = vars(config_attack.get_args())

  tf.set_random_seed(config['tf_seed'])
  np.random.seed(config['np_seed'])

  model_file = tf.train.latest_checkpoint(config['model_dir'])
  print("config['model_dir']: ", config['model_dir'])
  if model_file is None:
    print('No model found')
    sys.exit()
      
  if 'adv_trained_tinyimagenet_finetuned_on_c10_upresize' in config['model_dir']:
    print("finetuned tinyimagenet MODEL")
    from model_new import ModelTinyImagenetSourceExtendedLogits
    full_source_model_x_input = tf.placeholder(tf.float32, shape = [None, 32, 32, 3])
    upresized_full_source_model_x_input = tf.image.resize_images(full_source_model_x_input, size=[64, 64])
    if config['dataset'] == 'cifar10':
      model = ModelTinyImagenetSourceExtendedLogits(mode='train', dataset='tinyimagenet', target_task_class_num=10, train_batch_size=config['eval_batch_size'], input_tensor=upresized_full_source_model_x_input)
    elif config['dataset'] == 'cifar100':
      model = ModelTinyImagenetSourceExtendedLogits(mode='train', dataset='tinyimagenet', target_task_class_num=100, train_batch_size=config['eval_batch_size'], input_tensor=upresized_full_source_model_x_input)
  
    model.x_input = full_source_model_x_input

    t_vars = tf.trainable_variables()
    source_model_vars = [var for var in t_vars if ('discriminator' not in var.name and 'classifier' not in var.name and 'target_task_logit' not in var.name)]
    source_model_target_logit_vars = [var for var in t_vars if 'target_task_logit' in var.name]
    source_model_saver = tf.train.Saver(var_list=source_model_vars)
    finetuned_source_model_vars = source_model_vars + source_model_target_logit_vars
    finetuned_source_model_saver = tf.train.Saver(var_list=finetuned_source_model_vars)
  elif 'finetuned_on_cifar100' in config['model_dir']:
    print("finetuned MODEL")
    from model_original_cifar_challenge import ModelExtendedLogits
    model = ModelExtendedLogits(mode='train', target_task_class_num=100, train_batch_size=config['eval_batch_size'])
    
    t_vars = tf.trainable_variables()
    source_model_vars = [var for var in t_vars if ('discriminator' not in var.name and 'classifier' not in var.name and 'target_task_logit' not in var.name)]
    source_model_target_logit_vars = [var for var in t_vars if 'target_task_logit' in var.name]
    source_model_saver = tf.train.Saver(var_list=source_model_vars)
    finetuned_source_model_vars = source_model_vars + source_model_target_logit_vars
    finetuned_source_model_saver = tf.train.Saver(var_list=finetuned_source_model_vars)
  elif ('adv_trained' in config['model_dir'] or 'naturally_trained' in config['model_dir'] or 'a_very_robust_model' in config['model_dir']):
    print("original challenge MODEL")
    from free_model_original import Model
    model = Model(mode='eval', dataset=config['dataset'], train_batch_size=config['eval_batch_size'])
  elif 'IGAM' in config['model_dir']:
    print("IGAM MODEL")
    from model_new import Model
    model = Model(mode='train', dataset=config['dataset'], train_batch_size=config['eval_batch_size'], normalize_zero_mean=True)
  else:
    print("other MODEL")
    from free_model import Model
    model = Model(mode='eval', dataset=config['dataset'], train_batch_size=config['eval_batch_size'])
  
  attack = LinfPGDAttack(model,
                         config['epsilon'],
                         config['num_steps'],
                         config['step_size'],
                         config['random_start'],
                         config['loss_func'],
                         dataset=config['dataset'])
  saver = tf.train.Saver()

  data_path = config['data_path']
  
  if config['dataset'] == 'cifar10':
    print("load cifar10 dataset")
    cifar = cifar10_input.CIFAR10Data(data_path)
  else:
    print("load cifar100 dataset")
    cifar = cifar100_input.CIFAR100Data(data_path)

  with tf.Session() as sess:
    # Restore the checkpoint
    if 'adv_trained_tinyimagenet_finetuned_on_c10_upresize' in config['model_dir']:
      sess.run(tf.global_variables_initializer())
      source_model_file = tf.train.latest_checkpoint("models/model_AdvTrain-igamsource-IGAM-tinyimagenet_b16")
      source_model_saver.restore(sess, source_model_file)   
      finetuned_source_model_file = tf.train.latest_checkpoint(config['model_dir'])
      finetuned_source_model_saver.restore(sess, finetuned_source_model_file)
    elif 'finetuned_on_cifar100' in config['model_dir']:
      sess.run(tf.global_variables_initializer())
      source_model_file = tf.train.latest_checkpoint("models/adv_trained")
      source_model_saver.restore(sess, source_model_file)   
      finetuned_source_model_file = tf.train.latest_checkpoint(config['model_dir'])
      finetuned_source_model_saver.restore(sess, finetuned_source_model_file)
    else:
      saver.restore(sess, model_file)

    # Iterate over the samples batch-by-batch
    num_eval_examples = config['num_eval_examples']
    eval_batch_size = config['eval_batch_size']
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

    x_adv = [] # adv accumulator

    print('Iterating over {} batches'.format(num_batches))

    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)

      x_batch = cifar.eval_data.xs[bstart:bend, :]
      y_batch = cifar.eval_data.ys[bstart:bend]

      if config['attack_norm'] == 'inf':  
        x_batch_adv = attack.perturb(x_batch, y_batch, sess)
      elif config['attack_norm'] == '2':
        x_batch_adv = attack.perturb_l2(x_batch, y_batch, sess)

      x_adv.append(x_batch_adv)

    print('Storing examples')
    path = config['store_adv_path']
    if path == None:
      model_name = config['model_dir'].split('/')[1]
      if config['attack_name'] == None:
        if config['dataset'] == 'cifar10':
            path = "attacks/{}_attack.npy".format(model_name)
        else:
            path = "attacks/{}_c100attack.npy".format(model_name)
      else:
        if config['dataset'] == 'cifar10':
            path = "attacks/{}_{}_attack.npy".format(model_name, config['attack_name'])
        else:
            path = "attacks/{}_{}_c100attack.npy".format(model_name, config['attack_name'])

      if config['attack_norm'] == '2':
        path = path.replace("attack.npy", "l2attack.npy")
    
    if not os.path.exists("attacks/"):
      os.makedirs("attacks/")

    x_adv = np.concatenate(x_adv, axis=0)
    np.save(path, x_adv)
    print('Examples stored in {}'.format(path))

  if config['save_eval_log']:
    if not os.path.exists("attack_log/"):
      os.makedirs("attack_log/")
    date_str = datetime.now().strftime("%d_%b")
    log_dir = "attack_log/" + date_str
    if not os.path.exists(log_dir):
      os.makedirs(log_dir)
    log_filename = path.split("/")[-1].replace('.npy', '.txt') 
    log_file_path = os.path.join(log_dir, log_filename)
    with open(log_file_path, "w") as f:
      f.write('Saved model name: {} \n'.format(model_file))
    print('Model name saved at ', log_file_path)
