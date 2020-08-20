from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import collections
import os
import datetime as dt
import cPickle
from model import *
import argparse

#########################################
parser = argparse.ArgumentParser(description='train')

## required
parser.add_argument('beta', type=float, help='2nd order regularizer')
parser.add_argument('dim', type=int, help='hidden dimension')

parser.add_argument('lr', type=float, help='learning rate')

parser.add_argument('name', type=str, default='earthquake', help='dataset name')

args = parser.parse_args()

b = args.beta
dim = args.dim
lr = args.lr
data_name = args.name

#####################################
if args.name == 'earthquake':
    N = 648
    sub_name = 'eq_1997-2018'
    dia = 13
elif args.name == 'email':
    N = 2634
    sub_name = 'enron_0'
    dia = 11
elif args.name == 'twitter':
    N = 393
    sub_name = 'safrica'
    dia = 8
elif args.name == 'rand-1':
    N = 291
    sub_name = 'rand_0'
    dia = 57
elif args.name == 'rand-2':
    N = 1599
    sub_name = 'rand_0'
    dia = 23
else:
    raise Exception('Unknown dataset. Please provide your own dataset.')


time_step = 32 # 32 events
n_window = 30
batch_size = 32
num_epochs = 10

num_layers_event = 2
hidden_size_event = dim
num_layers_bg = 2
hidden_size_bg = N
embedd_dim = dim

event_path = 'data/event_%s_train.pkl' % (sub_name)
model_path = 'checkpoint/%s' % (sub_name)
embedding_path = 'checkpoint/embedding.pkl'
shortest_path = 'data/shortest.pkl'
background_path = 'data/background_%s_train.pkl' % (sub_name)
timeseries_path = 'data/timeseries_%s.pkl' % (sub_name)

learning_rate = lr
max_lr_epoch = 5
lr_decay = 0.9

dropout=0.5
init_scale=0.05

is_training = True
print_iter = 10

# 2nd order stat regulation
dvg = 'kl'
diameter = dia # graph diameter
scale = 0.1   # kde scale
ord = 2       # norm order
beta = b    # regulation weight


######################
# initialization
#####################
tf.reset_default_graph()

with open(event_path, 'r') as file:
    event = cPickle.load(file)

with open(background_path, 'r') as file:
    background = cPickle.load(file)

# load shortest distance matrix
with open(shortest_path, 'r') as file:
    shortest = cPickle.load(file)

with open(timeseries_path, 'r') as file:
    base_rate = cPickle.load(file)

with tf.device('/device:GPU:0'):
    input_obj = Input(event, background, batch_size)

    embedd = tf.random_uniform([N, embedd_dim], -init_scale, init_scale)

    m = Model(embedd, base_rate, shortest, is_training, batch_size, time_step, n_window, N,
            hidden_size_event, num_layers_event, hidden_size_bg, num_layers_bg,
            dropout=dropout, init_scale=init_scale,
            dvg=dvg, diameter=diameter, scale=scale, ord=ord, beta=beta)

init_op = tf.global_variables_initializer()
orig_decay = lr_decay

saver = tf.train.Saver()

#print(tf.trainable_variables())

#######################
# training
#########################
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    sess.run([init_op])
    cst_1 = []
    cst_2 = []
    cst_3 = []

    for epoch in range(num_epochs):
        new_lr_decay = orig_decay ** max(epoch + 1 - max_lr_epoch, 0.0)
        m.assign_lr(sess, learning_rate * new_lr_decay)
        # print(m.learning_rate.eval(), new_lr_decay)

        for step in range(input_obj.epoch_size):
            current_state_event = np.zeros((num_layers_event, 2, batch_size, hidden_size_event))
            current_state_bg = np.zeros((num_layers_bg, 2, batch_size, hidden_size_bg))

            next_example, next_label, next_bg = input_obj.next_batch()

            if step % print_iter != 0:
                cost, _ = sess.run([m.cost, m.train_op],
                feed_dict={m.x:next_example, m.y:next_label, m.z:next_bg, m.init_state_event: current_state_event, m.init_state_bg: current_state_bg})
            else:
                cost, _, acc, cost_h, h_real, h_fake, dist_real, dist_fake = sess.run([m.cost,
                m.train_op, m.accuracy, m.h_loss, m.h_real, m.h_fake, m.dist_real, m.dist_fake],
                feed_dict={m.x:next_example, m.y:next_label, m.z:next_bg, m.init_state_event: current_state_event, m.init_state_bg: current_state_bg})

                print("Epoch {}, Step {}, cost: {:.3f}, cost 1st: {:.3f}, cost 2nd: {:.3f}, accuracy: {:.3f}".format(epoch, step, cost, cost-beta*cost_h, beta*cost_h, acc))
                cst_1.append(cost)
                cst_2.append(cost-beta*cost_h)
                cst_3.append(beta*cost_h)

                print('dist real')
                print(dist_real)
                print('dist fake')
                print(dist_fake)
                print('---------------')
        input_obj.shuffle()
        # save a model checkpoint
        #saver.save(sess, model_path, global_step=epoch)

        #print(sess.run(m.embedding)[0])
    # do a final save
    saver.save(sess, model_path)

    # write embedding
    with open(embedding_path, 'w') as file:
        cPickle.dump(sess.run(m.embedding), file)

    with open('./cst/cst_1_%s_%s_%s.pkl' % (beta, dim, lr), 'w') as file:
        cPickle.dump(np.array(cst_1), file)

    with open('./cst/cst_2_%s_%s_%s.pkl' % (beta, dim, lr), 'w') as file:
        cPickle.dump(np.array(cst_2), file)

    with open('./cst/cst_3_%s_%s_%s.pkl' % (beta, dim, lr), 'w') as file:
        cPickle.dump(np.array(cst_3), file)


