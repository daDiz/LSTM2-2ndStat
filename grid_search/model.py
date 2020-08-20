from __future__ import print_function, absolute_import, division
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import os
import networkx as nx
import cPickle
import random

def load_adj(graph_name, normalized=True):
    g = nx.read_edgelist(graph_name, delimiter=' ', nodetype=int)
    A = nx.adj_matrix(g).todense()

    if normalized:
        A = A + np.eye(A.shape[0])
        rowsum = np.array(A.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        A = d_mat_inv_sqrt.dot(A).dot(d_mat_inv_sqrt)

    return A


# x1 : [batch_size, time_step, N], current
# x2 : [batch_size, time_step, N], next
# d : [N, N], pairwise distance
def hd(x1, x2, d):
    N = d.get_shape().as_list()[0]

    right = tf.reshape(x1, [-1,N])
    #right = tf.transpose(right, perm=[1,0]) # [N,-1]

    left = tf.reshape(x2, [-1,N])

    mid = tf.matmul(d, right, transpose_b=True) # [N,-1]
    h = tf.matmul(left, mid) # [batch_size*time_step, batch_size*time_step]
    h = tf.diag_part(h)

    return h

# KL divergence
def kl_dvg(p, q, sigma=1e-3):
    return tf.reduce_sum(p * tf.log((p+sigma)/(q+sigma)))

# JS divergence
def js_dvg(p, q, sigma=1e-3):
    m = 0.5*(p+q)
    return 0.5*kl_dvg(p,m,sigma)+0.5*kl_dvg(q,m,sigma)

class Input(object):
    def __init__(self, event, background, batch_size):
        self.event = event
        self.background = background
        self.batch_size = batch_size
        self.num_epochs = 0

        self.n_samples = len(self.event)
        self.epoch_size = self.n_samples // self.batch_size

        self.cur = 0

    def shuffle(self):
        self.cur = 0
        idx = np.arange(self.n_samples, dtype=int)
        random.shuffle(idx)
        self.event = self.event[idx]
        self.background = self.background[idx]
        self.num_epochs += 1

    def next_batch(self):
        if self.cur + self.batch_size > self.n_samples:
            raise Exception('epoch exhausts')


        # [batch_size, n_window, N]
        bg = self.background[self.cur:self.cur+self.batch_size]

        event_ = self.event[self.cur:self.cur+self.batch_size,:]
        x = event_[:,:-1] # [batch_size, time_step]
        y = event_[:,1:]  # [batch_size, time_step]

        self.cur += self.batch_size

        return x, y, bg


class Model(object):
    def __init__(self, embedd, base_rate, shortest, is_training, batch_size, time_step, n_window, N,
                hidden_size_event, num_layers_event, hidden_size_bg, num_layers_bg,
                dropout=0.5, init_scale=0.05,
                dvg='norm', diameter=19, scale=0.5, ord=2, beta=1.0): # dvg = 'norm', 'kl', 'js'
        if is_training: # embedd should be tf.random_uniform([N, hidden_size],
                        #                                   -init_scale, init_scale])
            self.embedding = tf.Variable(initial_value=embedd, dtype=tf.float32, trainable=True,
            name='embedding')
        else: # load learned embedding
            self.embedding = tf.Variable(initial_value=embedd, dtype=tf.float32, trainable=False,
            name='embedding')

        self.br = tf.Variable(initial_value=base_rate, dtype=tf.float32, trainable=True,
        name='base_rate')

        self.shortest = tf.Variable(initial_value=shortest, dtype=tf.float32, trainable=False, name='shortest')


        self.is_training = is_training
        self.batch_size = batch_size
        self.time_step = time_step
        self.n_window = n_window
        self.hidden_size_event = hidden_size_event
        self.hidden_size_bg = hidden_size_bg
        self.N = N
        self.num_layers_event = num_layers_event
        self.num_layers_bg = num_layers_bg

        self.dropout = dropout
        self.init_scale = init_scale

        self.diameter = diameter
        self.scale = scale
        self.ord = ord
        self.beta = beta

        self.x = tf.placeholder(tf.int32, [batch_size, time_step])
        self.y = tf.placeholder(tf.int32, [batch_size, time_step])
        self.z = tf.placeholder(tf.int32, [batch_size])

        inputs = tf.nn.embedding_lookup(self.embedding, self.x)
        br = tf.nn.embedding_lookup(self.br, self.z) # [batch_size, window_size, N]

        events_cur = tf.one_hot(self.x, self.N, on_value=1.0, off_value=0.0,dtype=tf.float32)
        events_nxt = tf.one_hot(self.y, self.N, on_value=1.0, off_value=0.0,dtype=tf.float32)


        if is_training and dropout < 1:
            inputs = tf.nn.dropout(inputs, dropout)
            br = tf.nn.dropout(br, dropout)

        ####################
        # event
        ################
        with tf.variable_scope("event") as scope:
            self.init_state_event = tf.placeholder(tf.float32,
                                            [num_layers_event, 2, self.batch_size, self.hidden_size_event])

            state_per_layer_list_event = tf.unstack(self.init_state_event, axis=0)

            rnn_tuple_state_event = tuple(
                [tf.contrib.rnn.LSTMStateTuple(state_per_layer_list_event[idx][0],
                state_per_layer_list_event[idx][1])
                for idx in range(num_layers_event)]
            )

            cell_event = tf.contrib.rnn.LSTMCell(hidden_size_event, forget_bias=1.0)

            if is_training and dropout < 1:
                cell_event = tf.contrib.rnn.DropoutWrapper(cell_event, output_keep_prob=dropout)

            if num_layers_event > 1:
                cell_event = tf.contrib.rnn.MultiRNNCell([cell_event for _ in range(num_layers_event)],
                                                        state_is_tuple=True)


            output_event, _ = tf.nn.dynamic_rnn(cell_event, inputs, dtype=tf.float32,
                                                initial_state=rnn_tuple_state_event)

        ########################
        # background
        #########################
        with tf.variable_scope("background") as scope:
            self.init_state_bg = tf.placeholder(tf.float32,
                                        [num_layers_bg, 2, self.batch_size, self.hidden_size_bg])

            state_per_layer_list_bg = tf.unstack(self.init_state_bg, axis=0)

            rnn_tuple_state_bg = tuple(
                [tf.contrib.rnn.LSTMStateTuple(state_per_layer_list_bg[idx][0],
                state_per_layer_list_bg[idx][1])
                for idx in range(num_layers_bg)]
            )

            cell_bg = tf.contrib.rnn.LSTMCell(hidden_size_bg, forget_bias=1.0)

            if is_training and dropout < 1:
                cell_bg = tf.contrib.rnn.DropoutWrapper(cell_bg, output_keep_prob=dropout)

            if num_layers_bg > 1:
                cell_bg = tf.contrib.rnn.MultiRNNCell([cell_bg for _ in range(num_layers_bg)],
                state_is_tuple=True)


            output_bg, _ = tf.nn.dynamic_rnn(cell_bg, br, dtype=tf.float32,
                                        initial_state=rnn_tuple_state_bg)

            # take the last step
            output_bg = tf.slice(output_bg, [0,self.n_window-1,0], [-1,-1,-1]) # [batch_size, 1, hidden_size_bg]
            output_bg = tf.tile(output_bg, [1, self.time_step, 1])


        # reshape to (batch_size*time_step,hidden_size)
        output = tf.concat([output_event, output_bg], axis=2)
        hidden_size = hidden_size_event + hidden_size_bg
        output = tf.reshape(output, [-1, hidden_size])

        # squeeze results from two lstms to ensure in the same scale
        output = tf.nn.tanh(output)

        softmax_w = tf.Variable(tf.random_uniform([hidden_size, N], -self.init_scale,self.init_scale))
        softmax_b = tf.Variable(tf.random_uniform([N], -self.init_scale, self.init_scale))
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)

        # reshape logits to be a 3-D tensor for sequence loss
        logits = tf.reshape(logits, [self.batch_size, -1, N])

        # use contrib sequence loss and average over the batches
        loss = tf.contrib.seq2seq.sequence_loss(logits, self.y,
                                    tf.ones([self.batch_size, self.time_step], dtype=tf.float32),
                                    average_across_timesteps=False,
                                    average_across_batch=True)


        # 2nd order stat
        p = tf.nn.softmax(logits, axis=2)
        h_fake = hd(events_cur, p, self.shortest)
        h_real = hd(events_cur, events_nxt, self.shortest)

        #self.h_fake = tf.reshape(h_fake, [self.batch_size, self.time_step])
        #self.h_real = tf.reshape(h_real, [self.batch_size, self.time_step])
        self.h_fake = h_fake
        self.h_real = h_real

        dist_fake = []
        dist_real = []

        tfd = tfp.distributions
        for i in range(diameter):
            dist = tf.distributions.Normal(loc=float(i), scale=self.scale)
            #dist = tfd.TruncatedNormal(loc=float(i), scale=self.scale,
            #                            low=float(i)-0.5, high=float(i)+0.5)

            dist_fake.append(tf.reduce_sum(dist.prob(h_fake)))
            dist_real.append(tf.reduce_sum(dist.prob(h_real)))

        dist_fake = tf.stack(dist_fake)
        dist_real = tf.stack(dist_real)

        self.dist_fake = tf.divide(dist_fake, tf.reduce_sum(dist_fake))
        self.dist_real = tf.divide(dist_real, tf.reduce_sum(dist_real))

        if dvg == 'norm':
            self.h_loss = tf.norm(dist_fake-dist_real, ord=self.ord)
        elif dvg == 'kl':
            self.h_loss = kl_dvg(self.dist_fake, self.dist_real, sigma=1e-6)
        elif dvg == 'js':
            self.h_loss = js_dvg(self.dist_fake, self.dist_real, sigma=1e-6)
        else:
            raise Exception('unknown divergence. select form norm, kl, js.')

        # update the cost
        self.cost = tf.reduce_sum(loss) + self.beta * self.h_loss

        # get the prediction accuracy
        self.softmax_out = tf.nn.softmax(tf.reshape(logits, [-1, N]))
        self.predict = tf.cast(tf.argmax(self.softmax_out, axis=1), tf.int32)
        correct_prediction = tf.equal(self.predict, tf.reshape(self.y,[-1]))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        if not is_training:
            return

        self.learning_rate = tf.Variable(0.0, trainable=False)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        # optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())
        # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

        self.new_lr = tf.placeholder(tf.float32, shape=[])
        self.lr_update = tf.assign(self.learning_rate, self.new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self.lr_update, feed_dict={self.new_lr: lr_value})




