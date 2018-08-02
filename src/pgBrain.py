#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 17:53:30 2018

@author: Arpit
"""

import os.path, threading, time
import numpy as np
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras import backend as K

MIN_BATCH = 256
LEARNING_RATE = 5e-3
LOSS_V = .5			# v loss coefficient
LOSS_ENTROPY = .01 	# entropy coefficient

class Brain:
    train_queue = [ [], [], [], [], [] ]	# s, a, r, s', s' terminal mask
    lock_queue = threading.Lock()
    
    def __init__(self, name, game, **kwargs):
        self.filename = str(name) + '.h5'
        self.stateCnt, self.actionCnt = game.getStateActionCnt()
        
        self.gamma = kwargs['gamma']
        self.n_step = kwargs['n_step']
        self.gamma_n = self.gamma ** self.n_step
        self.min_batch = kwargs['min_batch'] if "min_batch" in kwargs else MIN_BATCH
        
        self.session = tf.Session()
        K.set_session(self.session)
        K.manual_variable_initialization(True)

        self.model = self._build_model() if "model" not in kwargs else kwargs['model']
        self.graph = self._build_graph()
        
        self.session.run(tf.global_variables_initializer())
        if "loadWeights" in kwargs and kwargs['loadWeights']:
            self.loadWeights()
        
        self.default_graph = tf.get_default_graph()
        self.default_graph.finalize()

    def _build_model(self):
        l_input = Input( batch_shape=(None, self.stateCnt) )
        l_dense = Dense(24, kernel_initializer='random_uniform', bias_initializer='random_uniform', 
                        activation='relu')(l_input)
        l_dense = Dense(24, kernel_initializer='random_uniform', bias_initializer='random_uniform', 
                        activation='relu')(l_dense)
        
        out_actions = Dense(self.actionCnt, activation='softmax')(l_dense)
        out_value   = Dense(1, activation='linear')(l_dense)
        
        model = Model(inputs=[l_input], outputs=[out_actions, out_value])
        model._make_predict_function()	# have to initialize before threading
        
        return model
    
    def _build_graph(self):
        
        if type(self.stateCnt) is tuple:
            s_t = tf.placeholder(tf.float32, shape=(None, *self.stateCnt[0:len(self.stateCnt)]))
        else:
            s_t = tf.placeholder(tf.float32, shape=(None, self.stateCnt))

        a_t = tf.placeholder(tf.float32, shape=(None, self.actionCnt))
        q_t = tf.placeholder(tf.float32, shape=(None, 1)) # not immediate, but discounted n step reward
        
        p, v = self.model(s_t)
        
        log_prob = tf.log( tf.reduce_sum(p * a_t, axis=1, keepdims=True) + 1e-10)
        advantage = q_t - v
        
        loss_policy = - log_prob * tf.stop_gradient(advantage)									# maximize policy
        loss_value  = LOSS_V * tf.square(advantage)												# minimize value error
        entropy = LOSS_ENTROPY * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1, keepdims=True)	# maximize entropy (regularization)
        
        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)
        
        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99)
        minimize = optimizer.minimize(loss_total)
        
        return s_t, a_t, q_t, minimize

    def optimize(self):
        if len(self.train_queue[0]) < self.min_batch:
            time.sleep(0)	# yield
            return

        with self.lock_queue:
            if len(self.train_queue[0]) < self.min_batch:	# more thread could have passed without lock
                return 									# we can't yield inside lock

            s, a, r, s_, s_mask = self.train_queue
            self.train_queue = [ [], [], [], [], [] ]

        s = np.stack(s)
        a = np.vstack(a)
        r = np.vstack(r)
        s_ = np.stack(s_)
        s_mask = np.vstack(s_mask)

        if len(s) > 5*self.min_batch: print("Optimizer alert! Minimizing batch of %d" % len(s))
        
        v = self.predict_v(s_)
        q = r + self.gamma_n * v * s_mask	# set v to 0 where s_ is terminal state
        
        s_t, a_t, q_t, minimize = self.graph
        self.session.run(minimize, feed_dict={s_t: s, a_t: a, q_t: q})

    def train_push(self, sample):
        s = sample[0]
        a = sample[1]
        r = sample[2]
        s_ = sample[3]

        with self.lock_queue:
            self.train_queue[0].append(s)
            self.train_queue[1].append(a)
            self.train_queue[2].append(r)

            if s_ is None:
                self.train_queue[3].append(np.zeros(self.stateCnt))
                self.train_queue[4].append(0.)
            else:	
                self.train_queue[3].append(s_)
                self.train_queue[4].append(1.)
                
    def predict(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return p, v

    def predict_p(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)		
            return p

    def predict_v(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)		
            return v
        
    def save(self):
        self.model.save(self.filename)
        
    def loadWeights(self):
        if os.path.exists(self.filename):
            print (self.filename + " weights loaded")
            self.model.load_weights(self.filename)
            
    def load(self):
        if os.path.exists(self.filename):
            print (self.filename + " model loaded")
            self.model = load_model(self.filename)