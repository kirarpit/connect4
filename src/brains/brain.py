#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 12:30:56 2018

@author: Arpit
"""
import os.path
from keras.models import load_model
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, LeakyReLU, add
from keras import regularizers
import tensorflow as tf

def softmax_cross_entropy_with_logits(y_true, y_pred):
    p = y_pred
    pi = y_true
    
    zero = tf.zeros(shape = tf.shape(pi), dtype=tf.float32)
    where = tf.equal(pi, zero)
    negatives = tf.fill(tf.shape(pi), -100.0)
    p = tf.where(where, negatives, p)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels = pi, logits = p)
    
    return loss

class Brain:
    def __init__(self, name, game, **kwargs):
        self.name = name
        self.stateCnt, self.actionCnt = game.getStateActionCnt()
        self.filename = str(name) + '.h5'
        
        self.conv = True if type(self.stateCnt) is tuple else False
        
        self.hidden_layers = kwargs['hidden_layers'] if "hidden_layers" in kwargs else None
        self.batch_size = kwargs['batch_size'] if "batch_size" in kwargs else 64
        self.epochs = kwargs['epochs'] if "epochs" in kwargs else 1
        self.reg_const = kwargs['reg_const'] if "reg_const" in kwargs else 1e-4
        self.momentum = kwargs['momentum'] if "momentum" in kwargs else 0.9
        self.learning_rate = kwargs['learning_rate'] if "learning_rate" in kwargs else 1e-3

        self.gamma = kwargs['gamma'] if "gamma" in kwargs else 0.99
        self.n_step = kwargs['n_step'] if "n_step" in kwargs else 1
        self.gamma_n = self.gamma ** self.n_step
        self.min_batch = kwargs['min_batch'] if "min_batch" in kwargs else 256
        
        self.model = kwargs['model'] if "model" in kwargs else self._build_model()

    def conv_layer(self, x, filters, kernel_size):
        x = Conv2D(filters = filters, kernel_size = kernel_size, data_format="channels_first",
                   padding = 'same', use_bias=False, activation='linear', 
                   kernel_regularizer = regularizers.l2(self.reg_const))(x)
        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)
        return x
    
    def residual_layer(self, input_block, filters, kernel_size):
        x = self.conv_layer(input_block, filters, kernel_size)
        x = Conv2D(filters = filters, kernel_size = kernel_size, data_format="channels_first",
                   padding = 'same', use_bias=False, activation='linear',
                   kernel_regularizer = regularizers.l2(self.reg_const))(x)
        x = BatchNormalization(axis=1)(x)
        x = add([input_block, x])
        x = LeakyReLU()(x)
        return x
    
    def value_head(self, x):
        x = self.conv_layer(x, 1, (1,1))
        x = Flatten()(x)
        x = Dense(self.actionCnt, use_bias=False, activation='linear',
                  kernel_regularizer=regularizers.l2(self.reg_const))(x)
        x = LeakyReLU()(x)
        x = Dense(1, use_bias=False, activation='tanh',
                  kernel_regularizer=regularizers.l2(self.reg_const), name = 'value_head')(x)
        return x
    
    def policy_head(self, x):
        x = self.conv_layer(x, 2, (1,1))
        x = Flatten()(x)
        x = Dense(self.actionCnt, use_bias=False, activation='softmax',
                  kernel_regularizer=regularizers.l2(self.reg_const), name = 'policy_head')(x)
        return x
    
    def predict(self, s):
        return self.model.predict(s)
    
    def predict_p(self, s):
        result = self.predict(s)
        if len(result) > 1:
            return result[0]
        else:
            return result

    def get_weights(self):
        return self.model.get_weights()
    
    def set_weights(self, weights):
        self.model.set_weights(weights)
    
    def load_weights(self, filename=None):
        filename = self.getFileName(filename)

        if os.path.exists(filename):
            print (filename + " weights loaded")
            self.model.load_weights(filename)
        else:
            print("Error: file " + filename + " not found")

    def save(self, filename=None):
        filename = self.getFileName(filename)
        self.model.save(filename)
        
    def getFileName(self, filename):
        if filename is not None:
            filename = str(filename) + '.h5'
        else:
            filename = self.filename

        return filename

    @staticmethod
    def load_model(filename):
        filename = str(filename) + '.h5'

        if os.path.exists(filename):
            model = load_model(filename, custom_objects={'softmax_cross_entropy_with_logits': softmax_cross_entropy_with_logits})
            print(filename + " model loaded")
        else:
            print("Error: file " + filename + " not found")
            
        return model
