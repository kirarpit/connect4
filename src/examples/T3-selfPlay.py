#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 19:02:25 2018

@author: Arpit
"""

from games.t3Game import T3Game
from environment import Environment
from players.qPlayer import QPlayer
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten
from mathEq import MathEq
from keras import optimizers

game = T3Game()
BATCH_SIZE = 64
N_STEP_RETURN = 3

#opt: set custom ANN model
ann = Sequential()
ann.add(Dense(units = 12, kernel_initializer='random_uniform', bias_initializer='random_uniform',
              activation = 'relu',
              input_dim = game.stateCnt))
ann.add(Dense(units = 12, kernel_initializer='random_uniform', bias_initializer='random_uniform',
              activation = 'relu'))
ann.add(Dense(units = game.actionCnt, kernel_initializer='random_uniform', bias_initializer='random_uniform',
              activation = 'linear'))
#rmsprop = optimizers.RMSprop(lr=0.005, decay=0.99)
ann.compile(optimizer = 'rmsprop', loss = 'logcosh', metrics = ['accuracy'])

ann2 = Sequential()
ann2.add(Dense(units = 12, kernel_initializer='random_uniform', bias_initializer='random_uniform',
              activation = 'relu',
              input_dim = game.stateCnt))
ann2.add(Dense(units = 12, kernel_initializer='random_uniform', bias_initializer='random_uniform',
              activation = 'relu'))
ann2.add(Dense(units = game.actionCnt, kernel_initializer='random_uniform', bias_initializer='random_uniform',
              activation = 'linear'))
#rmsprop = optimizers.RMSprop(lr=0.005, decay=0.99)
ann2.compile(optimizer = 'rmsprop', loss = 'logcosh', metrics = ['accuracy'])


eq1 = MathEq({"min":0.10, "max":0.10, "lambda":0})
eq2 = MathEq({"min":0.10, "max":0.10, "lambda":0})

p1 = QPlayer(1, game, model=ann, tModel=ann2, eEq=eq1, batch_size=BATCH_SIZE, n_step=N_STEP_RETURN)
p2 = QPlayer(2, game, model=ann, tModel=ann2, eEq=eq1, batch_size=BATCH_SIZE, n_step=N_STEP_RETURN)
env = Environment(game, p1, p2)
env.run()