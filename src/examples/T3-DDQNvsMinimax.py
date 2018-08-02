#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 20:40:00 2018

@author: Arpit
"""

from games.t3Game import T3Game
from environment import Environment
from players.minimaxT3Player import MinimaxT3Player
from players.qPlayer import QPlayer
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten
from mathEq import MathEq

game = T3Game()
BATCH_SIZE = 64

#opt: set custom ANN model
ann = Sequential()
ann.add(Dense(units = 24, kernel_initializer='random_uniform', bias_initializer='random_uniform',
              activation = 'relu',
              input_dim = game.stateCnt))
ann.add(Dense(units = 24, kernel_initializer='random_uniform', bias_initializer='random_uniform',
              activation = 'relu'))
ann.add(Dense(units = game.actionCnt, kernel_initializer='random_uniform', bias_initializer='random_uniform',
              activation = 'linear'))
ann.compile(optimizer = 'rmsprop', loss = 'logcosh', metrics = ['accuracy'])

eq1 = MathEq({"min":0.10, "max":0.10, "lambda":0})
eq2 = MathEq({"min":0, "max":0, "lambda":0})

p1 = QPlayer(1, game, model=ann, eEq=eq1, batch_size=BATCH_SIZE)
p2 = MinimaxT3Player(2, game, eEq=eq2)
env = Environment(game, p1, p2)
env.run()