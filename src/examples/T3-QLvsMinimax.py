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

#Example 1
game = T3Game()

#opt: set custom ANN model
ann = Sequential()
ann.add(Dense(units = 12,
              kernel_initializer='random_uniform',
              bias_initializer='random_uniform',
              activation = 'relu',
              input_dim = game.stateCnt))
ann.add(Dense(units = 12,
              kernel_initializer='random_uniform',
              bias_initializer='random_uniform',
              activation = 'relu'))
ann.add(Dense(units = game.actionCnt,
              kernel_initializer='random_uniform',
              bias_initializer='random_uniform',
              activation = 'linear'))
ann.compile(optimizer = 'rmsprop', loss = 'logcosh', metrics = ['accuracy'])

eq1 = MathEq({"min":0.05, "max":0.3, "lambda":0.001})
eq2 = MathEq({"min":0, "max":0.05, "lambda":0.0001})

p1 = QPlayer(1, game, model=ann, eEq=eq1)
p2 = MinimaxT3Player(2, game, eEq=eq2)
env = Environment(game, p1, p2)
env.run()

#Example 2
game = T3Game()

#OPT: set custom ANN model
ann = Sequential()
ann.add(Dense(units = 12,
              kernel_initializer='random_uniform',
              bias_initializer='random_uniform',
              activation = 'relu',
              input_dim = game.stateCnt))
ann.add(Dense(units = 12,
              kernel_initializer='random_uniform',
              bias_initializer='random_uniform',
              activation = 'relu'))
ann.add(Dense(units = game.actionCnt,
              kernel_initializer='random_uniform',
              bias_initializer='random_uniform',
              activation = 'linear'))
ann.compile(optimizer = 'rmsprop', loss = 'logcosh', metrics = ['accuracy'])

p1 = QPlayer(1, game, model=ann)

#OPT: set custom exploration rate equation
eq = MathEq({"min":0, "max":0.3, "lambda":0.001})
p2 = MinimaxT3Player(2, game, eEq=eq)
env = Environment(game, p1, p2)
env.run()

#Example 3 with convolutional network
game = T3Game(3, isConv=True)

ann = Sequential()
ann.add(Convolution2D(8, (2, 2), padding='valid', strides=(1, 1), activation='relu', input_shape=game.stateCnt, data_format="channels_first"))
#ann.add(MaxPooling2D(pool_size = (2, 2), strides=1, padding='same', data_format="channels_first"))
ann.add(Flatten())
ann.add(Dense(units = 32, activation = 'relu'))
ann.add(Dense(units = game.actionCnt, activation = 'linear'))
ann.compile(optimizer = 'rmsprop', loss = 'logcosh', metrics = ['accuracy'])

debug = False
p1 = QPlayer(1, game, model=ann, debug=debug)
eq = MathEq({"min":0, "max":0.3, "lambda":0.001})
p2 = MinimaxT3Player(2, game)
env = Environment(game, p1, p2, debug)
env.run()

if debug:
    w1 = env.p1.brain.model.get_weights()
    sample1 = env.p1.memory.sample(64)
    sampleG1 = env.p1.goodMemory.sample(64)
    locals().update(env.p1.logs)
