#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 19:20:07 2018

@author: Arpit
"""
from games.t3Game import T3Game
from environment import Environment
from players.minimaxT3Player import MinimaxT3Player as M2T3
from players.qPlayer import QPlayer
from players.hoomanPlayer import HoomanPlayer
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten
from mathEq import MathEq

scoreAgent = True
loadWeights = True
game = T3Game()

model = Sequential()
model.add(Dense(units = 12, kernel_initializer='random_uniform', bias_initializer='random_uniform',
              activation = 'relu',
              input_dim = game.stateCnt))
model.add(Dense(units = 12, kernel_initializer='random_uniform', bias_initializer='random_uniform',
              activation = 'relu'))
model.add(Dense(units = game.actionCnt, kernel_initializer='random_uniform', bias_initializer='random_uniform',
              activation = 'linear'))
model.compile(optimizer = 'rmsprop', loss = 'logcosh', metrics = ['accuracy'])

eq1 = MathEq({"min":0.05, "max":0, "lambda":0})
eq2 = MathEq({"min":0.05, "max":0.05, "lambda":0})

p1 = QPlayer("1", game, eEq=eq1, model=model, targetNet=False, loadWeights=loadWeights)

if scoreAgent:
    p2 = M2T3(1, game, eEq=eq2)
else:
    p2 = HoomanPlayer(2, game)
    
env = Environment(game, p1, p2, training=False, observing=False)

if scoreAgent:
    while game.gameCnt < 999:
        env.runGame()
else:
    env.run()
    
env.printEnv()

