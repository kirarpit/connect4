#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 16:55:37 2018

@author: Arpit
"""

from games.c4Game import C4Game
from environment import Environment
from players.qPlayer import QPlayer
from keras.models import Sequential
from keras.layers import Dense
from players.minimaxC4Player import MinimaxC4Player
from mathEq import MathEq

game = C4Game(4, 5)

ann = Sequential()
ann.add(Dense(units = 36, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'relu', input_dim = game.stateCnt))
ann.add(Dense(units = 36, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'relu'))
ann.add(Dense(units = game.actionCnt, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'linear'))
ann.compile(optimizer = 'rmsprop', loss = 'logcosh', metrics = ['accuracy'])

p1 = QPlayer(1, game, debug=True, model=ann)
p1.brain.load("test2")

eq1 = MathEq({"min":0.05, "max":0.05, "lambda":0})
p2 = MinimaxC4Player(2, game, eEq=eq1)

env = Environment(game, p1, p2, debug=True)

while game.gameCnt < 999:
    env.runGame()
    
env.printEnv()