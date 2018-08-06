#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 23:11:45 2018

@author: Arpit
"""

from games.c4Game import C4Game
from environment import Environment
from players.qPlayer import QPlayer
from keras.models import Sequential
from keras.layers import Dense
from mathEq import MathEq
from memory.pMemory import PMemory

loadWeights = False
randBestMoves = False

N_STEP_RETURN = 6

memory = PMemory(100000)
goodMemory = PMemory(100000)
threads = []
ROWS = 6
COLUMNS = 7
BATCH_SIZE = 64

game = C4Game(ROWS, COLUMNS, name="dummy")

ann = Sequential()
ann.add(Dense(units = 45, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'relu', input_dim = game.stateCnt))
ann.add(Dense(units = 25, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'relu'))
ann.add(Dense(units = 16, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'relu'))
ann.add(Dense(units = game.actionCnt, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'linear'))
ann.compile(optimizer = 'rmsprop', loss = 'logcosh', metrics = ['accuracy'])

ann2 = Sequential()
ann2.add(Dense(units = 45, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'relu', input_dim = game.stateCnt))
ann2.add(Dense(units = 25, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'relu'))
ann2.add(Dense(units = 16, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'relu'))
ann2.add(Dense(units = game.actionCnt, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'linear'))
ann2.compile(optimizer = 'rmsprop', loss = 'logcosh', metrics = ['accuracy'])

eq1 = MathEq({"min":0.10, "max":1, "lambda":0.0001})
eq2 = MathEq({"min":0.10, "max":1, "lambda":0.0001})

p1 = QPlayer(1, game, model=ann, tModel=ann2, memory=memory, goodMemory=goodMemory, eEq=eq1, batch_size=BATCH_SIZE, n_step=N_STEP_RETURN)
p2 = QPlayer(2, game, model=ann, tModel=ann2, memory=memory, goodMemory=goodMemory, eEq=eq2, batch_size=BATCH_SIZE, n_step=N_STEP_RETURN)
env = Environment(game, p1, p2)
env.run()