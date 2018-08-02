#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 00:37:35 2018

@author: Arpit
"""

from games.t3Game import T3Game
from environment import Environment
from players.minimaxT3Player import MinimaxT3Player
from players.qPlayer import QPlayer
from keras.models import Sequential
from keras.layers import Convolution2D, Flatten
from keras.layers import Dense
from mathEq import MathEq
from myThread import MyThread
from memory.pMemory import PMemory
from brain import Brain

memory = PMemory(500)
goodMemory = PMemory(500)
isConv = False
threads = []

game = T3Game(3, name="dummy", isConv=isConv)

ann = Sequential()

if isConv:
    ann.add(Convolution2D(16, (2, 2), padding='valid', strides=(1, 1), 
                          activation='relu', input_shape=game.stateCnt, data_format="channels_first"))
    ann.add(Flatten())
else:
    ann.add(Dense(units = 12, kernel_initializer='random_uniform', bias_initializer='random_uniform',
                  activation = 'relu', input_dim = game.stateCnt))
    
ann.add(Dense(units = 12, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'relu'))
ann.add(Dense(units = game.actionCnt, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'linear'))
ann.compile(optimizer = 'rmsprop', loss = 'logcosh', metrics = ['accuracy'])

brain = Brain('t3infmem', game, model=ann)

config = {}
config[1] = {"min":0.05, "max":0.05, "lambda":0}
config[2] = {"min":0.05, "max":0.25, "lambda":0}
config[3] = {"min":0.05, "max":0.35, "lambda":0}
config[4] = {"min":0.05, "max":0.45, "lambda":0}

eq2 = MathEq({"min":0, "max":0.05, "lambda":0})

i = 1
threads = []
while i <= 4:
    name = "asyncDQNT3" + str(i)
    game = T3Game(3, name=name, isConv=isConv)
    p1 = QPlayer(name, game, brain=brain, eEq=MathEq(config[i]), memory=memory, goodMemory=goodMemory, targetNet=False)
    p2 = MinimaxT3Player(2, game, eEq=eq2)

    env = Environment(game, p1, p2, ePlot=False)
    threads.append(MyThread(env))
    i += 1

for t in threads:
    t.start()