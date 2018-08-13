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
from brains.qBrain import QBrain

GAMMA = 0.90
N_STEP_RETURN = 3

loadWeights = False
memory = PMemory(1000)
goodMemory = PMemory(1000)
isConv = False

game = T3Game(3, name="dummy", isConv=isConv)
ann = Sequential()
if isConv:
    ann.add(Convolution2D(16, (2, 2), padding='valid', strides=(1, 1), 
            activation='relu', input_shape=game.stateCnt, data_format="channels_first"))
    ann.add(Flatten())
else:
    ann.add(Dense(units = 24, kernel_initializer='random_uniform', bias_initializer='random_uniform',
                  activation = 'relu', input_dim = game.stateCnt))
    
ann.add(Dense(units = 24, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'relu'))
ann.add(Dense(units = game.actionCnt, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'linear'))
ann.compile(optimizer = 'rmsprop', loss = 'logcosh', metrics = ['accuracy'])

brain = QBrain('t3AsyncDQN', game, model=ann, loadWeights=loadWeights)

epsilons = {}
epsilons[1] = {"min":0.05, "max":0.05, "lambda":0}
epsilons[2] = {"min":0.05, "max":0.25, "lambda":0}
epsilons[3] = {"min":0.05, "max":0.35, "lambda":0}
epsilons[4] = {"min":0.05, "max":0.45, "lambda":0}

eq2 = MathEq({"min":0, "max":0.05, "lambda":0})
env_config = {"ePlotFlag":False, "evaluate":True, "evalPer":200}

i = 1
threads = []
while i <= 4:
    name = "asyncDQNT3-1" + str(i)
    game = T3Game(3, name=name, isConv=isConv)
    p1 = QPlayer(name, game, brain=brain, eEq=MathEq(epsilons[i]), memory=memory, 
                 goodMemory=goodMemory, targetNet=False, gamma=GAMMA, n_step=N_STEP_RETURN)
    p2 = MinimaxT3Player(2, game, eEq=eq2)

    if  False and i == 1:
        env = Environment(game, p1, p2, **env_config)
    else:
        env = Environment(game, p1, p2, ePlotFlag=False)
        
    threads.append(MyThread(env))
    i += 1

for t in threads:
    t.start()