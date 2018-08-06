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
from players.pgPlayer import PGPlayer
from players.hoomanPlayer import HoomanPlayer
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten
from mathEq import MathEq
from keras.layers import Input, Dense
from keras.models import Model
from pgBrain import Brain


scoreAgent = False
loadWeights = True
filename = "pgbraint3"

game = T3Game()

l_input = Input( batch_shape=(None, game.stateCnt) )
l_dense = Dense(24, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation='relu')(l_input)
l_dense = Dense(24, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation='relu')(l_dense)
out_actions = Dense(game.actionCnt, activation='softmax')(l_dense)
out_value   = Dense(1, activation='linear')(l_dense)
model = Model(inputs=[l_input], outputs=[out_actions, out_value])
model._make_predict_function()	# have to initialize before threading

brain = Brain(filename, game, model=model, loadWeights=loadWeights)

eq1 = MathEq({"min":0.05, "max":0, "lambda":0})
eq2 = MathEq({"min":0.05, "max":0.05, "lambda":0})

p1 = PGPlayer(1, game, brain=brain, eEq=eq1, sampling=False)

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

