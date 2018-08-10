#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 23:11:45 2018

@author: Arpit
"""

from games.c4Game import C4Game
from environment import Environment
from players.zeroPlayer import ZeroPlayer
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from memory.dictTree import DictTree

game = C4Game(6, 7)
simCnt = 100
iterCnt = 100
tree = DictTree()
LR = 5e-3
load_weights = False

l_input = Input( batch_shape=(None, game.stateCnt) )
l_dense = Dense(24, kernel_initializer='random_uniform', bias_initializer='random_uniform', 
                activation='relu')(l_input)
l_dense = Dense(24, kernel_initializer='random_uniform', bias_initializer='random_uniform', 
                activation='relu')(l_dense)

out_actions = Dense(game.actionCnt, activation='softmax', name="P")(l_dense)
out_value   = Dense(1, activation='linear', name="V")(l_dense)

model = Model(inputs=[l_input], outputs=[out_actions, out_value])
model.compile(loss=['categorical_crossentropy','mean_squared_error'], 
              optimizer=Adam(LR))

p1 = ZeroPlayer(1, game, model=model, tree=tree, simCnt=simCnt, iterCnt=iterCnt, load_weights=load_weights)
p2 = ZeroPlayer(2, game, model=model, tree=tree, simCnt=simCnt, iterCnt=iterCnt, load_weights=load_weights)
env = Environment(game, p1, p2)
env.run()