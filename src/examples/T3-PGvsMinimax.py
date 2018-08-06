#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 22:00:44 2018

@author: Arpit
"""

from games.t3Game import T3Game
from environment import Environment
from players.minimaxT3Player import MinimaxT3Player
from players.pgPlayer import PGPlayer
from mathEq import MathEq
from pgBrain import Brain
from optimizer import Optimizer
from myThread import MyThread
from keras.layers import Input, Dense
from keras.layers import Convolution2D, Flatten
from keras.models import Model

GAMMA = 0.8
N_STEP_RETURN = 1
MIN_BATCH = 125
isConv = False
loadWeights = True
filename = "pgbraint3"

#Example 1
game = T3Game(3, isConv=isConv)

if isConv:
    l_input = Input( shape=game.stateCnt )
    l_dense = Convolution2D(8, (2, 2), strides=(1,1), padding='valid', activation='relu', data_format="channels_first")(l_input)
    l_dense = Flatten()(l_dense)
else:
    l_input = Input( batch_shape=(None, game.stateCnt) )
    l_dense = Dense(24, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation='relu')(l_input)

l_dense = Dense(24, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation='relu')(l_dense)

out_actions = Dense(game.actionCnt, activation='softmax')(l_dense)
out_value   = Dense(1, activation='linear')(l_dense)

model = Model(inputs=[l_input], outputs=[out_actions, out_value])
model._make_predict_function()	# have to initialize before threading

brain = Brain(filename, game, model=model, loadWeights=loadWeights, gamma=GAMMA, n_step=N_STEP_RETURN)

config = {}
config[1] = {"min":0.05, "max":0.0, "lambda":0}
config[2] = {"min":0.05, "max":0.15, "lambda":0}
config[3] = {"min":0.05, "max":0.25, "lambda":0}
config[4] = {"min":0.05, "max":0.35, "lambda":0}

eq2 = MathEq({"min":0, "max":0.05, "lambda":0})

i = 1
threads = []
while i <= 4:
    name = "test" + str(i)
    game = T3Game(3, name=name, isConv=isConv)
    p1 = PGPlayer(name, game, brain=brain, eEq=MathEq(config[i]))
    p2 = MinimaxT3Player(2, game, eEq=eq2)

    env = Environment(game, p1, p2, ePlot=False)
    threads.append(MyThread(env))
    i += 1

opts = [Optimizer(brain) for i in range(1)]
for o in opts:
    o.start()
for t in threads:
    t.start()
