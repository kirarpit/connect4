#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 18:23:00 2018

@author: Arpit
"""

from games.c4Game import C4Game
from environment import Environment
from players.minimaxC4Player import MinimaxC4Player
from players.pgPlayer import PGPlayer
from mathEq import MathEq
from pgBrain import Brain
from optimizer import Optimizer
from envThread import EnvThread
from keras.layers import Input, Dense
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.models import Model
import games.c4Solver as C4Solver

GAMMA = 0.93
N_STEP_RETURN = 10
MIN_BATCH = 64
ROWS = 5
COLUMNS = 6
isConv = True
loadWeights = True
filename = "pgbrain67"

game = C4Game(ROWS, COLUMNS, name="dummy", isConv=isConv)

if isConv:
    l_input = Input( shape=game.stateCnt )
    l_dense = Convolution2D(32, (3, 3), strides=(1,1), padding='valid', activation='relu', 
                            data_format="channels_first")(l_input)
    l_dense = Convolution2D(32, (2, 2), strides=(1,1), padding='valid', activation='relu', 
                            data_format="channels_first")(l_dense)
    l_dense = Flatten()(l_dense)
else: 
    l_input = Input( batch_shape=(None, game.stateCnt) )
    l_dense = Dense(48, kernel_initializer='random_uniform', bias_initializer='random_uniform', 
                    activation='relu')(l_input)

l_dense = Dense(46, kernel_initializer='random_uniform', bias_initializer='random_uniform', 
                activation='relu')(l_dense)
l_dense = Dense(46, kernel_initializer='random_uniform', bias_initializer='random_uniform', 
                activation='relu')(l_dense)

out_actions = Dense(game.actionCnt, activation='softmax')(l_dense)
out_value   = Dense(1, activation='linear')(l_dense)

model = Model(inputs=[l_input], outputs=[out_actions, out_value])
model._make_predict_function()	# have to initialize before threading

brain = Brain(filename, game, model=model, loadWeights=loadWeights, min_batch=MIN_BATCH, 
              gamma=GAMMA, n_step=N_STEP_RETURN)

config = {}
config[1] = {"min":0.05, "max":0.05, "lambda":0}
config[2] = {"min":0.05, "max":0.25, "lambda":0}
config[3] = {"min":0.05, "max":0.35, "lambda":0}
config[4] = {"min":0.05, "max":0.45, "lambda":0}

eq2 = MathEq({"min":0.05, "max":0, "lambda":0})

i = 1
threads = []
while i <= 4:
    name = "test" + str(i)
    game = C4Game(ROWS, COLUMNS, name=name, isConv=isConv)
    p1 = PGPlayer(name, game, brain=brain, eEq=MathEq(config[i]),
                  gamma=GAMMA, n_step=N_STEP_RETURN)
    p2 = MinimaxC4Player(2, game, eEq=eq2, solver=C4Solver)

    env = Environment(game, p1, p2, ePlot=False)
    threads.append(EnvThread(env))
    i += 1

opts = [Optimizer(brain) for i in range(2)]
for o in opts:
    o.start()
for t in threads:
    t.start()
