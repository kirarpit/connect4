#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 00:37:35 2018

@author: Arpit
"""

from games.c4Game import C4Game
from environment import Environment
from players.minimaxC4Player import MinimaxC4Player
from players.qPlayer import QPlayer
from keras.models import Sequential
from keras.layers import Dense
from mathEq import MathEq
import games.c4Solver as C4Solver
from myThread import MyThread

threads = []
#Example 1
game = C4Game(4,5, name="test1")

ann = Sequential()
ann.add(Dense(units = 24, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'relu', input_dim = game.stateCnt))
ann.add(Dense(units = 24, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'relu'))
ann.add(Dense(units = game.actionCnt, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'linear'))
ann.compile(optimizer = 'rmsprop', loss = 'logcosh', metrics = ['accuracy'])

eq1 = MathEq({"min":0.05, "max":0.3, "lambda":0.001})
eq2 = MathEq({"min":0, "max":0.10, "lambda":0})

p1 = QPlayer(1, game, model=ann, eEq=eq1)
p2 = MinimaxC4Player(2, game, eEq=eq2, solver=C4Solver)
env = Environment(game, p1, p2)
threads.append(MyThread(env))

#Example 2
game = C4Game(4,5, name="test2")

ann = Sequential()
ann.add(Dense(units = 24, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'relu', input_dim = game.stateCnt))
ann.add(Dense(units = 24, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'relu'))
ann.add(Dense(units = game.actionCnt, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'linear'))
ann.compile(optimizer = 'rmsprop', loss = 'logcosh', metrics = ['accuracy'])

eq1 = MathEq({"min":0.05, "max":0.3, "lambda":0.001})
eq2 = MathEq({"min":0, "max":0.20, "lambda":0})

p1 = QPlayer(1, game, model=ann, eEq=eq1)
p2 = MinimaxC4Player(2, game, eEq=eq2, solver=C4Solver)
env = Environment(game, p1, p2)
#env.run()
threads.append(MyThread(env))

#Example 3
game = C4Game(4,5, name="test3")

ann = Sequential()
ann.add(Dense(units = 24, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'relu', input_dim = game.stateCnt))
ann.add(Dense(units = 24, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'relu'))
ann.add(Dense(units = game.actionCnt, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'linear'))
ann.compile(optimizer = 'rmsprop', loss = 'logcosh', metrics = ['accuracy'])

eq1 = MathEq({"min":0.05, "max":0.3, "lambda":0.001})
eq2 = MathEq({"min":0, "max":0.30, "lambda":0})

p1 = QPlayer(1, game, model=ann, eEq=eq1)
p2 = MinimaxC4Player(2, game, eEq=eq2, solver=C4Solver)
env = Environment(game, p1, p2)
threads.append(MyThread(env))


#Example 4
game = C4Game(4,5, name="test4")

ann = Sequential()
ann.add(Dense(units = 24, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'relu', input_dim = game.stateCnt))
ann.add(Dense(units = 24, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'relu'))
ann.add(Dense(units = game.actionCnt, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'linear'))
ann.compile(optimizer = 'rmsprop', loss = 'logcosh', metrics = ['accuracy'])

eq1 = MathEq({"min":0.05, "max":0.3, "lambda":0.001})
eq2 = MathEq({"min":0, "max":0.40, "lambda":0})

p1 = QPlayer(1, game, model=ann, eEq=eq1)
p2 = MinimaxC4Player(2, game, eEq=eq2, solver=C4Solver)
env = Environment(game, p1, p2)
threads.append(MyThread(env))

for t in threads:
    t.start()