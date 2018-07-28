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
from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten
from mathEq import MathEq

#Example 1
game = C4Game(4,5, name="test1")

ann = Sequential()
ann.add(Dense(units = 36, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'relu', input_dim = game.stateCnt))
ann.add(Dense(units = 36, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'relu'))
ann.add(Dense(units = game.actionCnt, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'linear'))
ann.compile(optimizer = 'rmsprop', loss = 'logcosh', metrics = ['accuracy'])

eq1 = MathEq({"type":1, "eqNo":3})
eq2 = MathEq({"type":1, "eqNo":4})

p1 = QPlayer(1, game, model=ann, eEq=eq1)
p2 = MinimaxC4Player(2, game, eEq=eq2)
env = Environment(game, p1, p2)
env.run()

#Example 2
game = C4Game(4,5, name="test2")

ann = Sequential()
ann.add(Dense(units = 36, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'relu', input_dim = game.stateCnt))
ann.add(Dense(units = 36, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'relu'))
ann.add(Dense(units = game.actionCnt, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'linear'))
ann.compile(optimizer = 'rmsprop', loss = 'logcosh', metrics = ['accuracy'])

eq1 = MathEq({"min":0.05, "max":1, "lambda":0.001})
eq2 = MathEq({"min":0.05, "max":0.3, "lambda":0})

p1 = QPlayer(1, game, model=ann, eEq=eq1)
p2 = MinimaxC4Player(2, game, eEq=eq2)
env = Environment(game, p1, p2)
env.run()

#Example 3
game = C4Game(4,5, name="test3")

ann = Sequential()
ann.add(Dense(units = 36, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'relu', input_dim = game.stateCnt))
ann.add(Dense(units = 36, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'relu'))
ann.add(Dense(units = game.actionCnt, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'linear'))
ann.compile(optimizer = 'rmsprop', loss = 'logcosh', metrics = ['accuracy'])

eq1 = MathEq({"min":0.05, "max":1, "lambda":0.001})
eq2 = MathEq({"min":0.05, "max":0.05, "lambda":0})

p1 = QPlayer(1, game, model=ann, eEq=eq1)
p2 = MinimaxC4Player(2, game, eEq=eq2)
env = Environment(game, p1, p2)
env.run()

#Example 4
name = "test4"
game = C4Game(4,5, name=name)

ann = Sequential()
ann.add(Dense(units = 36, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'relu', input_dim = game.stateCnt))
ann.add(Dense(units = 36, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'relu'))
ann.add(Dense(units = 36, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'relu'))
ann.add(Dense(units = game.actionCnt, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'linear'))
ann.compile(optimizer = 'rmsprop', loss = 'logcosh', metrics = ['accuracy'])

eq1 = MathEq({"min":0.1, "max":1, "lambda":0.001})
eq2 = MathEq({"type":1, "eqNo":5})

p1 = QPlayer(name, game, model=ann, eEq=eq1)
p2 = MinimaxC4Player(2, game, eEq=eq2)
env = Environment(game, p1, p2)
env.run()


#Example 5
name = "test5"
game = C4Game(4,5, name=name)

ann = Sequential()
ann.add(Dense(units = 36, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'relu', input_dim = game.stateCnt))
ann.add(Dense(units = 36, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'relu'))
ann.add(Dense(units = 36, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'relu'))
ann.add(Dense(units = game.actionCnt, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'linear'))
ann.compile(optimizer = 'rmsprop', loss = 'logcosh', metrics = ['accuracy'])

eq1 = MathEq({"min":0.05, "max":1, "lambda":0.0001})
eq2 = MathEq({"min":0.05, "max":0.3, "lambda":0})

p1 = QPlayer(name, game, model=ann, eEq=eq1)
p2 = MinimaxC4Player(2, game, eEq=eq2)
env = Environment(game, p1, p2)
env.run()

#Example 6
name = "test6"
game = C4Game(4,5, name=name)

ann = Sequential()
ann.add(Dense(units = 36, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'relu', input_dim = game.stateCnt))
ann.add(Dense(units = 36, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'relu'))
ann.add(Dense(units = 36, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'relu'))
ann.add(Dense(units = game.actionCnt, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation = 'linear'))
ann.compile(optimizer = 'rmsprop', loss = 'logcosh', metrics = ['accuracy'])

eq1 = MathEq({"min":0.05, "max":1, "lambda":0.0001})
eq2 = MathEq({"min":0.05, "max":0.05, "lambda":0})

p1 = QPlayer(name, game, model=ann, eEq=eq1)
p2 = MinimaxC4Player(2, game, eEq=eq2)
env = Environment(game, p1, p2)
env.run()