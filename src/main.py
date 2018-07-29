#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 18:46:51 2018

@author: Arpit
"""
from games.c4Game import C4Game
from environment import Environment
from players.minimaxC4Player import MinimaxC4Player
from players.qPlayer import QPlayer
from mathEq import MathEq

#Example 1
game = C4Game(4,5)

debug = True
p1 = QPlayer(1, game, debug=debug)
eq = MathEq({"min":0, "max":0.3, "lambda":0.001})
p2 = MinimaxC4Player(2, game)
env = Environment(game, p1, p2, debug=debug)
env.run()

if debug:
    w1 = env.p1.brain.model.get_weights()
    sample1 = env.p1.memory.sample(64)
    sampleG1 = env.p1.goodMemory.sample(64)
    locals().update(env.p1.logs)