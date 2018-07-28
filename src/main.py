#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 18:46:51 2018

@author: Arpit
"""
from envs.vsMinimaxEnv import VSMinimaxEnv
#from envs.selfPlayEnv import SelfPlayEnv
#from games.c4Game import C4Game
from games.t3Game import T3Game

debug = True
#game = C4Game(6,7)
game = T3Game()

env = VSMinimaxEnv(game, debug)
#env = SelfPlayEnv(game, debug)
env.run()

if debug:
    w1 = env.p1.ANN.ann.get_weights()
    sample1 = env.p1.memory.sample(64)
    sampleG1 = env.p1.goodMemory.sample(64)
    locals().update(env.p1.logs)