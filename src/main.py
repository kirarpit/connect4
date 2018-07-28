#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 18:46:51 2018

@author: Arpit
"""
from environment import Environment
from games.c4Game import C4Game

debug = False
game = C4Game(6,7)

env = Environment(game, p1, p2, debug)
env.run()

if debug:
    w1 = env.p1.brain.model.get_weights()
    sample1 = env.p1.memory.sample(64)
    sampleG1 = env.p1.goodMemory.sample(64)
    locals().update(env.p1.logs)