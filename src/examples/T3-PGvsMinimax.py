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

GAMMA = 0.99
N_STEP_RETURN = 2
GAMMA_N = GAMMA ** N_STEP_RETURN

#Example 1
game = T3Game()
brain = Brain('pgbrain', game, gamma=GAMMA, n_step=N_STEP_RETURN, gamma_n=GAMMA_N)

config = {}
config[1] = {"min":0.05, "max":0.05, "lambda":0}
config[2] = {"min":0.05, "max":0.25, "lambda":0}
config[3] = {"min":0.05, "max":0.35, "lambda":0}
config[4] = {"min":0.05, "max":0.45, "lambda":0}

eq2 = MathEq({"min":0.05, "max":0.05, "lambda":0})

i = 1
threads = []
while i <= 4:
    name = "test" + str(i)
    game = T3Game(3, name=name)
    p1 = PGPlayer(name, game, brain=brain, eEq=MathEq(config[i]))
    p2 = MinimaxT3Player(2, game, eEq=eq2)

    env = Environment(game, p1, p2)
    threads.append(MyThread(env))
    i += 1

opts = [Optimizer(brain) for i in range(1)]
for o in opts:
    o.start()
for t in threads:
    t.start()