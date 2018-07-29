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
N_STEP_RETURN = 1
GAMMA_N = GAMMA ** N_STEP_RETURN

#Example 1
game = T3Game()
brain = Brain('pgbrain', game, gamma=GAMMA, n_step=N_STEP_RETURN, gamma_n=GAMMA_N)

config = {"min":0.05, "max":0.05, "lambda":0}
eq2 = MathEq({"min":0.05, "max":0.05, "lambda":0})

i = 0
threads = []
while i < 4:
    name = "T3-thread-" + str(i)
    game = T3Game(3, name=name)
    p1 = PGPlayer(name, game, brain=brain, eEq=MathEq(config))
    p2 = MinimaxT3Player(2, game, eEq=eq2)

    env = Environment(game, p1, p2, thread=True)
    threads.append(MyThread(env))
    i += 1
    config['max'] += 0.1

opts = [Optimizer(brain) for i in range(1)]
for o in opts:
    o.start()
for t in threads:
    t.start()

for t in threads:
    t.stop()
for o in opts:
    o.stop()
