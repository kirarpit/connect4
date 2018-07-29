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
import games.c4Solver as C4Solver
from myThread import MyThread

GAMMA = 0.99
N_STEP_RETURN = 3
GAMMA_N = GAMMA ** N_STEP_RETURN

#Example 1
game = C4Game(4, 5)
brain = Brain('pgbrain', game, gamma=GAMMA, n_step=N_STEP_RETURN, gamma_n=GAMMA_N)

eq1 = MathEq({"min":0.05, "max":1, "lambda":0.001})
eq2 = MathEq({"min":0.05, "max":0.3, "lambda":0})

i = 0
threads = []
while i < 2:
    name = "thread-" + str(i)
    game = C4Game(4, 5, name=name)
    p1 = PGPlayer(name, game, brain=brain, eEq=eq1)
    p2 = MinimaxC4Player(2, game, eEq=eq2, solver=C4Solver)

    env = Environment(game, p1, p2)
    threads.append(MyThread(env))
    i += 1

opts = [Optimizer(brain) for i in range(1)]
for o in opts:
    o.start()
for t in threads:
    t.start()

for t in threads:
    t.stop()
for o in opts:
    o.stop()
