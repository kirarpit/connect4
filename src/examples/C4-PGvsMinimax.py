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
from brains.pgBrain import PGBrain
from optimizer import Optimizer
from envThread import EnvThread
import games.c4Solver as C4Solver

ROWS = 4
COLUMNS = 5
gamma = 0.90
n_step = 6
isConv = True
layers = [
	{'filters':32, 'kernel_size': (3,3)}
	 , {'filters':32, 'kernel_size': (3,3)}
	]

game = C4Game(ROWS, COLUMNS, isConv=isConv)

brain_config = {"gamma":gamma, "n_step":n_step, "min_batch":64,
                "layers":layers, "load_weights":False}
brain = PGBrain('c4PG', game, **brain_config)

player_config = {"batch_size":64, "gamma":gamma, "n_step":n_step}
epsilons = {0.05, 0.15, 0.25, 0.40}

i = 1
threads = []
while i <= 4:
    game = C4Game(ROWS, COLUMNS, name=i, isConv=isConv)
    p1 = PGPlayer(name, game, brain=brain, epsilon=epsilons[i], **player_config)
    p2 = MinimaxC4Player(2, game, epsilon=0.05, solver=C4Solver)
    env = Environment(game, p1, p2)
    threads.append(EnvThread(env))
    i += 1

opts = [Optimizer(brain) for i in range(2)]
for o in opts:
    o.start()
for t in threads:
    t.start()
