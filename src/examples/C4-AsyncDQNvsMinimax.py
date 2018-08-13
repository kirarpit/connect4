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
import games.c4Solver as C4Solver
from envThread import EnvThread
from memory.pMemory import PMemory
from brains.qBrain import QBrain

ROWS = 6
COLUMNS = 7
hidden_layers = [
	{'filters':64, 'kernel_size': (4,4)}
	 , {'filters':64, 'kernel_size': (4,4)}
	 , {'filters':64, 'kernel_size': (4,4)}
	 , {'filters':64, 'kernel_size': (4,4)}
	]

game = C4Game(ROWS, COLUMNS)
brain = QBrain('c4AsyncDQN', game, hidden_layers = hidden_layers, load_weights=False)

player_config = {"memory":PMemory(20000), "goodMemory":PMemory(20000), "targetNet":False,
                "batch_size":64, "gamma":0.99, "n_step":13}
epsilons = {0.05, 0.15, 0.25, 0.40}

i = 1
threads = []
while i <= 4:
    game = C4Game(ROWS, COLUMNS, name=i)
    p1 = QPlayer(name, game, brain=brain, epsilon=epsilons[i], **player_config)
    p2 = MinimaxC4Player(2, game, epsilon=0.05, solver=C4Solver)
    env = Environment(game, p1, p2)
    threads.append(EnvThread(env))
    i += 1

for t in threads:
    t.start()
