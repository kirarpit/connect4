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
from settings import charts_folder
from keras.utils import plot_model

ROWS = 4
COLUMNS = 5
isConv = False
layers = [
	{'filters':64, 'kernel_size': (4,4), 'size':24}
	 , {'filters':64, 'kernel_size': (4,4), 'size':24}
#	 , {'filters':64, 'kernel_size': (4,4)}
#	 , {'filters':64, 'kernel_size': (4,4)}
	]

game = C4Game(ROWS, COLUMNS, isConv=isConv)
brain = QBrain('c4AsyncDQN', game, layers=layers, load_weights=False)
plot_model(brain.model, show_shapes=True, to_file=charts_folder + 'model.png')

player_config = {"memory":PMemory(5000), "goodMemory":PMemory(5000), "targetNet":False,
                "batch_size":32, "gamma":0.95, "n_step":5}
epsilons = [0.05, 0.15, 0.25, 0.35]

i = 0
threads = []
while i < 4:
    game = C4Game(ROWS, COLUMNS, name=i, isConv=isConv)
    p1 = QPlayer(i, game, brain=brain, epsilon=epsilons[i], **player_config)
    p2 = MinimaxC4Player(2, game, epsilon=0.05, solver=C4Solver)
    env = Environment(game, p1, p2)
    threads.append(EnvThread(env))
    i += 1

for t in threads:
    t.start()
