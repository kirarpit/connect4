#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 00:37:35 2018

@author: Arpit
"""

from games.t3Game import T3Game
from environment import Environment
from players.minimaxT3Player import MinimaxT3Player
from players.qPlayer import QPlayer
from brains.qBrain import QBrain
from envThread import EnvThread
from memory.pMemory import PMemory

isConv = True
layers = [
	{'filters':64, 'kernel_size': (4,4)}
	 , {'filters':64, 'kernel_size': (4,4)}
	 , {'filters':64, 'kernel_size': (4,4)}
	 , {'filters':64, 'kernel_size': (4,4)}
	]

game = T3Game(3, isConv=isConv)
brain = QBrain('t3AsyncDQN', game, layers=layers, load_weights=False)

player_config = {"memory":PMemory(20000), "goodMemory":PMemory(20000), "targetNet":False,
                "batch_size":64, "gamma":0.99, "n_step":13}
epsilons = {0.05, 0.15, 0.25, 0.40}

i = 1
threads = []
while i <= 4:
    game = T3Game(3, name=i, isConv=isConv)
    p1 = QPlayer(name, game, brain=brain, epsilon=epsilons[i], **player_config)
    p2 = MinimaxT3Player(2, game, epsilon=0.05)
    env = Environment(game, p1, p2)
    threads.append(EnvThread(env))
    i += 1

for t in threads:
    t.start()
