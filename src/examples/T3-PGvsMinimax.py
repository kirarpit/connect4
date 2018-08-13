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
from brains.pgBrain import PGBrain
from optimizer import Optimizer
from envThread import EnvThread

gamma = 0.90
n_step = 6
isConv = True
layers = [
	{'filters':32, 'kernel_size': (3,3)}
	 , {'filters':32, 'kernel_size': (3,3)}
	]

game = T3Game(3, isConv=isConv)

brain_config = {"gamma":gamma, "n_step":n_step, "min_batch":64,
                "layers":layers, "load_weights":False, "epochs":10}
brain = PGBrain('c4PG', game, **brain_config)

player_config = {"batch_size":64, "gamma":gamma, "n_step":n_step}
epsilons = {0.05, 0.15, 0.25, 0.40}

i = 1
threads = []
while i <= 4:
    game = T3Game(3, name=i, isConv=isConv)
    p1 = PGPlayer(name, game, brain=brain, epsilon=epsilons[i], **player_config)
    p2 = MinimaxT3Player(2, game, epsilon=0.05)
    env = Environment(game, p1, p2)
    threads.append(EnvThread(env))
    i += 1

opts = [Optimizer(brain) for i in range(2)]
for o in opts:
    o.start()
for t in threads:
    t.start()
