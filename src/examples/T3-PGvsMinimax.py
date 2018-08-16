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
from settings import charts_folder
from keras.utils import plot_model

gamma = 0.90
n_step = 2
isConv = False
layers = [
	{'filters':32, 'kernel_size': (3,3), 'size':24}
	 , {'filters':32, 'kernel_size': (3,3), 'size':24}
	]

game = T3Game(3, isConv=isConv)

brain_config = {"gamma":gamma, "n_step":n_step, "min_batch":64,
                "layers":layers, "load_weights":False, "epochs":10}
brain = PGBrain('c4PG', game, **brain_config)
plot_model(brain.model, show_shapes=True, to_file=charts_folder + 'model.png')

player_config = {"gamma":gamma, "n_step":n_step}
epsilons = [0.05, 0.15, 0.25, 0.35]

i = 0
threads = []
while i < 4:
    game = T3Game(3, name=i, isConv=isConv)
    p1 = PGPlayer(i, game, brain=brain, epsilon=epsilons[i], **player_config)
    p2 = MinimaxT3Player(2, game, epsilon=0.05)
    env = Environment(game, p1, p2)
    threads.append(EnvThread(env))
    i += 1

opts = [Optimizer(brain) for i in range(2)]
for o in opts:
    o.start()
for t in threads:
    t.start()
