#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 20:40:00 2018

@author: Arpit
"""

from games.t3Game import T3Game
from environment import Environment
from players.minimaxT3Player import MinimaxT3Player
from players.qPlayer import QPlayer
from brains.qBrain import QBrain
from settings import charts_folder
from keras.utils import plot_model

isConv = True
layers = [
        {'filters':16, 'kernel_size': (2,2), 'size':24}
        , {'filters':16, 'kernel_size': (2,2), 'size':24}
	]

game = T3Game(3, isConv=isConv)
brain = QBrain('t3AsyncDQN', game, layers=layers, load_weights=False)
tBrain = QBrain('t3AsyncDQN', game, layers=layers)
plot_model(brain.model, show_shapes=True, to_file=charts_folder + 'model.png')

player_config = {"mem_size":5000, "brain":brain, "tBrain":tBrain, "epsilon":0.05,
                 "batch_size":32, "gamma":0.93, "n_step":1}

p1 = QPlayer(1, game, **player_config)
p2 = MinimaxT3Player(2, game, epsilon=0.05)
env = Environment(game, p1, p2)
env.run()