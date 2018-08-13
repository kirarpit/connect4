#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 23:11:45 2018

@author: Arpit
"""

from games.c4Game import C4Game
from environment import Environment
from players.zeroPlayer import ZeroPlayer
from memory.dictTree import DictTree
from brains.zeroBrain import ZeroBrain
from keras.utils import plot_model
from collections import deque

game = C4Game(6, 7, isConv=True)
hidden_layers = [
	{'filters':64, 'kernel_size': (4,4)}
	 , {'filters':64, 'kernel_size': (4,4)}
	 , {'filters':64, 'kernel_size': (4,4)}
	 , {'filters':64, 'kernel_size': (4,4)}
	]

player_config = {"tree":DictTree(), "longTermMem":deque(maxlen=20000), "load_weights":False, 
                 "epsilon":0.25, "dirAlpha":0.3, "simCnt":50, "iterPer":20, "turnsToTau0":8}
brain_config = {"learning_rate":0.001, "momentum":0.9, "batch_size":32, "epochs":10,
                "hidden_layers":hidden_layers}
env_config = {"switchFTP":False, "evaluate":True, "evalPer":100}

brain = ZeroBrain("1", game, **brain_config)
plot_model(brain.model, show_shapes=True, to_file='/Users/Arpit/Desktop/model.png')

p1 = ZeroPlayer(1, game, brain=brain, **player_config)
p2 = ZeroPlayer(2, game, brain=brain, **player_config)
env = Environment(game, p1, p2, **env_config)
env.run()
