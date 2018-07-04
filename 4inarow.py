#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 18:46:51 2018

@author: Arpit
"""

import game as c4game
from player import Player
import requests
import yaml
from functools import lru_cache

@lru_cache(maxsize=1024)
def getP2Move(gameString):
    r = requests.get('http://kevinalbs.com/connect4/back-end/index.php/getMoves?board_data='
                     + gameString + '&player=2')
    moves = yaml.safe_load(r.text)
    return int(max(moves, key=moves.get))
    
debug = False

game = c4game.Game(7, 7)
p1 = Player(1, game, debug)

while True or game.gameCnt < 1:
    game.newGame()

    while not game.isOver:
        if game.turnCnt % 2 == 0:
            p1.play(game)
        else:
            game.dropDisc(getP2Move(game.toString()))

    p1.play(game)

    if game.gameCnt % 10 == 0:
        game.printGameState()
        print ("Exploration Rate: " + str(p1.epsilon))
        print (getP2Move.cache_info())
        if not debug and game.gameCnt % 50 == 0:
            p1.saveWeights()
            
if debug:
    w1 = p1.ANN.ann.get_weights()
    locals().update(p1.logs)
    samples = p1.memory.samples