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
import random

@lru_cache(maxsize=None)
def getP2Move(gameColumnString):
    r = requests.get('http://connect4.gamesolver.org/solve?pos=' + str(gameColumnString))
    data = yaml.safe_load(r.text)
    moves = data['score']
    
    choices = []
    maxi = float("-inf")
    for index, value in enumerate(moves):
        if value != 100 and maxi <= value:
            if maxi<value:
                choices = [index]
                maxi = value
            else:
                choices.append(index)
        
    return choices

debug = True

game = c4game.Game(6, 7)
p1 = Player(1, game, debug)
#p2 = Player(2, game, debug)

p1WinCnt = 0
while game.gameCnt < 1:
    game.newGame()

    while not game.isOver:
        if game.turnCnt % 2 == 0:
            p1.play(game)
        else:
#            p2.play(game)
            game.dropDisc(random.sample(getP2Move(game.columnString), 1)[0])

    p1.play(game)
#    p2.play(game)

    if game.gameCnt % 1 == 0:
        game.printGameState()
        print ("Exploration Rate: " + str(p1.epsilon))
        print (getP2Move.cache_info())
        if not debug and game.gameCnt % 100 == 0:
            p1.saveWeights()
#            p2.saveWeights()
            
    p1WinCnt = p1WinCnt + 1 if game.isOver == 1 else 0
    if p1WinCnt == 100:
        break
        

if debug:
    w1 = p1.ANN.ann.get_weights()
    locals().update(p1.logs)
#    moves2 = p2.logs['moves']