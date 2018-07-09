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

debug = False

game = c4game.Game(6, 7)
p1 = Player(1, game, debug)

p1WinCnt = 0
while True or game.gameCnt < 1:
    game.newGame()

    while not game.isOver:
        if game.turnCnt % 2 == 0:
            p1.play(game)
        else:
#            game.dropDisc(random.sample(getP2Move(game.columnString), 1)[0])
            game.dropDisc(getP2Move(game.columnString)[0])

    p1.play(game)

    if game.gameCnt % 10 == 0:
        game.printGameState()
        print ("Exploration Rate: " + str(p1.epsilon))
        print (getP2Move.cache_info())
        if not debug and game.gameCnt % 100 == 0:
            p1.saveWeights()
            
    p1WinCnt = p1WinCnt + 1 if game.isOver == 1 else 0
    if p1WinCnt == 100:
        break
        

if debug:
    w1 = p1.ANN.ann.get_weights()
    locals().update(p1.logs)