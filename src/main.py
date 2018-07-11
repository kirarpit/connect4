#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 18:46:51 2018

@author: Arpit
"""

from c4game import C4Game
from player import Player

debug = False

game = C4Game()
stateCnt, actionCnt = game.getStateActionCnt()
p1 = Player(1, stateCnt, actionCnt, debug)

while True or game.gameCnt < 1:
    game.newGame()
    
    while not game.isOver():
        s = game.getCurrentState()
        a = p1.act(s, game.getIllMoves())
        s_, r = game.getNextState(a)
        sample = (s, a, r, s_)
    
        p1.observe(sample, game.gameCnt)
        p1.replay()
        
    if game.gameCnt % 10 == 0 or debug:
        game.printGame()
        print ("Exploration Rate: " + str(p1.epsilon))
        if not debug and game.gameCnt % 100 == 0:
            p1.saveWeights()

if debug:
    w1 = p1.ANN.ann.get_weights()
    locals().update(p1.logs)