#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 18:46:51 2018

@author: Arpit
"""

import game as c4game
from player import Player

debug = False

game = c4game.Game(6, 7)
p1 = Player(1, game, debug)
p2 = Player(2, game, debug)

while True or game.gameCnt < 2:
    game.newGame()
    
    while not game.isOver:
        if game.turnCnt % 2 == 0:
            p1.play(game)
        else:
            p2.play(game)

    p1.play(game)
    p2.play(game)
    
    if game.gameCnt % 10 == 0:
        game.printGameState()
            
        if debug == False:
            p1.saveWeights()
            p2.saveWeights()

if debug == True:
    batch = p1.batch
    s = p1.s
    s_ = p1.s_
    p = p1.p
    p_ = p1.p_
    fx = p1.fx
    fy = p1.fy
    x1 = p1.x_old
    x2 = p2.x_old
    y1 = p1.y_old
    y2 = p2.y_old
    m1 = p1.m_old
    m2 = p2.m_old
    w1 = p1.ANN.ann.get_weights()
    w2 = p2.ANN.ann.get_weights()