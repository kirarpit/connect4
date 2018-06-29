#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 18:46:51 2018

@author: Arpit
"""

import game as c4game
import player

game = c4game.Game(6, 7)
p1 = player.Player(1, game)
p2 = player.Player(2, game)

stats = {1:0, 2:0, 3:0}
while game.gameCnt <= 50000:
    game.newGame()
    
    while not game.isOver():
        if game.turnCnt % 2 == 0:
            p1.play(game)
        else:
            p2.play(game)

    verbosity = 0
    stats[game.winner] += 1
    if game.gameCnt % 1000 == 0:
        print "Game " + str(game.gameCnt) + ":"
        game.printGameState()
        print "Winner: " + str(game.winner)
        print "No. of turns: " + str(game.turnCnt)
        print "Illegal moves count: " + str(game.illMovesCnt)
        print stats
        verbosity = 2
        stats = {1:0, 2:0, 3:0}
        p1.saveExp()
        p2.saveExp()
            
    p1.train(game, verbosity)
    p2.train(game, verbosity)
    
x1 = p1.x_old
x2 = p2.x_old
y1o = p1.y_orig
y2o = p2.y_orig
y1 = p1.y_old
y2 = p2.y_old
m1 = p1.m_old
m2 = p2.m_old
w1 = p1.ANN.ann.get_weights()
w2 = p2.ANN.ann.get_weights()
game.printGameState()