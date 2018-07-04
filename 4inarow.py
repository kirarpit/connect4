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

debug = True

game = c4game.Game(7, 7)
p1 = Player(1, game, debug)

while game.gameCnt < 1:
    game.newGame()

    while not game.isOver:
        if game.turnCnt % 2 == 0:
            p1.play(game)
        else:
            r = requests.get('http://kevinalbs.com/connect4/back-end/index.php/getMoves?board_data='
                             + game.toString() + '&player=2')
            moves = yaml.safe_load(r.text)
            move = int(max(moves, key=moves.get))
            game.dropDisc(move)

    p1.play(game)

    if game.gameCnt % 1 == 0:
        game.printGameState()
        print "Learning Rate: " + str(p1.alpha)
        print "Exploration Rate: " + str(p1.epsilon)
        print "Reward Sample Rate: " + str(p1.sampleRatio)
        
        cnt=0.0
        for o in p1.batch:
            if o[2] != 0:
                cnt += 1
                
        print "Batch reward sample ratio: " + str(cnt/len(p1.batch))

        if debug == False and game.gameCnt % 50 == 0:
            p1.saveWeights()
            
if debug == True:
    batch = p1.batch
    s = p1.s
    s_ = p1.s_
    P1 = p1.p
    fy1 = p1.fy
    
    x1 = p1.x_old
    y1 = p1.y_old
    m1 = p1.m_old
    w1 = p1.ANN.ann.get_weights()
