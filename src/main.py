#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 18:46:51 2018

@author: Arpit
"""
import time
from games.c4Game import C4Game
from player import Player

def observeSample(lastS, lastA):
    p = p1 if (game.turnCnt + flag) % 2 == 0 else p2
    
    r = game.getReward(p.name)
    s_ = game.getCurrentState() if not game.isOver() else None

    sample = (lastS, lastA, r, s_)
    if sample[2] > 0: print(sample)
    p.observe(sample, game)
    p.replay()

start = time.time()
debug = True
game = C4Game(6,7)

stateCnt, actionCnt = game.getStateActionCnt()
p1 = Player(1, stateCnt, actionCnt, debug)
p2 = Player(2, stateCnt, actionCnt, debug)

while not debug or game.gameCnt < 10:
    game.newGame()
    
    flag = 0
    if game.gameCnt % 2 == 0:
        game.setFirstToPlay(2)
        flag = 1
        
    lastS = None
    lastA = None
    while not game.isOver():
        p = p1 if (game.turnCnt + flag) % 2 == 0 else p2
        
        s = game.getCurrentState()
        a = p.act(s, game.getIllMoves())
        game.step(a)

        if lastS is not None:
            observeSample(lastS, lastA)
        
        lastS = s
        lastA = a
        
    game.turnCnt += 1
    observeSample(lastS, lastA)
    
    if game.gameCnt % 100 == 0 or debug:
        game.printGame()
        print ("Exploration Rate: " + str(p1.epsilon))
        print ("Learning Rate: " + str(p1.alpha))
        game.clearStats()
        print("Time since beginning: " + str(time.time() - start))

if debug:
    w1 = p1.ANN.ann.get_weights()
    sample1 = p1.memory.sample(64)
    locals().update(p1.logs)
