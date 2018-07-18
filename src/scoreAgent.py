#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 16:55:37 2018

@author: Arpit
"""

from games.c4Game import C4Game
#from games.t3Game import T3Game
from player import Player

#game = T3Game()
game = C4Game()
stateCnt, actionCnt = game.getStateActionCnt()
p = Player(1, stateCnt, actionCnt, True)

epsilon = 0.05

while game.gameCnt < 1000:
    game.newGame() 
    
    if game.gameCnt % 2 == 0:
        game.setFirstToPlay(2)
        game.p2act(epsilon)
    
    while not game.isOver():
        s = game.getCurrentState()
        a = p.act(s, game.getIllMoves())
        game.step(a)
        
        if not game.isOver():
            game.p2act(epsilon)
            
game.printGame()