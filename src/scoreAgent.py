#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 16:55:37 2018

@author: Arpit
"""

from games.t3Game import T3Game
from player import Player

game = T3Game()
stateCnt, actionCnt = game.getStateActionCnt()
p = Player(2, stateCnt, actionCnt, True)

epsilon = 0.05

while game.gameCnt < 5000:
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