#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 19:47:06 2018

@author: Arpit
"""

import numpy as np

LOSER_R = -10
WINNER_R = 10

class Game:
    
    def __init__(self, rows, columns):
        self.rows = rows
        self.columns = columns
        self.gameCnt = -1
        self.newGame()
        self.deltas = {
                "N":(0, -1),
                "NE":(1, -1),
                "NW":(-1, -1),
                "E":(1, 0),
                "W":(-1, 0),
                "SE":(1, 1),
                "SW":(-1, 1),
                "S":(0, 1)
                }
        self.clearStats()
    
    def newGame(self):
        self.isOver = False
        self.rewards = {}
        self.gameCnt += 1
        self.toPlay = 1
        self.turnCnt = 0
        self.illMovesCnt = 0
        self.arrayForm = np.zeros((1, self.rows * self.columns * 2), dtype=int)
#        self.arrayForm[True] = -1
        self.gameState = np.zeros((self.rows, self.columns), dtype=int)

    def dropDisc(self, column):
        if self.isOver:
            print "Game's over already."
            return -1
        
        row = 0
        while row < self.rows:
            if self.gameState[row][column] != 0:
                break
            row += 1
        
        #illegal move. row full.
        if row == 0:
            self.illMovesCnt += 1
            self.setWinner(self.getNextPlayer(self.toPlay))
            return -2
        else:
            self.gameState[row - 1][column] = self.toPlay
            self.updateArrayForm(row - 1, column)
            self.switchTurn()
            self.checkEndStates(row - 1, column)
            return self.toPlay
        
    def switchTurn(self):
        self.turnCnt += 1
        self.toPlay = self.getNextPlayer(self.toPlay)
        
    def checkEndStates(self, row, column):
        player = self.gameState[row][column]
        directions = [('N', 'S'), ('E', 'W'), ('NE', 'SW'), ('SE', 'NW')]

        for dpair in directions:
            cnt = 1
            for direction in dpair:
                (dx, dy) = self.deltas[direction]
                x = column + dx
                y = row + dy
                
                while x>=0 and x<self.columns and y>=0 and y<self.rows:
                    if player == self.gameState[y][x]:
                        cnt += 1
                    else:
                        break

                    x += dx
                    y += dy
                
            if cnt >= 4:
                self.setWinner(player)
                break
            
        if self.turnCnt == self.rows * self.columns:
            self.isOver = True

        return
    
    def printGameState(self):
        print "#" * 19
        print "Total Games Played: " + str(self.gameCnt)
        print "Winner Stats: " + str(self.stats)
        print "-" * 19
        print "Game " + str(self.gameCnt) + ":"
        for x in range(0, self.rows):
            for y in range(0, self.columns):
                print str(self.gameState[x][y]) + " ",
            print "\n"
        print "Winner: " + str(self.isOver)
        print "No. of turns: " + str(self.turnCnt)
        print "Illegal moves count: " + str(self.illMovesCnt)
        print "-" * 19
    
    def updateArrayForm(self, row, column):
        pos = row * self.columns + column
        if self.gameState[row][column] == 2:
            pos += self.rows * self.columns
    
        self.arrayForm[0][pos] = 1
        
    def setWinner(self, player):
        self.isOver = player
        if self.illMovesCnt == 0:
            self.rewards[player] = WINNER_R
        self.rewards[self.getNextPlayer(player)] = LOSER_R
        self.stats[player] += 1
        
    def getNextPlayer(self, player):
        if player == 1:
            return 2
        else:
            return 1
    
    def demo(self, moves):
        self.newGame()
        for move in moves:
            self.dropDisc(move)
            
        self.printGameState()
        
    def clearStats(self):
        self.stats = {1:0, 2:0}
        
    def toString(self):
        lStr = ""
        for x in range(0, self.rows):
            for y in range(0, self.columns):
                lStr += str(self.gameState[x][y])
        return lStr
