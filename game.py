#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 19:47:06 2018

@author: Arpit
"""

import numpy as np

class Game:
    
    def __init__(self, rows, columns):
        self.gameState = np.zeros((rows, columns), dtype=int)
        self.rows = rows
        self.columns = columns
        self.toPlay = 1
        self.winner = 0
        self.turnCnt = 0
        self.arrayForm = np.zeros((1, rows * columns * 2), dtype=int)
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
    
    def isOver(self):
        if self.winner != 0:
            return self.winner
        else:
            return False
        
    def dropDisc(self, column):
        
        if self.isOver():
            print "Game's over already."
            return -1
        
        row = 0
        while row < self.rows:
            if self.gameState[row][column] != 0:
                break
            row += 1
        
        if row == 0:
#            print "Invalid move. Column " + str(column) + " full."
            return -2
        else:
            self.gameState[row - 1][column] = self.toPlay
            self.updateArrayForm(row - 1, column)
            self.switchTurn()
            self.checkEndStates(row - 1, column)
            return self.toPlay
        
    def switchTurn(self):
        self.turnCnt += 1
        if self.toPlay == 1:
            self.toPlay = 2
        else:
            self.toPlay = 1
        
    def checkEndStates(self, row, column):
        player = self.gameState[row][column]
        directions = [('N', 'S'), ('E', 'W'), ('NE', 'SW'), ('SE', 'NW')]

        for dpair in directions:
            cnt = 1
            for direction in dpair:
#                print "checking for direction" + direction
                (dx, dy) = self.deltas[direction]
                x = column + dx
                y = row + dy
                
                
                while x>=0 and x<self.columns and y>=0 and y<self.rows:
#                    print "trying x,y " + str(x) + "," + str(y)

                    if player == self.gameState[y][x]:
                        cnt += 1
#                        print "count: " + str(cnt)
                    else:
                        break

                    x += dx
                    y += dy
                
            if cnt >= 4:
                self.winner = player
                break
            
        if self.turnCnt == self.rows * self.columns:
            self.winner = 3

        return
    
    def printGameState(self):
        for x in range(0, self.rows):
            for y in range(0, self.columns):
                print str(self.gameState[x][y]) + " ",
            print "\n"
        print "-" * 20
    
    def updateArrayForm(self, row, column):
        pos = row * self.columns + column
        if self.gameState[row][column] == 2:
            pos += self.rows * self.columns
    
        self.arrayForm[0][pos] = 1
    
    
    