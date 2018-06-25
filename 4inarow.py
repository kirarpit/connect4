#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 18:46:51 2018

@author: Arpit
"""

# Importing the libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import game as c4game
import os.path

def create_ann():
    ann = Sequential()
    ann.add(Dense(units = 25, kernel_initializer = "uniform", activation = 'relu', input_dim = 84))
    ann.add(Dense(units = 25, kernel_initializer = "uniform", activation = 'relu'))
    ann.add(Dense(units = 7, kernel_initializer = "uniform", activation = 'softmax'))
    ann.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    return ann

def save_config():
    json = ann.to_json()
    file = open("config.json", "w")
    file.write(json)
    file.close()
    
def test():
    game = c4game.Game(6, 7)
    game.dropDisc(3)
    game.dropDisc(4)
    game.dropDisc(3)
    game.dropDisc(4)
    game.dropDisc(3)
    ans = ann.predict(game.arrayForm)
    print np.argmax(ans)

ann = create_ann()

weights_file = "weights.hdf5"
if os.path.exists(weights_file):
    ann.load_weights(weights_file, by_name=False)

gameNo = 1
while gameNo <= 100000:
    game = c4game.Game(6, 7)

    X_train = game.arrayForm
    y_train = ann.predict(game.arrayForm)
    result = []

    while not (game.isOver()):
        
        ans = ann.predict(game.arrayForm)
    
        X_train = np.vstack([X_train, game.arrayForm])
        y_train = np.vstack([y_train, ans])
        result.append(np.argmax(ans))

        if game.dropDisc(np.argmax(ans)) == -2:
            ans[0][np.argmax(ans)] -= 2
            ann.fit(game.arrayForm, ans, verbose=0)
        
    X_train = np.delete(X_train, 0, 0)
    y_train = np.delete(y_train, 0, 0)
    
    ## promoting winner player moves and vice versa
    winner = game.isOver()

    if winner == 1:
        p1 = 1
        p2 = -1
    elif winner == 2:
        p1 = -1
        p2 = 1
    else:
        p1 = -0.5
        p2 = -0.5
    
    i = 0
    for row in y_train:
        if i%2 == 0:
            delta = p1
        else:
            delta = p2

        row[result[i]] += delta
        i += 1
        
    ## training
    if gameNo % 500 == 0:
        print "Game " + str(gameNo) + ":"
        game.printGameState()
        print "Winner: " + str(game.winner)
        ann.fit(X_train, y_train)
    else:
        ann.fit(X_train, y_train, verbose=0)

        
    gameNo += 1
    
ann.save_weights(weights_file)
save_config()
