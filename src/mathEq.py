#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 17:29:15 2018

@author: Arpit
"""

class MathEq:
    def __init__(self, eqNo):
        self.eqNo = eqNo
        
    def getValue(self, x):
        if self.eqNo == 1:
            value = -0.06603804 + (0.9999994 - -0.06603804)/(1 + (x/2580.584) ** 0.6631246)
        elif self.eqNo == 2:
            value = -0.7433052 + (1.199923 - -0.7433052)/(1 + (x/33837.78)**0.3576007)
            
        return value