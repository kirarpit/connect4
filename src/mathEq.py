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
            value = -0.8242347 + (0.99722 - -0.8242347)/(1 + (x/97103.41) ** 0.6462744)
        elif self.eqNo == 2:
            value = -0.1860666 + (0.9925219 - -0.1860666)/(1 + (x/19241.1) ** 0.779946)
            
        return value