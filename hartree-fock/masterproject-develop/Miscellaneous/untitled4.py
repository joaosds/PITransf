#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 13:16:52 2021

@author: Michael Perle

just playing around to understand inheritance in python
"""

class Base:
    def __init__(self, val1, val2):
        self.val1 = val1
        self.val2 = val2
        self.val3 = 10
        self.none = None
        
    def multiply_by(self, val):
        self.val1 *= val
        self.val2 *= val
        self.val3 *= val
        if self.none is not None:
            self.none *= 3
    def set_none(self, val):
        self.none = val
        return f"setted none to {val}"
    
class Derived(Base):
    def __init__(self,val1, val2, val4, val5):
        super().__init__(val1, val2)
        self.val4 = val4
        self.val5 = val5
        
    def multiple_by(self, val):
        super().multiply_by(val*2)