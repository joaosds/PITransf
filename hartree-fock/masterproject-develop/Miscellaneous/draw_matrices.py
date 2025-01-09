#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 15:39:12 2021

@author: Michael Perle

this is just a file that writes matrices such that they can be displayed in latex. I was thinking of doing this for the presentation
"""

size = 10

string = ""
for i in range(size):
    for j in range(size):
        string += f"\\omega_{{{i}{j}}}"
        if j <9:
            string+=" & "
    string += "\\\\ "
    
print(string)

string = ""
for i in range(size):
    string += f"a_{i}\\\\ "
    
print(string)

string = ""
for i in range(size):
    string += f"b_{i}\\\\ "
    
print(string)