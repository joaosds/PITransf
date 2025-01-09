#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 13:22:32 2021

@author: Michael Perle

just playing around with the locals() function. No connection to anything physics related
"""

class FilePrinter:
    def __init__(self, 
                 print_frequency):
        self.print_frequency = print_frequency
        
    def print(self, value, i):
        if not (i%self.print_frequency):
            print(value)
        
class Calculation:
    def __init__(self, seed):
        self.seed = seed
        
    def generate_value(self, print_function):
        for i in range(0,10):
            value = i * self.seed
            print_function(value, i)
            
            
def function(calculation_instance, print_function):
    calculation_instance.generate_value(print_function)
    
def write_internal_parameter(mc_step, attempt, gradient_iteration_step, h_loc = None, energy_sum = None, energy_norm_sum = None, energy_qm = None, energy_classic = None, theta = None, wf_dif=None):
    function_dictionary = locals()
    for parameter in function_dictionary.keys():
       print(parameter)
       #print(function_dictionary[parameter])
            
        
    

file_printer = FilePrinter(2)
calculation_instance = Calculation(1)
function(calculation_instance, file_printer.print)
write_internal_parameter(1, 2, 3)