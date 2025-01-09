#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 13:34:39 2021

@author: Michael Perle

fJ this file contains the functions which are used to compute O_loc(s) in the case of the transverse field ising model
fJ it also contains the network derivatives apart from d_W_ij since this is composed of d_a_i and d_b_j (see composed observable)
"""

import numpy as np
np.seterr(all='raise')

#defining the observables that will be calculated

def d_a_i(state, i, j=None):
    return state.chain.configuration[i]

def d_b_j(state, j, i=None):
    return np.tanh(state.theta[j])


def c_s_delta(state, delta):
    correlation = 0
    for i in range(state.chain.length):
        correlation += state.chain.configuration[i] *state.chain.configuration[(i+delta)%state.chain.length]
    return correlation/state.chain.length
    
def c_x_delta(state, delta):
    correlation = 0
    for i in range(state.chain.length):
        correlation += state.calculate_wf_div(i,(i+delta)%state.chain.length)
    return correlation/state.chain.length
    

def h_loc(state):
        #calculates H_loc(s)
        # first term: classical ising
        energy_classic = 0.0
        #interaction of every neighbouring particles
        for index in range(state.chain.length):
            neighbour = state.chain.pbc(index+1)
            energy_classic +=  state.chain.configuration[index] * state.chain.configuration[neighbour]
        #multiplying with interaction strength
        energy_classic *= -state.chain.J 
        #second term: qm
        energy_qm = 0.0
        for i in range(state.chain.length):
            #no actual spin flip needed   
            energy_qm += state.calculate_wf_div(i)
        energy_qm *= -state.chain.h
        #adding the energies
        return (energy_classic + energy_qm)