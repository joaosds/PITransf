#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 13:24:35 2021

@author: Michael Perle
"""
import random

import numpy as np

np.seterr(all='raise')


class IsingModel:
    # @param J: classical coupling coefficient
    # @param h: strength of the external field
    # @param cain_length: number of spin sites
    # @param exact_configuration
    def __init__(self, J, h, length=None, exact_configuration=None):
        """
        :param J: classical coupling coefficient
        :param h: strength of the external field
        :param length: number of sites
        :param exact_configuration: initial occupation. if None, the initial occupation is random
        """
        # if no exact configuration is given, all spins are initialized as one per default
        self.J = J
        self.h = h
        if exact_configuration is None and length is not None:
            self.configuration = np.array([random.choice([-1,1]) for _ in range(length)], dtype=int)
            self.length = length
        elif all(spin == -1 or spin == 1 for spin in exact_configuration):
            self.length = len(exact_configuration)
            self.configuration = exact_configuration
        else:
            raise ValueError("Failed to create a SpinConfiguration")

    def flip_spin(self, index):
        self.configuration[index] *= -1

    def pbc(self, index):
        return int(int(index) % self.length)

    def reset_configuration_to_random(self):
        self.configuration = np.array([random.choice([-1, 1]) for _ in range(self.length)], dtype=int)

    def printConfiguration(self):
        # just for fun, not really needed anywhere
        # prints up arrow for +1 and down arrow for -1
        up = u'\u2191'
        down = u'\u2193'
        arrow_list = [up if spin == 1 else down if spin == -1 else None for spin in self.configuration]
        return str(arrow_list)

    def getBinary(self):
        list1 = [1 if x == 1 else 0 for x in self.configuration]
        return int("".join(map(str, list1)), 2)

    def __str__(self):
        return f"J: {self.J}, h: {self.h}, length: {self.length}, current_configuration: {self.printConfiguration()}"
