#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 13:03:41 2021

@author: Michael Perle
"""

import numpy as np
import warnings


class ObservableBase:

    def __init__(self, name, save_most_abundant=True, save_list=False):
        self.o_sum = 0.
        self.o_norm = 0
        self.name = name
        self.o_remember = None
        self.most_abundant_values = {}
        self.save_most_abundant = save_most_abundant
        self.save_list = None
        if save_list:
            self.save_list = []

    def sample(self, observed, probability = None):
        """
        :param observed: gets the result from the o_loc calculation adds the observed value to the sum counts how
        often it was called, this is the current norm. the counter allows to not call an observable after every local
        update event or to read out an observable during an MC step (both features currently not used)
        an observable remembers its last value. in case that the next local update is denied, the saved value will be used
        fJ probability is not None if full sampling is used. I apologize for the mess here, especially when it comes to composed observables
        """
        if probability is None:
            self.o_remember = observed
            self.o_sum += self.o_remember
            self.o_norm += 1
        else:
            self.o_remember = observed * probability
            self.o_sum += self.o_remember
            self.o_norm += probability
        if self.save_list is not None:
            self.save_list.append(self.o_remember)

    def sample_last(self):
        """
        is called if a local update is denied. the previously observed value is added to the sum
        """
        self.o_sum += self.o_remember
        self.o_norm += 1

    def get(self, expectComplex = True):  # float
        """
        matrix elements are generally complex numbers. The operators themselves are hermitesch. For a sufficiently large
        number of local updates, the imaginary part should go to 0.
        :return: o_loc which is the sum divided by the norm
        """
        if expectComplex:
            return self.o_sum / self.o_norm
        elif isinstance(self.o_sum, complex):
            return np.real(self.o_sum) / self.o_norm
        else:
            return self.o_sum / self.o_norm

    def reset(self):
        """
        resets the observable to zero
        """
        self.o_sum = 0.
        self.o_norm = 0
        self.most_abundant_values.clear()

    def __repr__(self):
        return repr("Observable " + self.name)

    def __str__(self):
        return self.name + ": " + str(self.get())

    def roundstr(self):
        """
        :return: a string representation of the observable
        """
        return self.name + ": " + f"{self.get():.5e}"

    def complex_str(self):
        """
        :return: does the same as __str__ but calling get with complex parameter
        """
        return self.name + ": " + str(self.get(expectComplex=True))

    def print_internals(self):
        """
        a function that can be helpful for debugging
        :return: a string representation of the observable: each attribute with its value is printed
        """
        [print("obj.%s = %r" % (attr, getattr(self, attr))) for attr in dir(self)]
