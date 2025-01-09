#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 13:29:39 2021

@author: Michael Perle
"""
from RBM.ObservableBase import ObservableBase


class Observable(ObservableBase):
    def __init__(self, name: str, o_loc, i=None, j=None, delta=None, save_list=False):
        """
        :param name: of the observable, i.e. "hloc"
        :param o_loc: function that is used to calculate the observable
        :param i: determines point where configuration is accessed
        :param j: determines second point where configuration is accessed
        :param delta: used for correlation observable
        """
        super().__init__(name, save_list=save_list)
        self.o_loc = o_loc
        self.i = i
        self.j = j
        self.delta = delta
        self.value = None

    def sample(self, state, probability=None):
        """
        o_loc is evaluated for given state
        :param state: is used to evaluate o_loc
        :param probability: probability of the state
        """

        def apply_o_loc():
            if self.i is None and self.j is None and self.delta is None:
                return self.o_loc(state)
            elif self.i is not None and self.j is None:
                return self.o_loc(state, self.i)
            elif self.i is None and self.j is not None:
                return self.o_loc(state, self.j)
            elif self.i is not None and self.j is not None:
                return self.o_loc(state, self.i, self.j)
            elif self.delta is not None and self.i is None and self.j is None:
                return self.o_loc(state, self.delta)
            else:
                raise ValueError("Unsupported Observable")

        if self.save_most_abundant:
            try:
                observed = self.most_abundant_values[state.chain.getBinary()]
            except KeyError:
                observed = apply_o_loc()
                if len(self.most_abundant_values) < 100:
                    self.most_abundant_values[state.chain.getBinary()] = observed
        else:
            observed = apply_o_loc()

        self.value = observed  # Set the value attribute
        super().sample(observed, probability)
        return observed

    def __str__(self):
        # Assuming the result of o_loc is stored in an attribute called 'value'
        # If it's not, you'll need to modify this to access the correct attribute
        if hasattr(self, "value"):
            return f"{self.value.real:.15f}"
        else:
            return "Observable not sampled yet"

    def __repr__(self):
        return self.__str__()
