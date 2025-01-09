#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 13:36:11 2021
@author: Michael Perle
"""
import numpy as np
from ObservableBase import ObservableBase


class ComposedObservable(ObservableBase):
    def __init__(self, name, obs1: ObservableBase, obs2: ObservableBase, conjugateFirst=False):
        super().__init__(name)
        self.obs1 = obs1
        self.obs2 = obs2
        self.test_mapping = False
        self.test_order = False
        self.conjugateFirst = conjugateFirst

    def sample(self, state, probability=None):
        """
        a composed observable is a product of two already calculated observables
        beside this simple logic, the method contains two test-features
        fJ it is tested and it works
        :param state: unused parameter
        :return:
        """
        if self.test_order:
            if not self.o_norm + 1 == self.obs1.o_norm and self.o_norm + 1 == self.obs2.o_norm:
                raise TypeError("Composed observable can not be called before underlying observables have been sampled")
        if self.test_mapping:
            if not str(self.name) == f"{self.obs1.name}*{self.obs2.name}":
                print(f"composed: {self.name}, underlying: {self.obs1.name}*{self.obs2.name}")
                print(f"composed: {self.name}, underlying: {self.obs1.name}*{self.obs2.name}")
                raise TypeError("Mapping of composed and underlying observables is not correct")
        if self.conjugateFirst:
            observed = np.conj(self.obs1.o_remember) * self.obs2.o_remember
        else:
            observed = self.obs1.o_remember * self.obs2.o_remember
        if probability is None:
            super().sample(observed)
        else:
            # fJ This is a mess. observed = o_loc(configuration) and basically I multiply it with (probability/probability) because only then it is consistent with the sampling function
            # fJ which is also called by a "normal", i.e. not composed observable. My apologies, maybe you throw this whole observable as object idea over board and resort to numpy arrays
            # fJ but it works also like this :)
            super().sample(observed * (0 if probability < 1e-12 else 1/(probability*probability)), probability)
