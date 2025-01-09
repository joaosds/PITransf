#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 13:29:21 2021

@author: Michael Perle
"""

import random
import sys  # for printing of progress
import warnings

import numpy as np

from Observable import Observable
from State import State


class Sampler:
    """
    contains the logic of the full sampler
    """

    def __init__(self):
        self.isImportanceSampler = False
        self.occurrences = {}
        self.loops = 1


    def sample_state(self, state: State, number_of_mc_steps=1024, observables: [Observable] = None,
                     save_occurrences=False, assign_energy_to_occurrence = False):  # float
        self.number_of_mc_steps = 1 << state.chain.length
        """
        calculates the expectation value from a list of local observables
        :param assign_energy_to_occurrence: used to label a histogram with energies
        :param state: contains neural network and configuration
        :param number_of_mc_steps: not used
        :param observables: list of local observables. can be None. the purpose of sampling is then to equilibrate to new
        network parameters after a rbm step
        :param save_occurrences: indicates if the wf_divs of the #self.save_configurations most abundant configurations should be saved
        :return: number of basis states
        """
        if observables is None:
            return "no equilibrium sampling needed"
        for binary_state in np.arange(2**state.chain.length):
            state.updateWithBinary(binary_state)
            exact_wf = state.exact_wf()
            p_configuration = exact_wf * np.conj(exact_wf)
            update_iterator = map(lambda o: o.sample(state, probability=p_configuration), observables)
            try:
                while True:
                    next(update_iterator)
            except StopIteration:
                pass
            if save_occurrences:
                if assign_energy_to_occurrence:
                    if len(observables) != 1 and observables[0].name != "hf_h_loc":
                        raise SystemError("Assigning energy to occurrences is not possible with these settings")
                    configuration_energy = observables[0].o_remember/p_configuration
                    self.occurrences[state.chain.getBinary()] = [str(p_configuration), str(configuration_energy)]
                else:
                    self.occurrences[state.chain.getBinary()] = [str(p_configuration)]

        if len(observables) > 0:
            print("\n")
        return 2**state.chain.length

    def __str__(self):
        return "sampler: full sampler, sum over 2^N configurations"
