#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 13:29:21 2021

@author: Michael Perle
"""

import random
import warnings

import numpy as np

from RBM.Observable import Observable
from RBM.State import State


def abs_squared(x):
    return x * np.conj(x)


class Sampler:
    """
    contains the logic of the metropolis algorithm

    Attributes:

    - :class:`int` mc_step --> corresponds to n_{mc} is multiplied with system size to obtain N_{mc}=n_{mc}*N (see thesis)
    - :class:`int` attempt --> internal counter for current number of local update attempt
    - :class:`dict` occurrences --> Stores the probability distribution that metropolis algorithm generates (to save it for later analysis)
    - :class:`bool` save_most_abundant--> Another probably failed attempt to increase efficiency by storing the most abundant configurations (or rather the first 100 obtained)
    """

    def __init__(self, save_most_abundant=True):
        self.number_of_mc_steps = None
        self.isImportanceSampler = True
        self.mc_step = 0
        self.attempt = 0
        self.occurrences = {}
        self.errors = 0
        self.save_most_abundant = save_most_abundant
        self.loops = 0
    
    
    def sample_given_wf(self, state: State, number_of_mc_steps: int, observables: [Observable] = None,
                        return_configs: bool = False):
        """
        fJ
        this method can be used for test purposes to obtain observables of a given wave function
        it is called by Supplementary.sample_given_wf.py
        I used it mainly to test if the ferromagnet in HF-basis returns the mean field energy
        :param state: state that contains a configuration and does not have to contain a neural network
        :param number_of_mc_steps:
        :param observables:
        :param return_configs: flag that states if the configuration should be added to a list after every accapted
        local update and be returned in the end
        :return: None
        """
        config_list = None
        if return_configs:
            config_list = list()
        if observables is None:
            observables = []
        accepted_moves = 0
        if state.given_wavefunction is None:
            raise ValueError("no given wf given")
        for self.mc_step in range(number_of_mc_steps):
            for self.attempt in range(state.chain.length):
                random_index = random.choice(range(0, state.chain.length))
                p_update = abs_squared(state.calculate_wf_div(k=random_index))
                # if p_update > random.random():
                #     state.update(random_index)
                #     if return_configs:
                #         config_list.append(state.chain.configuration)
                #     accepted_moves += 1
                #     update_iterator = map(lambda o: o.sample(state), observables)
                if self.mc_step == 0 and self.attempt == 0:
                    update_iterator = map(lambda o: o.sample(state), observables)
                else:
                    update_iterator = map(lambda o: o.sample_last(), observables)
                try:
                    while True:
                        next(update_iterator)
                except StopIteration:
                    pass
        return config_list

    def sample_state(self, state: State, number_of_mc_steps = None, observables: [Observable] = None,
                     save_occurrences=False, assign_energy_to_occurrence = False):  # float
    
        p_update_dict = {}
        self.number_of_mc_steps = number_of_mc_steps
        """
        calculates the expectation value from a list of local observables
        fJ here the metropolis algorithm is implemented. Most complexity is probably added by the save occurences feature
        fJ Also we have the problem of having very little or even no accepted local updates when using the local update algorithm
        fJ If no local updates are accepted, it also makes no sense to perform an expensive GD or SR minimization step (since all (co-)variances) will be zero
        fJ This is why I keep trying to get accepted local updates by looping over the sampler
        fJ this function is used in the equilibrium sampling process (meaning the [Observables] list is empty)
        fJ and in the normal sampling process (where [Observable] list at minimum contains all observables necessary to minimize the energy. other observables like correlators are optional)
        fJ this was the first motivation to implement a Observable as class since each has a seperate counter in that case
        fJ some sources suggest to only use every Nth obtained configuration for sampling as this reduces correlation
        fJ this is not done here
        :param assign_energy_to_occurrence: used to label a histogram with energies
        :param state: contains neural network and configuration
        :param number_of_mc_steps: one mc step contains of number of sites local update attempts
        :param observables: list of local observables. can be None. the purpose of sampling is then to equilibrate to new
        network parameters after a rbm step
        :param save_occurrences: indicates if the wf_divs of the #self.save_configurations most abundant configurations should be saved
        :return: the number of accepted local updates
        """
        if observables is None:
            observables = []


        def update_dict(configuration_energy=None):
            """
            counts the occurrence of each configuration to determine the #self.save_configurations most abundant occurrences
            :return:
            """
            if configuration_energy is None:
                if state.chain.getBinary() in self.occurrences:
                    self.occurrences[state.chain.getBinary()] += 1
                else:
                    self.occurrences[state.chain.getBinary()] = 1
            else:
                if state.chain.getBinary() in self.occurrences:
                    self.occurrences[state.chain.getBinary()][0] += 1
                else:
                    self.occurrences[state.chain.getBinary()] = [1, configuration_energy]


        accepted_local_updates = 0
        while (not accepted_local_updates) and self.loops < 2:
            self.loops += 1
            # overcounts loops by one because the same function is called during equilibrium sampling after which we have loops=1 and if
            # the observable sampling is done we add 1 in the first iteration which yields loops=2 after one observable sampling iteration
            # this is subtracted again when plotting the files
            # fJ sorry for this, this was towards the very end of the project when I implemented/noticed this
            for self.mc_step in range(number_of_mc_steps):
                # fJ see section 6.4: N_{mc} = n_{mc} * N, that is the reason for the nested loop
                for self.attempt in range(state.chain.length):
                    # one-step consist of chain.length spin flip (local update) attempts
                    # the local updates follow the metropolis algorithm
                    # select random spin to flip
                    random_index = random.choice(range(0, state.chain.length))
                    # update condition
                    try:
                        if self.save_most_abundant and len(observables):
                            try:
                                p_update = p_update_dict[(state.chain.getBinary(), random_index)]
                            except KeyError:
                                p_update = abs_squared(state.calculate_wf_div(random_index))
                                if len(p_update_dict) < 100:
                                    p_update_dict[(state.chain.getBinary(), random_index)] = p_update
                        else:
                            p_update = abs_squared(state.calculate_wf_div(random_index))
                    except FloatingPointError as e:
                        #fJ this only occurs if highly unlikely configurations are obtained which can make sampling in HF basis or sampling with more variational parameter than basis states unstable
                        if "underflow" in str(e).lower():
                            p_update = 0.
                        elif "overflow" in str(e).lower():
                            p_update = 1
                        else:
                            raise e
                        self.errors += 1
                    except OverflowError:
                        p_update = 1
                        self.errors += 1
                    if p_update > random.random():
                        # success: update parameters, energy and spin configuration
                        state.update(random_index)
                        accepted_local_updates += 1
                        update_iterator = map(lambda o: o.sample(state), observables)
                    elif self.mc_step == 0 and self.attempt == 0:
                        update_iterator = map(lambda o: o.sample(state), observables)
                    else:
                        update_iterator = map(lambda o: o.sample_last(), observables)
                    try:
                        while True:
                            next(update_iterator)
                    except StopIteration:
                        pass
                    if save_occurrences:
                        # fJ this allows to compute a Histogram (occurrences as approx of the squared wave function amplitudes) with local energy for each configuration
                        # fJ intented to test the hypotheses "sampling in d-basis might be better because we have equally likely configurations s with very similar H_loc(s)"
                        if assign_energy_to_occurrence:
                            if len(observables) != 1 and observables[0].name != "hf_h_loc":
                                raise SystemError("Assigning energy to occurrences is not possible with these settings")
                            update_dict(observables[0].o_remember)
                        else:
                            update_dict()
            if self.errors:
                warnings.warn(f"\n{self.errors} over- and/or underflow errors encountered in sampler!")
            if not len(observables):
                break
        if len(observables) > 0:
            print("\n")

        return accepted_local_updates

    def __str__(self):
        return f"sampler: mc_step={self.mc_step}, attempt = {self.attempt}"
