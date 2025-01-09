#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 13:30:49 2021

@author: Michael Perle
"""

import numpy as np
from RBM.NeuralNetwork import NeuralNetwork
from RBM.IsingModel import IsingModel

np.seterr("raise")


class State:
    ####### theta and wf ########
    def __init__(self, neural_network: NeuralNetwork, chain: IsingModel, given_wf=None):
        self.neural_network = neural_network
        self.chain = chain
        if neural_network is not None:
            self.__initialize_theta()
        self.given_wavefunction = given_wf

    def __initialize_theta(self):
        # private method
        # theta is calculated anew after every change in RBM parameter
        # fJ this is what is referred to in the thesis as "effective angle"
        self.theta = np.zeros(
            self.neural_network.n_hidden_neurons,
            dtype=complex if self.neural_network.complex_parameter else float,
        )
        for j in range(self.neural_network.n_hidden_neurons):
            self.theta[j] = self.neural_network.get_local_field_b(j)
        for i in range(self.chain.length):
            for j in self.neural_network.get_connection_to_neurons(i):
                self.theta[j] += self.chain.configuration[
                    i
                ] * self.neural_network.get_weight(i, j)

    def __update_theta(self, flip_spin):
        for j in self.neural_network.get_connection_to_neurons(flip_spin):
            self.theta[j] -= (
                2
                * self.chain.configuration[flip_spin]
                * self.neural_network.get_weight(flip_spin, j)
            )

    def exact_wf(self):
        # fJ needed in the case of using the full sampler
        exponent_sum = sum(
            [
                self.neural_network.get_local_field_a(i) * self.chain.configuration[i]
                for i in range(self.chain.length)
            ]
        )
        prod = np.prod(
            [
                2
                * np.cosh(
                    self.neural_network.get_local_field_b(j)
                    + sum(
                        [
                            (self.neural_network.get_weight(i, j))
                            * self.chain.configuration[i]
                            for i in range(self.chain.length)
                        ]
                    )
                )
                for j in range(self.neural_network.n_hidden_neurons)
            ]
        )
        return np.exp(exponent_sum) * prod

    def calculate_wf_div(self, k=None, l=None):
        """
        fJ luckily the computation of all local observables and 
        acceptence probabilities come down to either calculate
        fJ div1() --> see (3.24) in the thesis or div2() -->
        \psi_\lambda([s]_{i,j})/\psi_\lambda(s) with [s]_{i,j} defined in (6.11)
        :param k: indicates (s')_k = -(s)_k
        :param l: indicates (s')_l = -(s)_l
        otherwise (s')_i = (s)_i for all i neq k,l
        :return: the frequently needed expression psi_lambda(s')/psi_lambda(s)
        where s indicates self.configuration
        leaves the configuration unaltered
        """
        if self.given_wavefunction is not None:
            if self.neural_network is not None:
                raise ValueError(
                    "using a test wave function together with a neural network is contradictive"
                )
            if l is None:
                denominator = self.given_wavefunction(self.chain)
                self.chain.flip_spin(k)  # flip to calculate
                numerator = self.given_wavefunction(self.chain)
                div1 = numerator / denominator
                self.chain.flip_spin(k)  # flip back get original state
                return div1
            elif k == l:
                return 1
            else:
                denominator = self.given_wavefunction(self.chain)
                self.chain.flip_spin(k)  # flip to calculate
                self.chain.flip_spin(l)  # flip to calculate
                numerator = self.given_wavefunction(self.chain)
                div2 = numerator / denominator
                self.chain.flip_spin(k)  # flip back get original state
                self.chain.flip_spin(l)  # flip back get original state
                return div2

        def __div1():
            """
            helper function
            :return: psi_lambda(s')/psi_lambda(s) if only k is given
            """
            exponential_term_div = np.exp(
                self.neural_network.get_local_field_a(k)
                * -2
                * self.chain.configuration[k]
            )
            product_term_div = 1
            for j in self.neural_network.get_connection_to_neurons(k):
                theta_new_j = self.theta[j] - 2 * self.chain.configuration[
                    k
                ] * self.neural_network.get_weight(k, j)
                product_term_div *= np.cosh(theta_new_j) / np.cosh(self.theta[j])
            return product_term_div * exponential_term_div

        def __div2():
            """
            helper function
            :return: psi_lambda(s')/psi_lambda(s) if k and l are given
            """
            if k == l:
                return 1
            else:
                # no actual spin flip needed to calculate div
                exponential_term_div = np.exp(
                    self.neural_network.get_local_field_a(k)
                    * -2
                    * self.chain.configuration[k]
                ) * np.exp(
                    self.neural_network.get_local_field_a(l)
                    * -2
                    * self.chain.configuration[l]
                )
                product_term_div = 1
                connections_from_k = set(
                    self.neural_network.get_connection_to_neurons(k)
                )
                connections_from_l = set(
                    self.neural_network.get_connection_to_neurons(l)
                )
                common_connections = connections_from_k & connections_from_l
                for j in common_connections:
                    theta_new_j = (
                        self.theta[j]
                        - 2
                        * self.chain.configuration[k]
                        * self.neural_network.get_weight(k, j)
                        - 2
                        * self.chain.configuration[l]
                        * self.neural_network.get_weight(l, j)
                    )
                    product_term_div *= np.cosh(theta_new_j) / np.cosh(self.theta[j])
                for j in connections_from_k - common_connections:
                    theta_new_j = self.theta[j] - 2 * self.chain.configuration[
                        k
                    ] * self.neural_network.get_weight(k, j)
                    product_term_div *= np.cosh(theta_new_j) / np.cosh(self.theta[j])
                for j in connections_from_l - common_connections:
                    theta_new_j = self.theta[j] - 2 * self.chain.configuration[
                        l
                    ] * self.neural_network.get_weight(l, j)
                    product_term_div *= np.cosh(theta_new_j) / np.cosh(self.theta[j])
                return product_term_div * exponential_term_div

        if l is None:
            return __div1()
        else:
            return __div2()

    def updateWithBinary(self, binary):
        # fJ used by full sampler
        stringRepresentation = format(binary, f"0{self.chain.length}b")
        arrayRepresentation = np.array(
            [1 if string == "1" else -1 for string in stringRepresentation]
        )
        self.chain.configuration = arrayRepresentation
        self.__initialize_theta()

    def update(self, flip_spin):
        # important to call after every update since calculating wf_div leaves the state (or rather configuration) unaltered
        if self.neural_network is not None:
            self.__update_theta(flip_spin)
        self.chain.flip_spin(flip_spin)
