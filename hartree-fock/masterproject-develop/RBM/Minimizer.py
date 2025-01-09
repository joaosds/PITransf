#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 09:12:25 2021

@author: Michael Perle
"""
from NeuralNetwork import VectorizedNetworkParameter
from State import State
from Observable import Observable
import numpy as np
# from threadpoolctl import threadpool_info, threadpool_limits

np.seterr(all='raise')


def regularize_S_carleo(S, iteration, l0, b, lmin):
    # fJ see https://arxiv.org/abs/1606.02318
    lp = max(l0 * b**iteration, lmin)
    S[np.diag_indices_from(S)] = S[np.diag_indices_from(S)] * (1 + lp)


def regularize_S_constant(S, lmin):
    # fJ see equation (3.43) and discussion of damping
    S[np.diag_indices_from(S)] = S[np.diag_indices_from(S)] + lmin


class Minimizer:
    def __init__(self, learning_rate, sr: bool, complex_parameter: bool, regularization_function: str = "constant"):
        """
        The purpose of the minimizer is to minimize the cost function which is in our case the energy
        :param learning_rate: basically determines the stepsize towards the minimum. see eta in theory for explanation
        :param sr: True if stochastic reconfiguration should be used to minimize the cost function. False for normal
        gradient descent.
        fJ explicitly constructing and inverting the covariance produces a bottleneck as demonstrated in Figure 5 in my thesis
        fJ Carleo refers in https://arxiv.org/abs/1606.02318 to a method of performing SR with same costs as doing only gradient descent
        """
        if isinstance(learning_rate, float):
            # would allow for a learning rate depending on the iteration as some authors suggest
            self.learning_rate = lambda iteration: learning_rate
        else:
            self.learning_rate = learning_rate

        self.sr = sr
        self.complex_parameter = complex_parameter
        if regularization_function is None or regularization_function.lower() == "none":
            self.regularization_function = lambda S, iteration: None
        elif regularization_function.lower() == "carleo":
            self.regularization_function = lambda S, iteration: regularize_S_carleo(S, iteration, l0=100, b=0.95, lmin=1e-4)
        elif regularization_function.lower() == "constant":
            self.regularization_function = lambda S, iteration: regularize_S_constant(S, lmin=1e-4)
        else:
            raise ValueError

    def update_rbm_parameters(self, state: State, h_loc_val, gradient_observables: VectorizedNetworkParameter,
                              composed_gradient_observables: VectorizedNetworkParameter, composed_sr_observables, iteration=None):
        """

        :param state: that contains configuration and neural network
        :param h_loc_val: current sampled energy
        :param gradient_observables: observables defined for gradient descent, see (3.29)
        :param composed_gradient_observables: composed observables defined for gradient descent, see (3.29)
        :param composed_sr_observables: composed observables defined for sr, see (3.34)
        :return:
        """

        # fJ see section 6.4
        effective_learning_rate = self.learning_rate(iteration) / state.chain.length
        # fJ according to (3.31)
        F = np.empty(gradient_observables.numberOfNetworkParameter, dtype= complex if self.complex_parameter else float)
        for k in range(gradient_observables.numberOfNetworkParameter):
            F[k] = composed_gradient_observables.get_vector_parameter(k).get(self.complex_parameter) - h_loc_val * np.conj(gradient_observables.get_vector_parameter(k).get(self.complex_parameter))


        if self.sr:
            # fJ according to (3.33)
            covariance_values = np.empty([gradient_observables.numberOfNetworkParameter, gradient_observables.numberOfNetworkParameter], dtype=complex if self.complex_parameter else float)
            for k in range(gradient_observables.numberOfNetworkParameter):
                for ks in range(gradient_observables.numberOfNetworkParameter):
                    covariance_values[k][ks] = composed_sr_observables[k][ks].get(self.complex_parameter) - np.conj(gradient_observables.get_vector_parameter(k).get()) *\
                                               gradient_observables.get_vector_parameter(ks).get()

            self.regularization_function(covariance_values, iteration)
            print("Inverting covariance matrix")
            inverse_S = np.linalg.pinv(covariance_values, hermitian=True)
            print("Multiplying with F")
            update_values = np.matmul(inverse_S, F)

        else:
            update_values = F

        state.neural_network.vectorised_parameter -= effective_learning_rate * update_values
        print("Update finished")
        return np.linalg.norm(update_values)

    def __str__(self):
        return f"gradient descent: learning rate: {self.learning_rate}"
