#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 13:25:09 2021

@author: Michael Perle
"""

import random
import numpy as np

np.seterr(all='raise')


class VectorizedNetworkParameter:
    """
    fJ I basically defined my own vector class to make computations (at least for me) more transparent
    fJ Using numpy arrays instead might be in the end the better and more efficient option
    fJ Doing it like this helped me to ensure consistency and to be better able to define sparsely connected RBMs
    """
    def __init__(self, n_visible_neurons, n_hidden_neurons, vectorized_parameter_type, fully_connected, weights_per_visible_neuron):

        if not fully_connected and n_visible_neurons != n_hidden_neurons:
            raise ValueError("N must match M in a loosely connected RBM")

        self.n_hidden_neurons = n_hidden_neurons
        self.n_visible_neurons = n_visible_neurons
        self.fully_connected = fully_connected

        if self.fully_connected:
            if weights_per_visible_neuron is not None:
                if weights_per_visible_neuron != n_hidden_neurons:
                    raise ValueError(f"Invalid Argument: weights_per_visible_neuron={weights_per_visible_neuron}. Fully connected means that the weights per visible neuron equal the number of hidden neurons: {n_hidden_neurons}.")
            self.weights_per_visible_neuron = self.n_hidden_neurons
        else:
            self.weights_per_visible_neuron = weights_per_visible_neuron

        if self.fully_connected:
            self.get_connection_to_neurons = lambda i: [j for j in range(self.n_hidden_neurons)]
        else:
            self.get_connection_to_neurons = lambda i: [(i + c) % self.n_visible_neurons for c in range(self.weights_per_visible_neuron)]

        self.numberOfNetworkParameter = self.n_hidden_neurons + self.n_visible_neurons + self.n_visible_neurons * self.weights_per_visible_neuron

        self.vectorised_parameter = np.empty(self.numberOfNetworkParameter, dtype=vectorized_parameter_type)

    def get_local_field_a(self, i):
        if i >= self.n_visible_neurons or not isinstance(i, int):
            raise IndexError
        return self.vectorised_parameter[i]

    def get_local_field_b(self, j):
        if j >= self.n_hidden_neurons or not isinstance(j, int):
            raise IndexError
        return self.vectorised_parameter[self.n_visible_neurons + j]

    def get_weight(self, i, j):
        if j >= self.n_hidden_neurons or not isinstance(j, int):
            raise IndexError
        if i >= self.n_visible_neurons or not isinstance(i, int):
            raise IndexError
        if not self.fully_connected:
            if j not in self.get_connection_to_neurons(i):
                return 0
        if self.fully_connected:
            return self.vectorised_parameter[self.n_visible_neurons + self.n_hidden_neurons + i * self.weights_per_visible_neuron + j]
        else:
            return self.vectorised_parameter[self.n_visible_neurons + self.n_hidden_neurons + i * self.weights_per_visible_neuron + (j-i) % self.n_hidden_neurons]

    def get_vector_parameter(self, k):
        return self.vectorised_parameter[k]

    def set_local_field_a(self, i, value):
        if i >= self.n_visible_neurons:
            raise IndexError
        self.vectorised_parameter[i] = value

    def set_local_field_b(self, j, value):
        if j >= self.n_hidden_neurons:
            raise ValueError
        self.vectorised_parameter[self.n_visible_neurons + j] = value
        pass

    def set_weight(self, i, j, value):
        # the order is: row0, row1, ... i.e. w[0,0], w[0,1], w[0,2], ..., w[0,N-1], w[1;0], ..., w[N-1,N-1]
        if j >= self.n_hidden_neurons or not isinstance(j, int):
            raise IndexError(f"invalid j: {j}")
        if i >= self.n_visible_neurons or not isinstance(i, int):
            raise IndexError(f"invalid i: {i}")
        if not self.fully_connected:
            if j not in self.get_connection_to_neurons(i):
                raise ValueError(f"w[{i}][{j}] does not exist in a loosely connected RBM with {self.weights_per_visible_neuron} connections per visble neuron")
        if self.fully_connected:
            self.vectorised_parameter[self.n_visible_neurons + self.n_hidden_neurons + i * self.weights_per_visible_neuron + j] = value
        else:
            self.vectorised_parameter[self.n_visible_neurons + self.n_hidden_neurons + i * self.weights_per_visible_neuron + (j-i) % self.n_hidden_neurons] = value


    def set_vector_parameter(self, k, value):
        self.vectorised_parameter[k] = value


class NeuralNetwork(VectorizedNetworkParameter):
    """
    this class is a model class beside the logic in the init function
    internally all network parameters are stored in the vectorised parameter attribute. the other attributes just exist
    to give easier access to the desired parameter
    """

    def __init__(self, n_visible_neurons, n_hidden_neurons, initial_vectorised_parameter=None, complex_parameter: bool = False, fully_connected: bool = True, weights_per_visible_neuron=None):
        """
        :param n_visible_neurons: number of electron or spin sites.
        :param n_hidden_neurons: initializes the network with a given, consistent number of hidden neurons. can't be changed in an instance.
        per default all network parameters are set to random. random means in that context that each a,b,w is initialized
        with a random number in the range of [-1e-4, +1e-4] (in case of real parameter, otherwise also multiplied with random phase). setting to zero can lead to an undesired bias, therefore this noise
        was suggested. experiments with bias lead to inferior results
        """

        self.complex_parameter = complex_parameter

        super().__init__(n_visible_neurons=n_visible_neurons, n_hidden_neurons=n_hidden_neurons, vectorized_parameter_type= complex if complex_parameter else float, fully_connected=fully_connected, weights_per_visible_neuron=weights_per_visible_neuron)



        initial_magnitude = 0.0001

        # initialising local_field_a
        for i in range(n_visible_neurons):
            if self.complex_parameter:
                self.set_local_field_a(i=i, value=np.exp(complex(0, random.uniform(0, 2 * np.pi))) * initial_magnitude)
            else:
                self.set_local_field_a(i=i, value=random.uniform(-1, 1) * initial_magnitude)

        # initialising local_field_b
        for j in range(n_hidden_neurons):
            if self.complex_parameter:
                self.set_local_field_b(j=j, value=np.exp(complex(0, random.uniform(0, 2 * np.pi))) * initial_magnitude)
            else:
                self.set_local_field_b(j=j, value=random.uniform(-1, 1) * initial_magnitude)

        # initialising weights
        for i in range(self.n_visible_neurons):  # row
            for j in self.get_connection_to_neurons(i):  # column
                if complex_parameter:
                    self.set_weight(i=i, j=j, value=np.exp(complex(0, random.uniform(0, 2 * np.pi))) * initial_magnitude)
                else:
                    self.set_weight(i=i, j=j, value=random.uniform(-1, 1) * initial_magnitude)


        if initial_vectorised_parameter is not None:
            if len(initial_vectorised_parameter) != self.numberOfNetworkParameter:
                raise ValueError
            else:
                for i in range(self.numberOfNetworkParameter):
                    self.vectorised_parameter[i] = initial_vectorised_parameter[i]
