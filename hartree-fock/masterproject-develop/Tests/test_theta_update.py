import random
from unittest import TestCase

import numpy as np
from RBM.ImportanceSampler import Sampler
from RBM.NeuralNetwork import NeuralNetwork
from RBM.FermionModel import FermionModel
from RBM.State import State

"""
fJ This is something I used in a debugging process and which is not relevant since the bug I found using this test
fJ is fixed
"""


class TestTheta(TestCase):

    def testThetaUpdatesFullyConnected(self):
        N = 10
        t = 0.11
        equilibrium_steps = 200
        ff1 = lambda k, q: 1
        ff2 = lambda k, q: 1 * np.sin(q) * (np.sin(k) + np.sin((k + q)))
        ff3 = lambda k, q: 0
        ff4 = lambda k, q: 0
        potential_function = lambda q: 1 / (1 + q * q) / (2 * N)

        random.seed(5)
        M = N
        model = FermionModel(potential_function=potential_function, ff1=ff1, ff2=ff2, ff3=ff3, ff4=ff4, h=float(t), length=N, sumOverG=False)
        neural_network = NeuralNetwork(n_visible_neurons=N,
                                       n_hidden_neurons=M,
                                       complex_parameter=True,
                                       fully_connected=True,
                                       initial_vectorised_parameter=[random.random() for _ in range(N)] + [random.random() for _ in range(M)] + [random.random() for _ in range(N*M)])
        state = State(neural_network, model)

        theta_initial = [value for value in state.theta]

        sampler = Sampler()
        print('sampling', end="")
        sampler.sample_state(state, equilibrium_steps)
        theta_after_sampling = np.array([value for value in state.theta])
        state_new = State(neural_network, model)
        print("diff between sampled theta and newly initialized theta with config after sampling")
        print(state_new.theta - theta_after_sampling)
        print("diff to initial theta")
        print(state_new.theta - theta_initial)
        np.testing.assert_almost_equal(theta_after_sampling, state_new.theta)

    def testThetaUpdatesLooselyConnected(self):
        N = 10
        t = 0.11
        equilibrium_steps = 250
        ff1 = lambda k, q: 1
        ff2 = lambda k, q: 1 * np.sin(q) * (np.sin(k) + np.sin((k + q)))
        ff3 = lambda k, q: 0
        ff4 = lambda k, q: 0
        potential_function = lambda q: 1 / (1 + q * q) / (2 * N)
        M = N
        for w in range(2,M+1):
            with self.subTest(w=w):

                random.seed(1)
                model = FermionModel(potential_function=potential_function, ff1=ff1, ff2=ff2, ff3=ff3, ff4=ff4, h=float(t), length=N, sumOverG=False)
                neural_network = NeuralNetwork(n_visible_neurons=N,
                                               n_hidden_neurons=M,
                                               complex_parameter=True,
                                               fully_connected=False,
                                               weights_per_visible_neuron=w,
                                               initial_vectorised_parameter=[random.random() for _ in range(N)] + [random.random() for _ in range(M)] + [random.random() for _ in range(w*N)])
                state = State(neural_network, model)

                theta_initial = [value for value in state.theta]

                sampler = Sampler()
                print('sampling', end="\n")
                sampler.sample_state(state, equilibrium_steps)
                theta_after_sampling = np.array([value for value in state.theta])
                state_new = State(neural_network, model)
                print("diff between sampled theta and newly initialized theta with config after sampling")
                print(state_new.theta - theta_after_sampling)
                print("diff to initial theta")
                print(state_new.theta - theta_initial)
                np.testing.assert_almost_equal(theta_after_sampling, state_new.theta)