#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 12:34:04 2022

@author: Michael Perle
"""
from typing import Optional
import sys
import numpy as np
from IsingModel import IsingModel

pi = np.pi

sigma_0 = np.eye(2)
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, complex(0, -1)], [complex(0, 1), 0]])
sigma_z = np.array([[1, 0], [0, -1]])


# Should we add np.complex128 here?
# Y = sparse.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=np.complex128))
# Z = sparse.csr_matrix(np.array([[1, 0], [0, -1]], dtype=np.float64))


class FermionModel(IsingModel):
    def __init__(
        self,
        potential_function,
        ff1,
        ff2,
        ff3,
        ff4,
        h,
        length=None,
        exact_configuration=None,
        hf_unitary: np.ndarray = None,
        potential_over_brillouin_zones=1,
        sumOverG=False,
    ):
        """
        The fermionic model is defined by...
        :param potential_function: entirely repulsive
        :param ff1: formfactor even expects k,q in units of pi
        :param ff2: formfactor odd expects k,q in units of pi
        :param ff3: expected to be lambda k, q: 0
        :param ff4: expected to be lambda k, q: 0
        :param h: strength of kinetic energy. reffered to as t in theory
        :param length: number of sites
        :param exact_configuration: initial occuptaion. if None, the initial occupation is random
        :param hf_unitary: unitary which transforms the d-fermions. None means calculation in
        :param sumOverG: flag that states if the hamiltonian includes reciprocal lattice points or not
        fJ then again: sumOverG feature requires fixing if you want to use from factors where (6.15) does not hold
        basis d. if hf_unitary = [np.eye for i in range(N)]
        """
        super().__init__(
            J=None, h=h, length=length, exact_configuration=exact_configuration
        )
        self.hf_unitary = hf_unitary
        """a list of matrices"""
        print("this is your tau matrix")
        if self.hf_unitary is not None:
            self.Tau = [
                np.conjugate(hf_unitary_k).T @ sigma_x @ hf_unitary_k
                for hf_unitary_k in hf_unitary
            ]
            print(np.shape(self.tau))
        else:
            self.Tau = None
        self.potential = potential_function
        self.ff1 = ff1
        self.ff2 = ff2
        self.ff3 = ff3
        self.ff4 = ff4
        self.potential_over_brillouin_zones = potential_over_brillouin_zones
        self.sumOverG = sumOverG

        def __construct_G():
            G_pot = []
            G_index_blank = []
            G_units_of_2pi = []
            for i in range(
                -self.potential_over_brillouin_zones,
                self.potential_over_brillouin_zones + 1,
            ):
                potential_G = self.potential(i * 2 * pi)
                G_pot.append(potential_G)
                G_index_blank.append(i * self.length)
                G_units_of_2pi.append(i * 2 * pi)
            return np.array((G_index_blank, G_units_of_2pi, G_pot)).T

        """
        fJ k is implemented as list of [index, value] pairs where the index is an integer and value corresponds to the actual definition of k in units of pi
        fJ q is implemented as list of [index, value, V(q)] triples where the index is an integer and value corresponds to the actual definition of q in units of pi
        fJ V(q) is the precomputed value of the potential for each q
        """

        def __construct_q():
            q_pot = []
            q_index_blank = []
            for i in range(
                -self.potential_over_brillouin_zones * self.length,
                self.potential_over_brillouin_zones * self.length + 1,
            ):
                if i % self.length:  # q must not be elem RL
                    potential_q = self.potential(i * 2 * pi / self.length)
                    q_pot.append(potential_q)
                    q_index_blank.append(i)
            return np.array(
                (
                    q_index_blank,
                    np.array([i * 2 * pi / self.length for i in q_index_blank]),
                    q_pot,
                )
            ).T

        def __construct_k():
            k_index = np.array([int(i) for i in range(self.length)])
            k_value = np.linspace(
                start=-pi, stop=pi * (1 - 2 / self.length), num=self.length
            )
            return np.array((k_index, k_value)).T

        self.G = __construct_G()
        self.q = __construct_q()
        self.k = __construct_k()

        print(np.shape(self.G))
        print(np.shape(self.q))
        print(np.shape(self.k))
        print("test")
        print(self.q)
        print("test")
        print(self.k)
        print("test")
        print(self.G)

    def bigF(self, k, q):
        return (
            sigma_0 * self.ff1(k, q)
            + 1j * sigma_z * self.ff2(k, q)
            + sigma_y * self.ff3(k, q)
            + sigma_x * self.ff4(k, q)
        )
