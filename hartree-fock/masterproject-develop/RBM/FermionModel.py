#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapted by João Sobral from main author Michael Perle.
Brief Description: Definition of functions, parameters,
and momentum points for the fermionic model.
"""
import numpy as np
from RBM.IsingModel import IsingModel

pi = np.pi

sigma_0 = np.eye(2)
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, complex(0, -1)], [complex(0, 1), 0]])
sigma_z = np.array([[1, 0], [0, -1]])


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
        nbz=1,
        sumOverG=False,
    ):
        """
        Initialize a fermionic model based on the IsingModel class.

        Parameters
        ----------
        potential_function : callable The potential energy function that defines the interaction between fermions.
        ff1 : callable
            Form factor function for even parity, expects k and q in units of pi.
        ff2 : callable
            Form factor function for odd parity, expects k and q in units of pi.
        ff3 : callable
            Form factor function, expected to be lambda k, q: 0.
        ff4 : callable
            Form factor function, expected to be lambda k, q: 0.
        h : float
            Strength of the kinetic energy, often referred to as 't' in fermionic
            model and 'h' in transverse ising model.
        length : int, optional
            Number of sites (electrons) in the system.
        exact_configuration : array_like or None, optional
            Initial occupation configuration. If None, the occupation is randomized.
        hf_unitary : np.ndarray or None, optional
            Unitary matrix for transforming d-fermions. If None, calculations are done in the d-basis.
        nbz : int, optional
            Number of Brillouin zones for potential energy calculation.
        sumOverG : bool, optional
            Flag indicating whether the Hamiltonian includes reciprocal lattice points.

        Notes
        -----
        This class is built upon the IsingModel class.
        fJ then again: sumOverG feature requires fixing if you want to use from factors where (6.15) does not hold
        basis d. if hf_unitary = [np.eye for i in range(N)]
        """
        super().__init__(
            J=None, h=h, length=length, exact_configuration=exact_configuration
        )

        self.hf_unitary = hf_unitary

        # List of matrices for the Tau_alpha,beta(k) hartree fock functions 3.11
        if self.hf_unitary is not None:
            self.Tau = [
                np.conjugate(hf_unitary_k).T @ sigma_x @ hf_unitary_k
                for hf_unitary_k in hf_unitary
            ]
        else:
            self.Tau = None


        self.potential = potential_function

        self.ff1 = ff1
        self.ff2 = ff2
        self.ff3 = ff3
        self.ff4 = ff4
        self.nbz = nbz
        self.sumOverG = sumOverG


        # def V(q):
        #     return (1 / (q**2 + 1))

        def construct_G(nbz, potential, length):
            """
            Defines the array of momentum values for potential G = np.array(i, G_{i}, V(G)) where i = 1,..., N,
            and N = self.length = number of sites or electrons.
            Main difference from this to construct_q is that G can also be in the boundaries.
            """
            G_pot = []
            G_index_blank = []
            G_units_of_2pi = []
            for i in range(
                -nbz,
                nbz + 1,
            ):
                potential_G = potential(i * 2 * np.pi, length)
                G_pot.append(potential_G)
                G_index_blank.append(i * length)
                G_units_of_2pi.append(i * 2 * np.pi)
            return np.array((G_index_blank, G_units_of_2pi, G_pot)).T

        def custom_sort(arr):
            # Identify the pattern within the array
            min_val = min(arr)
            max_val = max(arr)

            # Sort the array based on the identified pattern
            sorted_arr = sorted(arr, key=lambda x: (abs(x), x))

            return np.array(sorted_arr)

        def construct_q(nbz, potential, length):
            """
            Defines the array of momentum values for potential q = np.array(i, q_{i}, V(q)) where i = 1,..., N,
            and N = self.length = number of sites or electrons
            """
            q_pot = []
            q_index_blank = []
            for i in range(
                -nbz * length,
                nbz * length + 1,
            ):
                if i % length:  # q must not be in boundaries of BZ
                    # potential_q = potential(float(i) * 2.0 * pi / float(length), float(length))
                    # q_pot.append(potential_q)
                    q_index_blank.append(i)
            qtemp = np.array(
                [float(i) * 2.0 * pi / float(length) for i in q_index_blank]
            )
            # qtemp = custom_sort(qtemp)
            for i in range(len(qtemp)):
                potential_q = potential(qtemp[i], float(length))
                q_pot.append(potential_q)
            # print(q_pot)
            # print(custom_sort(qtemp))
            # return 1
            return np.array(
                (
                    q_index_blank,
                    qtemp,
                    q_pot,
                )
            ).T

        def construct_k(length):
            """
            Defines the array of momentum values k = np.array(i, k_{i}) where i = 1,..., N,
            and N = self.length = number of sites or electrons
            """
            k_index = np.array([int(i) for i in range(length)])
            k_value = np.linspace(
                start=-np.pi, stop=np.pi * (1.0 - 2.0 / float(length)), num=length
            )
            # print(custom_sort(k_value))
            return np.array((k_index, k_value)).T
            # return np.array((k_index, custom_sort(k_value))).T

        # self.G = construct_G(length)
        # self.q = construct_q(length)
        # self.k = construct_k(length)
        self.G = construct_G(self.nbz, self.potential, length)
        self.q = construct_q(self.nbz, self.potential, length)
        self.k = construct_k(length)
        # print(self.k)
        # print(self.q)
        # print(self.G)

    def bigF(self, k, q):
        """
        Full form factor in the pauli matrices basis in band space. See eq.
        3.13 from final Mathia's notes.
        """
        return (
            sigma_0 * self.ff1(k, q)
            + 1j * sigma_z * self.ff2(k, q)
            + sigma_y * self.ff3(k, q)
            + sigma_x * self.ff4(k, q)
        )
