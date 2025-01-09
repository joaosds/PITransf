#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 12:42:05 2022

@author: Michael Perle


just some unstructered test if the formfactor/unitary fulfill conditions
I think I used it to test some theory
This script is rather irrelavent now
"""
import numpy as np
from RBM.FermionModel import FermionModel
from scipy.stats import unitary_group

pi = np.pi
N = 10
h = 0
U = None

test_config = FermionModel(potential_function=lambda q: 1 / (q * q + 1),
                           ff1=lambda k, q: 1,
                           ff2=lambda k, q: np.sin(q) * (np.sin(k) + np.sin(k + q)),
                           h=float(h),
                           exact_configuration=[-1 for i in range(N)],
                           hf_unitary=U)

for k in test_config.k:
    for q in test_config.q:
        test_config.bigF(k[1], q[1])
        if not np.all(np.round(test_config.bigF(k[1], -q[1]), 10) == np.round(
                np.conjugate(test_config.bigF(k[1] - q[1], q[1])), 10)):
            print("relation (2.8) violated")

for _ in range(8):
    U = unitary_group.rvs(2)
    U_dag = np.conjugate(U).T
    up_holds = np.all(np.eye(2) == np.round(U_dag @ U, 12))
    if not up_holds:
        print("random matrix is not unitary")
    for k in test_config.k:
        for q in test_config.q:
            unitary_does_nothing = np.all(
                np.round(test_config.bigF(k[1], q[1]) @ np.conjugate(test_config.bigF(k[1], q[1])), 12) == np.round(
                    U_dag @ test_config.bigF(k[1], q[1]) @ np.conjugate(test_config.bigF(k[1], q[1])) @ U, 12))
            if not unitary_does_nothing:
                # F @ F_dag yields a* identity. U identity U_dag = identity
                print("UNITARY changes identity")

for _ in range(10000):
    ferromagnetic_contribution = 0
    list_of_unitary = [unitary_group.rvs(2) for _ in test_config.k]
    for q in test_config.q:
        for k in test_config.k:
            k_index = int(k[0])
            kmq_index = test_config.pbc(k[0] - q[0])
            big_F_kmq_q = test_config.bigF(k[1] - q[1], q[1])
            tilde_F_k_mq = np.conj(list_of_unitary[kmq_index]).T @ np.conjugate(big_F_kmq_q) @ list_of_unitary[k_index]
            tilde_F_kmq_q = np.conjugate(list_of_unitary[k_index]).T @ big_F_kmq_q @ list_of_unitary[kmq_index]
            ferromagnetic_contribution += tilde_F_k_mq[0][1] * tilde_F_kmq_q[1][0]
    print(ferromagnetic_contribution)
    if ferromagnetic_contribution <= 0:
        raise ValueError
