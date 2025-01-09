#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 13:32:49 2022

@author: Michael Perle
"""
import random
import sys

sys.path.append(".")  # Adds higher directory to python modules path.
import numpy as np

from RBM.Observable import Observable
from RBM.ImportanceSampler import Sampler
from RBM.State import State
from RBM.FermionModel import FermionModel

# from RBM.fermionObservables import h_loc
from RBM.fermionHfObservablebs import h_loc as hf_h_loc
from RBM.fermionObservables import h_loc

random.seed(1)

# print("test modified matrix elements")
# h = 0
# length = 10
# # ff2 = lambda k, q: 0
# ff2 = lambda k, q: np.sin(q) * (np.sin(k) + np.sin(k + q))
# # U = np.load(
# #     f"C:\\Users\\Hester\\PycharmProjects\\masterproject\\Results\\HF_results\\ultimate_results\\Uk_N={length}_t={h:.5e}.npy")
# # U = [np.eye(2) for i in range(length)]
# if any(
#     [
#         np.linalg.norm(np.eye(2) - np.matmul(np.conjugate(Uk).T, Uk)) > 1e-12
#         or np.linalg.norm(np.eye(2) - np.matmul(Uk, np.conjugate(Uk).T)) > 1e-12
#         for Uk in U
#     ]
# ):
#     raise ValueError(f"loaded U={U} is not unitary")
#
energy_observable = Observable("h_loc", lambda given_state: hf_h_loc(given_state))

# sign of U doesn't change result
# for Uk in U:
#     if np.matmul(np.conjugate(Uk).T, Uk)[0][0]<0 and np.matmul(np.conjugate(Uk).T, Uk)[1][1] <0:
#         Uk *= -1


# print("test d basis matrix elements")
h = 0
length = 6
N = 6
# ff2 = lambda k, q: 0
ff2 = lambda k, q: np.sin(q) * (np.sin(k) + np.sin(k + q))
U = None
U = np.eye(2)
# energy_observable = Observable("h_loc", lambda given_state: h_loc(given_state))

chain_only_kinetic = FermionModel(
    potential_function=lambda q, N: 1 / (q * q + 1) / (2 * N),
    ff1=lambda k, q: 1,
    ff2=ff2,
    ff3=lambda k, q: 0,
    ff4=lambda k, q: 0,
    h=float(h),
    length=length,
    exact_configuration=[random.choice([-1, 1]) for i in range(length)],
    hf_unitary=U,
)

chain_normal = FermionModel(
    potential_function=lambda q, length: 1 / (q * q + 1) / (2 * length),
    ff1=lambda k, q: 1,
    ff2=ff2,
    ff3=lambda k, q: 0,
    ff4=lambda k, q: 0,
    h=float(h),
    length=length,
    exact_configuration=[random.choice([-1, 1]) for i in range(length)],
    hf_unitary=U,
)

chain_only_potential = FermionModel(
    potential_function=lambda q, N: 1 / (q * q + 1) / (2 * N),
    ff1=lambda k, q: 1,
    ff2=ff2,
    ff3=lambda k, q: 0,
    ff4=lambda k, q: 0,
    length=length,
    h=0,
    exact_configuration=[random.choice([1]) for i in range(length)],
    hf_unitary=U,
)

ferromagnetic_test_chain = FermionModel(
    potential_function=lambda q, N: 1 / (q * q + 1) / (2 * N),
    ff1=lambda k, q: 1,
    ff2=ff2,
    ff3=lambda k, q: 0,
    ff4=lambda k, q: 0,
    length=length,
    h=float(h),
    exact_configuration=[random.choice([-1]) for i in range(length)],
    hf_unitary=U,
)


def given_wf_only_kinetic(fermionSpinConfiguration: FermionModel):
    product = 1
    for k in fermionSpinConfiguration.k:
        product *= (
            1 if fermionSpinConfiguration.configuration[int(k[0])] == 1 else 0
        ) - (
            np.sign(fermionSpinConfiguration.h * np.cos(k[1]))
            if fermionSpinConfiguration.configuration[int(k[0])] == -1
            else 0
        )
    return product


def given_wf_all_configs_equal(fermionSpinConfiguration: FermionModel):
    """
    :param fermionSpinConfiguration:
    :return: 1 always
    """
    return 1


def ferromagnetic_wf(fermionSpinConfiguration: FermionModel):
    """
    :param fermionSpinConfiguration:
    :return: the groundstate wavefunction if only potential, no kinetic energy is present
    """
    product1 = 1
    product2 = 1
    for k in fermionSpinConfiguration.k:
        product1 *= 1 if fermionSpinConfiguration.configuration[int(k[0])] == 1 else 0
        product2 *= 1 if fermionSpinConfiguration.configuration[int(k[0])] == -1 else 0
    return product1 + product2


"""
# Option 1: only kinetic
chain = chain_normal
given_wf = given_wf_only_kinetic
"""

"""
# Option 2: only potential
chain = chain_only_potential
given_wf = ferromagnetic_wf
"""


# # Option 3: test if ferromagnetic yields mean field
# # chain = ferromagnetic_test_chain
# chain = chain_only_kinetic
# given_wf = ferromagnetic_wf
# given_wf = given_wf_only_kinetic
# print(chain, given_wf)


# Option 4: test if given wf for t>>|V| yields same results for modified matrix elements if t=0 or generally U is set
# manually to eye
chain = chain_normal
given_wf = given_wf_all_configs_equal
# no physical reasoning behind the test. it's just that given_wf_all_configs_equal accepts
# all configurations and works with
# t=0 and v \neq 0

# # Option 5: custom test for t=3
# chain = chain_normal
# chain.configuration = -np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1])
# given_wf = (
#     lambda fermionSpinConfiguration: 1
#     if all(
#         fermionSpinConfiguration.configuration
#         == -np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1])
#     )
#     else 0
# )
# restore final rbm+hf state for t=3
# t=0 and v \neq 0

# begin of test

print(f"t = {chain.h}")
print("pacoca")

state = State(neural_network=None, chain=chain, given_wf=given_wf)
print(state.chain.configuration)
sampler = Sampler()

sampler.sample_given_wf(state=state, number_of_mc_steps=100)

print(state.chain.configuration)

# wf_div_list = [state.calculate_wf_div(k=int(k[0]), given_wavefunction=given_wf) for k in state.chain.k]
# p_update_list = [wf_div**2 for wf_div in wf_div_list]
# print(p_update_list)


configs = sampler.sample_given_wf(
    state=state,
    number_of_mc_steps=1500,
    observables=[energy_observable],
    return_configs=True,
)

print(state.chain.configuration)
print(energy_observable)
print(energy_observable)
print(energy_observable)
# print(f"imag part = {np.imag(imaginary_uneven_part(state))}")
# print(f"h1b energy contr = {compareh(state)}")
print("end")
print(f"check if h={h}, N = {length}, ff2 and U were set properly")
# print(configs)
