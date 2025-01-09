#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 13:32:49 2022

@author: Michael Perle
"""
import random
import numpy as np
from RBM.Observable import Observable
from RBM.ImportanceSampler import Sampler
from RBM.State import State
from RBM.FermionModel import FermionModel  # ok
from RBM.fermionHfObservablebs import h_loc
import sys

# from RBM.fermionObservables import h_loc
h = float(sys.argv[1])
length = int(sys.argv[2])
path0 = str(sys.argv[3])
basis = str(sys.argv[4])

print(f"Path where files are saved: {path0}")

# h = 100000000000
# length = 6
# U = np.eye(2)
# chiral
if basis == "chiral":
    U = [np.eye(2) for i in range(length)]
elif basis == "band":
    U = [
        np.array(
            [
                [1, -1j],
                [1, 1j],
            ]
            / np.sqrt(2)
        )
        for i in range(length)
    ]


energy_observable = Observable("h_loc", lambda given_state: h_loc(given_state) / length)


def ff2(k, q):
    return np.sin(q) * (np.sin(k) + np.sin(k + q))


chain_normal = FermionModel(
    # potential_function=V,
    potential_function=lambda q, N: 1 / (1 + q * q) / (2*N),
    ff1=lambda k, q: 1,
    ff2=ff2,
    ff3=lambda k, q: 0,
    ff4=lambda k, q: 0,
    h=float(h),
    length=length,
    exact_configuration=[-1 for i in range(length)],
    hf_unitary=U,
)

# For FM chain
# exact_configuration=[random.choice([-1]) for i in range(length)],


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


chain = chain_normal
given_wf = given_wf_all_configs_equal

print(chain, given_wf)
print(f"t = {chain.h}")

state = State(neural_network=None, chain=chain, given_wf=given_wf)
sampler = Sampler()

sampler.sample_given_wf(state=state, number_of_mc_steps=100)


configs = sampler.sample_given_wf(
    state=state,
    number_of_mc_steps=2500,
    observables=[energy_observable],
    return_configs=False,
)
print("Getting energy for the state of reference: ", state.chain.configuration)
print(energy_observable)
with open(
    path0 + "enhf.txt",
    "w",
) as file1:
    file1.write(f"{energy_observable}")
    # file1.write("0")
