#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 11:57:22 2022

@author: Michael Perle
"""

from HFNumerics import HartreeFockCalculations
from RBM.FermionModel import FermionModel
import os
import numpy as np
import sys
import inspect

sys.path.append(os.path.normpath(os.getcwd() + os.sep + os.pardir))


def get_random_projector():
    """
    :return: 2x2 matrix that fulfills idempotency and conj(P).T = P
    see (6.40)
    """
    normalize = (
        lambda vec: 1
        / np.sqrt(vec[0] * np.conj(vec[0]) + vec[1] * np.conj(vec[1]))
        * vec
    )
    ketbra = lambda vec1: np.array(
        [
            [vec1[0] * np.conj(vec1[0]), vec1[0] * np.conj(vec1[1])],
            [vec1[1] * np.conj(vec1[0]), vec1[1] * np.conj(vec1[1])],
        ]
    )
    random_vector = np.random.rand(1, 2) + 1j * np.random.rand(1, 2)
    random_vector = normalize(random_vector[0])
    return ketbra(random_vector)


def get_interactive_edge_case_projector():
    # see (6.40)
    return np.array([[0, 0], [0, 1]])


def get_kinetic_edge_case_projector():
    # see (6.40)
    return np.array([[1, -1], [-1, 1]]) * 0.5


# Invoke iterate
def main(
    ff1,
    ff2,
    ff3,
    ff4,
    potential_function,
    t,
    N,
    identifier,
    Uflag,
    path0,
    acc,
    info_string="",
    p0=None,
):
    """
    fJ this calls the HF routine but writes a config file before
    fJ you could simply call
    HFClass = HartreeFockCalculations(model=model, p0=p0)
    converged_energy, final_HF_Hamiltonian, final_projector = HFClass.iterate(1e-12, 3000, path=result_path)
    """
    result_path = (
        os.path.normpath(os.getcwd() + os.sep + os.pardir)
        + f"/masterproject-develop/RawResults/newest/{identifier}/"
    )
    try:
        os.makedirs(result_path)
        os.mkdir(result_path)
        print(f"creating directory: {result_path}")
    except FileExistsError:
        print(f"saving plot to already existing directory: {result_path}")
    if p0 is None:
        p0 = [np.array([[0, 0], [0, 1]]) for _ in range(N)]

    model = FermionModel(
        potential_function=potential_function,
        ff1=ff1,
        ff2=ff2,
        ff3=ff3,
        ff4=ff4,
        h=float(t),
        length=N,
        sumOverG=False,
    )
    with open(result_path + "config.txt", "a") as config_file:
        formfactor_string = [
            inspect.getsource(ff1),
            inspect.getsource(ff2),
            inspect.getsource(ff3),
            inspect.getsource(ff4),
            inspect.getsource(potential_function),
            f"t={t}\n",
            f"N={N}\n",
            f"p0=\n{p0}\n",
        ]
        config_file.writelines(formfactor_string + [info_string])
    HFClass = HartreeFockCalculations(Uflag, model=model, path0=path0, p0=p0)
    converged_energy, final_HF_Hamiltonian, final_projector = HFClass.iterate(
        acc, 60000, path=result_path
    )
    return converged_energy


# fj I leave this here as an example how I called the functions starting the actual hf-algorithm (sequentially)
t = float(sys.argv[1])
Ni = int(sys.argv[2])
path0 = str(sys.argv[3])
Uflag = str(sys.argv[4])  # Decides if you should save the HF-basis U transformation
acc = float(sys.argv[5])

print(f"Starting generation of HF and ED results with t={t}, N={Ni} \n")
print(f"Path where files are saved: {path0}")
for N in [Ni]:
    identifier = f"asd-N={Ni}"
    ff1 = lambda k, q: 1
    ff2 = lambda k, q: 0.9 * np.sin(q) * (np.sin(k) + np.sin((k + q)))
    ff3 = lambda k, q: 0
    ff4 = lambda k, q: 0
    potential_function = lambda q, N: 1 / (1 + q * q) / (2 * N)

    randomVar1 = 0
    randomVar2 = 0
    randomVar3 = 0
    randomVar4 = 0
    randomVar5 = 0
    randomVar6 = 0
    randomVar7 = 0
    randomFreq1 = 0
    randomFreq2 = 0
    randomFreq3 = 0
    randomFreq4 = 0
    randomFreq5 = 0

    info_string = f"randomVar1 = {randomVar1}\nrandomVar2 = {randomVar2}\nrandomVar3 = {randomVar3}\nrandomVar4 = {randomVar4}\nrandomVar5 = {randomVar5}\nrandomVar6 = {randomVar6}\nrandomVar7 = {randomVar7}\n"
    info_string += f"randomFreq1 = {randomFreq1}\nrandomFreq2 = {randomFreq2}\nrandomFreq3 = {randomFreq3}\nrandomFreq4 = {randomFreq4}\nrandomFreq5 = {randomFreq5}\n"
    for t in [t]:
        main(
            ff1,
            ff2,
            ff3,
            ff4,
            potential_function,
            t,
            N,
            identifier,
            Uflag,
            path0=path0,
            acc=acc,
            info_string="",
            p0=[get_random_projector() for _ in range(N)],
            # p0=None,
        )
