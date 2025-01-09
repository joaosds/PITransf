#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 17:09:55 2021

@author: Michael Perle
"""

import numpy as np
import os
import multiprocessing

np.seterr(all="raise")
import sys

# sys.path = ["/net/mungo3/csat6559/masterproject/", "/net/mungo3/csat6559/masterproject/RBM"]
sys.path.append(os.path.normpath(os.getcwd()))
from FermionMain import main as fermion_main


def start_calculation(
    h,
    N,
    identifier,
    basis,
    doImportanceSampling,
    eta,
    takeOtherUWhere_t_is,
    M,
    weights_per_visible_neuron,
    initial_vectorised_parameter,
    used_model,
    seed,
    complex_param,
    sr,
):
    used_model = used_model
    seed = seed

    lr = eta
    sr = sr
    gradient_descent = not sr

    """
    fJ
    we experimented with using different form factors, i.e. also ff3 and ff4.
    !!! NB !!!
    correct output is not guaranteed if you set ff3 and ff4 not equal to zero
    correct output is also not guaranteed if you violate the condition ff2(k,G) = 0
    this is because equation 6.15 in my thesis must hold for correct output. if you want to experiment with ff that violate that constraints you might have to
    implement the matrix elements that include the sum over reciprocal lattice points
    """

    randomVar1 = 1
    randomVar2 = 1
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
    ff1 = lambda k, q: randomVar1
    ff2 = lambda k, q: randomVar2 * np.sin(q) * (np.sin(k) + np.sin((k + q)))
    ff3 = lambda k, q: 0
    ff4 = lambda k, q: 0
    potential_function = lambda q: 1 / (1 + q * q) / (2 * N)

    M = M
    doFullConnection = False
    # fJ N=M is the condition of sparsely connected RBM. If you want to simulate using fully connected RBM with \alpha>1, you have to state this here

    """
    jF
    this is printed to config file in FermionMain.py
    """

    info_string = f"randomVar1 = {randomVar1}\nrandomVar2 = {randomVar2}\nrandomVar3 = {randomVar3}\nrandomVar4 = {randomVar4}\nrandomVar5 = {randomVar5}\nrandomVar6 = {randomVar6}\nrandomVar7 = {randomVar7}\n"
    info_string += f"randomFreq1 = {randomFreq1}\nrandomFreq2 = {randomFreq2}\nrandomFreq3 = {randomFreq3}\nrandomFreq4 = {randomFreq4}\nrandomFreq5 = {randomFreq5}\n"
    info_string += f"doFullConnection = {doFullConnection}\n"

    # fJ this path says where to look for the unitary transformation obtained by HF which is loaded in FermionMain.py
    hf_result_path = os.path.normpath(
        os.getcwd()
        + f"/RawResults/HF_Results/{'original0_p0rand_longConv_final_N=6-N=6'}"
    )
    # this path tells where to store the results
    RBM_result_path = os.path.normpath(os.getcwd() + "/RawResults/newest/")

    # fJ this contains the unitaries required do the calculations in c/d-basis
    explict_U = None
    if basis == "d_bar":
        explicit_U = None
        a_bias = None
    elif basis == "d":
        explicit_U = [np.eye(2) for _ in range(N)]
        a_bias = None
    elif basis == "c":
        explicit_U = [np.array([[1, -1j], [1, 1j]]) / np.sqrt(2) for _ in range(N)]
        a_bias = None
    else:
        raise ValueError(f"No Basis specified: basis={basis}")

    # fJ you can specify which matrix elements to use, i.e. which model to use by either setting used_model to "fermionic model" or "ising model"
    # fj here it only determines the file name, everything else happens in FermionMain.py
    if used_model == "fermionic model":
        add_file_label = (
            hf_result_path[hf_result_path.rfind("/") + hf_result_path.rfind("\\") + 1 :]
            + "_"
            + basis
            + "-basis_sampler-"
            + ("importance" if doImportanceSampling else "full")
            + f"_weights_per_visible_neuron={N if doFullConnection else weights_per_visible_neuron}"
            + f"_unitary_t={takeOtherUWhere_t_is}"
            + f"_seed={seed}_proof_of_concept_fermionic_allow_MF"
        )
    elif used_model == "ising model":
        add_file_label = (
            "ising_model_"
            + identifier
            + "_"
            + "sampler-"
            + ("importance" if doImportanceSampling else "full")
            + f"_weights_per_visible_neuron={N if doFullConnection else weights_per_visible_neuron}_seed={seed}_sr={sr}_complex={complex_param}"
        )
    else:
        raise ValueError("No model specified")

    print(40 * "!" + "\n" + 40 * "!")
    print(f"N={N}, h={h}")
    # fJ: important to do that!! the ff you used to obtain the unitary should be ofc the same that you load here
    print(
        f"manually check if ff2 in simulation parameter is consistent with path= {hf_result_path}"
    )
    print(40 * "!" + "\n" + 40 * "!")

    # fJ by loading initial vectorised parameterm you can "continue" a simulation
    # initial_vectorised_parameter = np.load("C:\\Users\\Hester\\PycharmProjects\\masterproject\\RawResults\\newest\\Complex_basis_hfElementsButUisIdentity_Continued_t=1_N=10_M=15_sr=T_eta=0.1_mc-steps=100_network_files-h=1.00\\t=150.npy")

    # fJ this contains all parameter needed to start a simulation
    # fJ network_occurrences_parameter_freq determines after how many sr/gd steps occurrence (approx of prob distribution) and network parameter are stored
    # fJ !!! The cluster has a file limit !!! You either keep this frequency "high" or you change the output method (would be better to write things in one files as in separate files)
    # fJ I tried to name the parameter consistent with the thesis or to write out the full name
    # fJ "h" is in case of the fermionic model "t", i.e. the kinetic strength, the band scaling parameter
    fermion_main(
        N=N,
        M=M,
        h=h,
        potential_function=potential_function,
        ff1=ff1,
        ff2=ff2,
        ff3=ff3,
        ff4=ff4,
        eta=lr,
        sr=sr,
        gradient_descent=gradient_descent,
        equilibrium_steps=50,
        observable_steps=50,
        number_of_gradient_steps=int(1e6),
        seed=seed,
        energy_freq=1,
        other_obs_freq=30,
        network_occurrences_parameter_freq=int(1500 / weights_per_visible_neuron)
        if basis != "d_bar"
        else int(10000 / weights_per_visible_neuron),
        file_label=add_file_label,
        result_path=RBM_result_path,
        hf=True,
        hf_result_path=hf_result_path,
        timelimit=1000,
        takeOtherUWhere_t_is=takeOtherUWhere_t_is,
        complex_param=complex_param,
        initial_a=[a_bias for _ in range(N)] if a_bias is not None else None,
        initial_vectorised_parameter=initial_vectorised_parameter,
        rotateUBy=None,
        explicit_U=explicit_U,
        doImportanceSampling=doImportanceSampling,
        doFullConnection=doFullConnection,
        info_string=info_string,
        weights_per_visible_neuron=weights_per_visible_neuron,
        used_model=used_model,
    )


"""
fJ
the start of a simulation is done using the multiprocessing module
the nested loops create the arguments (kwargs) that are used to call a function (start_calculation) which later start a simulation (fermion_main)
multiprocessing.Process(kwargs, start_calculation)
Those process do not communicate with each other. Using multiprocessing like this is equivalent to starting multiple terminals
You can type "export OMP_NUM_THREADS=1" in the terminal before starting multiple processes. This restricts each simulation to exactly one core (for example to calc benchmarks or if a machine is 
so full with processes that additional multithreading would crash a simulation)
multiprocessing is not compatible with ipython console (pycharm), thus you have to start in in a terminal if you want to do it like this
"""
if __name__ == "__main__":
    args = {}
    run = 0
    N = 6
    distorted_t = None
    # basis = "d_bar"
    doImportanceSampling = True
    for basis in ["d_bar", "d", "c"]:
        # for basis in ["d_bar", "d", "c"]:
        for sr in [True]:
            for h in [0.04, 0.12, 0.14, 0.16, 0.5]:
                for eta in [1.0]:
                    for complex_param in [True]:
                        for seed in [1, 2, 3, 4]:
                            for M in [6]:
                                for weights_per_visible_neuron in [6]:
                                    continue_simulation_with_network_param_from_id = (
                                        None
                                    )
                                    args[run] = dict(
                                        h=h,
                                        N=N,
                                        M=M,
                                        identifier="proof_of_concept_fermionic_small_t_damping_fixed",
                                        basis=basis,
                                        doImportanceSampling=doImportanceSampling,
                                        eta=eta,
                                        takeOtherUWhere_t_is=distorted_t,
                                        weights_per_visible_neuron=weights_per_visible_neuron,
                                        initial_vectorised_parameter=continue_simulation_with_network_param_from_id,
                                        used_model="fermionic model",
                                        seed=seed,
                                        complex_param=complex_param,
                                        sr=sr,
                                    )
                                    run += 1
    processes = []
    for run in args:
        p = multiprocessing.Process(target=start_calculation, kwargs=args[run])
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
