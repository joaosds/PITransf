#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 17:09:55 2021

@author: Michael Perle
"""
import os

import numpy as np
import json
import random
from datetime import datetime, timedelta
import inspect
from FermionModel import FermionModel
from IsingModel import IsingModel
from NeuralNetwork import NeuralNetwork, VectorizedNetworkParameter
from Minimizer import Minimizer
from FullSampler import Sampler as FullSampler
from ImportanceSampler import Sampler as ImportanceSampler
from Observable import Observable
from ComposedObservable import ComposedObservable
from State import State
from TFICObservables import d_a_i, d_b_j, c_s_delta, c_x_delta
from TFICObservables import h_loc as ising_h_loc
from fermionHfObservablebs import (
    h_loc as hf_h_loc,
    occupation_number as hf_occupation_number,
)

np.seterr(all="raise")


"""
fJ without question the messiest file and the one that 
I had to edit most often as it connects all components with each
other and therefore is dependent on every other component
see doc in each function for infos
"""


def generate_file_id(file_label, sr, h, N, M, eta, observable_steps):
    """
    :param file_label:
    :param sr:
    :param h:
    :param N:
    :param M:
    :param eta:
    :param observable_steps:
    :return: a file id based on the above given parameter
    """
    if file_label is None:
        file_label = str(datetime.now())[:-7].replace(" ", "_")
    h_label = f"t={h}"
    file_id = f"{file_label}_{h_label}_N={N}_M={M}_sr={str(sr)[0]}_eta={eta}_mc-steps={observable_steps if 'lambda' not in str(observable_steps) else 'lambda'}"
    return file_id


def initialize_config_file(
    M,
    N,
    other_obs_freq,
    energy_freq,
    equilibrium_steps,
    eta,
    h,
    identifier,
    initial_a,
    initial_configuration,
    network_parameter_freq,
    number_of_gradient_steps,
    observable_steps,
    path,
    potential_function,
    sr,
    ff1,
    ff2,
    ff3,
    ff4,
    takeOtherUWhere_t_is,
    rotateUBy,
    complex_param,
    initial_vectorised_parameter,
    explicit_U,
    info_string,
):
    """
    :param M:
    :param N:
    :param other_obs_freq:
    :param energy_freq:
    :param equilibrium_steps:
    :param eta:
    :param h:
    :param identifier:
    :param initial_a:
    :param initial_configuration:
    :param network_parameter_freq:
    :param number_of_gradient_steps:
    :param observable_steps:
    :param path:
    :param potential_function:
    :param sr:
    :return: a config file containing the above stated parameter
    fJ the config file stays open until the end of a simulation, that is probably not good practice
    """
    config_file = open(path + f"/{identifier}_config.txt", "w")
    # write config
    config_file.write(f"N: {N}\t M: {M}\n")
    config_file.write(
        f"equilibrium_steps: {equilibrium_steps} \t observable_steps: {observable_steps}\n"
    )
    config_file.write(f"max_gradient_steps: {number_of_gradient_steps}\n")
    config_file.write(f"eta: {eta}\n")
    config_file.write(f"h: {str(h)}\n")
    config_file.writelines(
        [
            inspect.getsource(ff1),
            inspect.getsource(ff2),
            inspect.getsource(ff3),
            inspect.getsource(ff4),
            inspect.getsource(potential_function),
        ]
    )
    config_file.write(info_string)
    config_file.write(
        f"initial_configuration:\n a = {str(initial_a)} \n chain: {str(initial_configuration)}\n"
    )
    config_file.write(f"sr: {sr}\n")
    config_file.write(
        f"other_obs_freq: {other_obs_freq}, energy_freq: {energy_freq}, network_parameter_freq: {network_parameter_freq}\n"
    )
    config_file.write(
        f"takeOtherUWhere_t_is={takeOtherUWhere_t_is},"
        f"rotateUBy={rotateUBy}, complex_param={complex_param},"
        f"initial_vectorised_parameter={initial_vectorised_parameter}, explicit_U={explicit_U}"
    )
    config_file.flush()
    return config_file


def define_observables(model, neural_network, sr):
    """
    :param model: the model
    :param neural_network: instance containing RBM parameter
    :param sr: indicates if stochastic reconfiguration is used or GD
    :return: observables needed for minimization and that give insight into the model
    fJ an observable is a class in my implementation
    This is probably not really necessary and the first step in increasing performance would probably be to store all this things (probably more efficiently) in numpy arrays and thus also being able to
    perform more efficient calculations. Maybe less would be more here.
    """

    energy_observable = Observable("hf_h_loc", hf_h_loc)
    occupation_number_observale = [
        Observable(f"N_{k_index}", hf_occupation_number, k_index)
        for k_index in range(model.length)
    ]

    squared_energy_observable = ComposedObservable(
        energy_observable.name + "**2",
        energy_observable,
        energy_observable,
        conjugateFirst=True,
    )

    gradient_observables = VectorizedNetworkParameter(
        n_visible_neurons=neural_network.n_visible_neurons,
        n_hidden_neurons=neural_network.n_hidden_neurons,
        vectorized_parameter_type=object,
        fully_connected=neural_network.fully_connected,
        weights_per_visible_neuron=neural_network.weights_per_visible_neuron,
    )

    for i in range(neural_network.n_visible_neurons):
        gradient_observables.set_local_field_a(i, Observable(f"d_a_{i}", d_a_i, i=i))

    for j in range(neural_network.n_hidden_neurons):
        gradient_observables.set_local_field_b(j, Observable(f"d_b_{j}", d_b_j, j=j))

    for i in range(neural_network.n_visible_neurons):  # row
        for j in neural_network.get_connection_to_neurons(i):  # column
            gradient_observables.set_weight(
                i=i,
                j=j,
                value=ComposedObservable(
                    f"d_w_{i}{j}",
                    gradient_observables.get_local_field_a(i),
                    gradient_observables.get_local_field_b(j),
                ),
            )

    composed_gradient_observables = VectorizedNetworkParameter(
        n_visible_neurons=neural_network.n_visible_neurons,
        n_hidden_neurons=neural_network.n_hidden_neurons,
        vectorized_parameter_type=object,
        fully_connected=neural_network.fully_connected,
        weights_per_visible_neuron=neural_network.weights_per_visible_neuron,
    )

    for k in range(gradient_observables.numberOfNetworkParameter):
        gradient_observable_k = gradient_observables.get_vector_parameter(k)
        composed_gradient_observables.set_vector_parameter(
            k,
            ComposedObservable(
                f"{gradient_observable_k.name}*{energy_observable.name}",
                gradient_observable_k,
                energy_observable,
                conjugateFirst=True,
            ),
        )

    if sr:
        composed_sr_observables = np.empty(
            [
                gradient_observables.numberOfNetworkParameter,
                gradient_observables.numberOfNetworkParameter,
            ],
            dtype=object,
        )
        for k in range(gradient_observables.numberOfNetworkParameter):
            for ks in range(gradient_observables.numberOfNetworkParameter):
                gradient_observable_k = gradient_observables.get_vector_parameter(k)
                gradient_observable_ks = gradient_observables.get_vector_parameter(ks)
                composed_sr_observables[k][ks] = ComposedObservable(
                    f"{gradient_observable_k.name}*{gradient_observable_ks.name}",
                    gradient_observable_k,
                    gradient_observable_ks,
                    conjugateFirst=True,
                )
    else:
        composed_sr_observables = np.empty(0)

    functional_observables = (
        [energy_observable, squared_energy_observable]
        + [
            gradient_observables.get_vector_parameter(k)
            for k in range(gradient_observables.numberOfNetworkParameter)
        ]
        + [
            composed_gradient_observables.get_vector_parameter(k)
            for k in range(composed_gradient_observables.numberOfNetworkParameter)
        ]
    )

    if sr:
        for k in range(gradient_observables.numberOfNetworkParameter):
            for ks in range(gradient_observables.numberOfNetworkParameter):
                functional_observables.append(composed_sr_observables[k][ks])

    all_observables = functional_observables + occupation_number_observale

    return (
        all_observables,
        composed_gradient_observables,
        energy_observable,
        functional_observables,
        gradient_observables,
        composed_sr_observables,
        occupation_number_observale,
        squared_energy_observable,
    )


def get_convergence_files(h, identifier, path):
    """
    and creates directory for magnitude files
    :param h: the strength of kinetic energy, called t in theory
    :param identifier: global idintifier for all files of this output
    :param path: where results are stored
    :return: multiple file_pointer
    """
    magnitude_directory = path + f"/{identifier}_magnitude_files-h={h:.5e}"
    network_directory = path + f"/{identifier}_network_files-h={h:.5e}"
    os.mkdir(magnitude_directory)
    os.mkdir(network_directory)
    energy_convergence_file = open(
        path + f"/{identifier}_energy_convergence_h={h:.5e}.txt", "w"
    )
    occupation_file = open(path + f"/{identifier}_occupation_h={h:.5e}.txt", "w")
    other_observable_file = open(
        path + f"/{identifier}_other_obeservable_convergence_h={h:.5e}.txt", "w"
    )
    return (
        energy_convergence_file,
        network_directory,
        occupation_file,
        other_observable_file,
        magnitude_directory,
    )


def close_files(energy_convergence_file, occupation_file, other_observable_file):
    """
    closes files at the end of simulation
    :param energy_convergence_file: file pointer
    :param occupation_file: file pointer
    :param other_observable_file: file pointer
    :return:
    """
    energy_convergence_file.close()
    other_observable_file.close()
    occupation_file.close()


def write_to_files(
    accepted_moves,
    other_obs_freq,
    energy_convergence_file,
    energy_freq,
    energy_observable,
    squared_energy_observable,
    gradient_observables,
    network_directory,
    network_occurrences_parameter_freq,
    number_of_gradient_steps,
    occupation_file,
    occupation_number_observale,
    other_observable_file,
    precision_reached,
    state,
    t,
    magnitude_directory,
    sampler,
    neural_network=None,
):
    """
    dumps magnitude to file
    :param sampler: contains occurrences of configurations which are written to magnitude file
    :param magnitude_directory: directory in which magnitude json files are stored
    :param accepted_moves: indicates how many local updates were accepted in the preveous mc step
    :param other_obs_freq: how often occupation is calculated and printed to file and how often d_a_i is written to file
    :param energy_convergence_file: file pointer
    :param energy_freq: indicates how often energy is printed to file
    :param energy_observable: value of <E>
    :param gradient_observables: value of <O_loc> that are needed for GD TODO
    :param network_directory: file pointer
    :param network_occurrences_parameter_freq: indicates how often network parameter and occurrences are printed to file
    :param number_of_gradient_steps: how often RBM is updated
    :param occupation_file: file pointer
    :param occupation_number_observale: <N_loc>
    :param other_observable_file: <d_a_i>
    :param precision_reached: TODO
    :param state: RBM with model
    :param t: timestep
    :return: None, prints to files
    """
    if not t % energy_freq or t == number_of_gradient_steps - 1:
        energy_convergence_file.write(f"t: {t}\n")
        energy_convergence_file.write(energy_observable.complex_str() + "\n")
        energy_convergence_file.write(squared_energy_observable.complex_str() + "\n")
        energy_convergence_file.write(f"eps: {precision_reached}\n")
        energy_convergence_file.write(
            f"acc_mov: {accepted_moves} in {sampler.loops*sampler.number_of_mc_steps} mc steps\n"
        )
        energy_convergence_file.write("\n")
        energy_convergence_file.flush()

    if not (t % other_obs_freq or t == 0) or t == number_of_gradient_steps - 1:
        other_observable_file.write(f"t: {t}\n")
        other_observable_file.writelines(
            [
                gradient_observables.get_local_field_a(i).complex_str() + "\n"
                for i in range(neural_network.n_visible_neurons)
            ]
        )
        other_observable_file.write("\n")
        occupation_file.write(f"t: {t}\n")
        occupation_file.writelines(
            [o.complex_str() + "\n" for o in occupation_number_observale]
        )
        occupation_file.write("\n")
        other_observable_file.flush()
        occupation_file.flush()
    if not (t % network_occurrences_parameter_freq):
        np.save(
            file=network_directory + f"/t={t}.npy",
            arr=state.neural_network.vectorised_parameter,
        )
        json.dump(sampler.occurrences, open(magnitude_directory + f"/t={t}.txt", "w"))


def load_hf_unitary(N, h, hf, hf_result_path, takeOtherUWhere_t_is, rotateUBy):
    """
    :param N: the number of fermions that was also used in HF mean field calculation
    :param h: the strength of kinetic energy, called t in theory
    :param hf: True for 'HF+RBM', False for 'RBM'
    :param hf_result_path: path where HF mean field calculation is stored
    :return: hf_unitary that transforms from basis d --> d barred
    """
    hf_unitary = None
    if hf:
        hOfloadedUnitary = h
        if takeOtherUWhere_t_is is not None and rotateUBy is not None:
            raise ValueError(
                "simultaneously rotating and importing of U is not supported"
            )
        if takeOtherUWhere_t_is is not None:
            hOfloadedUnitary = takeOtherUWhere_t_is
        hf_unitary = np.load(f"{hf_result_path}/Uk_N={N}_t={hOfloadedUnitary:.5e}.npy")
        if rotateUBy is not None:
            hf_unitary = [U_k @ rotateUBy for U_k in hf_unitary]
        # unitary test:
        if any(
            [
                np.linalg.norm(np.eye(2) - np.matmul(np.conjugate(Uk).T, Uk)) > 1e-13
                or np.linalg.norm(np.eye(2) - np.matmul(Uk, np.conjugate(Uk).T)) > 1e-13
                for Uk in hf_unitary
            ]
        ):
            raise ValueError(f"loaded hf_unitary={hf_unitary} is not unitary")
    print("simulating with U=")
    print(hf_unitary)
    return hf_unitary


def generate_file_id_ising(file_label, sr, h, N, M, eta, observable_steps):
    return f"sr_{sr}_h={h}_N={N}_M={M}_lr_{eta}_obssteps_{observable_steps}_id_{file_label}"


def define_observables_ising(chain, neural_network, sr, weights_per_visible_neuron):
    # fj either that or the define_observables_fermions function is called depending on used model
    # fj contains duplicated code which could be extracted in another function
    energy_observable = Observable("h_loc", ising_h_loc)
    squared_energy_observable = ComposedObservable(
        "h_loc**2", energy_observable, energy_observable
    )
    gradient_observables = VectorizedNetworkParameter(
        n_visible_neurons=neural_network.n_visible_neurons,
        n_hidden_neurons=neural_network.n_hidden_neurons,
        vectorized_parameter_type=object,
        fully_connected=neural_network.fully_connected,
        weights_per_visible_neuron=weights_per_visible_neuron,
    )
    for i in range(neural_network.n_visible_neurons):
        gradient_observables.set_local_field_a(i, Observable(f"d_a_{i}", d_a_i, i=i))
    for j in range(neural_network.n_hidden_neurons):
        gradient_observables.set_local_field_b(j, Observable(f"d_b_{j}", d_b_j, j=j))
    for i in range(neural_network.n_visible_neurons):  # row
        for j in neural_network.get_connection_to_neurons(i):  # column
            gradient_observables.set_weight(
                i=i,
                j=j,
                value=ComposedObservable(
                    f"d_w_{i}{j}",
                    gradient_observables.get_local_field_a(i),
                    gradient_observables.get_local_field_b(j),
                ),
            )
    composed_gradient_observables = VectorizedNetworkParameter(
        n_visible_neurons=neural_network.n_visible_neurons,
        n_hidden_neurons=neural_network.n_hidden_neurons,
        vectorized_parameter_type=object,
        fully_connected=neural_network.fully_connected,
        weights_per_visible_neuron=weights_per_visible_neuron,
    )
    for k in range(gradient_observables.numberOfNetworkParameter):
        gradient_observable_k = gradient_observables.get_vector_parameter(k)
        composed_gradient_observables.set_vector_parameter(
            k,
            ComposedObservable(
                f"{gradient_observable_k.name}*{energy_observable.name}",
                gradient_observable_k,
                energy_observable,
                conjugateFirst=True,
            ),
        )
    if sr:
        composed_sr_observables = np.empty(
            [
                gradient_observables.numberOfNetworkParameter,
                gradient_observables.numberOfNetworkParameter,
            ],
            dtype=object,
        )
        for k in range(gradient_observables.numberOfNetworkParameter):
            for ks in range(gradient_observables.numberOfNetworkParameter):
                gradient_observable_k = gradient_observables.get_vector_parameter(k)
                gradient_observable_ks = gradient_observables.get_vector_parameter(ks)
                composed_sr_observables[k][ks] = ComposedObservable(
                    f"{gradient_observable_k.name}*{gradient_observable_ks.name}",
                    gradient_observable_k,
                    gradient_observable_ks,
                    conjugateFirst=True,
                )
    else:
        composed_sr_observables = np.empty(0)
    functional_observables = (
        [energy_observable, squared_energy_observable]
        + [
            gradient_observables.get_vector_parameter(k)
            for k in range(gradient_observables.numberOfNetworkParameter)
        ]
        + [
            composed_gradient_observables.get_vector_parameter(k)
            for k in range(composed_gradient_observables.numberOfNetworkParameter)
        ]
    )
    if sr:
        for k in range(gradient_observables.numberOfNetworkParameter):
            for ks in range(gradient_observables.numberOfNetworkParameter):
                functional_observables.append(composed_sr_observables[k][ks])
    correlation_observables = [
        Observable(f"c_s_{delta}", c_s_delta, delta=delta)
        for delta in range(int(chain.length / 2) + 1)
    ] + [
        Observable(f"c_x_{delta}", c_x_delta, delta=delta)
        for delta in range(int(chain.length / 2) + 1)
    ]
    all_obervables = functional_observables + correlation_observables
    return (
        all_obervables,
        composed_gradient_observables,
        composed_sr_observables,
        correlation_observables,
        energy_observable,
        functional_observables,
        gradient_observables,
        squared_energy_observable,
    )


def main(
    N,
    M,
    h,
    potential_function,
    ff1,
    ff2,
    ff3,
    ff4,
    eta,
    sr: bool,
    gradient_descent: bool,
    equilibrium_steps=1000,
    observable_steps=200,
    number_of_gradient_steps=100,
    seed=1,
    energy_freq=1,
    other_obs_freq=100,
    network_occurrences_parameter_freq=100,
    file_label=None,
    result_path="",
    hf: bool = True,
    timelimit=72,
    hf_result_path=None,
    takeOtherUWhere_t_is=None,
    rotateUBy=None,
    complex_param=False,
    initial_a=None,
    initial_vectorised_parameter=None,
    explicit_U=None,
    doImportanceSampling=True,
    doFullConnection=True,
    info_string="",
    weights_per_visible_neuron=None,
    used_model="fermionic model",
):
    """
    This is the core function of the package, the simulation takes place here
    TODO np.save the network instead of printing exact str to file?
    :param initial_vectorised_parameter:
    :param initial_a: if a bias is added to initial start
    :param rotateUBy: rotation matrix applied to U_HF --> U_HF@rotateUBy
    :param takeOtherUWhere_t_is: float. If given, t_1 is taken to obtain U_HF and the simulation is done with t_2
    :param hf_result_path: path that links to results of the mean field calculation
    :param timelimit: number of hours that calculation is supposed to run maximally
    :param N: Size of input data. Equivalent to spin chain N (ising), number of electrons (fermions)
    :param M: Number of hidden neurons. the right number remains to be calculated
    :param h: contains strength of the transverse magnetic field (Ising) or kinetic parameter t (fermions)
    :param potential_function: See potential_function in :class: FermionSpinConfiguration.
    :param ff1: See ff1 in :class: FermionSpinConfiguration.
    :param ff2: See ff2 in :class: FermionSpinConfiguration.
    :param ff3: See ff3 in :class: FermionSpinConfiguration.
    :param ff4: See ff4 in :class: FermionSpinConfiguration.
    :param eta: See lr in :class: FermionSpinConfiguration.
    :param sr: set to true if stochastic reconfiguration should be used
    :param gradient_descent: set to true if normal gradient descent should be used
    :param equilibrium_steps: number of mc steps without observable calculation.
    :param observable_steps: number of steps that calculate observables
    :param number_of_gradient_steps: number of times that neuronal network parameters are updated based on sr or gg
    :param seed: seed used for numpy and random
    :param energy_freq: every freq gd steps the energy is printed to a file. energy is calculated every gd step indepently from this setting
    :param other_obs_freq: every freq minimzer steps the occupation is calculated and printed together with output of d_a_i
    :param network_occurrences_parameter_freq: every freq gd steps state.neural_network.exact_str() is printed to a file
    :param file_label: this parameter adds a label to the auto generated begin of the file id
    :param result_path: determines where the file should be saved
    :param hf: True for 'HF+RBM'. False for 'RBM'
    :return: None. prints files
    """

    if isinstance(initial_vectorised_parameter, str):
        # fj initial_vectorised_parameter can be given as array (or list) or path to a stored npy numpy array
        last_network_file = sorted(
            os.listdir(
                result_path
                + "/"
                + initial_vectorised_parameter
                + f"_network_files-h={h:.5e}/"
            ),
            key=lambda x: int(x.replace("t=", "").replace(".npy", "")),
        )[-1]
        initial_vectorised_parameter = np.load(
            result_path
            + "/"
            + initial_vectorised_parameter
            + f"_network_files-h={h:.5e}/{last_network_file}"
        )
        print("Succesfully loaded old nn file")

    if used_model == "ising model":
        identifier = generate_file_id_ising(
            file_label, sr, h, N, M, eta, observable_steps
        )
    elif used_model == "fermionic model":
        identifier = generate_file_id(file_label, sr, h, N, M, eta, observable_steps)
    else:
        raise ValueError("model unspecified")

    if isinstance(observable_steps, int):
        observable_steps_int = observable_steps
        observable_steps = lambda iteration: observable_steps_int

    if doImportanceSampling:
        # fJ both is supported, FullSampler works for up to N=10
        Sampler = ImportanceSampler
    else:
        Sampler = FullSampler

    start = datetime.now()
    random.seed(seed)
    np.random.seed(seed)

    initial_configuration = [random.choice([-1, 1]) for _ in range(N)]

    if sr == gradient_descent:
        raise SystemExit("No or more than one minimization method specified")

    config_file = initialize_config_file(
        M,
        N,
        other_obs_freq,
        energy_freq,
        equilibrium_steps,
        eta,
        h,
        identifier,
        initial_a,
        initial_configuration,
        network_occurrences_parameter_freq,
        number_of_gradient_steps,
        observable_steps,
        result_path,
        potential_function,
        sr,
        ff1=ff1,
        ff2=ff2,
        ff3=ff3,
        ff4=ff4,
        takeOtherUWhere_t_is=takeOtherUWhere_t_is,
        rotateUBy=rotateUBy,
        complex_param=complex_param,
        initial_vectorised_parameter=initial_vectorised_parameter,
        explicit_U=explicit_U,
        info_string=info_string,
    )
    (
        energy_convergence_file,
        network_directory,
        occupation_file,
        other_observable_file,
        magnitude_directory,
    ) = get_convergence_files(h, identifier, result_path)
    if used_model == "fermionic model":
        if explicit_U is None:
            hf_unitary = load_hf_unitary(
                N, h, hf, hf_result_path, takeOtherUWhere_t_is, rotateUBy
            )
        else:
            hf_unitary = explicit_U
    elif used_model == "ising model":
        hf_unitary = None
    else:
        raise ValueError("model unspecified")

    # network and model
    # ising integration here
    if used_model == "ising model":
        model = IsingModel(1, h, exact_configuration=initial_configuration)
    elif used_model == "fermionic model":
        model = FermionModel(
            potential_function=potential_function,
            ff1=ff1,
            ff2=ff2,
            ff3=ff3,
            ff4=ff4,
            h=h,
            exact_configuration=initial_configuration,
            hf_unitary=hf_unitary,
        )
    else:
        raise ValueError("model unspecified")
    neural_network = NeuralNetwork(
        model.length,
        M,
        initial_vectorised_parameter=initial_vectorised_parameter,
        complex_parameter=complex_param,
        fully_connected=doFullConnection,
        weights_per_visible_neuron=weights_per_visible_neuron,
    )
    if initial_a is not None:
        if len(initial_a) != N:
            raise ValueError
        for i in range(N):
            neural_network.set_local_field_a(i, value=initial_a[i])

    # ising integration here
    if used_model == "ising model":
        (
            all_obervables,
            composed_gradient_observables,
            composed_sr_observables,
            correlation_observables,
            energy_observable,
            functional_observables,
            gradient_observables,
            squared_energy_observable,
        ) = define_observables_ising(
            model, neural_network, sr, weights_per_visible_neuron
        )
    elif used_model == "fermionic model":
        (
            all_obervables,
            composed_gradient_observables,
            energy_observable,
            functional_observables,
            gradient_observables,
            composed_sr_observables,
            occupation_number_observale,
            squared_energy_observable,
        ) = define_observables(model, neural_network, sr)
    else:
        raise ValueError("model unspecified")

    if gradient_descent:
        minimizer = Minimizer(
            eta,
            sr=False,
            complex_parameter=complex_param,
            regularization_function="constant",
        )
    else:
        minimizer = Minimizer(
            eta,
            sr=True,
            complex_parameter=complex_param,
            regularization_function="constant",
        )

    iterate_until = datetime.now() + timedelta(hours=timelimit)

    try:
        for t in range(number_of_gradient_steps):
            model.reset_configuration_to_random()  # this is not optimal since a reset without new state initialisation would break the simulation. an update within state class would be better
            print("random config: " + str(model.configuration))
            state = State(neural_network, model)
            sampler = Sampler()
            if t == 0 and initial_vectorised_parameter is not None:
                print("equilibrate with given network param")
                sampler.sample_state(state, int(equilibrium_steps * 500))
            # eq. sampling
            print("sampling", end="")
            sampler.sample_state(state, equilibrium_steps)
            print("...")
            print(
                "model configuration after equilibrium sampling: "
                + str(model.configuration)
            )
            if not (t % other_obs_freq or t == 0) or t == number_of_gradient_steps - 1:
                accepted_moves = sampler.sample_state(
                    state,
                    number_of_mc_steps=observable_steps(t),
                    observables=all_obervables,
                    save_occurrences=not t % network_occurrences_parameter_freq,
                )
            else:
                accepted_moves = sampler.sample_state(
                    state,
                    number_of_mc_steps=observable_steps(t),
                    observables=functional_observables,
                    save_occurrences=not t % network_occurrences_parameter_freq,
                )

            # minimizer applied
            print("update rbm")
            precision_reached = minimizer.update_rbm_parameters(
                state,
                energy_observable.get(complex_param),
                gradient_observables,
                composed_gradient_observables,
                composed_sr_observables,
                iteration=t,
            )

            # console output
            print(
                str(
                    [
                        gradient_observables.get_local_field_a(i).roundstr()
                        for i in range(neural_network.n_visible_neurons)
                    ]
                )
            )
            print(f"h={h:.5e}, t={t}, {energy_observable}")
            print(
                f"current precision = {precision_reached}, accepted moves = {accepted_moves}"
                + f" in {observable_steps(t)*sampler.loops} mc_steps"
            )
            # print(f"current precision = {precision_reached}, accepted moves = {accepted_moves}" + f" in {observable_steps(t)*sampler.loops} mc_steps")

            # file output
            # ising integration
            write_to_files(
                accepted_moves,
                other_obs_freq,
                energy_convergence_file,
                energy_freq,
                energy_observable,
                squared_energy_observable,
                gradient_observables,
                network_directory,
                network_occurrences_parameter_freq,
                number_of_gradient_steps,
                occupation_file,
                occupation_number_observale
                if used_model == "fermionic model"
                else correlation_observables,
                other_observable_file,
                precision_reached,
                state,
                t,
                magnitude_directory,
                sampler,
                neural_network,
            )

            if accepted_moves:
                [o.reset() for o in all_obervables]
            if datetime.now() > iterate_until:
                break

    finally:
        close_files(energy_convergence_file, occupation_file, other_observable_file)
        config_file.write(f"\nruntime: {datetime.now() - start}")
        config_file.close()
        print(datetime.now() - start)
