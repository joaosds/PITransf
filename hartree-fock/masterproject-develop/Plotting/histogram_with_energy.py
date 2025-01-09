

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
# import pyblock
import pandas as pd

from RBM.FermionModel import FermionModel
from RBM.ImportanceSampler import Sampler
from RBM.IsingModel import IsingModel
from RBM.NeuralNetwork import NeuralNetwork
from RBM.Observable import Observable
from RBM.State import State
from RBM.TFICObservables import h_loc as ising_h_loc
from RBM.fermionHfObservablebs import h_loc as hf_h_loc


def plot_file(identifier, path = "C:\\Users\\Hester\\PycharmProjects\\masterproject\\RawResults\\newest", equilibrium_steps=int(1e3), observable_steps=int(1<<12),
              save=False, do_plot_network_convergence=False, do_plot_histogram=False, last_ones=True, sr_step1=None, sr_step2=None,
              do_calculate_energy=False, weights_per_visible_neuron=None, do_binning=False):

    return_data = {}

    config_file_variables = {}
    config_file_variables["np"] = np

    if not "ising" in identifier:
        with open(path + "\\" + identifier + "_config.txt", "r") as config_file_pointer:
            config_file = config_file_pointer.read()


        exec(config_file[config_file.find("h: "):config_file.find("h: ") + config_file[config_file.find("h: "):].find(
            "\n")].replace(": ", "="), config_file_variables)

        exec(config_file[config_file.find("ff1 = "):config_file.find("initial_configuration")].replace("    ",""), config_file_variables)

        exec(config_file[:config_file.find("\n")].replace("\t ", "\n").replace(": ", "="), config_file_variables)
        h = config_file_variables["h"]

    else:
        config_file_variables["M"] = int(identifier[identifier.find("M=") + len("M="):identifier.find("_lr")])
        config_file_variables["N"] = int(identifier[identifier.find("N=") + len("N="):identifier.find("_M")])
        h = float(identifier[identifier.find("h=") + len("h="):identifier.find("_N=")])
    if "do_full_connection=True" in identifier:
        fully_connected = True
    else:
        fully_connected = False
        weights_per_visible_neuron = int(identifier[identifier.find("weights_per_visible_neuron=") + len("weights_per_visible_neuron="):])



    network_parameter_files = os.listdir(path+"\\"+identifier+f"_network_files-h={h:.5e}"+"\\")
    network_parameter_files = sorted(network_parameter_files, key = lambda name: int(name[name.find("t=") + len("t="):name.find(".npy")]))
    last_param = np.load(path + "\\" + identifier + f"_network_files-h={h:.5e}" + "\\" + network_parameter_files[
        -1 if last_ones else sr_step2])

    if do_plot_network_convergence:
        second_last_param = np.load(path + "\\" + identifier + f"_network_files-h={h:.5e}" + "\\" + network_parameter_files[-2 if last_ones else sr_step2])
        rel_change = (second_last_param-last_param)/second_last_param
        fig0, axes0 = plt.subplots(2, 1)
        l00, = axes0[0].plot(np.arange(config_file_variables["N"]), np.real(rel_change[:config_file_variables["N"]]), label="$Re(\Delta a_i)$")
        l01, = axes0[0].plot(np.arange(config_file_variables["N"]), np.imag(rel_change[:config_file_variables["N"]]), label="$Im(\Delta a_i)$")
        axes0[0].set_xticks(np.arange(config_file_variables["N"]), [f"a_{i}" for i in np.arange(config_file_variables["N"])])
        l00.set_linestyle("None")
        l00.set_marker(".")
        l01.set_linestyle("None")
        l01.set_marker(".")
        l10, = axes0[1].plot(np.arange(config_file_variables["M"]), np.real(rel_change[config_file_variables["N"]:config_file_variables["N"]+config_file_variables["M"]]), label="$Re(b_i)$")
        l11, = axes0[1].plot(np.arange(config_file_variables["M"]), np.imag(rel_change[config_file_variables["N"]:config_file_variables["N"] + config_file_variables["M"]]), label="$Im(b_i)$")
        axes0[1].set_xticks(np.arange(config_file_variables["M"]),
                           [f"b_{i}" for i in np.arange(config_file_variables["M"])])
        l10.set_linestyle("None")
        l10.set_marker(".")
        l11.set_linestyle("None")
        l11.set_marker(".")
        axes0[0].legend()
        axes0[1].legend()
        fig0.suptitle("Bias convergence")
        fig0.tight_layout()
        fig1, axes1 = plt.subplots(1, 2)
        rel_change_weights = rel_change[config_file_variables["N"]+config_file_variables["M"]:].reshape(config_file_variables["N"], weights_per_visible_neuron)
        weights_image_real = axes1[0].imshow(np.real(rel_change_weights).T, interpolation="none", origin="lower")
        weights_image_imag = axes1[1].imshow(np.imag(rel_change_weights).T, interpolation="none", origin="lower")
        fig1.colorbar(weights_image_real, ax=axes1[0], label="$Re(\Delta\omega_{i,j})$")
        fig1.colorbar(weights_image_imag, ax=axes1[1], label="$Im(\Delta\omega_{i,j})$")
        axes1[0].set_ylabel("$j$", rotation=0)
        axes1[1].set_ylabel("$j$", rotation=0)
        axes1[0].set_xlabel("$i$", rotation=0)
        axes1[1].set_xlabel("$i$", rotation=0)
        fig1.set_size_inches((fig0.get_size_inches() * np.array([1, 0.7])))
        fig1.suptitle("Weights convergence")
        fig1.tight_layout()
        fig0.show()
        fig1.show()
        print("fin")
        return_data["network_convergence"] = [fig0,fig1]

    if do_plot_histogram:
        converged_param = last_param if last_ones else np.load(path + "\\" + identifier + f"_network_files-h={h:.5e}" + "\\" + network_parameter_files[-1])
        neural_network = NeuralNetwork(n_visible_neurons=config_file_variables["N"], n_hidden_neurons=config_file_variables["M"], initial_vectorised_parameter=converged_param, complex_parameter=True, fully_connected=fully_connected, weights_per_visible_neuron=weights_per_visible_neuron)
        if not do_calculate_energy:
            if not "ising" in identifier:
                model = FermionModel(potential_function=config_file_variables["potential_function"],
                                     ff1=config_file_variables["ff1"], ff2=config_file_variables["ff2"],
                                     ff3=config_file_variables["ff3"], ff4=config_file_variables["ff4"],
                                     h=float(h), length=config_file_variables["N"], sumOverG=False,
                                     exact_configuration = [-1 for _ in range(config_file_variables["N"])])
            else:
                model = IsingModel(J=1, h=h,length=config_file_variables["N"])
            state = State(neural_network, model)

            sampler = Sampler()
            print('sampling', end="")
            sampler.sample_state(state, equilibrium_steps)
            print("...")

            up = u'\u2191'
            down = u'\u2193'
            sampler.sample_state(state, observable_steps, save_occurrences=True)
            dictionary = sampler.occurrences
            print("sorted by occurence:")
            print("identifier  configuration  rel occurence")
            for key, value in sorted(dictionary.items(), key=lambda item: item[1], reverse=True):
                print("%s\t%s\t%s" % (
                key, "".join(down if x == "0" else up for x in f"{key:0{config_file_variables['N']}b}"), value / (observable_steps * config_file_variables['N'])))
            print("sorted by identifier:")
            print("\n identifier  configuration  rel occurence")
            for key, value in sorted(dictionary.items()):
                print("%s\t%s\t%s" % (
                key, "".join(down if x == "0" else up for x in f"{key:0{config_file_variables['N']}b}"), value / (observable_steps * config_file_variables['N'])))

            plt.plot([key for key in sorted(dictionary)],
                     [dictionary.get(key) / (observable_steps * config_file_variables['N']) for key in sorted(dictionary)], linestyle=None,
                     marker=".")
            plt.title(identifier)
            plt.xlabel("configuration identifier")
            plt.ylabel("rel. occurence")
            plt.yscale('log')
            plt.grid()
            if save:
                plt.savefig(identifier + ".png", bbox_inches="tight")
            plt.show()
            plt.close()

            print("identifiers that never occured:")
            print([item for item in [x for x in range(64)] if item not in sorted(dictionary)])
        else:
            if not "ising" in identifier:
                if "hf-basis" in identifier or "d_bar-basis" in identifier:
                    basis_str = "hf-basis"
                    hf_unitary = np.load(os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + "RawResults\\HF_Results\\" + identifier[:identifier.find("hf-basis") if identifier.find("hf-basis")>0 else identifier.find("_d_bar-basis")] + os.sep + f"Uk_N={config_file_variables['N']}_t={h:.5e}.npy"))
                elif "c-basis" in identifier:
                    basis_str = "c-basis"
                    hf_unitary = [np.array([[1, -1j],[1, 1j]])/np.sqrt(2) for _ in range(config_file_variables['N'])]
                elif "d-basis" in identifier:
                    basis_str = "d-basis"
                    hf_unitary = [np.eye(2) for _ in range(config_file_variables['N'])]
                else:
                    raise ValueError("Unknown hf_unitary")
                model = FermionModel(potential_function=config_file_variables["potential_function"],
                                     ff1=config_file_variables["ff1"], ff2=config_file_variables["ff2"],
                                     ff3=config_file_variables["ff3"], ff4=config_file_variables["ff4"],
                                     h=float(h), length=config_file_variables["N"], sumOverG=False, hf_unitary=hf_unitary,
                                     exact_configuration = [-1 for _ in range(config_file_variables["N"])])
            else:
                model = IsingModel(J=1,h=h,length=config_file_variables["N"])
                basis_str = "ising"
            state = State(neural_network, model)
            sampler = Sampler()
            print('sampling', end="")
            sampler.sample_state(state, equilibrium_steps)
            print("...")
            if not "ising" in identifier:
                energy_observable = Observable("hf_h_loc", hf_h_loc, save_list=do_binning)
            else:
                energy_observable = Observable("ising_h_loc", ising_h_loc, save_list=do_binning)
            sampler.sample_state(state, observable_steps, observables=[energy_observable], save_occurrences=True, assign_energy_to_occurrence=True)
            dictionary = sampler.occurrences
            normalisation = sum(entry[0] for entry in dictionary.values())
            # if that does not work in both cases
            # normalisation = observable_steps * config_file_variables['N'] if sampler.isImportanceSampler else sum(entry[0] for entry in dictionary.values())
            fig_hist_0, axes_hist_0 = plt.subplots(1, 1)
            bar_container_0 = axes_hist_0.bar([key for key in sorted(dictionary)], [dictionary.get(key)[0] / normalisation for key in sorted(dictionary)], color='b', label = "occurrence and energy")
            axes_hist_0.set_yscale("log")
            axes_hist_0.bar_label(bar_container_0, [""] + [np.round(dictionary.get(key)[1], 4) for key in sorted(dictionary)[1:]], rotation=90, label_type="edge", padding= 10, annotation_clip="True")
            axes_hist_0.bar_label(bar_container_0, [np.round(dictionary.get(sorted(dictionary)[0])[1], 4)] + ["" for _ in sorted(dictionary)[1:]], rotation=90, label_type="center", padding=-40, annotation_clip="True", color="white")
            axes_hist_0.set_xlabel("Configuration identifier")
            axes_hist_0.set_ylabel("Relative Occurrence")
            axes_hist_0.legend()
            tuples_sorted_by_ocurence = sorted(zip(dictionary.keys(), [dictionary.get(key)[0] / normalisation for key in dictionary.keys()]), key=lambda tuple: tuple[1], reverse=True)
            fig_hist_1, axes_hist_1 = plt.subplots(1, 1)
            bar_container_1 = axes_hist_1.bar(
                [str(config_occ_tuple[0]) for config_occ_tuple in tuples_sorted_by_ocurence[:30]],
                [config_occ_tuple[1] for config_occ_tuple in tuples_sorted_by_ocurence[:30]], color='b',
                label="histogram, first 30 by occurrence")
            axes_hist_1.set_yscale("log")
            axes_hist_1.set_xlabel("Configuration identifier")
            axes_hist_1.set_ylabel("Relative Occurrence")
            axes_hist_1.legend()
            axes_hist_1.bar_label(bar_container_1, [np.round(dictionary.get(entry[0])[1], 4) for entry in tuples_sorted_by_ocurence[:30]], rotation=90, label_type="edge", padding=10, annotation_clip="True")
            axes_hist_1.set_xticks(axes_hist_1.get_xticks(), [str(config_occ_tuple[0]) for config_occ_tuple in tuples_sorted_by_ocurence[:30]], rotation=90)
            fig_hist_1.suptitle(f"RBM in {basis_str}, {'full sampler'} configurations, t={h}\n" + "$E_{RBM}=$" + str(energy_observable.get()))
            fig_hist_1.tight_layout()
            fig_hist_0.suptitle(f"RBM in {basis_str}, {'full sampler'} configurations, t={h}\n" + "$E_{RBM}=$" + str(energy_observable.get()))
            fig_hist_0.tight_layout()
            print("fin")
            fig_hist_1.show()
            fig_hist_0.show()
            return_data["energy_histogram"] = [fig_hist_0, fig_hist_1, dictionary]

        if do_binning and do_calculate_energy and do_plot_histogram:
            energy_every_step = pd.Series([energy.real for energy in energy_observable.save_list])
            (data_length, reblock_data, covariance) = pyblock.pd_utils.reblock(energy_every_step)
            pyblock.plot.plot_reblocking(reblock_data)
            return_data["binning_data"] = (data_length, reblock_data, covariance)
            return_data["energy_every_step"] = energy_every_step



    return return_data



def plot_ed_histogram(identifier, N, t):
    return_data =[]
    H_ED = np.load(os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + "RawResults\\ED_Results\\" + identifier + f"_N={N}-N={N}_t={t:.5e}.npy"))
    w, v = eigh(H_ED)
    E_0 = w[0]
    p_0 = v[:, 0] * np.conj(v[:, 0]) # psi**2 = p
    fig_ed, axes_ed = plt.subplots(2, 1)
    axes_ed[0].plot(np.arange(2 ** N), p_0, color='b', label="$|\psi(s)|^2$", marker=".", linestyle="None")
    axes_ed[0].set_yscale("log")
    axes_ed[0].set_xlabel("$s$")
    axes_ed[0].set_ylim(top=1)
    # axes_ed[0].set_ylabel("Relative Occurrence")
    axes_ed[0].legend()
    tuples_sorted_by_ocurence = sorted(zip(np.arange(2 ** N), p_0), key=lambda tuple: tuple[1], reverse=True)
    bar_container_1 = axes_ed[1].bar([str(config_occ_tuple[0]) for config_occ_tuple in tuples_sorted_by_ocurence[:35]], [config_occ_tuple[1] for config_occ_tuple in tuples_sorted_by_ocurence[:35]], color='b',
                                     label="$|\psi(s)|^2$")
    axes_ed[1].set_yscale("log")
    axes_ed[1].set_xlabel("s")
    # axes_ed[1].set_ylabel("Relative Occurrence")
    axes_ed[1].legend()
    # axes_ed[1].set_ylim(top=1)
    axes_ed[1].set_xticks(axes_ed[1].get_xticks(), [str(config_occ_tuple[0]) for config_occ_tuple in tuples_sorted_by_ocurence[:35]], rotation=90)
    fig_ed.suptitle(f"$t={round(t,8)}$" + "\n" + "$E_{ED}=$" + str(E_0))
    fig_ed.tight_layout()
    print("fin")
    fig_ed.show()
    fig_ed.savefig("C:\\Users\\Hester\\PycharmProjects\\masterproject\\Text\\" + f"ExactDiagonalization_t={t}" + ".eps")
    return_data.append([fig_ed, tuples_sorted_by_ocurence])

    return return_data

start = datetime.now()
for t in [0.01, 0.13, 0.5, 1]:
    local_data = plot_ed_histogram("original0", 10, t)
    # local_data = plot_file("original0_N=6_d_bar-basis_sampler-importance_weights_per_visible_neuron=4_unitary_t=None_state_old_t=0.11_N=6_M=6_sr=T_eta=1.0_mc-steps=20", do_plot_histogram=True, do_calculate_energy=True)
    # local_data = plot_file("sr_True_h=1.0_N=20_M=20_lr_1.0_obssteps_20_id_ising_model_lastN20varyw_sampler-importance_weights_per_visible_neuron=2", do_plot_network_convergence=True, do_calculate_energy=True, do_plot_histogram=True, do_binning=True)
    print(f"runtime: {datetime.now() - start}")

