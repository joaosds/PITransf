#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 10:57:54 2021

@author: Michael Perle
"""

import os
from matplotlib import pyplot as plt
import numpy as np
# from Supplementary.exact_diagonalization import finite_gs_energy




def plot_file(identifier, save_all = False, avg_over = 25, given_seed=None):
    path = "C:\\Users\\Hester\\PycharmProjects\\masterproject\\RawResults\\newest\\"
    convergence_files = sorted([file_name for file_name in os.listdir(path=path) if identifier in file_name and "occupation" in file_name])
    h = convergence_files[0][convergence_files[0].find("occupation_h=") + len("occupation_h="):convergence_files[0].find(".txt")]
    N = int(convergence_files[0][convergence_files[0].find("_N=") + len("_N="):convergence_files[0].find("_M=")])

    with open("C:\\Users\\Hester\\PycharmProjects\\masterproject\\Supplementary\\" + f"l_{N}_h_{float(h):.1f}_ed_observables.txt") as ED_correlator_file_pointer:
        if f"L: {N}, J: 1" not in ED_correlator_file_pointer.readline():
            raise ValueError
        if f"h: {float(h):.1f}" not in ED_correlator_file_pointer.readline():
            raise ValueError
        E_ED_str = ED_correlator_file_pointer.readline()
        c_s_ed = [float(ED_correlator_file_pointer.readline().split(": ")[1]) for _ in range(0, N+1)][:(int((N+1)/2) + 1)]
        c_x_ed = [float(ED_correlator_file_pointer.readline().split(": ")[1]) for _ in range(0, N + 1)][:(int((N+1)/2) + 1)]

    best_avg = None
    best_seed = None
    best_min_x = 1000
    best_min_z = 1000

    for seed in [1,2,3]:
        seed_convergence_files = [file_name for file_name in convergence_files if f"seed={seed}" in file_name]
        if len(seed_convergence_files) != 1:
            print(f"unspecified files: " + str(seed_convergence_files))
            raise ValueError
        convergence_file = seed_convergence_files[0]

        with open(path+convergence_file, "r") as NQS_correlator_file_pointer:
            delta_list = [int(i) for i in range(int((N+1)/2)+1)]
            c_s_list = [[] for delta in delta_list]
            c_x_list = [[] for delta in delta_list]
            while True:
                try:
                    t = NQS_correlator_file_pointer.readline().split(": ")[1]
                except IndexError:
                    break
                [c_s_list[delta].append(float(NQS_correlator_file_pointer.readline().split(": ")[1])) for delta in delta_list]
                [c_x_list[delta].append(float(NQS_correlator_file_pointer.readline().split(": ")[1])) for delta in delta_list]
                NQS_correlator_file_pointer.readline()


        for avg_over in range(1,30):
            c_s_nqs = [sum(c_s[-avg_over:]) / len(c_s[-avg_over:]) for c_s in c_s_list]
            c_x_nqs = [sum(c_x[-avg_over:]) / len(c_x[-avg_over:]) for c_x in c_x_list]
            accuracy_x  = np.mean([abs((c_x_nqs[delta] - c_x_ed[delta]) / c_x_ed[delta]) for delta in delta_list])
            accuracy_z = np.mean([abs((c_s_nqs[delta] - c_s_ed[delta]) / c_s_ed[delta]) for delta in delta_list])
            if accuracy_x < best_min_x and accuracy_z < best_min_z:
                best_min_x = accuracy_x
                best_min_z = accuracy_z
                best_avg = avg_over
                best_seed = seed

    seed_convergence_files = [file_name for file_name in convergence_files if f"seed={best_seed}" in file_name]
    if len(seed_convergence_files) != 1:
        print(f"unspecified files: " + str(seed_convergence_files))
        raise ValueError
    convergence_file = seed_convergence_files[0]

    with open(path + convergence_file, "r") as NQS_correlator_file_pointer:
        delta_list = [int(i) for i in range(int((N + 1) / 2) + 1)]
        c_s_list = [[] for delta in delta_list]
        c_x_list = [[] for delta in delta_list]
        while True:
            try:
                t = NQS_correlator_file_pointer.readline().split(": ")[1]
            except IndexError:
                break
            [c_s_list[delta].append(float(NQS_correlator_file_pointer.readline().split(": ")[1])) for delta in delta_list]
            [c_x_list[delta].append(float(NQS_correlator_file_pointer.readline().split(": ")[1])) for delta in delta_list]
            NQS_correlator_file_pointer.readline()

    c_s_nqs = [sum(c_s[-best_avg:]) / len(c_s[-best_avg:]) for c_s in c_s_list]
    c_x_nqs = [sum(c_x[-best_avg:]) / len(c_x[-best_avg:]) for c_x in c_x_list]

    loc_identifier = f"seed={seed}_h={float(h)}"

    fig, ax = plt.subplots(1)
    ax.plot(delta_list[0:], c_s_ed[0:], marker = "o", fillstyle="none", linestyle="None", label="$(c_\Delta^z)_{ED}$", color="blue")
    ax.plot(delta_list[0:], c_s_nqs[0:], marker = "x", linestyle="None", label="$(c_\Delta^z)_{NQS}$", color="blue")
    ax.plot(delta_list[0:], c_x_ed[0:], marker = "o", fillstyle="none", linestyle="None", label="$(c_\Delta^x)_{ED}$", color="red")
    ax.plot(delta_list[0:], c_x_nqs[0:], marker = "x", linestyle="None", label="$(c_\Delta^x)_{NQS}$", color="red")
    ax.set_xlabel("$\Delta$")
    ax.set_title(f"$N = {N}, h = {float(h):.1f}, avg = {best_avg}, \chi = {best_seed}$")
    ax.set_ylim(-0.1,1.1)
    ax.legend()
    fig.tight_layout()
    fig.show()
    # fig.savefig(f"N = {N} h = {float(h):.1f} avg = {best_avg} chi = {best_seed}.pdf")


    # ##absolute correlators
    # plt.plot(delta_list[1:], c_s_ed[1:], marker = "o", linestyle=(0, (1, 10)), label="ED") #losely dotted
    # plt.plot(delta_list[1:], c_s_nqs[1:], marker = "x", linestyle="dashdot", label="NQS") #losely dotted
    # #plt.title(identifier_nqs + f"_s_correlation_h={h_nqs}")
    # plt.title(f" abs_c_s_{loc_identifier}")
    # plt.xlabel("$\Delta$")
    # plt.ylabel("$c_\Delta$")
    # plt.grid()
    # plt.legend()
    # if save_all:
    #     plt.savefig(f"abs_c_s_{loc_identifier}.png", bbox_inches = "tight")
    # plt.show()
    # plt.close()
    #
    #
    # plt.plot(delta_list[1:], c_x_ed[1:], marker = "o", linestyle=(0, (1, 10)), label="ED") #losely dotted
    # plt.plot(delta_list[1:], c_x_nqs[1:], marker = "x", linestyle="dashdot", label="NQS") #losely dotted
    # #plt.title(identifier_nqs + f"_x_correlation_h={h_nqs}")
    # plt.title(f"abs_c_x_{loc_identifier}")
    # plt.xlabel("$\Delta$")
    # plt.ylabel("$c_\Delta$")
    # plt.grid()
    # plt.legend()
    # if save_all:
    #     plt.savefig(f"abs_c_x_{loc_identifier}.png", bbox_inches = "tight")
    # plt.show()
    # plt.close()



    ##relative correlators
    # try:
    #     plt.plot(delta_list, [(c_s_nqs[delta]-c_s_ed[delta])/c_s_ed[delta] for delta in delta_list], marker = "o", linestyle=(0, (1, 10)), label="ED") #losely dotted
    #     #plt.title(identifier_nqs + f"_s_correlation_h={h_nqs}")
    #     plt.title(f"rel_c_s_{loc_identifier}")
    #     plt.xlabel("$\Delta$")
    #     plt.ylabel("$(c_\Delta^{NQS}-c_\Delta^{ED})/c_\Delta^{ED}$")
    #     plt.grid()
    #     # if save_all:
    #     #     plt.savefig(f"rel_c_s_{loc_identifier}.svg", bbox_inches = "tight")
    #     # plt.show()
    #     plt.close()
    #     plt.plot(delta_list, [(c_x_nqs[delta]-c_x_ed[delta])/c_x_ed[delta] for delta in delta_list], marker = "o", linestyle=(0, (1, 10)), label="ED") #losely dotted
    #     #plt.title(identifier_nqs + f"_x_correlation_h={h_nqs}")
    #     plt.title(f"rel_c_x_{loc_identifier}")
    #     plt.xlabel("$\Delta$")
    #     plt.ylabel("$(c_\Delta^{NQS}-c_\Delta^{ED})/c_\Delta^{ED}$")
    #     plt.grid()
    #     # if save_all:
    #     #     plt.savefig(f"rel_c_x_{loc_identifier}.svg", bbox_inches = "tight")
    #     # plt.show()
    #     plt.close()
    # except ZeroDivisionError:
    #     print(f'plot error in h: {h}')
    #     plt.close()


for h in [0.3, 1., 1.3]:

    if h in [0.1, 0.3]:
        identifier = f"sr_True_h={h:.1f}_N=20_M=20_lr_1.0_obssteps_20_id_ising_model_ising_sr_vs_gd_small_t_damping_fixed_sampler-importance_weights_per_visible_neuron=6"
    else:
        identifier = f"sr_True_h={h:.1f}_N=20_M=20_lr_1.0_obssteps_20_id_ising_model_ising_sr_vs_gd_sampler-importance_weights_per_visible_neuron=6"
        
    plot_file(identifier, save_all=True)

