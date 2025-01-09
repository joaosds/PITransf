#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 14:27:46 2022

@author: Michael Perle

Entspricht 1:1 plot_occupation_i
"""

import os
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from matplotlib import pyplot as plt
from RBM import FermionModel
import numpy as np


def plot_file(path="", identifier="", h_list=[], save_all=False, avg_over=10):
    # files_in_directory = os.listdir()
    if path:
        convergence_files = sorted(
            [file_name for file_name in os.listdir(path=path) if identifier in file_name and "occupation" in file_name])
    else:
        convergence_files = sorted(
            [file_name for file_name in os.listdir() if identifier in file_name and "fermion_occupation" in file_name])
    if h_list:
        convergence_files = [file_name for file_name in convergence_files if
                             any([f"h={h:.2f}" in file_name for h in h_list])]
    if identifier != "":
        convergence_files = [file_name for file_name in convergence_files if identifier in file_name]

    for file in convergence_files:
        loc_identifier = file[:file.find(".txt")] + "_"

        try:
            file_pointer = open(path + file, "r")
            h = file[file.find("h=") + len("h="):file.find(".txt")]
            try:
                N = int(file[file.find("_l_") + len("_l_"):file.find("_alpha")])
            except ValueError:
                try:
                    N = int(file[file.find("N=") + len("N="):file.find("_M")])
                except ValueError:
                    N = int(file[file.find("_l_") + len("_l_"):file.find("_lton")])

            i_list = [int(i) for i in range(N)]

            occupation_i_list_real = [[] for i in i_list]
            occupation_i_list_imag = [[] for i in i_list]

            while True:
                try:
                    t = file_pointer.readline().split(": ")[1]
                except IndexError:
                    break
                occupation_str = [file_pointer.readline().split(": ")[1] for _ in i_list]
                complex_occupation = [complex(occupation_str[i].replace(")", "").replace("(", "")) for i in i_list]
                real_part_occupation = [np.real(complex_occupation[i]) for i in i_list]
                imag_part_occupation = [np.imag(complex_occupation[i]) for i in i_list]
                [occupation_i_list_real[i].append(real_part_occupation[i]) for i in i_list]
                [occupation_i_list_imag[i].append(imag_part_occupation[i]) for i in i_list]
                file_pointer.readline()

        except IndexError:
            print(loc_identifier)
            print(f"Trimming lists after t: {t}")
            shortest = min(
                [len(occupation_i_list_real[delta]) for delta in i_list] + [len(occupation_i_list_imag[delta]) for delta
                                                                            in i_list])
            print(f"list length = {shortest}")
            for delta in i_list:
                del occupation_i_list_real[delta][shortest:]
                del occupation_i_list_imag[delta][shortest:]

        occupation_i_list_real = [occupation_i[-avg_over:] for occupation_i in occupation_i_list_real]
        occupation_i_list_imag = [occupation_i[-avg_over:] for occupation_i in occupation_i_list_imag]

        try:
            occupation_nqs_real = [sum(occupation_i) / len(occupation_i) for occupation_i in occupation_i_list_real]
            occupation_nqs_imag = [sum(occupation_i) / len(occupation_i) for occupation_i in occupation_i_list_imag]


        except ZeroDivisionError:
            print(f'skipping: {loc_identifier} - might be unfinished')
            continue

        chain = FermionModel(h=h, potential_function=lambda q: 1 / (q * q + 1), ff1=lambda k, q: 1,
                             ff2=lambda k, q: np.sin(q) * (np.sin(k) + np.sin(k + q)), ff3=lambda k, q: 0,
                             ff4=lambda k, q: 0,
                             length=N)

        print(occupation_nqs_real)
        plt.plot(i_list, occupation_nqs_real, linestyle="None", marker='o', label='$Re\{<N_k>\}$')
        plt.plot(i_list, occupation_nqs_imag, linestyle="None", marker='o', fillstyle="none", label='$Im\{<N_k>\}$')
        plt.plot([k[0] for k in chain.k], [np.cos(k[1]) for k in chain.k], linestyle="None", marker='x',
                 label='$\pm cos(k)$', color='red')
        plt.plot([k[0] for k in chain.k], [-np.cos(k[1]) for k in chain.k], linestyle="None", marker='x', color='red')
        plt.title(f"{file[:file.find('_')]}_t = {h}, avg over {avg_over}")
        plt.xlabel("$k$")
        plt.xticks([k[0] for k in chain.k], [f"{round(round(k[1], 10) / round(np.pi, 10), 2)}$\pi$" for k in chain.k])
        plt.grid()
        plt.legend()
        if save_all:
            base_expression = loc_identifier.replace("_occupation", "")
            plot_dir = path + base_expression + "plot"
            try:
                os.mkdir(plot_dir)
            except FileExistsError:
                print("saving plot to already existing directory")
            plt.savefig(plot_dir + "\\" + loc_identifier[:loc_identifier.find('_')] + "_occupation" + f"t = {h}.jpg",
                        bbox_inches="tight")
        plt.show()
        plt.close()


# plot_file(path= "fermions_easter/fermions_ff2-0/", save_all=True)
plot_file(path="/Users/jass/Documents/oldlinux/phd/projects/perle2/RawResults/newest/asd-N=6/", save_all=True, avg_over=50,
          h_list=[1], identifier="asd")
print("end")
