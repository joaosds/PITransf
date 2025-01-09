#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 14:13:20 2022

@author: Michael Perle
"""


import os
from matplotlib import pyplot as plt
from RBM.FermionModel import FermionModel
import numpy as np



def plot_file(path = "", identifier="", h_list = [], save_all = False, avg_over = 20):
    
    """
    Plots d_a_i over cos(k)
    :param path:
    :param identifier:
    :param h_list:
    :param save_all:
    :param avg_over:
    :return:
    """

    #files_in_directory = os.listdir()
    if path:
        convergence_files = sorted([file_name for file_name in os.listdir(path=path) if identifier in file_name and "other_obeservable_convergence" in file_name])
    else:
        convergence_files = sorted([file_name for file_name in os.listdir() if identifier in file_name and "other_obeservable_convergence" in file_name])
    if h_list:
        convergence_files = [file_name for file_name in convergence_files if any(f"h={h:.2f}" in file_name for h in h_list)]
    if identifier != "":
        convergence_files = [file_name for file_name in convergence_files if identifier in file_name]
    for file in convergence_files:
        try:
            file_pointer = open(path+file, "r")
            h = file[file.find("h=") + len("h="):file.find(".txt")]
            try:
                L = int(file[file.find("_l_") + len("_l_"):file.find("_lton")])
            except ValueError:
                try:
                    L = int(file[file.find("_l_") + len("_l_"):file.find("_alpha")])
                except ValueError:
                    L = int(file[file.find("N=") + len("N="):file.find("_M=")])

            i_list = [int(i) for i in range(L)]
            

            d_a_i_list = [[] for i in i_list]

            while True:
                try:
                    t = file_pointer.readline().split(": ")[1]
                except IndexError:
                    break
                [d_a_i_list[i].append(float(file_pointer.readline().split(": ")[1])) for i in i_list]
                file_pointer.readline()
                
        except IndexError:
            print(f"Trimming lists after t: {t}")
            shortest = min([len(d_a_i_list[delta]) for delta in i_list])
            print(f"list length = {shortest}")
            for delta in i_list:
                del d_a_i_list[delta][shortest:]


        
        d_a_i_list = [d_a_i[len(d_a_i)-avg_over:] for d_a_i in d_a_i_list]

        
        try:
            d_a_nqs = [sum(d_a_i)/len(d_a_i) for d_a_i in d_a_i_list]


        except ZeroDivisionError:
            print(f'skipping h: {h}')
            continue
        
        chain = FermionModel(h = h, potential_function = lambda q : 1 / (q * q + 1), ff1 = lambda k, q: 1, ff2 = lambda k, q: np.sin(q) * (np.sin(k) + np.sin(k + q)), ff3=lambda k, q: 0, ff4=lambda k, q: 0, length=L)

        plt.plot(i_list, d_a_nqs, linestyle="None", marker = 'o', label= 'band occupation')
        plt.plot([k[0] for k in chain.k],[np.cos(k[1]) for k in chain.k], linestyle = "None", marker = 'x', label= '$\pm cos(k)$', color='red')
        plt.plot([k[0] for k in chain.k],[-np.cos(k[1]) for k in chain.k], linestyle = "None", marker = 'x', color='red')
        plt.title(f"{file[:file.find('_')]}_t = {h}, avg over {avg_over}")
        plt.xlabel("$k$")
        plt.xticks([k[0] for k in chain.k], [f"{round(round(k[1],10)/round(np.pi,10),2)}$\pi$" for k in chain.k])
        plt.grid()
        plt.legend(loc=None)
        plt.ylim(top=None)
        if save_all:
            plt.savefig(f"d_a_i, t = {h}.jpg", bbox_inches = "tight")
        plt.show()
        plt.close()


# plot_file(path= "fermions_easter/fermions_ff2-0/", save_all=True)
plot_file("C:\\Users\\Hester\\PycharmProjects\\masterproject\\RawResults\\newest\\", save_all=False, h_list=[1], identifier="d-basis", avg_over=50)
