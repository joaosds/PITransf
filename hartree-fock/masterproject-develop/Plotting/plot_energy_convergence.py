#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 10:57:54 2021

@author: Michael Perle
"""

import os
from matplotlib import pyplot as plt
from Supplementary.exact_diagonalization import finite_gs_energy
import numpy as np



def plot_file(identifier="", h_list = [], save_all = False, path = os.getcwd()+"/../RBM/"):
    

    #files_in_directory = os.listdir()
    if bool(path):
        convergence_files = sorted([file_name for file_name in os.listdir(path=path) if "_energy_convergence_" in file_name and all(id_elem in file_name for id_elem in identifier.split("*"))])
    else:
        convergence_files = sorted([file_name for file_name in os.listdir() if "_energy_convergence_" in file_name and all(id_elem in file_name for id_elem in identifier.split("*"))])
    if h_list:
        convergence_files = [file_name for file_name in convergence_files if any(f"h-{h:.2f}" in file_name for h in h_list)]
   
    ordered_list= []
    all_in_one_list = []
    identifier_list = []
    glob_identifier_list = []
    for file in convergence_files:
        if identifier == "":
            loc_identifier = file[:file.find(".txt")] + "_"
        else:
            glob_identifier_list.append(file[:file.find(".txt")] + "_")
            loc_identifier = identifier
        try:
            file_pointer = open(path+file, "r")
            h = file[file.find("h-") + len("h-"):file.find(".txt")]
            try:
                L = int(file[file.find("_l_") + len("_l_"):file.find("_lton")])
            except ValueError:
                L = int(file[file.find("_l_") + len("_l_"):file.find("_alpha")])
            if int(L)== 20 and round(float(h), 5) == 1:
                E_ed = -25.49098968636473
            else:
                E_ed = finite_gs_energy(int(L), 1, float(h))[0]
            
            t_list = []
            E_list = []
            E_square_list = []
            eps_list = []
            acc_mov_list = []
            while True:
                try:
                    t = file_pointer.readline().split(": ")[1]
                except IndexError:
                    break
                t_list.append(int(t))
                E_string = file_pointer.readline().split(": ")[1]
                try:
                    E_list.append(float(E_string))
                except ValueError:
                    E_list.append(complex(E_string))
                E_square_string = file_pointer.readline().split(": ")[1]
                try:
                    E_square_list.append(float(E_square_string))
                except ValueError:
                    E_square_list.append(complex(E_square_string))
                eps_list.append(float(file_pointer.readline().split(": ")[1]))
                acc_mov_list.append(int(file_pointer.readline().split(": ")[1]))
                file_pointer.readline()
                
        except IndexError:
            print("Trimming lists")
            shortest = min(len(E_list), len(eps_list), len(acc_mov_list), len(t_list))
            del E_list[shortest:]
            del E_square_list[shortest:]
            del eps_list[shortest:]
            del acc_mov_list[shortest:]
            del t_list[shortest:]
            print(f"{len(E_list)}, {len(eps_list)}, {len(acc_mov_list)}, {len(t_list)}")
        except ValueError as e:
            print(e)
            print(file)
            print(loc_identifier)
            print("will be skipped\n")
            continue
        if len(E_list) > 0:
            config, conax = plt.subplots(1)
            conax.set_title(f"_h={h}")
            #plt.plot(t_list, [(E-E_ed)/abs(E_ed) for E in E_list], marker = ".", linestyle="None")
            rel_energy = [(E-E_ed)/abs(E_ed) for E in E_list]
            conax.plot(t_list, rel_energy)
            conax.plot(t_list[30:], [np.average(rel_energy[i-30:i]) for i in range(30, len(t_list))])
            conax.set_xlabel("t")
            conax.set_ylabel("$(E_{NQS}-E_{ED})/E_{ED}$")
            conax.set_xscale('log')
            conax.set_yscale('symlog', linthresh=1e-5)
            conax.set_ylim(bottom=-1e-5)
            conax.grid()
            if save_all:
                plt.savefig(f"h-{h}_relError_full_{identifier}.svg", bbox_inches = "tight")
            config.tight_layout()
            config.show()

            fig, ax = plt.subplots(1)
            ax.plot(t_list, np.array(E_square_list) - np.array(E_list)**2)
            ax.plot(t_list, [0 for _ in t_list], label="goal")

            """
            ###########################################
            plt.title(loc_identifier + f"_h={h}")
            #plt.plot(t_list, [(E-E_ed)/abs(E_ed) for E in E_list], marker = ".", linestyle="None")
            plt.plot(t_list[0:1750], [(E-E_ed)/abs(E_ed) for E in E_list[0:1750]])
            plt.xlabel("t")
            plt.ylabel("$(E_{NQS}-E_{ED})/E_{ED}$")
            #plt.yscale('log')
            plt.grid()
            if save_all:
                plt.savefig(f"h-{h}_relError_uppper_{identifier}.svg", bbox_inches = "tight")
            plt.show()
            plt.close()
            ###########################################
            
            plt.title(loc_identifier + f"_h={h}")
            #plt.plot(t_list, [(E-E_ed)/abs(E_ed) for E in E_list], marker = ".", linestyle="None")
            plt.plot(t_list[-100:], [(E-E_ed)/abs(E_ed) for E in E_list[-100:]])
            plt.xlabel("t")
            plt.ylabel("$(E_{NQS}-E_{ED})/E_{ED}$")
            #plt.yscale('log')
            plt.grid()
            if save_all:
                plt.savefig(f"h-{h}_relError_upper_{identifier}.svg", bbox_inches = "tight")
            plt.show()
            plt.show()
            plt.close()
            """
            
            """
            ###########################################
            plt.title(loc_identifier + f"_h={h}")
            #plt.plot(t_list, [(E-E_ed)/abs(E_ed) for E in E_list], marker = ".", linestyle="None")
            plt.plot(t_list, [-E/L for E in E_list], label="$E_{NQS}$")
            plt.axhline(y=-E_ed/L, color='r', linestyle='-', linewidth = 0.8, label="$E_{ED}$")
            print(E_ed/L)
            plt.xlabel("t")
            plt.ylabel("$-E_{NQS}/L$")
            plt.yscale('log')
            plt.grid()
            plt.legend()
            if save_all:
                plt.savefig(f"h-{h}_absError_full_{identifier}.svg", bbox_inches = "tight")
            plt.show()
            plt.close()
            ##########################################
            plt.title(loc_identifier + f"_h={h}")
            #plt.plot(t_list, [(E-E_ed)/abs(E_ed) for E in E_list], marker = ".", linestyle="None")
            plt.plot(t_list, eps_list)
            plt.xlabel("t")
            plt.ylabel("$\epsilon$")
            plt.yscale('log')
            plt.grid()
            if save_all:
                plt.savefig(f"h-{h}_eps_{identifier}.svg", bbox_inches = "tight")
            plt.show()
            plt.close()
            ##########################################
            plt.title(loc_identifier + f"_h={h}")
            #plt.plot(t_list, [(E-E_ed)/abs(E_ed) for E in E_list], marker = ".", linestyle="None")
            plt.plot(t_list, acc_mov_list)
            plt.xlabel("t")
            plt.ylabel("$acc_moves$")
            plt.yscale('log')
            plt.grid()
            if save_all:
                plt.savefig(f"h-{h}_acc_moves_{identifier}.svg", bbox_inches = "tight")
            plt.show()
            plt.close()
            """


            """
            last_result_prec = (sum(E_list[-10:])/10-E_ed)/abs(E_ed)
            print(loc_identifier)
            print(f"h={h}, mean deviation from ED for last 10 results: {last_result_prec} from iteration = {t_list[-1]}\n")
            ordered_list.append([last_result_prec, h, loc_identifier, t_list[-1:][0]])
            all_in_one_list.append(np.array([(E-E_ed)/abs(E_ed) for E in E_list]))
            identifier_list.append(loc_identifier)
            """
        #print(t_list)
        #print(E_list)
    """    
    plt.title(f"GD vs SR: h={h}")
    shortest = min([len(entry) for entry in all_in_one_list])
    #shortest = 300
    identifier_iter = iter(glob_identifier_list)
    for entry in all_in_one_list:
        plot_name = next(identifier_iter)
        if plot_name == "sr_False_1_hhigh_1_l_6_lton_0.50_lr_0.1_reuse_False_obssteps_200_id_60050_energy_convergence_h-1.00_":
            plot_name = "standard gradient descent"
            
        elif plot_name == "sr_True_1_hhigh_1_l_6_lton_0.50_lr_0.1_reuse_False_obssteps_200_id_not_dynamic_sr_energy_convergence_h-1.00_":
            plot_name = "SR using regularization constant 1e-4"
            continue
        elif plot_name == "sr_True_1_hhigh_1_l_6_lton_0.50_lr_0.1_reuse_False_obssteps_200_id_dynamic_sr_energy_convergence_h-1.00_":
            continue
            plot_name = "SR with decaying regulatization constant"
        #plt.plot(list(range(300)), entry[:300], label=f"{plot_name}")
        plt.plot(list(range(shortest-100, shortest)), entry[shortest-100:shortest], label=f"{plot_name}")
    plt.xlabel("t")
    plt.ylabel("$(E_{NQS}-E_{ED})/abs(E_{ED})$")
    plt.legend()
    #plt.yscale('logit')
    plt.grid()
    plt.show()
    plt.close
    """    
    return ["acc", "h", "identifier", "gradient_steps"] + sorted(ordered_list, key= lambda list: abs(list[0]), reverse=True), fig, config
