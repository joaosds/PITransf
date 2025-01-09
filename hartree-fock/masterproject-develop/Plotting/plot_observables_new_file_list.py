#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 18:17:24 2021

@author: Michael Perle
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_file(length, lton="", learning_rate="", precision="", identifier = "",
              plot_all_correlation = False, correlation_plot_list = [],
              plot_all_energy = False, energy_plot_list = [],
              save_correlation_files = False, save_energy_files=False):
    """
    I guess this was an attempt to unify the other plot scripts
    For now deprecated
    :param length:
    :param lton:
    :param learning_rate:
    :param precision:
    :param identifier:
    :param plot_all_correlation:
    :param correlation_plot_list:
    :param plot_all_energy:
    :param energy_plot_list:
    :param save_correlation_files:
    :param save_energy_files:
    :return:
    """
    
    
    if len(energy_plot_list) > 0:
        energy_plot_list = [round(h, 2) for h in energy_plot_list]
    
    if len(identifier)>1:
        identifier_nqs = identifier
        identifier_ed = identifier[:identifier.find("_lton")]
        #print(identifier_ed)
        identifier_ed = str(identifier_ed)[-3:]
    else:
        identifier_nqs = f"l_{length}_lton_{lton:.2f}_lr_{learning_rate}_precision_{precision}"
        identifier_ed = f"l_{length}"
    
    ed_pointer = open(identifier_ed + "_ed_observables.txt")
    nqs_pointer = open(identifier_nqs + "_observables.txt")
    
    iteration= 0
    h_ed_list = []
    E_ed_list = []
    h_nqs_list = []
    E_nqs_list = []
    
    eof_nqs = False
    eof_ed = False
    
    delta = [i for i in range(length+1)]
    
    while True:
        if iteration:
            ed_pointer.readline()
            nqs_pointer.readline()
        head_ed = ed_pointer.readline()
        head_nqs = nqs_pointer.readline()
        
        
        if not head_nqs:
            eof_nqs = True
            
        if not head_ed:
            eof_ed = True
            
        if eof_nqs:
            break
        
  
        if not eof_nqs:
            h_nqs = float(nqs_pointer.readline().split(": ")[1])
            E_nqs = float(nqs_pointer.readline().split(": ")[1])
            c_s_nqs = [float(nqs_pointer.readline().split(": ")[1]) for i in range(length + 1)]
            c_x_nqs = [float(nqs_pointer.readline().split(": ")[1]) for i in range(length + 1)]
            if plot_all_energy or round(h_nqs,2) in energy_plot_list:
                h_nqs_list.append(h_nqs)
                E_nqs_list.append(E_nqs)
                
                
        if not eof_ed:
            while True:
                h_ed = float(ed_pointer.readline().split(": ")[1])
                E_ed = float(ed_pointer.readline().split(": ")[1])
                c_s_ed = [float(ed_pointer.readline().split(": ")[1]) for i in range(length + 1)]
                c_x_ed = [float(ed_pointer.readline().split(": ")[1]) for i in range(length + 1)]
                if plot_all_energy or round(h_ed,2) in energy_plot_list:
                    h_ed_list.append(h_ed)
                    E_ed_list.append(E_ed)
                if round(h_ed,5) >= round(h_nqs, 5):
                    break
        
        if plot_all_correlation or round(h_ed, 2) in correlation_plot_list:
            if round(h_ed,5) == round(h_nqs, 5):        
                plt.plot(delta, c_s_ed, marker = "o", linestyle=(0, (1, 10)), label="ED") #losely dotted
            plt.plot(delta, c_s_nqs, marker = "x", linestyle="dashdot", label="NQS") #losely dotted
            plt.title(identifier_nqs + f"_s_correlation_h={h_nqs}")
            #plt.title(f"c_s for $h$ = {h_nqs}: $N = {6}$, $M = {10}$, $\eta = {0.1}$, $\epsilon = {0.0001}$, " + "$t_{max} = 250$")
            plt.xlabel("$\Delta$")
            plt.ylabel("$c_\Delta$")
            plt.grid()
            plt.legend()
            if save_correlation_files:
                plt.savefig(identifier_nqs + f"_c_s_h_{h_nqs}"+".svg")
            plt.show()
            plt.close()
            
            if round(h_ed,5) == round(h_nqs, 5):
                plt.plot(delta, c_x_ed, marker = "o", linestyle=(0, (1, 10)), label="ED") #losely dotted
            plt.plot(delta, c_x_nqs, marker = "x", linestyle="dashdot", label="NQS") #losely dotted
            plt.title(identifier_nqs + f"_x_correlation_h={h_nqs}")
            #plt.title(f"c_x for $h$ = {h_nqs}: $N = {6}$, $M = {10}$, $\eta = {0.1}$, $\epsilon = {0.0001}$, " + "$t_{max} = 250$")
            plt.xlabel("$\Delta$")
            plt.ylabel("$c_\Delta$")
            plt.grid()
            plt.legend()
            if save_correlation_files:
                plt.savefig(identifier_nqs + f"_c_x_h_{h_nqs}"+".svg")
            plt.show()
            plt.close()
        
        
        
        iteration += 1
        """
        print(f"h: {h_nqs}")
        print(f"E: {E_nqs}")
        print(f"c_s: {c_s_nqs}")
        print(f"c_x: {c_x_nqs}")
        """
    if len(h_nqs_list)>0:
        #Plot Energy over h
        plt.plot(h_nqs_list, np.array(E_nqs_list)/length, marker = "x", linestyle="None", color ='g', markersize=10, label="NQS")
        plt.plot(h_ed_list, np.array(E_ed_list)/length, color = 'k', marker = ".", linestyle="None", label="$ED$")
        plt.axhline(y=-1, color='r', linestyle='-', linewidth = 0.8, label="h=0")
        #limit_qm_h = np.linspace(min(h_nqs_list), max(h_nqs_list), 100)
        limit_qm_h = np.linspace(1, 1.3, 100)
        limit_qm_E = -1*limit_qm_h
        plt.plot(limit_qm_h, limit_qm_E, color='y', linewidth = 0.8, label="J=0")
        
        #plt.title(f"$N = {6}$, $M = {10}$, $\eta = {0.1}$, $\epsilon = {0.0001}$, " + "$t_{max} = 250$")
        plt.title(identifier_nqs + "_energy over h")
        plt.xlabel("$h$")
        plt.ylabel("$E/N$")
        plt.grid()
        plt.legend()
        plt.xticks()
        if save_energy_files:
            plt.savefig(identifier_nqs + f"_interval[{min(h_nqs_list)},{max(h_nqs_list)}]" + "_energy_full.svg")
        plt.show()

    
        


file_list_four = ["fast_l_4_lton_1.00_lr_0.1_precision_None_config.txt"]
file_list_six =["min_prec_l_6_lton_0.60_lr_0.1_precision_0.005_config.txt","l_4_lton_1.00_lr_0.1_precision_None_config.txt","min_prec_l_6_lton_0.60_lr_0.1_precision_0.01_config.txt","l_6_lton_0.60_lr_0.1_precision_None_config.txt","min_prec_max_move_l_6_lton_0.60_lr_0.1_precision_0.01_config.txt","l_6_lton_1.00_lr_0.1_precision_None_config.txt","without_div_l_6_lton_0.60_lr_0.01_precision_0.001_config.txt","min_prec_l_6_lton_0.60_lr_0.05_precision_0.005_config.txt","without_div_l_6_lton_0.60_lr_0.1_precision_0.001_config.txt"]

for file in file_list_six:
    identifier = file[:-len("_config.txt")]
    try:
        plot_file(6,identifier=identifier,plot_all_energy=False, correlation_plot_list=[1,0.5,1,3])
    except Exception as e:
        print(identifier)
        print(e)
"""
length = 6
lton = 6/10
learning_rate = 0.1
precision = -1
plot_file(length = 6, lton = 6/10, learning_rate = 0.1, precision = -1, energy_plot_list=list(np.linspace(0,1,11)))
"""