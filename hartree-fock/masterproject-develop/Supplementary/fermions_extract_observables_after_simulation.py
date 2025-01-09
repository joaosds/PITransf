#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:38:48 2022

@author: Michael Perle



!!!
deprecated
fJ was intended to do as the file name suggests. The way to do it now is starting a new simulation with vectorized (network) parameters
!!!

Dieses File rechnet auf die pbservablen d_a_i retour. Alternativ dazu könnten diese bei jeder Simulation auch in einem File ausgegeben werden
Hier wird mit Hilfe der neuronalen Netzwerk parameter am Ende einer Simulation erneut gesamplet. Bei den vorkommenden Konfigurationen wird d_a_i ermittelt
Das File network_convergence beinhaltet nach jeder Simulation die exakten netzwerkparameter. Diese werden hier eingelesen und dann wird gesamplet.
Nachteil: Keine Mittelung über mehrere Gradient steps, sondern nur über das finale Resultat
Kompromisslösung also
"""


from RBM.IsingModel import IsingModel
from RBM.FermionModel import FermionModel
from RBM.NeuralNetwork import  NeuralNetwork
from RBM.ImportanceSampler import Sampler
from RBM.State import State
import random
import matplotlib.pyplot as plt
import os
import numpy as np
import warnings
from RBM.Observable import Observable
from RBM.TFICObservables import d_a_i



def plot_file(identifier="", h_list = [], path = "", equilibrium_steps = int(1e4), observable_steps = int(1e5), save = False, plot_magnitude = False, plot_d_a_i = False):
    
    if identifier == "" or h_list == []:
        print("This will take some time")

    #files_in_directory = os.listdir()
    if bool(path):
        convergence_files = sorted([file_name for file_name in os.listdir(path=path) if identifier in file_name and "_network_convergence_" in file_name])
    else:
        convergence_files = sorted([file_name for file_name in os.listdir() if identifier in file_name and "_network_convergence_" in file_name])
    if h_list:
        convergence_files = [file_name for file_name in convergence_files if any(f"h-{h:.2f}" in file_name for h in h_list)]
   
    for file in convergence_files:
        if identifier == "":
            loc_identifier = file[:file.find(".txt")] + "_"
        else:
            loc_identifier = identifier
        
        file_pointer = open(path+file, "r")
        h = file[file.find("h-") + len("h-"):file.find(".txt")]
        try:
            length = int(file[file.find("_l_") + len("_l_"):file.find("_lton")])
        except ValueError:
            length = int(file[file.find("_l_") + len("_l_"):file.find("_alpha")])
        
        number_of_hidden_neurons = int(length * float(file[file.find("_alpha_") + len("_alpha_"):file.find("_lr_")]))
        
        
        while True:
            try:
                t = file_pointer.readline().split(": ")[1]
            except IndexError:
                break

            neural_str = file_pointer.readline()
            initial_configuration = file_pointer.readline().strip('][\n').split(', ')
        print(f"h={h}: simulation results for network config at t={t} with initial config")
        print(initial_configuration)

                


        ###############################read in param############################
        initial_a = []
        initial_b = []
        initial_w = [[None for j in range(number_of_hidden_neurons)] for i in range(length)]
        neural_var_list = [elem.split(": ") for elem in neural_str.split(", ")[1:]]
        for var_tuple in neural_var_list[:length]:
            initial_a.append(float(var_tuple[1]))
        
        for var_tuple in neural_var_list[length:length+number_of_hidden_neurons]:
            initial_b.append(float(var_tuple[1]))
            
        matrix_iterator = iter(neural_var_list[length+number_of_hidden_neurons:])
        for i in range(length):
            for j in range(number_of_hidden_neurons):
                initial_w[i][j] = float(next(matrix_iterator)[1])
        
        ##############################initialize classes##############################
        if not bool(initial_configuration):
            initial_configuration = [random.choice([-1,1]) for spin in range(length)]
        if "fermion" in file:
            warnings.warn("potential_function, ff1, ff2 are user provided. check for consistency")
            chain = FermionModel(h = h, potential_function = lambda q : 1 / (q * q + 1), ff1 = lambda k, q: 1, ff2 = lambda k, q: np.sin(q) * (np.sin(k) + np.sin(k + q)), exact_configuration = [int(x) for x in initial_configuration])
        else:
            warnings.warn("J is user provided. check for consistency")
            J = 1
            chain = IsingModel(J, h, exact_configuration = [int(x) for x in initial_configuration])
        neural_network = NeuralNetwork(chain.length, number_of_hidden_neurons, local_field_a= initial_a, local_field_b= initial_b, weights=initial_w)
        state = State(neural_network, chain)
        sampler = Sampler()
        up = u'\u2191'
        down = u'\u2193'
        
        print('sampling', end="")
        sampler.sample_state(state, equilibrium_steps)
        print("...")
        if plot_magnitude:
            dictionary = {}
            sampler.sample_state(state, observable_steps, save_occurrences= True)
            dictionary = sampler.occurrences
            print("sorted by occurence:")
            print("identifier  configuration  rel occurence")    
            for key, value in sorted(dictionary.items(), key=lambda item: item[1], reverse=True):
                print("%s\t%s\t%s" % (key ,"".join(down if x == "0" else up for x in f"{key:0{length}b}"), value/(observable_steps*length)))
            print("sorted by identifier:")
            print("\n identifier  configuration  rel occurence")
            for key, value in sorted(dictionary.items()):
                print("%s\t%s\t%s" % (key ,"".join(down if x == "0" else up for x in f"{key:0{length}b}"), value/(observable_steps*length)))    
            
            plt.plot([key for key in sorted(dictionary)], [dictionary.get(key)/(observable_steps*length) for key in sorted(dictionary)],linestyle=None, marker = ".")
            plt.title(loc_identifier)
            plt.xlabel("configuration identifier")
            plt.ylabel("rel. occurence")
            plt.yscale('log')
            plt.grid()
            if save:
                plt.savefig(loc_identifier+".png", bbox_inches = "tight")
            plt.show()
            plt.close()
            
            print("identifiers that never occured:")
            print([item for item in [x for x in range(64)] if item not in sorted(dictionary)])

        if plot_d_a_i:
            d_a_i_observables = [Observable(f"d_a_{i}",d_a_i, i=i) for i in range(chain.length)]
            sampler.sample_state(state, observable_steps, observables=d_a_i_observables)
            plt.plot([d_a_i_observable.i for d_a_i_observable in d_a_i_observables],[d_a_i_observable.get() for d_a_i_observable in d_a_i_observables], linestyle = "None", marker = 'o', label= 'spin_expectation')
            plt.plot([k[0] for k in state.chain.k],[np.cos(k[1]) for k in state.chain.k], linestyle = "None", marker = 'x', label= '$\pm cos(k)$', color='red')
            plt.plot([k[0] for k in state.chain.k],[-np.cos(k[1]) for k in state.chain.k], linestyle = "None", marker = 'x', color='red')
            plt.xticks([k[0] for k in state.chain.k], [f"{round(round(k[1],10)/round(np.pi,10),2)}$\pi$" for k in state.chain.k])
            #plt.title(loc_identifier)
            plt.title("ff2(k,q)=sin(q)*(sin(k)+sin(k+q)), h = 14")
            plt.xlabel("$k$")
            plt.legend()
            plt.grid()
            if save:
                plt.savefig("xtick_"+loc_identifier+".png", bbox_inches = "tight")
            #plt.show()
            plt.close()
            
plot_file(plot_d_a_i=True, path="fermions_easter/",save=True, h_list=[14])