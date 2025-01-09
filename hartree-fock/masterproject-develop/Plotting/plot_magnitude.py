#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 17:22:48 2021

@author: Michael Perle
"""

from RBM.IsingModel import IsingModel
from RBM.NeuralNetwork import  NeuralNetwork
from RBM.ImportanceSampler import Sampler
from RBM.State import State
import random
import matplotlib.pyplot as plt
import os




def plot_file(identifier="", h_list = [], save_all = False, path = "", equilibrium_steps = int(1e2), observable_steps = int(1e4), save = False, number_of_hidden_neurons = 12):
    
    if identifier == "" or h_list == []:
        print("This will take some time")

    #files_in_directory = os.listdir()
    if bool(path):
        convergence_files = sorted([file_name for file_name in os.listdir(path=path) if identifier in file_name and "_network_convergence_" in file_name])
    else:
        convergence_files = sorted([file_name for file_name in os.listdir() if identifier in file_name and "_network_convergence_" in file_name])
    if h_list:
        convergence_files = [file_name for file_name in convergence_files if any(f"h-{h:.5e}" in file_name for h in h_list)]
   
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
        J = 1
        
        while True:
            try:
                t = file_pointer.readline().split(": ")[1]
            except IndexError:
                break

            neural_str = file_pointer.readline()
            initial_configuration = file_pointer.readline().strip('][\n').split(', ')
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
        chain = IsingModel(J, h, exact_configuration = [int(x) for x in initial_configuration])
        neural_network = NeuralNetwork(chain.length, number_of_hidden_neurons, local_field_a= initial_a, local_field_b= initial_b, weights=initial_w)
        state = State(neural_network, chain)
        sampler = Sampler()
        up = u'\u2191'
        down = u'\u2193'
        
        print('sampling', end="")
        sampler.sample_state(state, equilibrium_steps)
        dictionary = {}
        print("...")
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


