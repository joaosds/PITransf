#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 17:22:48 2021

@author: Michael Perle
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_file(path, save_all_pictures=False, make_gif=True, basis=None):
    folder = path
    print(f"prepare to plot {folder}")
    up = u'\u2191'
    down = u'\u2193'


    print(f"making gif for {folder}")
    pictureNamesList = list()
    N = (folder[folder.find(f"t={t}_N=") + len(f"t={t}_N="):folder.find("_M=")])
    seed = int(folder[folder.find(f"_seed=") + len(f"_seed="):folder.find("_speed")])
    sorted_files = sorted(filter(lambda f: "txt" in f, os.listdir(path=folder + "/")), key=lambda f: int(f[2:-4]))[-1:]
    for file in sorted_files:
        iteration = int(file[2:-4])
        # if iteration % 10 and make_gif:
        #     continue
        with open(folder + "/" + file, "r") as file_pointer:
            dictionary = json.load(file_pointer)
        total_occurrences = sum(complex(value[0]).real for value in dictionary.values())

        for i in [0, (1 << int(N)) - 1]:
            try:
                dictionary[str(i)]
            except KeyError:
                dictionary[str(i)] = 0
        psi_squared = [complex(dictionary.get(key)[0]).real / total_occurrences for key in sorted(dictionary, key=int)]
        fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 4]})
        ax[1].plot([key for key in sorted(dictionary, key=int)], psi_squared, linestyle="None", marker=".")
        ax[1].set_xlabel("s")
        ax[1].set_ylabel("$|\psi(s)|^2$")
        # plt.ylim(1e-4, 3)
        ax[1].set_yscale('log')
        # plt.grid()
        # ylim = plt.ylim()
        # plt.vlines([int(1 << x) for x in range(int(N))], ylim[0], ylim[1], color = "red", alpha = 0.3)
        # ax[1].ylim(ylim)
        # plt.xticks([int(1 << x) for x in range(int(N))], [])
        # plt.draw()
        # current_ticks = plt.xticks(rotation=70)
        # plt.xticks(current_ticks[0][::10][:-1]+current_ticks[0][-1:], current_ticks[1][::10][:-1]+current_ticks[1][-1:], rotation=70)
        ax[1].set_xticks([int(list(dictionary.keys())[0]), int(list(dictionary.keys())[-1])])
        biggest_kv = [[key, complex(dictionary.get(key)[0]).real / total_occurrences] for key in sorted(dictionary, key=lambda k: complex(dictionary.get(k)[0]).real, reverse=True)[0:28]]
        ax[0].plot([kv[0] for kv in biggest_kv], [kv[1] for kv in biggest_kv], linestyle="None", marker=".")
        ax[0].set_yscale('log')
        ax[0].set_ylabel("$|\psi(s)|^2$")
        ax[0].set_xticklabels([kv[0] for kv in biggest_kv], rotation=70)
        # ax[0].set_yticks([1e-2,1e-3,1e-4])
        fig.suptitle(basis + f", $t = {t}, \chi = {seed}, p = {int(file[2:-4])}$" + f", $|\psi({np.argmax(psi_squared)})|^2 \\approx $" + f"${round(psi_squared[np.argmax(psi_squared)], 3)}$, full sampler")
        fig.tight_layout()
        if make_gif:
            plt.savefig(folder + "/occurrence_iteration=" + file[2:-4] + ".pdf")
            pictureNamesList.append(folder + "/occurrence_iteration=" + file[2:-4] + ".pdf")
        # fig.savefig(f"C:\\Users\\Hester\\PycharmProjects\\masterproject\\Text\\N={N} w={weights_per_visible_neuron} full sampler histogram seed={seed} basis={basis}.pdf")
        fig.show()

    # if False and make_gif and len(pictureNamesList) > 0:
    #     # build gif
    #     with imageio.get_writer(folder + '/occurrence.gif', mode='I') as writer:
    #         for filename in (pictureNamesList + 10 * [pictureNamesList[-1]]):
    #             image = imageio.imread(filename)
    #             writer.append_data(image)

        # Remove files
        # if not save_all_pictures:
        #     for filename in pictureNamesList[1:-10]:
        #         os.remove(filename)

        # print("identifiers that never occured:")
        # print([item for item in [x for x in range(64)] if item not in sorted(dictionary)])

t = 0.14
weights_per_visible_neuron = 10
path = "C:\\Users\\Hester\\PycharmProjects\\masterproject\\RawResults\\newest\\"
for basis in ["d_bar-basis"]:
    if basis == "d_bar-basis":
        seed = 1
    else:
        seed = 2
    rbm_identifier = f"original0_p0rand_longConv_final_N=10-N=10_{basis}_sampler-full_weights_per_visible_neuron={weights_per_visible_neuron}_unitary_t=None_fullN10_w3_seed={seed}_speed_test_t={t:.2f}_N=10_M=10_sr=T_eta=1.0_mc-steps=20_magnitude_files-h={t:.5e}\\"
    # rbm_identifier = f"original0_p0rand_longConv_final_N=16-N=16_{basis}_sampler-importance_weights_per_visible_neuron={weights_per_visible_neuron}_unitary_t=None_importanceN16_seed={seed}_more_param_t={t:.2f}_N=16_M=16_sr=T_eta=1.0_mc-steps=20_magnitude_files-h={t:.5e}\\"
    plot_file(path=path+rbm_identifier, basis=basis)
