#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 13:41:04 2022

@author: Michael Perle
"""

import os
from matplotlib import pyplot as plt
import numpy as np
import imageio
import sys
import HartreeFock.HFNumerics
from Plotting.plotUnitaryAndProjector import (
    get_projector_unitaries_kpoints,
    get_hartree_fock_functional,
)


# a path list is given to plot_energy
def plot_energy(
    pathList, length, identifier="", save_all=False, plot_convergence=False
):
    pi = np.pi
    k_value = np.linspace(start=-pi, stop=pi * (1 - 2 / length), num=length)

    t_list = []
    energy_list = []
    # offset_list = []
    constant_offset = None

    for path in pathList:
        if "t=" in path:
            hf_energy_files = sorted(
                [
                    file_name
                    for file_name in os.listdir(path=path)
                    if identifier in file_name
                    and "HF_N_energy_" in file_name
                    and not "png" in file_name
                ],
                key=lambda file: float(
                    file[
                        file.find("iteration=") + len("iteration=") : file.find(".dat")
                    ]
                ),
            )
            hf_energy_file_last = hf_energy_files[-1]
            # offset_file = [file_name for file_name in os.listdir(path=path) if identifier in file_name and "offset" in file_name and not "png" in file_name][0]
            t = path[path.find("t=") + len("t=") : path.find("/")]
            iteration = hf_energy_file_last[
                hf_energy_file_last.find("iteration=")
                + len("iteration=") : hf_energy_file_last.find(".dat")
            ]

        if plot_convergence:
            real_energy_over_iteration = []
            imag_energy_over_iteration = []
            for file in hf_energy_files:
                with open(path + file, "r") as energyfilepointer:
                    energyLines = energyfilepointer.readlines()
                energy_complex = 0.0
                for line in energyLines[1:]:
                    xi, y0i = line.strip("\n").split(" \t ")
                    energy_complex += complex(y0i.strip("(").strip(")"))
                real_energy_over_iteration.append(np.real(energy_complex))
                imag_energy_over_iteration.append(np.imag(energy_complex))
            plt.plot(
                range(len(real_energy_over_iteration)),
                np.array(real_energy_over_iteration),
                label="$Re\{E_{HF}\}$",
            )
            # plt.plot(range(len(real_energy_over_iteration)), imag_energy_over_iteration, label="$Im\{E_{HF}\}$")
            plt.xlabel("iteration")
            plt.ylabel("$E$")
            plt.legend()
            plt.grid()
            # plt.xlim((200,250))
            plt.title(f"t={t}, {hf_energy_file_last[:hf_energy_file_last.find('_')]}")
            plt.show()
            plt.close()

        energyfilepointer = open(path + hf_energy_file_last, "r")
        energyLines = energyfilepointer.readlines()
        energyfilepointer.close()
        energy = 0.0
        for line in energyLines[1:]:
            xi, y0i = line.strip("\n").split(" \t ")
            energy += np.real(complex(y0i.strip("(").strip(")")))

        t_list.append(float(t))
        energy_list.append(energy)
        # offset_list.append(offset)

    asymptotic_energy_list = [
        -float(t) * sum([abs(np.cos(k)) for k in k_value]) for t in t_list
    ]
    plt.plot(t_list, energy_list, marker="x", linestyle="None", label="HF_energy")
    plt.plot(
        t_list, asymptotic_energy_list, label="asymptotic", marker="o", fillstyle="none"
    )
    # plt.plot(t_list, energy_list, label="HF", marker="x")
    # plt.plot(t_list, np.array(energy_list) + np.array(offset_list), label = "offset acc to Tr[..]", marker = "x", linestyle="none")
    # plt.plot(t_list, [energy_t+constant_offset for energy_t in energy_list], marker = "x", linestyle = "None", label = "HF_const_offset")
    plt.title("Energy over t")
    plt.xlabel("t")
    plt.ylabel("E")
    plt.grid()
    plt.legend()
    plt.show()
    plt.close()
    return t_list, energy_list, asymptotic_energy_list


def plot_occupation(
    path0, path="", identifier="", h_list=[], save_all=False, make_gif=False
):

    if "t=" in path:
        hf_occupation_file = sorted(
            [
                file_name
                for file_name in os.listdir(path=path)
                if identifier in file_name
                and "HF_N_iteration" in file_name
                and not "png" in file_name
                and not "energy" in file_name
            ],
            key=lambda file: float(
                file[file.find("iteration=") + len("iteration=") : file.find(".dat")]
            ),
        )
    elif path:
        hf_occupation_file = sorted(
            [
                file_name
                for file_name in os.listdir(path=path)
                if identifier in file_name and "HF_N_iteration" in file_name
            ],
            key=lambda file: float(
                file[file.find("t=") + len("t=") : file.find(".dat")]
            ),
        )
    else:
        hf_occupation_file = sorted(
            [
                file_name
                for file_name in os.listdir()
                if identifier in file_name and "HF_N_iteration" in file_name
            ],
            key=lambda file: float(
                file[file.find("t=") + len("t=") : file.find(".dat")]
            ),
        )
    if h_list:
        hf_occupation_file = [
            file_name
            for file_name in hf_occupation_file
            if any(f"h-{h:.5e}" in file_name for h in h_list)
        ]

    if not make_gif:
        hf_occupation_file = hf_occupation_file[-1:]

    for file in hf_occupation_file:
        file_pointer = open(path + file, "r")
        print(file_pointer)

        if "t=" in path:
            h = path[path.find("t=") + len("t=") : path.find("/")]
            iteration = file[
                file.find("iteration=") + len("iteration=") : file.find(".dat")
            ]
            labelLine = 0
        else:
            h = file[file.find("t=") + len("t=") : file.find(".dat")]
            labelLine = 1

        try:
            Lines = file_pointer.readlines()
            if labelLine == 1:
                iteration = Lines[0][: Lines[0].find(" iterations")]
            # fileout.write("k \t N_x \t N_y \t N_z")
            xlabel = Lines[labelLine].split(" \t ")[0]
            y0label = Lines[labelLine].split(" \t ")[1]
            y1label = Lines[labelLine].split(" \t ")[2]
            y2label = Lines[labelLine].split(" \t ")[3].strip(" \n")
        except IndexError:
            file_pointer.close()
            continue
        x = []
        y0 = []
        y1 = []
        y2 = []
        flagy0 = flagy1 = flagy2 = False
        i = 0
        for line in Lines[(labelLine + 1) :]:
            xi, y0i, y1i, y2i = line.strip("\n").split(" \t ")
            y0i = complex(y0i.strip("(").strip(")"))
            y1i = complex(y1i.strip("(").strip(")"))
            y2i = complex(y2i.strip("(").strip(")"))
            x.append(float(xi))

            if np.imag(y0i) < 1e-8:
                y0.append(np.real(y0i))
            else:
                flagy0 = True

            if np.imag(y1i) < 1e-8:
                y1.append(np.real(y1i))
            else:
                flagy1 = True

            if np.imag(y2i) < 1e-8:
                y2.append(np.real(y2i))
            else:
                flagy2 = True

            i += 1
            test = np.array([y0, y1, y2])
            print(i)
            np.save(path0 + "nk.npy", test)


# def plot_order_paramater(
#     path="", identifier="", h_list=None, save_all=False, make_gif=False
# ):
#
#     hf_t_list = sorted(
#         [
#             file_name
#             for file_name in os.listdir(path=path)
#             if identifier in file_name and "HF_t=" in file_name
#         ],
#         key=lambda file: float(file[file.find("t=") + len("t=") :]),
#     )
#     if h_list is not None:
#         hf_t_list = [
#             file_name
#             for file_name in hf_t_list
#             if any(f"HF_t={h:.5e}" in file_name for h in h_list)
#         ]
#
#     N = int(path[path.find("N=") + len("N=") : path.find("-N")])
#     t_list = []
#     y0_sum = []
#     y1_sum = []
#     import torch
#
#     y2_sum = []
#     for hf_t in hf_t_list:
#         h = float(hf_t[hf_t.find("t=") + len("t=") :])
#         if h > 0.2:
#             break
#         last_hf_occupation_file = sorted(
#             [
#                 file_name
#                 for file_name in os.listdir(path=os.path.join(path, hf_t))
#                 if identifier in file_name
#                 and "HF_N_iteration" in file_name
#                 and not "png" in file_name
#                 and not "energy" in file_name
#             ],
#             key=lambda file: float(
#                 file[file.find("iteration=") + len("iteration=") : file.find(".dat")]
#             ),
#         )[-1]
#         path_of_last_occupation_file = os.path.normpath(
#             str(os.path.join(path, hf_t)) + "\\" + last_hf_occupation_file
#         )
#         file_pointer = open(path_of_last_occupation_file, "r")
#
#         iteration = last_hf_occupation_file[
#             last_hf_occupation_file.find("iteration=")
#             + len("iteration=") : last_hf_occupation_file.find(".dat")
#         ]
#         labelLine = 0
#
#         try:
#             Lines = file_pointer.readlines()
#             if labelLine == 1:
#                 # print(Lines[0])
#                 iteration = Lines[0][: Lines[0].find(" iterations")]
#             # fileout.write("k \t N_x \t N_y \t N_z")
#             xlabel = Lines[labelLine].split(" \t ")[0]
#             y0label = Lines[labelLine].split(" \t ")[1]
#             y1label = Lines[labelLine].split(" \t ")[2]
#             y2label = Lines[labelLine].split(" \t ")[3].strip(" \n")
#         except IndexError:
#             file_pointer.close()
#             continue
#         y0 = 0  # should be N_x
#         y1 = 0  # should be N_y
#         y2 = 0  # should be N_x
#
#         for line in Lines[(labelLine + 1) :]:
#             xi, y0i, y1i, y2i = line.strip("\n").split(" \t ")
#             y0i = complex(y0i.strip("(").strip(")"))
#             y1i = complex(y1i.strip("(").strip(")"))
#             y2i = complex(y2i.strip("(").strip(")"))
#
#             y0 += y0i
#             y1 += y1i
#             y2 += y2i
#         t_list.append(float(h))
#         y0_sum.append(y0)
#         y1_sum.append(y1)
#         y2_sum.append(y2)
#
#     fig, ax = plt.subplots(1)
#     # ax.plot(t_list, np.array(y0_sum)/N, label=y0label)
#     # ax.plot(t_list, np.array(y1_sum)/N, label=y1label)
#     # ax.plot(t_list, np.array(np.real(y2_sum))/N, label=f"${y2label}$", linestyle="none", marker = "o", fillstyle="none", color='C0')
#     ax.plot(
#         t_list,
#         np.array(np.real(y2_sum)) / N,
#         label=f"$\\xi$",
#         linestyle="none",
#         marker="o",
#         fillstyle="none",
#         color="C0",
#     )
#     # ax.plot(t_list, np.abs(np.real(np.array(y2_sum))) / N, label="$|$" + f"${y2label}$" + "$|$", marker="x", color='C0')
#     ax.plot(
#         t_list,
#         np.abs(np.real(np.array(y2_sum))) / N,
#         label="$|$" + f"$\\xi$" + "$|$",
#         marker="x",
#         color="C0",
#     )
#     ax.set_xlabel("$t$")
#     ax.set_title(f"N={N}")
#     ax.legend()
#     ax.grid()
#     fig.tight_layout()
#     fig.savefig(
#         "C:\\Users\\Hester\\PycharmProjects\\masterproject\\Text\\"
#         + path[path.find("HF_Results") + len("HF_Results") :]
#         .replace("\\", "_")
#         .replace("/", "")
#         + "_order_parameter"
#         + ".pdf"
#     )
#     fig.show()
#     return fig, ax


# for t in [0]:
#     hf_path= f"HF_t={t:.5e}/"
#     plot_occupation(path="C:\\Users\\Hester\\PycharmProjects\\masterproject\\RawResults\\HF_Results\\original0_p0rand_longConv_final_N=100-N=100\\"+hf_path, make_gif=False)


#
path0 = sys.argv[1]
Ni = int(sys.argv[2])
t = float(sys.argv[3])
print(t)
path = (
    os.path.normpath(os.getcwd() + os.sep + os.pardir)
    + f"/masterproject-develop/RawResults/newest/asd-N={Ni}/HF_t={t:.3e}/"
)
print(path)
print(path0)
plot_occupation(
    path0=path0,
    path=path,
)
