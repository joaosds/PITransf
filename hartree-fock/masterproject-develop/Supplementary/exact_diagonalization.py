#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 16:09:29 2021

origrinal idea taken from
https://tenpy.github.io/intro/examples/tfi_exact.html
under notice of Copyright 2019-2020 TeNPy Developers, GNU GPLv3
adapted by
@author: Michael Perle
fJ this is the exact_diagonalization file for the TFIM following section (6.1.1)
"""
import matplotlib.pyplot as plt
import numpy as np
np.seterr(all='raise')
import scipy.sparse as sparse
from scipy.sparse.linalg import eigsh
from datetime import datetime
import warnings
import time

    
"""Provides exact ground state energies for the transverse field ising model for comparison.

The Hamiltonian reads
.. math ::
    H = - J \\sum_{i} \\sigma^z_i \\sigma^z_{i+1} - h \\sum_{i} \\sigma^x_i
"""


def finite_gs_energy(L, J, h, plot_histogram=False, calculate_entropy=False):
    """For comparison: obtain ground state energy from exact diagonalization.

    Exponentially expensive in L, only works for small enough `L` <~ 20.
    """
    if L >= 20:
        warnings.warn("Large L: Exact diagonalization might take a long time!")
    # get single site operaors
    sx = sparse.csr_matrix(np.array([[0., 1.], [1., 0.]]))
    sz = sparse.csr_matrix(np.array([[1., 0.], [0., -1.]]))
    id = sparse.csr_matrix(np.eye(2))
    sx_list = []  # sx_list[i] = kron([id, id, ..., id, sx, id, .... id])
    sz_list = []
    X = sx
    Z = sz
    for i_site in range(L):
        if i_site > 0:
            X = Z = id
        for j in range(1, L):
            if j==i_site:
                x_ops = sx
                z_ops = sz
            else:
                x_ops = z_ops = id
            X = sparse.kron(X, x_ops, 'csr')
            Z = sparse.kron(Z, z_ops, 'csr')
        sx_list.append(X)
        sz_list.append(Z)
    H_zz = sparse.csr_matrix((2**L, 2**L))
    H_x = sparse.csr_matrix((2**L, 2**L))
    for i in range(L): #pbc
        H_zz = H_zz + sz_list[i] * sz_list[(i + 1) % L]
    
    
    s_zz =[]
    s_xx =[]
    for delta in range(L+1):
        c_delta_z = sparse.csr_matrix((2**L, 2**L))
        c_delta_x = sparse.csr_matrix((2**L, 2**L))
        for i in range(L):
            add_to_c_delta_z = sparse.csr_matrix(sz_list[i] * sz_list[(i + delta) % L])
            add_to_c_delta_x = sparse.csr_matrix(sx_list[i] * sx_list[(i + delta) % L])
            c_delta_z += add_to_c_delta_z
            c_delta_x += add_to_c_delta_x
        s_zz.append(c_delta_z)
        s_xx.append(c_delta_x)
        
    
    for i in range(L):
        H_x = H_x + sx_list[i]
    H = -J * H_zz - h * H_x
    E, V = eigsh(H, k=1, which='SA', return_eigenvectors=True, ncv=20)
    # V[:, i] being the ith eigenvector corresponding to w[i]
    c_delta_z = []
    c_delta_x =[]

    for delta in range(L+1):
        c_delta_z.append(V.transpose().dot(s_zz[delta].dot(V))/L)
        c_delta_x.append(V.transpose().dot(s_xx[delta].dot(V))/L)
    w = v = None
    if (plot_histogram or calculate_entropy) and L <=10:
        denseH = np.asarray(H.todense())
        w, v = np.linalg.eigh(denseH)
    if plot_histogram:
        psi_squared = v[:, 0] * np.conj(v[:, 0])  # psi**2 = p
        psi_squared = [psi.real for psi in psi_squared]
        ky = np.arange(2 << L)
        dictionary = dict(zip(ky, psi_squared))
        total_occurrences = sum(dictionary.values())
        fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 4]})
        ax[1].plot([int(key) for key in sorted(dictionary, key=int)], psi_squared, linestyle="None", marker=".", color='b')
        ax[1].set_xlabel("s")
        ax[1].set_ylabel("$|\psi(s)|^2$")
        ax[1].set_yscale('log')
        ax[1].set_xticks([0, (1 << int(L)) - 1], [0, (1 << int(L)) - 1])
        biggest_kv = [[str(key), dictionary.get(key) / total_occurrences] for key in sorted(dictionary, key=lambda k: dictionary.get(k), reverse=True)[0:28]]
        ax[0].plot([kv[0] for kv in biggest_kv], [kv[1] for kv in biggest_kv], linestyle="None", marker=".", color='b')
        ax[0].set_yscale('log')
        ax[0].set_ylabel("$|\psi(s)|^2$")
        ax[0].set_xticklabels([kv[0] for kv in biggest_kv], rotation=70)
        # ax[0].set_yticks([1e-2,1e-3])
        fig.suptitle("ED" + f", $h = {h}, |\psi({np.argmax(psi_squared)})|^2 \\approx $" + f"${round(psi_squared[np.argmax(psi_squared)], 3)}$, TFIM")
        fig.tight_layout()
        fig.savefig(f"C:\\Users\\Hester\\PycharmProjects\\masterproject\\Text\\N={L} h={h} TFIM ED histogram.pdf")
        fig.show()


    if calculate_entropy:
        rho = np.tensordot(np.conj(v[:,0]), v[:,0].T, axes=0)
        ground_state_wavefunction_ampl = v[:, 0]*np.conj(v[:, 0])
        entropy = -sum([(ground_state_wavefunction_ampl[i] * np.log2(ground_state_wavefunction_ampl[i])) if ground_state_wavefunction_ampl[i]>0 else ground_state_wavefunction_ampl[i] for i in range(len(ground_state_wavefunction_ampl))])
        try:
            entropy_by_matrix = -np.trace(rho@np.log2(rho))
        except FloatingPointError:
            entropy_by_matrix = None
        return entropy, entropy_by_matrix
    
    return E[0], [s[0][0] for s in c_delta_z], [s[0][0] for s in c_delta_x]


def write_file(L, J, h):

    start = datetime.now()
    identifier = f"l_{L}_h_{h}"
    E, c_delta_z, c_delta_x = finite_gs_energy(L,J,h)
    ed_observables = open(identifier + "_ed_observables.txt", "w")
    ed_observables.write(f"L: {L}, J: {J}\n")
    ed_observables.write(f"h: {h}\n")
    ed_observables.write(f"E: {E}\n")
    ed_observables.writelines([f"c_s_{delta}: {c_delta_z[delta]}\n" for delta in range(len(c_delta_z))])
    ed_observables.writelines([f"c_x_{delta}: {c_delta_x[delta]}\n" for delta in range(len(c_delta_x))])
    ed_observables.write("\n")
    ed_observables.close()
    print(datetime.now()-start)
    

# entropy = finite_gs_energy(5,1,0.8, plot_histogram=True)
# print("fin")
# h = 0.3
# print(f"h={h}")
# print(finite_gs_energy(20, 1, h, plot_histogram=False, calculate_entropy=False))
print(finite_gs_energy(10, 1, 0.03, plot_histogram=True, calculate_entropy=False))

# h = 0.1
# print(f"h={h}")
# print("start")
# write_file(20, 1, h)
