#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 12:25:08 2022

@author: Michael Perle
"""
import os
import sys

sys.path.append(os.path.normpath(os.getcwd() + os.sep + os.pardir))
import numpy as np
from scipy.linalg import eigh
from RBM.FermionModel import FermionModel
from numpy import conjugate as conj

np.set_printoptions(precision=16)
pi = np.pi
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, complex(0, -1)], [complex(0, 1), 0]])
sigma_z = np.array([[1, 0], [0, -1]])


def test_properties(Dk, H_MF, Uk, Uk_dag, error_fileout, k, t):
    # checking if H_MF is really hermitian, if diagonalization was correct, if resulting U is really unitary
    # fJ this is a test function which I wrote to ensure consistency when I implemented the routine
    if np.linalg.norm(conj(H_MF).T - H_MF) > 1e-12:
        error_fileout.write(
            f"{str(H_MF)} might not be hermitian, norm = {np.linalg.norm(conj(H_MF).T - H_MF)}\n"
        )
        print(
            f"{str(H_MF)} might not be hermitian, norm = {np.linalg.norm(conj(H_MF).T - H_MF)}\n"
        )
    dp_holds = np.all(np.round(Dk, 12) == np.round(Uk_dag @ H_MF @ Uk, 12))
    if not dp_holds:
        error_fileout.write(
            f"diagonalization properties do not hold in step {t} for k={k[1]} with norm = {np.linalg.norm(Dk - Uk_dag @ H_MF @ Uk)}\n"
        )
        error_fileout.write(
            f"Dk = {str(np.round(Dk, 12))}, Uk = {str(Uk)}, H_MF = {str(H_MF)}\n"
        )
    up_holds = np.all(np.eye(2) == np.round(Uk_dag @ Uk, 12))
    if not up_holds:
        error_fileout.write(
            f"unitary properties do not hold in step {t} for k={k[1]}\n"
        )
        error_fileout.write(f"Uk = {str(Uk)}\n")


def test_projector_properties(P_tp1_k, error_fileout, k, t):
    # fJ this is a test function which I wrote to ensure consistency when I implemented the routine
    pp_holds = np.all(
        np.round(P_tp1_k @ P_tp1_k, 12) == np.round(P_tp1_k, 12)
    ) and np.all(np.round(P_tp1_k, 12) == np.round(conj(P_tp1_k).T, 12))
    # print(f"projector properties hold: \t\t {pp_holds}")
    if not pp_holds:
        error_fileout.write(
            f"projector properties do not hold in step {t} for k={k[1]}\n"
        )
        error_fileout.write(f"P =\n{P_tp1_k}\n")


class HartreeFockCalculations:
    """
    fJ the class which performs the HF calculations according to section (6.2.3)
    fJ model.sumOverG == False means that (6.15) holds. Be careful when you set model.sumOverG=True, I can not promise that results will be correct without the (6.15) condition w
    fJ I'm highly confident that everything is correct if (6.15) holds. If (6.15) holds, you set model.sumOverG == False
    """

    def __init__(
        self,
        Uflag,
        model: FermionModel,
        path0,
        p0=None,
        occupation_freq=True,
        energy_freq=True,
    ):
        self.model = model
        self.Uflag = Uflag
        # fJ the frequencies determine how often a FILE IS CREATED AND WRITTEN (file limit on cluster!).
        # Allows to make a nice gif of the energy/occupation convergence towards
        # fJ the mean field solution
        # fJ occupation means N_x, N_y and N_z as visualized in Fig 8 in my thesis
        self.occupation_freq = occupation_freq
        # fJ energy means the (D_k)_{--}=E_k^- in (6.42) in each iteration
        self.energy_freq = energy_freq
        # fJ I recommend always initializing with random projector, i.e. giving p0_rand as argument when creating the object
        self.p0 = p0
        self.path0 = path0
        # fJ see (6.44)
        self.D = np.array([[0, 0], [0, 1]])
        self.p0label = ""
        if self.p0 is None:
            self.P = [np.array([[0, 0], [0, 1]], dtype=complex) for _ in self.model.k]
        else:
            self.P = p0
            self.p0label = "p0"

    def __constant_term(self, k):
        """
        needs to be calculated only once since it is independent of self.P
        :param k:
        :return: the constant offset term which arises when we use form factors that violate (6.15)
        # fJ since I always ensure that (6.15) holds, I don't want to promise that the implementation with "sumOverG==True" is correct
        """
        if self.model.sumOverG:
            return np.eye(2) * sum(
                [
                    G[2]
                    * sum([self.model.ff1(ks[1], G[1]) for ks in self.model.k])
                    * self.model.ff1(k[1], G[1])
                    for G in self.model.G
                ]
            )
        else:
            return np.zeros([2, 2])

    def __single_body_term(self, k):
        """
        needs to be calculated only once since it is independent of self.P
        :param k:
        :return: equation (6.30) in the thesis
        """
        kinetic_term = self.model.h * np.cos(k[1]) * sigma_x
        if self.model.sumOverG:
            single_body_G = -2 * sum(
                [
                    G[2]
                    * sum([self.model.ff1(ks[1], G[1]) for ks in self.model.k])
                    * self.model.bigF(k[1], G[1])
                    for G in self.model.G
                ]
            )
            single_body_qG = sum(
                [
                    conj(self.model.bigF(k[1], qG[1]).T)
                    @ self.model.bigF(k[1], qG[1])
                    * qG[2]
                    for qG in np.append(self.model.q, self.model.G, axis=0)
                ]
            )
        else:
            single_body_G = 0
            single_body_qG = sum(
                [
                    conj(self.model.bigF(k[1], q[1]).T)
                    @ self.model.bigF(k[1], q[1])
                    * q[2]
                    for q in self.model.q
                ]
            )
        return kinetic_term + single_body_qG + single_body_G

    def __test_energy_cancellation(self, k, hartree_term_k, fock_term_k):
        # fJ a test function that maybe becomes relevant if you want to violate (6.15)
        # fj otherwise you don't need it, or you can write your own
        # fj no usages in my code
        E_h1singleG = np.trace(
            self.P[int(k[0])].T
            @ (
                -2
                * sum(
                    [
                        G[2]
                        * sum([self.model.ff1(ks[1], G[1]) for ks in self.model.k])
                        * self.model.bigF(k[1], G[1])
                        for G in self.model.G
                    ]
                )
            )
        )
        E_h1const = np.trace(self.P[int(k[0])].T @ self.__constant_term(k))
        E_H = self.get_hartree_fock_energy(
            hartree_fock_term=hartree_term_k, k_index=k[0]
        )
        if abs(E_h1singleG + E_H + E_h1const) > 1e-16:
            raise ValueError
        E_h1singleqelemRL = np.trace(
            self.P[int(k[0])].T
            @ sum(
                [
                    conj(self.model.bigF(k[1], G[1]).T)
                    @ self.model.bigF(k[1], G[1])
                    * G[2]
                    for G in self.model.G
                ]
            )
        )

        F_G = np.array([[0, 0], [0, 0]], dtype=complex)
        for qG in self.model.G:
            F_G += (
                qG[2]
                * conj(self.model.bigF(k[1], qG[1])).T
                @ self.P[self.model.pbc(k[0] + qG[0])].T
                @ self.model.bigF(k[1], qG[1])
            )
        F_G *= -2
        E_F_G = self.get_hartree_fock_energy(F_G, k[0])
        if abs(E_h1singleqelemRL + E_F_G) > 1e-16:
            raise ValueError

    def __fock_term(self, k):
        # direct implementation of fock term
        # fJ implements (6.35)
        hf = np.array([[0, 0], [0, 0]], dtype=complex)
        if self.model.sumOverG:
            for qG in np.append(self.model.q, self.model.G, axis=0):
                hf += (
                    qG[2]
                    * conj(self.model.bigF(k[1], qG[1])).T
                    @ self.P[self.model.pbc(k[0] + qG[0])].T
                    @ self.model.bigF(k[1], qG[1])
                )
        else:
            for qG in self.model.q:
                hf += (
                    qG[2]
                    * conj(self.model.bigF(k[1], qG[1])).T
                    @ self.P[self.model.pbc(k[0] + qG[0])].T
                    @ self.model.bigF(k[1], qG[1])
                )
        return -2 * hf

    def __hartree_term(self, k):
        # direct implementation of hartree term
        # fJ which is nonzero if (6.15) holds
        if not self.model.sumOverG:
            return np.array([[0, 0], [0, 0]], dtype=complex)
        ht = np.array([[0, 0], [0, 0]], dtype=complex)
        for G in self.model.G:
            ks_sum = complex(0, 0)
            for ks in self.model.k:
                ks_sum += np.trace(
                    self.P[int(ks[0])] @ conj(self.model.bigF(ks[1], G[1]))
                )
            ht += G[2] * ks_sum * self.model.bigF(k[1], G[1])
        return 2 * ht

    def get_hartree_fock_energy(self, hartree_fock_term, k_index):
        """
        :param hartree_fock_term: matrix expression of the fock term + hartree term
        :param k_index:
        :return: fock energy with factor 1/2 that is due definition following mathias paper and should prevent overcounting the energy
        """
        return 0.5 * np.trace(self.P[int(k_index)].T @ hartree_fock_term)

    def get_single_energy(self, single_term, k_index):
        """
        :param single_offset_term: matrix expression of the single + offset term
        :param k_index:
        :return: kinetic energy
        """
        return np.trace(self.P[int(k_index)].T @ single_term)

    def get_hf_functional(self, k):
        constant_term = [
            self.__constant_term(k) for k in self.model.k
        ]  # is 0 if sumOverG not included
        single_body_term = [
            self.__single_body_term(k) for k in self.model.k
        ]  # is 0 if h is 0
        single_body_term_k = single_body_term[int(k[0])]
        constant_term_k = constant_term[int(k[0])]
        fock_term_k = self.__fock_term(k)
        hartree_term_k = self.__hartree_term(k)  # is 0 if sumOverG not included
        H_MF_k = single_body_term_k + constant_term_k + fock_term_k + hartree_term_k
        return H_MF_k, fock_term_k

    def iterate(self, desired_accuracy, max_number_of_iterations, path=""):
        pathtemp = path + f"HF_t={self.model.h:.3e}"
        if not (os.path.exists(pathtemp) and os.path.isdir(pathtemp)):
            os.makedirs(pathtemp)
        # os.mkdir(path + f"HF_t={self.model.h:.5e}")
        error_fileout = open(
            path + f"HF_t={self.model.h:.3e}/{self.p0label}HF_N_error.dat", "w"
        )
        lastIteration = False  # flag that everything (most importantly U_HF) should be written to a file
        occupation_fileout = energy_fileout = None
        Uk_list = (
            []
        )  # fJ this stores the output that we need in the end: the unitary U_HF transformation
        Dk_sum = 0
        final_HF_hamiltionen_list = []
        # fJ as mentioned those terms are independent of P and need only to be computed once
        constant_term = [self.__constant_term(k) for k in self.model.k]
        single_body_term = [self.__single_body_term(k) for k in self.model.k]

        for t in range(max_number_of_iterations):
            if lastIteration:
                occupation_fileout = open(
                    path
                    + f"HF_t={self.model.h:.3e}/{self.p0label}HF_N_iteration={int(t)}.dat",
                    "w",
                )
                occupation_fileout.write("k \t N_x \t N_y \t N_z \n")
            if lastIteration:
                energy_fileout = open(
                    path
                    + f"HF_t={self.model.h:.3e}/{self.p0label}HF_N_energy_iteration={int(t)}.dat",
                    "w",
                )
                energy_fileout.write("k \t E \n")

            P_tp1 = [
                np.zeros([2, 2], dtype=complex) for _ in self.model.k
            ]  # (tp1 means t+1) caches the projector that is calculated at the end of each iteration
            P_diff = [
                np.zeros([2, 2], dtype=complex) for _ in self.model.k
            ]  # stores the difference between P_t and P_tp1

            try:
                energy_sum = 0
                for k in self.model.k:
                    single_body_term_k = single_body_term[int(k[0])]
                    constant_term_k = constant_term[int(k[0])]
                    fock_term_k = self.__fock_term(k)
                    hartree_term_k = self.__hartree_term(k)
                    H_MF_k = (
                        single_body_term_k
                        + constant_term_k
                        + fock_term_k
                        + hartree_term_k
                    )
                    # diagonalizing H_MF
                    w, v = eigh(
                        H_MF_k
                    )  # v being the normalized (unit “length”) eigenvectors, such that the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
                    Dk = np.array(
                        [
                            [w[0] if np.real(w[0]) > np.real(w[1]) else w[1], 0],
                            [0, w[0] if np.real(w[0]) < np.real(w[1]) else w[1]],
                        ]
                    )  # following (6.42)
                    Uk = np.array(
                        [
                            v[:, 0] if np.real(w[0]) > np.real(w[1]) else v[:, 1],
                            v[:, 0] if np.real(w[0]) < np.real(w[1]) else v[:, 1],
                        ]
                    ).T  # following (6.43)
                    Uk_dag = conj(Uk).T

                    # Begin Tests
                    #test_properties(Dk, H_MF_k, Uk, Uk_dag, error_fileout, k, t)
                    # End Tests

                    # writing occupation number in band basis
                    #if not t % self.occupation_freq or lastIteration:
                    if lastIteration:
                        occupation_fileout.write(
                            f"{k[1]} \t {(Uk_dag @ sigma_x @ Uk)[1][1]} \t {(Uk_dag @ sigma_y @ Uk)[1][1]} \t {(Uk_dag @ sigma_z @ Uk)[1][1]}\n"
                        )
                        # print("k \t\t N_x \t N_y \t N_z \n")
                        # print(
                        #     f"{k[1]:.3f} \t {(Uk_dag @ sigma_x @ Uk)[1][1]} \t {(Uk_dag @ sigma_y @ Uk)[1][1]} \t {(Uk_dag @ sigma_z @ Uk)[1][1]}\n"
                        # )
                    # writing energy including offset to file
                    if lastIteration:
                        single_energy_k = self.get_single_energy(
                            single_body_term_k + constant_term_k, k_index=k[0]
                        )  # why??
                        hartree_fock_energy_k = self.get_hartree_fock_energy(
                            hartree_fock_term=hartree_term_k + fock_term_k, k_index=k[0]
                        )
                        trace_energy_k = single_energy_k + hartree_fock_energy_k
                        Dk_energy = Dk[1][1] - hartree_fock_energy_k
                        # self.__test_energy_cancellation(k, hartree_term_k, fock_term_k)
                        if lastIteration:
                            Dk_sum += Dk_energy
                            final_HF_hamiltionen_list.append(H_MF_k)
                        energy_fileout.write(f"{k[1]} \t {trace_energy_k}\n")
                        # print("k \t\t E \n")
                        # print(f"{k[1]:.3f} \t {trace_energy_k}\n")
                        energy_sum += trace_energy_k
                        """
                        fJ Dk_sum and energy_sum must have the same value upon convergence. before convergence, one is the energy from the step before since the projector is only up-
                        fJ dated at the end of each iteration. this is a little bit messy but having those two energies in the first place is only for test purposes because they
                        fJ are equivalent analytically (apart from one being calculated with the "old" and the other with the "new" projector)
                        """

                    # calculating Pk^(t+1)
                    P_tp1_k = conj(Uk) @ self.D @ Uk.T  # fJ see (6.45)
                    P_tp1[int(k[0])] = np.array(P_tp1_k)
                    P_diff[int(k[0])] = P_tp1[int(k[0])] - self.P[int(k[0])]
                    # Begin Test
                    test_projector_properties(P_tp1_k, error_fileout, k, t)
                    # End Test

                    # preparing to print U as a numpy file
                    if lastIteration:
                        Uk_list.append(Uk)
                # diff_list.append(abs(np.real(sum(# diag_energy_list))-np.real(sum(# tr_energy_list))))
            except KeyboardInterrupt:
                print("last iteration and file write follows")
                if lastIteration:
                    raise SystemExit(
                        "Hitting Strg+c twice brute force ends the program"
                    )
                lastIteration = True
                t -= 1
                continue  # executes finally and then moves on to next iteration
            finally:
                if lastIteration:
                    occupation_fileout.close()
                if lastIteration:
                    energy_fileout.close()

            max_norm = max(
                [np.linalg.norm(diff_matrix) for diff_matrix in P_diff]
            )  # max_norm<eps means convergence
            self.P = P_tp1

            if lastIteration:
                print(f"Acurracy after {t}  iterations = {max_norm}")
                accuracy_fileout = open(
                    path
                    + f"HF_t={self.model.h:.3e}/{self.p0label}HF_N_accuracy_info.txt",
                    "w",
                )
                accuracy_fileout.write(f"accuracy after {t} iterations = {max_norm}")
                accuracy_fileout.close()
                error_fileout.close()
                if self.Uflag == "complete":  # Get both the HF energy + basis
                    print("***Getting both E_HF and Uk for hf-basis***")
                    np.save(
                        path + f"Uk_N={self.model.length}_t={self.model.h:.3e}", Uk_list
                    )
                    np.save(
                        path
                        + f"bin_projector_N={self.model.length}_t={self.model.h:.3e}",
                        self.P,
                    )
                    np.save(
                        self.path0 + "uk.npy",
                        Uk_list,
                    )
                    with open(
                        self.path0 + "enhf.txt",
                        "w",
                    ) as file1:
                        file1.write(f"{energy_sum/self.model.length}")
                elif self.Uflag == "detuned":  # Get only the basis
                    print("***Getting only the detuned Uk for hf-basis***")
                    np.save(
                        path + f"Uk_N={self.model.length}_t={self.model.h:.3e}", Uk_list
                    )
                    np.save(
                        path
                        + f"bin_projector_N={self.model.length}_t={self.model.h:.3e}",
                        self.P,
                    )
                    np.save(
                        self.path0 + "uk.npy",
                        Uk_list,
                    )
                elif self.Uflag == "energy":  # Get only the HF energy
                    print("***Getting only the energy***")
                    with open(
                        self.path0 + "enhf.txt",
                        "w",
                    ) as file1:
                        file1.write(f"{energy_sum/self.model.length}")
                print("Ehf = ", energy_sum / self.model.length)
                return energy_sum, final_HF_hamiltionen_list, self.P

            elif (max_norm < desired_accuracy) or t == max_number_of_iterations - 2:
                # print(f"Acurracy after {t}  iterations reached with max_norm = {max_norm}")
                lastIteration = True
            # else:
            # print(f"t = {self.model.h} Iteration {t}. Current max_norm = {max_norm}")
