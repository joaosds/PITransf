#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 10:36:26 2022

@author: Michael Perle
fJ this file contains the functions which are used to compute O_loc(s) in the case of the fermionic model
"""

import numpy as np
from numpy import conj as conj
from RBM.State import State


def h_loc(state: State, only_kinetic=False):
    """
    :param state: current state containing model and current state of configuration and model
    :return: the local energy of the state
    """

    def get_wf_div(k, l=None, k_value=None, l_value=None):
        if k_value is not None and k_value not in [1, -1]:
            raise ValueError("unexpected k_value")
        if l_value is not None and (l_value not in [1, -1] or k_value is None):
            raise ValueError("unexpected l_value")
        """
        supplementary function that determines which wf_div_matrix entries or state.wf_div_list values are needed
        fJ in other words: (6.19) always comes down to (6.11) or (3.20)
        and this nested function determines which of (3.20)-->div1() or (6.11)-->div2() to call
        :param k: inidicating psi([s]_k)/psi(s) when given alone
        :param l: inidicating psi([s]_k, s_[l])/psi(s) when given only together with l
        :param k_value: indicating psi((s)_k=k_value,...)/psi(s)
        :param l_value: indicating psi(...,(s)_l=l_value)/psi(s)
        :return: psi(s')/psi(s)
        """
        if k_value is None and l is None and l_value is None:
            # if only k is given: simple precalculated spinflip at site k
            return state.calculate_wf_div(k)

        if k_value is not None and l is None and l_value is None:
            # k is given with k_value. if (s)_k = k_value, return 1, else flip
            if state.chain.configuration[k] == k_value:
                return 1
            return state.calculate_wf_div(k)  # this should be k=k_value no?

        if l is not None and k_value is None and l_value is None:
            # both k and l are given with no values. return psi([s]_k, s_[l])/psi(s)
            return state.calculate_wf_div(k=k, l=l)
            # return wf_div_matrix[k][l]

        if l is not None and k_value is not None and l_value is not None:
            # both k and l are given with values
            if state.chain.configuration[k] == k_value:
                # is holds (s)_k = (s)
                if state.chain.configuration[l] == l_value:
                    # is holds (s)_k,l = (s)
                    return 1
                # only l configuration differs from (s)
                return get_wf_div(k=l, k_value=l_value)
            if state.chain.configuration[l] == l_value:
                return get_wf_div(k=k, k_value=k_value)

            if (
                state.chain.configuration[l] != l_value
                and state.chain.configuration[k] != k_value
            ):
                # flip both
                return state.calculate_wf_div(k=k, l=l)
                # return wf_div_matrix[k][l]

        raise ValueError(
            f"No valid parameters given to get_wf_div: k={k}, l={l}, k_value={k_value}, l_value={l_value}"
        )

    # it follows the implementation of H_1^b_loc

    """

    equals the d matrix elements if HFUnitary = np.eye(2)
    maybe parallelization could be implemented here

    """

    h1bloc = 0.0
    for q in state.chain.q:
        cp_k = 0.0
        for k in state.chain.k:
            k_index = int(k[0])
            kmq_index = state.chain.pbc(int(k[0]) - int(q[0]))
            # fJ at one point I tried to store this in a list/dict/array
            # so that it doesn't have to be calculated anew in each call but this is "the safe way"
            bigF_k_mq = (
                conj(state.chain.hf_unitary[kmq_index]).T
                @ state.chain.bigF(k[1], -q[1])
                @ state.chain.hf_unitary[k_index]
            )
            bigF_kmq_q = (
                conj(state.chain.hf_unitary[k_index]).T
                @ state.chain.bigF(k[1] - q[1], q[1])
                @ state.chain.hf_unitary[kmq_index]
            )
            # speedup if they were only computed once and stored in an array or set
            beta = -state.chain.configuration[kmq_index]
            alpha = state.chain.configuration[k_index]
            a_index = get_index(alpha)
            b_index = get_index(beta)
            for delta in [1, -1]:
                d_index = get_index(delta)
                addTo = (
                    get_wf_div(k=k_index, k_value=delta) * bigF_k_mq[b_index][d_index]
                    - get_wf_div(k=k_index, l=kmq_index, k_value=delta, l_value=beta)
                    * bigF_k_mq[minus(b_index)][d_index]
                )
                addTo *= bigF_kmq_q[a_index][b_index]
                cp_k += addTo

        h1bloc += q[2] * cp_k

    # it follows the implementation of H_1^a_loc
    """
    fJ
    H_1^a_loc always calculates to zero:
    !!! IF f_3 and f_4 are ZERO !!!
    !!! IF f_2(k,G) = 0 !!!
    every summand in the G loop is zero
    see thesis (6.15)
    """
    # h1aloc = 0.0
    # if state.chain.sumOverG:
    #     for G in state.chain.G:
    #         sub_G = 0.0
    #         ff1_sum = sum([state.chain.ff1(k[1], G[1]) for k in state.chain.k])
    #         sub_G += ff1_sum * ff1_sum
    #         sub_k = 0.0
    #         bigF_minus = [conj(state.chain.hf_unitary[int(k[0])]).T @ state.chain.bigF(k[0], -G[1]) @ state.chain.hf_unitary[int(k[0])] for k in state.chain.k]
    #         bigF_plus = [conj(state.chain.hf_unitary[int(k[0])]).T @ state.chain.bigF(k[0], G[1]) @ state.chain.hf_unitary[int(k[0])] for k in state.chain.k]
    #         for k in state.chain.k:
    #             k_index = int(k[0])
    #             alpha = state.chain.configuration[k_index]
    #             a_index = get_index(alpha)
    #             sub_k += bigF_plus[k_index][a_index][a_index] + bigF_plus[k_index][a_index][
    #                 minus(a_index)] * get_wf_div(
    #                 k=k_index)
    #             sub_ks = 0.0
    #             for ks in state.chain.k:
    #                 ks_index = int(ks[0])
    #                 for delta in [-1, 1]:
    #                     d_index = get_index(delta)
    #                     if k_index == ks_index:
    #                         wf_div_ks_delta = get_wf_div(k=int(ks[0]), k_value=delta)
    #                         for beta in [-1, 1]:
    #                             b_index = get_index(beta)
    #                             sub_ks += wf_div_ks_delta * bigF_plus[k_index][a_index][b_index] * \
    #                                       bigF_minus[ks_index][b_index][d_index]
    #                     else:
    #                         gamma = state.chain.configuration[int(ks[0])]
    #                         g_index = get_index(gamma)
    #                         for beta in [-1, 1]:
    #                             b_index = get_index(beta)
    #                             sub_ks += get_wf_div(k=k_index, l=ks_index, k_value=beta, l_value=delta) * \
    #                                       bigF_plus[k_index][a_index][b_index] * bigF_minus[ks_index][g_index][d_index]
    #             sub_G += sub_ks
    #         sub_k *= -2*ff1_sum
    #         sub_G += sub_k
    #         sub_G *= G[2]

    #         h1aloc += sub_G
    # it follows the implementation of H_0^loc
    h0loc = 0.0
    if state.chain.h != 0:
        for k in state.chain.k:
            k_index = int(k[0])
            alpha = state.chain.configuration[k_index]
            a_index = get_index(alpha)
            # a_index determines which entry on the tau matrix will be accesed
            # alpha == 1 => 0
            # alpha == -1 => 1
            tau_k = state.chain.Tau[k_index]
            h0loc += np.cos(k[1]) * (
                tau_k[a_index][a_index]
                + tau_k[a_index][minus(a_index)] * get_wf_div(k=k_index)
            )
        # h0loc *= state.chain.h

    # print(f"id {state.chain.getBinary()}, h0loc: {h0loc}, h1aloc: {0}, h1bloc: {h1bloc}, h_loc={h0loc+h1bloc+0}")
    if only_kinetic:
        return h0loc
    return h0loc + h1bloc + 0


def occupation_number(state, k_index):
    """
    :param state:
    :param k_index: an integer which indicates the site
    :return: N_k_{loc}
    fJ according to definition (2.14) in the thesis
    """

    alpha = state.chain.configuration[k_index]
    a_index = get_index(alpha)
    tau_k = state.chain.Tau[k_index]
    N_k = tau_k[a_index][a_index] + tau_k[a_index][
        minus(a_index)
    ] * state.calculate_wf_div(k_index)
    return N_k


def minus(index):
    """
    :param index: + -> [0] or - -> [1]
    :return: - if + is given and vice versa
    :raises ValueError: if unexpected value is given
    """
    if index not in [0, 1]:
        raise ValueError("expected index, no index was given")
    return int(not index)


def get_index(a):
    """
    the notation is at follows:
    ++ -> 1,1 -> [0][0]
    -- -> -1,-1 -> [1][1]
    +- -> 1,-1 -> [0][1]
    -+ -> -1,1 -> [1][0]
    :param a: corresponds to +/-
    :return: matrix index 0,1
    :raises ValueError: if unexpected value is given
    """
    if a == 1:
        return 0
    if a == -1:
        return 1
    raise ValueError("expected 'spin value' but received number that is not in [-1,1]")
