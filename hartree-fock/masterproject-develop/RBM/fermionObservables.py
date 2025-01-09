# /usr/bin/env python3 -*- coding: utf-8 -*-
"""
Adapted by João Sobral from main author Michael Perle.
Brief Description: Calculates observables for fermionic model.
"""

import numpy as np
from RBM.State import State


def h_loc(state: State, zero_interaction=False) -> float:
    """
    Calculates H_loc(s) for given state.

    Parameters
    ----------
    zero_interaction : bolean
        True, if the interaction term is zero and thus can be neglected.
    state : float
        State for which energy will be calculated

    Returns
    -------
    float
        Energy of state.

    Notes
    -----
    fJ these are the matrix elements of H in (2.18), i.e. d-basis.
    # Numerically "proven" to be equal to fermionHFObservablebs if U = np.eye(2) forall k in BZ
    fJ could be labeled as deprecated but probably performance is
    better using "pure" d-basis matrix elements instead of generalized matrix elements in fermionHFObservablebs with U = np.eye(2) forall k in BZ

    Examples
    --------

    """
    """

    """

    energy_kinetic = 0.0

    if state.chain.h == 0:
        energy_kinetic = 0
    else:
        for k in state.chain.k:
            energy_kinetic += np.cos(k[1]) * state.calculate_wf_div(int(k[0]))
            p = state.chain.configuration[int(k[0])]
            c = state.chain.configuration[:]
        energy_kinetic *= state.chain.h

    if zero_interaction:
        return energy_kinetic

    correlation_part = 0.0

    for k in state.chain.k:
        sub_sum_q = 0.0
        # von jedem k ausgehend den qten nachbar außer sich selbst --> q ungl RL
        # starting from each k, the qth neighbor other than itself --> q ungl RL
        for q in state.chain.q:
            potential = q[2]
            kmq_index = state.chain.pbc(int(k[0]) - int(q[0]))
            if (
                state.chain.configuration[int(k[0])]
                == state.chain.configuration[kmq_index]
            ):
                contribution = 0  # If the configuration is the same, we don't need to calculate the energy
                # Test to improve this like calculating either way
                p = state.chain.configuration[int(k[0])]
                b = state.chain.configuration[kmq_index]
                c = state.chain.configuration[:]
            else:
                p = state.chain.configuration[int(k[0])]
                b = state.chain.configuration[kmq_index]
                c = state.chain.configuration[:]
                wf_div = state.calculate_wf_div(
                    int(k[0]), kmq_index
                )  # unglaublich ineffizient

                ff1_square = state.chain.ff1(k[1], -q[1]) * state.chain.ff1(k[1], -q[1])
                ff2_square = state.chain.ff2(k[1], -q[1]) * state.chain.ff2(k[1], -q[1])
                contribution = (
                    (1 - wf_div) * ff1_square + (1 + wf_div) * ff2_square
                ) + complex(
                    0,
                    2
                    * p
                    * state.chain.ff1(k[1], -q[1])
                    * state.chain.ff2(k[1], -q[1])
                    * wf_div,
                )

            sub_sum_q += potential * contribution

        correlation_part += sub_sum_q

    # h1aloc is also proven to be zero
    h1aloc = 0.0
    for G in state.chain.G:
        for k in state.chain.k:
            for ks in state.chain.k:
                p = state.chain.configuration[int(k[0])]
                ps = state.chain.configuration[int(ks[0])]
                h1aloc += complex(
                    p * ps * state.chain.ff2(k[1], G[1]) * state.chain.ff2(ks[1], G[1]),
                    p * state.chain.ff2(k[1], G[1]) * state.chain.ff1(ks[1], G[1])
                    - ps * state.chain.ff2(ks[1], G[1]) * state.chain.ff1(k[1], G[1]),
                )

    return correlation_part + energy_kinetic


def occupation_number(state, k_index):
    occupation_number = state.calculate_wf_div(k_index)
    return occupation_number
