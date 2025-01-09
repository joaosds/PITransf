import random
from unittest import TestCase

import numpy as np

from RBM.FermionModel import FermionModel


"""
fj Contains tests that are hopefully not relevant anymore since I tried a lot to ensure correctness of the simulation

"""



def find_which_FHF_entries_are_equal():

    ff2name = "ff2 eq 0"
    N = 12
    h = 3
    ff2 = 0, lambda k, q: np.sin(q) * (np.sin(k) + np.sin(k + q))
    U_HF = [np.eye(2) for _ in range(N)]
    exact_configuration = [random.choice([-1, 1]) for _ in range(N)]

    chain_HF = FermionModel(potential_function=lambda q: 1 / (q * q + 1),
                            ff1=lambda k, q: 1,
                            ff2=ff2,
                            h=float(h),
                            exact_configuration=exact_configuration,
                            hf_unitary=U_HF)
    index_list = []
    for k in range(len(chain_HF.FHF)):
        for q in range(len(chain_HF.FHF[k])):
            for ks in range(len(chain_HF.FHF)):
                for qs in range(len(chain_HF.FHF[ks])):
                    norm = np.linalg.norm(chain_HF.FHF[k][q] - np.conjugate(chain_HF.FHF[ks][qs]))
                    if norm < 1e-5:
                        index_list.append([k,q,ks,qs])
    return index_list

class TestFermionSpinConfiguration(TestCase):
    def test_big_f(self):
        self.fail()


    def test_get_fhf_correct_implementation(self):
        ff2name = "ff2 eq 0"
        N = 12
        h=3
        for ff2 in [lambda k, q: 0, lambda k, q: np.sin(q) * (np.sin(k) + np.sin(k + q))]:
            with self.subTest(ff2=ff2name):
                U_HF = [np.eye(2) for _ in range(N)]
                U_RBM = None
                exact_configuration = [random.choice([-1, 1]) for _ in range(N)]

                test_chain_HF = FermionModel(potential_function=lambda q: 1 / (q * q + 1),
                                             ff1=lambda k, q: 1,
                                             ff2=ff2,
                                             h=float(h),
                                             exact_configuration=exact_configuration,
                                             hf_unitary=U_HF)

                test_chain_RBM = FermionModel(potential_function=lambda q: 1 / (q * q + 1),
                                              ff1=lambda k, q: 1,
                                              ff2=ff2,
                                              h=float(h),
                                              exact_configuration=exact_configuration,
                                              hf_unitary=U_RBM)
                for k_index in range(N):
                    with self.subTest(k_index=k_index):
                        for q in test_chain_HF.q:
                            with self.subTest(q_index = q[0]):
                                np.testing.assert_array_equal(test_chain_HF.get_FHF(test_chain_HF.pbc(k_index-q[0]), q[0]), np.conjugate(test_chain_HF.get_FHF(k_index, -q[0])))

    def test_get_fhf_conjugation(self):
        ff2name = "ff2 neq 0"
        N = 10
        h=3
        for ff2 in [lambda k, q: np.sin(q) * (np.sin(k) + np.sin(k + q)), lambda k, q: 0]:
            with self.subTest(ff2=ff2name):
                U_HF = [np.eye(2) for _ in range(N)]
                U_RBM = None
                exact_configuration = [random.choice([-1, 1]) for _ in range(N)]

                test_chain_HF = FermionModel(potential_function=lambda q: 1 / (q * q + 1),
                                             ff1=lambda k, q: 1,
                                             ff2=ff2,
                                             h=float(h),
                                             exact_configuration=exact_configuration,
                                             hf_unitary=U_HF)

                test_chain_RBM = FermionModel(potential_function=lambda q: 1 / (q * q + 1),
                                              ff1=lambda k, q: 1,
                                              ff2=ff2,
                                              h=float(h),
                                              exact_configuration=exact_configuration,
                                              hf_unitary=U_RBM)
                for k_index in range(N):
                    with self.subTest(k_index=k_index):
                        for q in test_chain_HF.q:
                            with self.subTest(q_index = q[0]):
                                np.testing.assert_array_equal(test_chain_HF.get_FHF(test_chain_HF.pbc(k_index-q[0]), q[0]), np.conjugate(test_chain_HF.get_FHF(k_index, -q[0])))
            N = 12
            ff2name = "ff2 eq 0"

