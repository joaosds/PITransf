import random
import unittest

import numpy as np

from Plotting.plot_HF import get_hf_energy
from RBM.FermionModel import FermionModel
from RBM.NeuralNetwork import NeuralNetwork
from RBM.State import State
from RBM.fermionObservables import h_loc as rbm_h_loc
from RBM.fermionHfObservablebs import h_loc as hf_h_loc
import pytest
import os

"""
fJ these are some unit tests, the most relevant being "test_ferromagnet"
fJ it ensures that (4.15) has the energy E_HF (calculating using H_loc from fermionHFObservablebs). In other words it ensures consistency between H_loc implementation 
fJ and Hartree Fock calculation
"""


class Test(unittest.TestCase):
    random.seed(1)

    def test_compare_implementations(self):
        """
        Test that old and new matrix element implementation are equal
        """
        random.seed(1)
        given_wf = lambda config: 1
        for h in [0.5]:
            with self.subTest(h=h):
                N = 6
                M = 8
                for i in range(1):
                    with self.subTest(i=i):
                        exact_configuration = [random.choice([-1, 1]) for _ in range(N)]
                        potential_function = lambda q: 1 / (q * q + 1) / (2 * N)
                        ff1 = lambda k, q: 1
                        ff2 = lambda k, q: 0.9*np.sin(q) * (np.sin(k) + np.sin(k + q))
                        ff3 = lambda k, q: 0
                        ff4 = lambda k, q: 0
                        test_chain = FermionModel(
                            potential_function=potential_function,
                            ff1=ff1,
                            ff2=ff2,
                            ff3=ff3,
                            ff4=ff4,
                            h=float(h),
                            exact_configuration=exact_configuration,
                            hf_unitary=[np.eye(2) for _ in range(N)],
                        )

                        neural_network = NeuralNetwork(
                            test_chain.length, M, complex_parameter=True
                        )

                        test_state = State(
                            neural_network=neural_network, chain=test_chain
                        )
                        self.assertAlmostEqual(
                            np.real(hf_h_loc(test_state)),
                            np.real(rbm_h_loc(test_state)),
                            delta=1e-12,
                        )

    def test_ferromagnet(self):
        def ferromagnetic_wf(fermionSpinConfiguration: FermionModel):
            """
            :param fermionSpinConfiguration:
            :return: the groundstate wavefunction if only potential, no kinetic energy is present
            """
            product1 = 1
            product2 = 1
            for k in fermionSpinConfiguration.k:
                product1 *= (
                    1 if fermionSpinConfiguration.configuration[int(k[0])] == 1 else 0
                )
                product2 *= (
                    1 if fermionSpinConfiguration.configuration[int(k[0])] == -1 else 0
                )
            return product1 + product2

        random.seed(1)
        for h in [0.5]:
            with self.subTest(h=h):
                varList = ["original0"]
                for var in varList:
                    with self.subTest(var=var):
                        print(f"it follows var{var}")
                        N = 6

                        # result_path =                            + f"/RawResults/HF_Results/{identifier}/"
                        # )
                        path = (
                            os.path.normpath(os.getcwd() + os.sep + os.pardir)
                            + f"/masterproject-develop/RawResults/newest/{var}_N={int(N)}/"
                        )
                        U_HF = np.load(f"{path}Uk_N={N}_t={h:.5e}.npy")
                        i = -1
                        print(f"all spins are {i}")
                        exact_configuration = [i for _ in range(N)]
                        if var == "original0":
                            ff1 = lambda k, q: 1
                            ff2 = (
                                lambda k, q: 1 * np.sin(q) * (np.sin(k) + np.sin(k + q))
                            )
                            ff3 = lambda k, q: 0
                            ff4 = lambda k, q: 0
                            potential_function = lambda q: 1 / (1 + q * q) * 1 / (2 * N)

                        test_chain_HF = FermionModel(
                            potential_function=potential_function,
                            ff1=ff1,
                            ff2=ff2,
                            ff3=ff3,
                            ff4=ff4,
                            h=float(h),
                            exact_configuration=exact_configuration,
                            hf_unitary=U_HF,
                            sumOverG=False,
                        )

                        state_HF = State(
                            neural_network=None,
                            chain=test_chain_HF,
                            given_wf=ferromagnetic_wf,
                        )
                        kinetic_energy = hf_h_loc(state_HF, only_kinetic=True)
                        total_energy = hf_h_loc(state_HF)
                        interaction_energy = total_energy - kinetic_energy
                        print(f"E_kin(psi^-)={kinetic_energy}")
                        print(f"E_pot(psi^-)={interaction_energy}")
                        self.assertTrue(
                            np.round(interaction_energy, 16) >= 0,
                            msg=f"Ferromagnetic E contribution = {interaction_energy} smaller zero. E_kin = {kinetic_energy}, E_tot = {total_energy}",
                        )
                        self.assertAlmostEqual(
                            total_energy,
                            get_hf_energy(h, var, N),
                            delta=1e-11,
                            msg="E neq E_HF",
                        )


if __name__ == "__main__":
    unittest.main()
