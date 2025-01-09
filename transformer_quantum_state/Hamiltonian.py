# -*- coding: utf-8 -*-
import os
import torch

from Hamiltonian_utils import construct_G, construct_q, construct_k
from Hamiltonian_utils import V


sigma_0 = torch.eye(2)
sigma_x = torch.tensor([[0, 1.0], [1.0, 0]], dtype=torch.complex64)
sigma_y = torch.tensor([[0, -1.0j], [1.0j, 0]], dtype=torch.complex64)
sigma_z = torch.tensor([[1.0, 0], [0, -1.0]], dtype=torch.complex64)



class Hamiltonian:
    def __init__(self):
        self.system_size = None
        self.n = None
        self.H = None
        self.symmetry = None
        self.n_dim = None

    def update_param(self, param):
        assert len(param) == len(self.H)
        for i, param_i in enumerate(param):  # (1, )
            self.H[i][1][:] = [param_i] * len(self.H[i][1])

    # @torch.no_grad()
    # def ElocFM2(
    #     self, samples, Uk, model, k, q, g, alfa0, alfa1, alfa2, alfa3, use_symmetry=True
    # ):
    #     # samples: (seq, batch, input_dim)
    #     symmetry = self.symmetry if use_symmetry else None
    #     E = 0
    #     E2 = 0
    #     E3 = 0
    #     Et = 0
    #     Et2 = 0
    #     Et3 = 0
    #     # params = model.param  # (n_param, )
    #     # self.update_param(params)
    #     # return results, results2, results3, resultsnl, resultsnl2, resultsnl3, norm
    #     # return results, resultsnl, norm
    #     # VECTORIZE THIS
    #     for Hi in self.H:
    #         # list of tensors, (n_op, batch)
    #         h = float(Hi[0])
    #         print(h)
    #
    #         print(samples.shape, "pacoca")
    #         # return results, results2, results3, results3loss, resultsnl, resultsnl2, resultsnl3, resultsnl3loss, norm
    #         start_time = time.perf_counter()
    #         E, E2, E3, Onl, Onl2, Onl3, norm, Occ = compute_observableFMHF(
    #             model,
    #             samples,
    #             Uk,
    #             Hi,
    #             k,
    #             q,
    #             g,
    #             alfa1,
    #             alfa2,
    #             alfa3,
    #             batch_mean=False,
    #             symmetry=symmetry,
    #         )
    #         end_time = time.perf_counter()
    #         execution_time = end_time - start_time
    #         print(f"TEST1: {execution_time} seconds")
    #         #print(torch.stack(O3).shape)
    #         # for Oj in O:
    #         #     E += Oj.sum(dim=0)
    #         # for Oj2 in O2:
    #         #     E2 += Oj2.sum(dim=0)
    #         # for Oj3 in O3:
    #         #     E3 += Oj3.sum(dim=0)
    #         #     # Stack the tensors along a new dimension
    #         #O_stacked = torch.stack(O)
    #         #O2_stacked = torch.stack(O2)
    #         #O3_stacked = torch.stack(O3)
    #
    #         # Sum along the appropriate dimensions
    #         #E = O_stacked.sum(dim=(0, 1))
    #         #E2 = O2_stacked.sum(dim=(0, 1))
    #         #E3 = O3_stacked.sum(dim=(0, 1))
    #
    #         # Compute Et, Et2, and Et3
    #         Et = Onl[0] + E
    #         Et2 = Onl2[0] + E2
    #         Et3 = Onl3[0] + E3
    #         # _, batch = psi[0].shape
    #         # print(psi[0].shape)
    #         # for Ojj in psi:
    #         #     psis_psi += np.array(Ojj).sum(dim=1)
    #         # Et = - Onl[0] + E
    #         # print("aqui", Et)
    #         Et = Onl[0] + E
    #         Et2 = Onl2[0] + E2
    #         Et3 = Onl3[0] + E3
    #         # print(Et)
    #     return (
    #         Et,
    #         Et2,
    #         Et3,
    #         h,
    #         Onl[0],
    #         Onl2[0],
    #         Onl3[0],
    #         E,
    #         E2,
    #         E3,
    #         norm,
    #         Occ[0],
    #     )

    # def full_H(self, param=None):
    #     raise NotImplementedError
    #
    # def calc_E_ground(self, param=None):
    #     if param is None:
    #         full_Hamiltonian = self.full_H()
    #     else:
    #         full_Hamiltonian = self.full_H(param)
    #     [E_ground, psi_ground] = eigsh(full_Hamiltonian, k=1, which="SA")
    #     E_ground = E_ground[0]
    #     psi_ground = psi_ground[:, 0]
    #     self.E_ground = E_ground
    #     self.psi_ground = psi_ground
    #     return E_ground


class FermionicModel(Hamiltonian):
    def __init__(self, system_size, t, periodic=True):
        super().__init__()
        self.system_size = torch.tensor(system_size).reshape(-1)
        self.n_dim = len(self.system_size)  # Dimension of the model
        self.n = self.system_size.prod()  # Number of electrons
        self.param_dim = 1  # Only one tweakable parameter, hopping t
        self.param_range = torch.tensor([[0.05], [1.5]])

        self.t = t  # Hopping parameter
        self.n_dim = len(self.system_size)
        self.H = [
            ([self.t]),
        ]

        self.potential = V
        self.nbz = 1
        self.sumOverG = False

        self.G = construct_G(self.nbz, self.potential, self.n)
        self.q = construct_q(self.nbz, self.potential, self.n)
        self.k = construct_k(self.n)

    def update_param(self, param):
        # param: (1, )
        self.H[0][1][0] = param  # First element of self.t will be updated


if __name__ == "__main__":
    try:
        os.mkdir("results/")
    except FileExistsError:
        pass
