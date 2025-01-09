import torch

#  ----------------------      Utils for fermionic model -------------------------


device = torch.device(torch.get_default_device())


def V(q, n):
    return torch.tensor(1 / (1 + q * q) / (2 * n), dtype=torch.float32)


def construct_G(nbz, potential, length):
    """
    Defines the array of momentum values for potential G = np.array(i, G_{i}, V(G)) where i = 1,..., N,
    and N = self.length = number of sites or electrons.
    Main difference from this to construct_q is that G can also be in the boundaries.
    """
    G_pot = []
    G_index_blank = []
    G_units_of_2pi = []
    for i in range(
        -nbz,
        nbz + 1,
    ):
        potential_G = potential(i * 2 * torch.pi, length).clone().detach()
        index = (i * length).clone().detach()
        unitsofpi = torch.tensor(i * 2 * torch.pi).clone().detach()
        G_pot.append(potential_G)
        G_index_blank.append(index)
        G_units_of_2pi.append(unitsofpi)
    G_index_blank = torch.tensor(G_index_blank)
    G_pot = torch.tensor(G_pot).clone().detach()
    G_units_of_2pi = torch.tensor(G_units_of_2pi)
    # return np.array((G_index_blank, G_units_of_2pi, G_pot)).T
    return torch.stack((G_index_blank, G_units_of_2pi, G_pot)).transpose(0, 1)


def construct_q(nbz, potential, length):
    """
    Defines the array of momentum values for potential q = np.array(i, q_{i}, V(q)) where i = 1,..., N,
    and N = self.length = number of sites or electrons
    """
    q_pot = []
    q_index_blank = []
    for i in range(
        -nbz * length,
        nbz * length + 1,
    ):
        if i % length:  # q must not be in boundaries of BZ
            # potential_q = potential(float(i) * 2.0 * pi / float(length), float(length))
            # q_pot.append(potential_q)
            q_index_blank.append(i)
    # qtemp = np.array([float(i) * 2.0 * pi / float(length) for i in q_index_blank])
    qtemp = torch.tensor(
        [float(i) * 2.0 * torch.pi / float(length) for i in q_index_blank],
        dtype=torch.float32,
    )
    for i in range(len(qtemp)):
        potential_q = potential(qtemp[i], float(length))
        q_pot.append(potential_q)

    return torch.stack(
        (torch.tensor(q_index_blank), torch.tensor(qtemp), torch.tensor(q_pot))
    ).transpose(0, 1)


def construct_k(length):
    """
    Defines the array of momentum values k = np.array(i, k_{i}) where i = 1,..., N,
    and N = self.length = number of sites or electrons
    """
    k_index = torch.arange(length, dtype=torch.int32)
    k_value = torch.linspace(
        start=-torch.pi, end=torch.pi * (1.0 - 2.0 / float(length)), steps=length
    )
    return torch.stack(
        (
            torch.tensor(k_index, dtype=torch.uint16),
            torch.tensor(k_value, dtype=torch.float32),
        )
    ).transpose(0, 1)
