import os
import torch

# Get default device for tensor operations
device = torch.device(torch.get_default_device())

# Pauli matrices for spin operations
sigma_0 = torch.eye(2)
sigma_x = torch.tensor([[0, 1.0], [1.0, 0]], dtype=torch.complex64)
sigma_y = torch.tensor([[0, -1.0j], [1.0j, 0]], dtype=torch.complex64)
sigma_z = torch.tensor([[1.0, 0], [0, -1.0]], dtype=torch.complex64)


# ============================= UTILITY FUNCTIONS =============================

def V(q, n):
    """
    Interaction potential in momentum space
    
    Args:
        q: momentum transfer
        n: system size
        
    Returns:
        Interaction strength V(q)
    """
    return torch.tensor(1 / (1 + q * q) / (2 * n), dtype=torch.float32)


def construct_G(nbz, potential, length):
    """
    Construct momentum grid for potential interactions including boundaries
    
    Args:
        nbz: Number of Brillouin zones
        potential: Potential function V(q, n)
        length: System size (number of sites/electrons)
        
    Returns:
        torch.Tensor: Array of shape (N_G, 3) containing [index, momentum, V(momentum)]
    """
    G_pot = []
    G_index_blank = []
    G_units_of_2pi = []
    
    for i in range(-nbz, nbz + 1):
        potential_G = potential(i * 2 * torch.pi, length).clone().detach()
        index = (i * length).clone().detach()
        unitsofpi = torch.tensor(i * 2 * torch.pi).clone().detach()
        
        G_pot.append(potential_G)
        G_index_blank.append(index)
        G_units_of_2pi.append(unitsofpi)
    
    G_index_blank = torch.tensor(G_index_blank)
    G_pot = torch.tensor(G_pot).clone().detach()
    G_units_of_2pi = torch.tensor(G_units_of_2pi)
    
    return torch.stack((G_index_blank, G_units_of_2pi, G_pot)).transpose(0, 1)


def construct_q(nbz, potential, length):
    """
    Construct momentum grid for potential interactions excluding boundaries
    
    Args:
        nbz: Number of Brillouin zones
        potential: Potential function V(q, n)
        length: System size (number of sites/electrons)
        
    Returns:
        torch.Tensor: Array of shape (N_q, 3) containing [index, momentum, V(momentum)]
    """
    q_pot = []
    q_index_blank = []
    
    for i in range(-nbz * length, nbz * length + 1):
        if i % length:  # q must not be in boundaries of BZ
            q_index_blank.append(i)
    
    # Convert indices to momentum values
    qtemp = torch.tensor(
        [float(i) * 2.0 * torch.pi / float(length) for i in q_index_blank],
        dtype=torch.float32,
    )
    
    # Calculate potential for each momentum
    for i in range(len(qtemp)):
        potential_q = potential(qtemp[i], float(length))
        q_pot.append(potential_q)

    return torch.stack(
        (torch.tensor(q_index_blank), torch.tensor(qtemp), torch.tensor(q_pot))
    ).transpose(0, 1)


def construct_k(length):
    """
    Construct momentum grid for kinetic energy terms
    
    Args:
        length: System size (number of sites/electrons)
        
    Returns:
        torch.Tensor: Array of shape (length, 2) containing [index, momentum]
    """
    k_index = torch.arange(length, dtype=torch.int32)
    k_value = torch.linspace(
        start=-torch.pi, 
        end=torch.pi * (1.0 - 2.0 / float(length)), 
        steps=length
    )
    
    
    return torch.stack(
        (
            torch.tensor(k_index, dtype=torch.uint16),
            torch.tensor(k_value, dtype=torch.float32),
        )
    ).transpose(0, 1)


# ============================= HAMILTONIAN CLASSES =============================

class Hamiltonian:
    """
    Base class for quantum many-body Hamiltonians
    
    Each child class should specify:
    - self.n: system size
    - self.H: list of tuples describing Hamiltonian terms
    """
    
    def __init__(self):
        """
        Initialize base Hamiltonian class
        
        Attributes:
            system_size: Physical dimensions of the system
            n: Total number of degrees of freedom
            H: List of Hamiltonian terms
            symmetry: Symmetries of the system
            n_dim: Spatial dimensionality
        """
        self.system_size = None
        self.n = None
        self.H = None
        self.symmetry = None
        self.n_dim = None

    def update_param(self, param):
        """
        Update coefficients in the Hamiltonian
        
        Args:
            param: torch.Tensor of shape (n_param,) containing new parameters
            
        Note:
            Default implementation assumes coefficients are (n_op,) same in every group.
            Override this method for specific Hamiltonian forms.
        """
        assert len(param) == len(self.H)
        for i, param_i in enumerate(param):
            self.H[i][1][:] = [param_i] * len(self.H[i][1])


class FermionicModel(Hamiltonian):
    def __init__(self, system_size, t, periodic=True):
        """
        Initialize fermionic model
        
        Args:
            system_size: System dimensions (list or tensor)
            t: Hopping parameter
            periodic: Whether to use periodic boundary conditions
        """
        super().__init__()
        
        # System configuration
        self.system_size = torch.tensor(system_size).reshape(-1)
        self.n_dim = len(self.system_size)  # Spatial dimension
        self.n = self.system_size.prod()    # Total number of sites/electrons
        
        # Model parameters
        self.param_dim = 1  # Only hopping parameter t is tunable
        self.param_range = torch.tensor([[0.05], [1.5]])  # Valid range for t
        
        # Physical parameters
        self.t = t  # Hopping parameter
        self.H = [([self.t])]  # Hamiltonian terms
        
        # Interaction setup
        self.potential = V  # Interaction potential function
        self.nbz = 1        # Number of Brillouin zones
        self.sumOverG = False
        
        # Construct momentum space grids
        self.G = construct_G(self.nbz, self.potential, self.n)  # With boundaries
        self.q = construct_q(self.nbz, self.potential, self.n)  # Without boundaries  
        self.k = construct_k(self.n)                            # Kinetic momenta
        
    def update_param(self, param):
        """
        Update hopping parameter
        
        Args:
            param: New hopping parameter value
        """
        self.H[0][1][0] = param
        self.t = param

    def get_info(self):
        """
        Get model information for logging
        
        Returns:
            dict: Model parameters and configuration
        """
        return {
            'model_type': 'FermionicModel',
            'system_size': self.system_size.tolist(),
            'n_sites': self.n.item(),
            'hopping_t': self.t,
            'n_momentum_G': len(self.G),
            'n_momentum_q': len(self.q),
            'n_momentum_k': len(self.k),
            'param_range': self.param_range.tolist()
        }

