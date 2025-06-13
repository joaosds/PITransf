import os
import time
import torch
import torch.nn as nn
import random

from model_utils import treesampler, normalsampler, compute_grad, compute_psi
from model_utils import compute_observableFMHF
from soap import SOAP

torch.set_printoptions(precision=5)

# Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda:2")
    torch.set_default_device("cuda:2")
    torch.set_default_dtype(torch.float32)
else:
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float32)
    device = "cpu"


class Optimizer:
    """
    Main optimizer class for quantum state learning
    Combines neural network optimization with variational Monte Carlo
    """
    
    def __init__(self, model, Hamiltonians):
        """
        Initialize optimizer
        
        Args:
            model: Transformer neural network model
            Hamiltonians: List of Hamiltonian objects
        """
        self.model = model
        self.Hamiltonians = Hamiltonians
        self.model.param_range = Hamiltonians[0].param_range
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        
        # Neural network optimizer - using SOAP (Second-Order Approximation)
        self.optim = SOAP(
            self.model.parameters(),
            lr=10,
            betas=(0.95, 0.95),
            weight_decay=0.01,
            precondition_frequency=50,
        )
        
        # Alternative Adam optimizer (commented out)
        # self.optim = torch.optim.Adam(
        #     self.model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9
        # )
        
        # Saving frequencies
        self.save_freq = 100
        self.ckpt_freq = 1000

    @staticmethod
    def lr_schedule(step, model_size, factor=1, warmup=2000, start_step=0):
        """
        Learning rate schedule from "Attention is All You Need"
        """
        step = step + start_step
        if step < 1:
            step = 1
        return factor * (
            model_size ** (-0.5) * min(step ** (-0.75), step * warmup ** (-1.75))
        )

    @staticmethod
    def lr_schedule2(step, model_size, factor=10000, warmup=2000, start_step=0):
        """
        Alternative learning rate schedule for alpha parameter
        """
        step = step + start_step
        if step < 1:
            step = 1
        return factor * (
            model_size ** (-0.5) * min(step ** (-0.75), step * warmup ** (-1.75))
        )

    def get_unique_states_with_weights(self, tensor):
        """
        Process tensor of binary states to get unique states and their weights
        
        Args:
            tensor: torch.Tensor of shape (n_bits, n_samples) containing binary states
            
        Returns:
            unique_states: torch.Tensor of unique states
            sample_weights: torch.Tensor of normalized occurrence counts
        """
        # Convert (n_bits, n_samples) to (n_samples, n_bits)
        states = tensor.t()
        
        # Convert binary states to integers for unique identification
        powers = torch.pow(2, torch.arange(states.size(1) - 1, -1, -1, device=states.device))
        state_ints = (states * powers).sum(dim=1)
        
        # Get unique states and their counts
        unique_ints, inverse_indices, counts = torch.unique(
            state_ints, return_inverse=True, return_counts=True
        )
        
        # Calculate normalized weights
        sample_weights = counts.float() / len(state_ints)
        
        # Convert unique integers back to binary representation
        unique_states = torch.zeros(
            (len(unique_ints), states.size(1)),
            dtype=torch.long,
            device=states.device,
        )
        for i in range(states.size(1)):
            unique_states[:, i] = (unique_ints >> (states.size(1) - 1 - i)) & 1
            
        return unique_states, sample_weights

    def minimize_energy_step2(self, H, t, Uk, Ehf, batch, max_unique, i, 
                             alfa0, alfa1, alfa2, alfa3, sec_batch, sampler):
        """
        Perform one step of energy minimization
        
        Args:
            H: Hamiltonian object
            t: Hopping parameter
            Uk: Basis transformation matrix
            Ehf: Hartree-Fock energy
            batch: Batch size for sampling
            max_unique: Maximum number of unique samples
            i: Current iteration
            alfa0, alfa1, alfa2, alfa3: Mixing parameters
            sec_batch: Number of batch sections for memory management
            sampler: Sampling method ("tree" or "normal")
            
        Returns:
            Tuple containing energies, gradients, and other observables
        """
        ind = i

        # === SAMPLING PHASE ===
        start_time = time.perf_counter()

        if sampler == "tree":
            samples, sample_weight, idxns = treesampler(self.model, batch, max_unique)
        elif sampler == "normal":
            samples = normalsampler(self.model, batch)
            unique_states, sample_weight = self.get_unique_states_with_weights(samples)
            samples = unique_states.t()

        end_time = time.perf_counter()
        execution_time = end_time - start_time
        if ind % 100 == 0:
            print(f"Execution time for sampling: {execution_time} seconds")

        # === OBSERVABLE COMPUTATION ===
        start_time = time.perf_counter()

        n, batch = samples.shape
        
        # Initialize energy components
        E = torch.zeros(batch, dtype=torch.complex64)
        E2 = torch.zeros(batch, dtype=torch.complex64)
        E3 = torch.zeros(batch, dtype=torch.complex64)
        Eloc = torch.zeros(batch, dtype=torch.complex64)
        Eloc2 = torch.zeros(batch, dtype=torch.complex64)
        Eloc3 = torch.zeros(batch, dtype=torch.complex64)
        Einl = torch.zeros(batch, dtype=torch.complex64)
        Einl2 = torch.zeros(batch, dtype=torch.complex64)
        Einl3 = torch.zeros(batch, dtype=torch.complex64)
        occupation = torch.zeros(n, batch, 3)

        # Process samples in sections to manage memory
        sec_batch0 = max(1, int(batch % sec_batch))
        sec_batch = sec_batch0
        batch_per_section = batch // sec_batch

        start_time = time.perf_counter()

        for i in range(sec_batch):
            start_batch = i * batch_per_section
            end_batch = (i + 1) * batch_per_section if i < sec_batch - 1 else batch
            current_samples = samples[:, start_batch:end_batch]

            # Compute observables for current batch section
            (E_batch, E2_batch, E3_batch, Onl_batch, Onl2_batch, 
             Onl3_batch, Occ_batch) = compute_observableFMHF(
                self.model, current_samples, Uk, t, H.k, H.q, alfa2, alfa3
            )

            # Store results
            E[start_batch:end_batch] = E_batch + Onl_batch[0]
            E2[start_batch:end_batch] = E2_batch + Onl2_batch[0]
            E3[start_batch:end_batch] = E3_batch + Onl3_batch[0]
            Eloc[start_batch:end_batch] = E_batch
            Eloc2[start_batch:end_batch] = E2_batch
            Eloc3[start_batch:end_batch] = E3_batch
            Einl[start_batch:end_batch] = Onl_batch[0]
            Einl2[start_batch:end_batch] = Onl2_batch[0]
            Einl3[start_batch:end_batch] = Onl3_batch[0]
            occupation[:, start_batch:end_batch, :] = Occ_batch[0]

        end_time = time.perf_counter()
        execution_time = end_time - start_time
        #if ind % 100 == 0:
        #    print(f"TEST1: {execution_time} seconds")

        # === NORMALIZATION AND ENERGY CALCULATION ===
        start_time = time.perf_counter()

        # defining normalization
        log_amp, _, emb = compute_psi(self.model, samples, check_duplicate=True)
        normt = log_amp.clone().detach().exp().sum()
        norm = alfa1**2 + (1 - alfa1**2) * normt

        # Attributing the weights
        def e_mean(Etemp):
            return (Etemp * sample_weight / norm).sum()

        def e_var(Etemp, Etempmean):
            # Variance for E/N
            var1 = (
                (((Etemp.real - Etempmean.real) ** 2 * sample_weight).sum() / (H.n).real)
            ) / (sample_weight.sum())
            return var1

        # Apply mixing parameters
        Einl2 = Einl2 * alfa2
        Einl3 = Einl3 * alfa3
        Eloc2 = Eloc2 * alfa2
        Eloc3 = Eloc3 * alfa3

        Einl = Einl2 + Einl3
        Eloc = Eloc2 + Eloc3
        Eloc0 = Eloc2 + 2 * Eloc3.real
        Einl0 = Einl2 + 2 * Einl3.real
        Et = Einl + Eloc  # Total energy
        normsize = H.n * norm
        Et0 = Einl2 + Eloc2 + 2 * (Einl3 + Eloc3).real
        Ehfsp = 2 * e_mean(E3).real / H.n + 0j
        Essp = e_mean(E2).real / H.n  # take only real part since imaginary should be zero

        E2 = alfa2 * E2
        E3 = alfa3 * E3
        E = E2 + 2 * E3.real

        # Calculate mean energies
        E_mean0 = e_mean(Et0)
        E_mean_inl = e_mean(Einl0)
        E_mean_loc = e_mean(Eloc0)
        E_var = e_var(E, E_mean0)
        E_mean = E_mean0 + Ehf * H.n * alfa1**2 / norm
        
        # Real and imaginary parts
        Er = E_mean.real / H.n
        Ei = E_mean.imag / H.n
        Er_inl = E_mean_inl.real / H.n
        Ei_inl = E_mean_inl.imag / H.n
        Er_loc = E_mean_loc.real / H.n
        Ei_loc = E_mean_loc.imag / H.n

        # ----------------------- 2nd group
        E_mean2t = e_mean(E2)
        E_mean_inl2 = e_mean(Einl2)
        E_mean_loc2 = e_mean(Eloc2)
        E_var2 = e_var(E2, E_mean2t)
        E_mean2 = E_mean2t
        Er2 = E_mean2.real / H.n
        Ei2 = E_mean2.imag / H.n
        Er_inl2 = E_mean_inl2.real / H.n
        Ei_inl2 = E_mean_inl2.imag / H.n
        Er_loc2 = E_mean_loc2.real / H.n
        Ei_loc2 = E_mean_loc2.imag / H.n

        # ----------------------- 3rd group
        # The 2 factor reflects that we calculate only scatterings from Nhf to HF
        E_mean3t = 2 * e_mean(E3).real
        E_mean_inl3 = 2 * e_mean(Einl3).real + 0j
        E_mean_loc3 = 2 * e_mean(Eloc3).real + 0j
        E_var3 = e_var(2 * E3, E_mean3t)
        E_mean3 = E_mean3t + 0j  # 0j is necessary since E3 is real now
        Er3 = E_mean3.real / H.n
        Ei3 = E_mean3.imag / H.n
        Er_inl3 = E_mean_inl3.real / H.n
        Ei_inl3 = E_mean_inl3.imag / H.n
        Er_loc3 = E_mean_loc3.real / H.n
        Ei_loc3 = E_mean_loc3.imag / H.n

        # === GRADIENT COMPUTATION ===
        loss, log_amp, log_phase = compute_grad(
            self.model, samples, sample_weight, Et/(H.n), 0, idxns
        )

        # Alpha parameter gradient calculation
        def grad3(Ehf, Essp, Ehfsp, a0):
            """Calculate gradient with respect to alpha parameter"""
            def sech(x):
                return 1 / torch.cosh(x)
            
            def tanh(x):
                return torch.tanh(x)

            # Calculate common terms
            sech_a0_squared = sech(a0) ** 2
            tanh_term = 1 + tanh(a0)
            sqrt_term = torch.sqrt(1 - 0.25 * tanh_term**2)

            # Calculate each part of the expression
            part1 = 0.5 * Ehf * sech_a0_squared * tanh_term
            part2 = -0.5 * Essp * sech_a0_squared * tanh_term
            part3 = -(Ehfsp * sech_a0_squared * tanh_term**2) / (8 * sqrt_term)
            part4 = 0.5 * Ehfsp * sech_a0_squared * sqrt_term

            return part1 + part2 + part3 + part4

        grad2 = grad3(Ehf, Essp, Ehfsp, alfa0)

        # Update alpha parameter
        lambd = 100  # Learning rate for alpha
        alfanew = alfa0 - lambd * grad2
        alfa0 = alfanew.clone().detach()
        alfa1 = (1 + torch.tanh(alfanew)) / 2
        alfa2 = 1.0 - alfa1**2
        alfa3 = alfa1 * torch.sqrt(alfa2)

        # Calculate variance scores
        rand_ind = int(random.uniform(0, batch))
        einfty = 1
        vscore = H.n * E_var / (((E_mean.real) / H.n) ** 2)
        vscore2 = H.n * E_var * (H.n**2) / (((e_mean(Et0)) / H.n) ** 2)
        
        # Weight occupation numbers
        for i in range(len(sample_weight)):
            occupation[:, i, :] = occupation[:, i, :] * sample_weight[i]

        # Logging
        if ind % 100 == 0:
            #print("current nu", batch_per_section)
            #print("Local estimators are divided in ", sec_batch, "batches")
            print(
                f"alfa1={alfa1}, Ehf/norm={Ehf * (alfa1) ** 2 / norm}, norm2={norm}, "
                f"lamb={lambd}, batch={batch}"
            )
            #print(f"E_var={E_var}, vscore={vscore}, {vscore2}, {einfty/H.n}")
            print()

            end_time = time.perf_counter()
            execution_time = end_time - start_time
            print(f"Execution time for REST: {execution_time} seconds")

        return (
            loss, log_amp, log_phase, sample_weight, Er, Ei, E_var,
            Er_inl, Ei_inl, Er_loc, Ei_loc, Er2, Ei2, E_var2,
            Er_inl2, Ei_inl2, Er_loc2, Ei_loc2, Er3, Ei3, E_var3,
            Er_inl3, Ei_inl3, Er_loc3, Ei_loc3, t, samples,
            1 / norm, grad2, alfa0, alfa1, alfa2, alfa3,
            vscore, vscore2, occupation, emb, E
        )

    def train(self, n0, n_iter, t, Uk, Ehf, alfa0, alfa1, alfa2, alfa3,
              sec_batch, device, label, sampler, batch=10000, max_unique=1000,
              param_range=None, fine_tuning=False, use_SR=True, ensemble_id=0,
              start_iter=None):
        """
        Main training loop for quantum state learning
        
        Args:
            n0: System size
            n_iter: Number of training iterations
            t: Hopping parameter
            Uk: Basis transformation matrix
            Ehf: Hartree-Fock energy
            alfa0, alfa1, alfa2, alfa3: Mixing parameters
            sec_batch: Number of batch sections for memory management
            device: Computing device
            label: Label for saving files
            sampler: Sampling method ("tree" or "normal")
            batch: Batch size for sampling
            max_unique: Maximum number of unique samples
            param_range: Parameter range for the model
            fine_tuning: Whether this is fine-tuning
            use_SR: Whether to use stochastic reconfiguration (not used currently)
            ensemble_id: Ensemble identifier
            start_iter: Starting iteration
            
        Returns:
            Tuple containing training curves and final results
        """
        # Extract model information
        name, embedding_size, n_head, n_layers = (
            type(self.Hamiltonians[0]).__name__,
            self.model.embedding_size,
            self.model.n_head,
            self.model.n_layers,
        )
        
        if start_iter is None:
            start_iter = 0 if not fine_tuning else 100000
            
        system_sizes = self.model.system_sizes
        n_iter += 1  # Adjust for main loop
        
        if param_range is None:
            param_range = self.Hamiltonians[0].param_range
        self.model.param_range = param_range
        
        # Create filename string for saving
        save_str = (
            f"{name}_{embedding_size}_{n_head}_{n_layers}_{ensemble_id}_{t}_{n0}"
            if not fine_tuning
            else f"ft_{self.model.system_sizes[0].clone().detach().item()}_"
            f"{param_range[0].clone().detach().item():.2f}_"
            f"{name}_{embedding_size}_{n_head}_{n_layers}_{ensemble_id}_{t}_{n0}"
        )

        self.device = device

        # Create results directory outside the transformer_quantum_state folder
        results_dir = os.path.join("..", "results")
        os.makedirs(results_dir, exist_ok=True)

        # Setup optimizer and scheduler
        optim = self.optim
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optim,
            lambda step: self.lr_schedule(
                step, self.model.embedding_size, start_step=start_iter
            ),
        )

        # Initialize arrays to save curves
        E_curve = torch.zeros(n_iter)
        E_curve2 = torch.zeros(n_iter)
        E_curve3 = torch.zeros(n_iter)
        norm_curve = torch.zeros(n_iter)
        alfa1_curve = torch.zeros(n_iter)
        gradalfa1_curve = torch.zeros(n_iter)
        alfa2_curve = torch.zeros(n_iter)
        alfa3_curve = torch.zeros(n_iter)
        E_curve_i = torch.zeros(n_iter)
        E_curve_i2 = torch.zeros(n_iter)
        E_curve_i3 = torch.zeros(n_iter)
        E_vars = torch.zeros(n_iter)
        E_vars2 = torch.zeros(n_iter)
        E_vars3 = torch.zeros(n_iter)
        vscore_curve = torch.zeros(n_iter)
        vscore_curve2 = torch.zeros(n_iter)
        loss_curve = torch.zeros(n_iter)

        # === MAIN TRAINING LOOP ===
        for i in range(start_iter, start_iter + n_iter):
            start = time.time()
            self.model.set_param()
            size_idx = self.model.size_idx
            n = self.model.system_size.prod()
            H = self.Hamiltonians[size_idx]

            ind = i
            (
                loss, log_amp, log_phase, sample_weight,
                Er, Ei, E_var, Er_inl, Ei_inl, Er_loc, Ei_loc,
                Er2, Ei2, E_var2, Er_inl2, Ei_inl2, Er_loc2, Ei_loc2,
                Er3, Ei3, E_var3, Er_inl3, Ei_inl3, Er_loc3, Ei_loc3,
                t, samples, norm, grad, alfa0, alfa1, alfa2, alfa3,
                vscore, vscore2, occupation, emb, E,
            ) = self.minimize_energy_step2(
                H, t, Uk, Ehf, batch, max_unique, ind,
                alfa0, alfa1, alfa2, alfa3, sec_batch, sampler,
            )

            psi = (log_amp + 1j * log_phase).exp()
            t1 = time.time()

            # Optimization step
            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()
            t2 = time.time()

            # Logging
            if i % 100 == 0:
                print_str = f"E_real = {Er:.10f}\t E_imag = {Ei:.10f}\t E_var = {E_var:.10f}\t"
                print_strt2 = f"E_real2 = {Er2:.10f}\t E_imag2 = {Ei2:.10f}\t E_var2 = {E_var2:.10f}\t"
                print_strt3 = f"E_real3 = {Er3:.10f}\t E_imag3 = {Ei3:.10f}\t E_var3 = {E_var3:.10f}\t"
                print_str2 = f"E_r_inl= {Er_inl:.10f}\t Ei_inl = {Ei_inl:.10f}\t Er_loc = {Er_loc:.10f}\t Ei_loc = {Ei_loc:.10f}\t"
                
                print(
                    f"i = {i}\t {print_str} n = {n}\t lr = {scheduler.get_last_lr()[0]:.4e} "
                    f"t = {(t1-start):.6f}  t_optim = {t2-t1:.6f}"
                )
                print(f"i = {i}\t h={t:.4f}\t {print_str2} n = {n}\t")
                print(f"i = {i}\t h={t:.4f}\t {print_strt2} n = {n}\t")
                print(f"i = {i}\t h={t:.4f}\t {print_strt3} n = {n}\t")

            # Store results
            E_curve[i - start_iter] = Er
            loss_curve[i - start_iter] = loss
            E_curve2[i - start_iter] = Er2
            E_curve3[i - start_iter] = Er3
            E_curve_i[i - start_iter] = Ei
            E_curve_i2[i - start_iter] = Ei2
            E_curve_i3[i - start_iter] = Ei3
            E_vars[i - start_iter] = E_var
            E_vars2[i - start_iter] = E_var2
            E_vars3[i - start_iter] = E_var3
            norm_curve[i - start_iter] = norm
            alfa1_curve[i - start_iter] = alfa1**2
            gradalfa1_curve[i - start_iter] = grad
            alfa2_curve[i - start_iter] = alfa2
            alfa3_curve[i - start_iter] = alfa3
            vscore_curve[i - start_iter] = vscore
            vscore_curve2[i - start_iter] = vscore2

            # Save results periodically
            if i % self.save_freq == 0:
                results_dir = os.path.join("..", "results")
                with open(os.path.join(results_dir, f"E_{save_str}_{label}.pt"), "wb") as f:
                    torch.save(
                        {
                            "E_curve": E_curve,
                            "E_curve2": E_curve2,
                            "E_curve3": E_curve3,
                            "E_curve_i": E_curve_i,
                            "E_curve_i2": E_curve_i2,
                            "E_curve_i3": E_curve_i3,
                        },
                        os.path.join(results_dir, f"E_{save_str}_{label}.pt"),
                    )
                with open(os.path.join(results_dir, f"E_var_{save_str}_{label}.pt"), "wb") as f:
                    torch.save(
                        {"E_vars": E_vars, "E_vars2": E_vars2, "E_vars3": E_vars3},
                        os.path.join(results_dir, f"E_var_{save_str}_{label}.pt"),
                    )
                with open(os.path.join(results_dir, f"alpha_{save_str}_{label}.pt"), "wb") as f:
                    torch.save(
                        {
                            "alfa1_curve": alfa1_curve,
                            "alfa2_curve": alfa2_curve,
                            "alfa3_curve": alfa3_curve,
                            "norm": norm_curve,
                            "vscore": vscore_curve,
                            "vscore2": vscore_curve2,
                        },
                        os.path.join(results_dir, f"alpha_{save_str}_{label}.pt"),
                    )
                torch.save(self.model.state_dict(), os.path.join(results_dir, f"model_{save_str}_{label}.ckpt"))
                
                if i % self.ckpt_freq == 0:
                    torch.save(
                        self.model.state_dict(), os.path.join(results_dir, f"ckpt_{i}_{save_str}.ckpt")
                    )

        return (
            loss_curve, E_curve, E_curve_i, E_vars,
            E_curve2, E_curve_i2, E_vars2,
            E_curve3, E_curve_i3, E_vars3,
            abs(psi) ** 2, sample_weight, samples,
            norm_curve, gradalfa1_curve, alfa1_curve, alfa2_curve, alfa3_curve,
            vscore_curve, vscore_curve2, occupation, emb,
        )
