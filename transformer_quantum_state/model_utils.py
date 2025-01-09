import torch
from torch.distributions.binomial import Binomial
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt


if torch.cuda.is_available():
    device = torch.device("cuda:3")  ## specify the GPU id's, GPU id's start from 0.
    torch.set_default_device("cuda:3")
    torch.set_default_dtype(torch.float32)
else:
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cpu")
    device = "cpu"

sigma_0 = torch.eye(2, dtype=torch.complex64)
sigma_x = torch.tensor([[0, 1.0], [1.0, 0]], dtype=torch.complex64)
sigma_y = torch.tensor([[0, -1.0j], [1.0j, 0]], dtype=torch.complex64)
sigma_z = torch.tensor([[1.0, 0], [0, -1.0]], dtype=torch.complex64)
sig = torch.stack([sigma_x, sigma_y, sigma_z])


@torch.no_grad()
def treesamplerumap(model, batch=10000, max_unique=1000):
    batch0 = batch
    assert (
        model.phys_dim == 2
    ), "Only spin 1/2 systems are supported"  # Addapt to double occupancy in the future
    n = model.system_size.prod()
    samples = torch.zeros(0, 1, device=device, dtype=torch.uint8)
    sample_count = torch.tensor(
        [batch], dtype=torch.int32
    )  # This is N_batch in the text

    for _ in range(n):  # For each k
        (log_amp,), _ = model.forward(
            samples, compute_phase=False
        )  # (seq, batch, phys_dim)
        # Get amplitude from last batch

        amp = log_amp[-1].exp()  # (batch, phys_dim)

        # -----------------------------------------------------
        if (
            len(sample_count) < max_unique
        ):  # Max_unique = N_unique in the text / This is in parallel
            distribution = Binomial(
                total_count=sample_count, probs=amp[:, 0]
            )  # Sample from binomial
            zero_count = distribution.sample()  # (batch, ), ocurrences for s_{i+1}=0
            one_count = (
                sample_count - zero_count
            )  # ocurrences for s_{i+1}=1, from n_k0+nk1=1
            sample_count = torch.cat([zero_count, one_count], dim=0)  # Total ocurrences
            mask = (
                sample_count > 0
            )  # Just tell us about samples that have more than 1 ocurrence
            mask = sample_count == sample_count

            batch = samples.shape[1]
            samples = torch.cat(
                [
                    torch.cat(
                        [samples, torch.zeros(1, batch, dtype=torch.uint8)], dim=0
                    ),
                    torch.cat(
                        [samples, torch.ones(1, batch, dtype=torch.uint8)], dim=0
                    ),
                ],
                dim=1,
            )
            # First T serves to act with mask only on batch dimensionsself.
            # Mask plucks out the 0 configurations

            samples = samples.T[mask].T  # (seq, batch), with updated batch
            sample_count = sample_count[mask]  # (batch, )
            # print(sample_count)
            samples = torch.cat([samples], dim=0)

            # print(len(sample_count), sample_count)
        else:
            # do not generate new branches
            # Remanining spins are generated in the regular way
            sampled_spins = torch.multinomial(amp, 1).to(torch.uint8)  # (batch, 1)
            samples = torch.cat([samples, sampled_spins.T], dim=0)

    n, dim = samples.shape

    # Sum along the first dimension (axis=0) to get the count of 1s in each column
    ns = torch.sum(samples, dim=0, dtype=torch.int8)

    # Find indices where ns equals n
    idxns = torch.where(ns == n)[0]

    # Initial sample weight calculation

    sample_weight = sample_count / batch0

    # Handle the case where len(idxns) == 1
    if len(idxns) == 1:
        id = int(idxns)
        batch2 = batch0 * (1 - sample_weight[id])
        sample_weight = sample_count / batch2
        sample_weight[id] = 0
    else:
        batch2 = batch0
        sample_weight = sample_count / batch2

    # Final sample weight calculation

    return samples, sample_weight  # (n, batch), (batch, )


@torch.no_grad()
def treesampler(model, batch=10000, max_unique=1000):
    batch0 = batch
    assert (
        model.phys_dim == 2
    ), "Only spin 1/2 systems are supported"  # Addapt to double occupancy in the future
    n = model.system_size.prod()
    samples = torch.zeros(0, 1, device=device, dtype=torch.uint8)
    sample_count = torch.tensor(
        [batch], dtype=torch.int32
    )  # This is N_batch in the text

    for _ in range(n):  # For each k
        (log_amp,), src = model.forward(
            samples, compute_phase=False
        )  # (seq, batch, phys_dim)
        # Get amplitude from last batch

        amp = log_amp[-1].exp()  # (batch, phys_dim)

        # -----------------------------------------------------
        if (
            len(sample_count) < max_unique
        ):  # Max_unique = N_unique in the text / This is in parallel
            distribution = Binomial(
                total_count=sample_count, probs=amp[:, 0]
            )  # Sample from binomial
            zero_count = distribution.sample()  # (batch, ), ocurrences for s_{i+1}=0
            one_count = (
                sample_count - zero_count
            )  # ocurrences for s_{i+1}=1, from n_k0+nk1=1
            sample_count = torch.cat([zero_count, one_count], dim=0)  # Total ocurrences
            mask = (
                sample_count > 0
            )  # Just tell us about samples that have more than 1 ocurrence
            # mask = sample_count == sample_count

            batch = samples.shape[1]
            samples = torch.cat(
                [
                    torch.cat(
                        [samples, torch.zeros(1, batch, dtype=torch.uint8)], dim=0
                    ),
                    torch.cat(
                        [samples, torch.ones(1, batch, dtype=torch.uint8)], dim=0
                    ),
                ],
                dim=1,
            )
            # First T serves to act with mask only on batch dimensionsself.
            # Mask plucks out the 0 configurations

            samples = samples.T[mask].T  # (seq, batch), with updated batch
            sample_count = sample_count[mask]  # (batch, )
            # print(sample_count)
            samples = torch.cat([samples], dim=0)

            # print(len(sample_count), sample_count)
        else:
            # do not generate new branches
            # Remanining spins are generated in the regular way
            sampled_spins = torch.multinomial(amp, 1).to(torch.uint8)  # (batch, 1)
            samples = torch.cat([samples, sampled_spins.T], dim=0)

    n, dim = samples.shape

    # Sum along the first dimension (axis=0) to get the count of 1s in each column
    ns = torch.sum(samples, dim=0, dtype=torch.int8)

    # Find indices where ns equals n
    idxns = torch.where(ns == n)[0]

    # Initial sample weight calculation

    sample_weight = sample_count / batch0

    # Handle the case where len(idxns) == 1
    if len(idxns) == 1:
        id = int(idxns)
        batch2 = batch0 * (1 - sample_weight[id])
        sample_weight = sample_count / batch2
        sample_weight[id] = 0
    else:
        batch2 = batch0
        sample_weight = sample_count / batch2

        # Final sample weight calculation

    return samples, sample_weight, idxns  # (n, batch), (batch, )


@torch.no_grad()
def normalsampler(model, batch=10000):
    n = model.system_size.prod()
    samples = torch.zeros((0, batch), dtype=torch.uint8)
    for i in range(n):
        (log_amp,), _ = model.forward(
            samples, compute_phase=False
        )  # (seq, batch, phys_dim)
        amp = log_amp[-1].exp()  # (batch, phys_dim)

        sampled_spins = torch.multinomial(amp, 1).to(torch.uint8)  # (batch, 1)
        samples = torch.cat([samples, sampled_spins.T], dim=0)

    return samples


def compute_psi(model, samples, check_duplicate=True):
    if check_duplicate:
        samples, inv_idx = torch.unique(samples, dim=1, return_inverse=True)

    n, batch = samples.shape
    n_idx = torch.arange(n).reshape(n, 1)
    batch_idx = torch.arange(batch).reshape(1, batch)

    spin_idx = samples.to(torch.int32)

    (log_amp, log_phase), emb = model.forward(
        samples, compute_phase=True
    )  # (seq, batch, phys_dim)

    log_amp = log_amp[:-1]  # (n, batch, phys_dim)
    log_phase = log_phase[:-1]  # (n, batch, phys_dim)

    log_amp = log_amp[n_idx, batch_idx, spin_idx].sum(dim=0)  # (batch, )
    log_phase = log_phase[n_idx, batch_idx, spin_idx].sum(dim=0)  # (batch, )

    if check_duplicate:
        # Directly calculate just this part in the future
        log_amp = log_amp[inv_idx]
        log_phase = log_phase[inv_idx]

    return log_amp, log_phase, emb


@torch.no_grad()
def Hloc(
    samples,
    Uk,
    k,
    alfa2,
    alfa3,
    log_amp,
    log_amp_2,
    log_phase,
    log_phase_2,
    tp,
    batch,
    n,
    ns,
    nsp,
    sig,
    k_indices,
):
    indices = torch.arange(len(log_amp_2)).reshape(
        -1, n
    )  # Reshape from (batch*ne) to (batch, ne)

    # Broadcast log_amp and log_phase for subtraction
    log_amp_diff = log_amp_2[indices] - log_amp.unsqueeze(1)  # (batch, ne)
    log_phase_diff = log_phase_2[indices] - log_phase.unsqueeze(1)  # (batch,ne)

    # Calculate the psi(sp)/psi for each group
    psisp_s = torch.exp((log_amp_diff + 1j * log_phase_diff) / 2)
    psis = torch.exp((log_amp + 1j * log_phase) / 2)
    # Create empty list to organize energies into groups (that scatter to HF or not)
    results = []
    results2 = []
    results3 = []

    result = torch.zeros(n, batch, dtype=torch.cfloat)
    result_i = torch.zeros(n, batch, dtype=torch.cfloat)
    result_i2 = torch.zeros(n, batch, dtype=torch.cfloat)
    result_i3 = torch.zeros(n, batch, dtype=torch.cfloat)
    result_i2temp = torch.zeros(n, batch, dtype=torch.cfloat)
    result_i3temp = torch.zeros(n, batch, dtype=torch.cfloat)

    occupation = []
    occupation_i = torch.zeros(n, batch, 3, dtype=torch.cfloat)
    occupation_in = torch.zeros(n, batch, 3, dtype=torch.cfloat)
    occupation_inz = torch.zeros(n, batch, dtype=torch.cfloat)
    occupation_inx = torch.zeros(n, batch, dtype=torch.cfloat)
    occupation_iny = torch.zeros(n, batch, dtype=torch.cfloat)

    batch = int(batch)
    cos_kinds = torch.tensor([torch.cos(kind[1]) for kind in k])  # (ne)
    alpha2_all = samples[k_indices, :]  # (ne, batch)
    alphat = alpha2_all[:, :batch].long()  # (ne, batch)
    alphatnot = 1 - alphat  # (ne, batch)

    tau_k = (
        torch.conj(Uk[k_indices]).transpose(-1, -2) @ sig[0] @ Uk[k_indices]
    )  # (ne, 2, 2)
    tau_ky = (
        torch.conj(Uk[k_indices]).transpose(-1, -2) @ sig[1] @ Uk[k_indices]
    )  # (ne, 2, 2)
    tau_kz = (
        torch.conj(Uk[k_indices]).transpose(-1, -2) @ sig[2] @ Uk[k_indices]
    )  # (ne, 2, 2)

    # # Masks
    sf_hf = nsp[:] == n  # Final state sf=HF
    sf_hf = sf_hf.view(batch, n).T  # (n, batch)

    si_nhf = ns[:] != n  # Initial state si \neq HF
    si_nhf2 = si_nhf[:].expand(n, batch)
    sf_nhf = nsp[:] != n  # Final state sf \neq HF
    sf_nhf = sf_nhf.view(batch, n).T  # (n, batch)

    batch_indices = torch.arange(n).unsqueeze(1).expand(n, batch)  # shape: [ne, batch]
    h0nloc = tau_k[
        batch_indices, alphat, alphatnot
    ]  # (ne, batch). Here, ne is the batch dimension, where independen t data points are grouped

    # --------------------------------------------------------------------
    # # (1) Scatter from HF to NHF
    # mask = si_hf2 & sf_nhf
    # psisp = psisp.T  # (ne,batch)
    # h0nloc = tau_k[batch_indices, alphat, alphatnot] * psisp  # (ne, batch)
    # result[:, :] = (
    #     cos_kinds[:] * h0nloc[:, :].T
    # ).T  # (ne)*(ne, batch).T) = (ne, batch)
    # occupation_inx[:, :] = alfa3 * h0nloc
    # occupation_iny[:, :] = alfa3 * tau_ky[batch_indices, alphat, alphatnot] * psisp
    # occupation_inz[:, :] = alfa3 * tau_kz[batch_indices, alphat, alphatnot] * psisp
    # # First, get all the ones with sf_nhf, this is needed since the dimension of this is
    # # n*batch > dim(si_hf) = batch
    #
    # occupation_inx[:, :] = torch.where(
    #     mask, occupation_inx, torch.zeros_like(occupation_inx)
    # )
    # occupation_iny[:, :] = torch.where(
    #     mask, occupation_iny, torch.zeros_like(occupation_iny)
    # )
    # occupation_inz[:, :] = torch.where(
    #     mask, occupation_inz, torch.zeros_like(occupation_inz)
    # )
    # result_i3temp[:, :] = torch.where(mask, result, torch.zeros_like(result))
    # result_i3[:, :] += result_i3temp[:, :]
    #
    # occupation_in = torch.stack(
    #     [occupation_inx, occupation_iny, occupation_inz], dim=-1
    # )
    #
    # occupation_in = torch.where(
    #     mask.unsqueeze(-1), occupation_in, torch.zeros_like(occupation_in)
    # )
    #
    # occupation_i[:, :, :] += occupation_in[:, :, :]
    #
    # occupation_in = torch.zeros(n, batch, 3, dtype=torch.cfloat)
    # occupation_inz = torch.zeros(n, batch, dtype=torch.cfloat)
    # occupation_inx = torch.zeros(n, batch, dtype=torch.cfloat)
    # occupation_iny = torch.zeros(n, batch, dtype=torch.cfloat)

    # -------------------------------------------------------------------------------
    result_i3temp = torch.zeros(n, batch, dtype=torch.cfloat)

    # # (2) Scatter from NHF to HF
    mask = si_nhf2 & sf_hf
    h0nloc = tau_k[batch_indices, alphat, alphatnot] / psis  # (ne, batch)
    result[:, :] = (
        cos_kinds[:] * h0nloc[:, :].T
    ).T  # (ne)*(ne, batch).T) = (ne, batch)
    occupation_inx[:, :] = alfa3 * h0nloc
    occupation_iny[:, :] = alfa3 * tau_ky[batch_indices, alphat, alphatnot] / psis
    occupation_inz[:, :] = alfa3 * tau_kz[batch_indices, alphat, alphatnot] / psis
    occupation_inx[:, :] = (
        2 * torch.where(mask, occupation_inx, torch.zeros_like(occupation_inx)).real
    )
    occupation_iny[:, :] = (
        2 * torch.where(mask, occupation_iny, torch.zeros_like(occupation_iny)).real
    )
    occupation_inz[:, :] = (
        2 * torch.where(mask, occupation_inz, torch.zeros_like(occupation_inz)).real
    )
    # First, get all the ones with sf_nhf, this is needed since the dimension of this is
    # n*batch > dim(si_hf) = batch
    result_i3temp[:, :] = torch.where(mask, result, torch.zeros_like(result))
    result_i3[:, :] += result_i3temp[:, :]

    occupation_in = torch.stack(
        [occupation_inx, occupation_iny, occupation_inz], dim=-1
    )
    occupation_in = torch.where(
        mask.unsqueeze(-1), occupation_in, torch.zeros_like(occupation_in)
    )

    occupation_i[:, :, :] += occupation_in[:, :, :]

    occupation_in = torch.zeros(n, batch, 3, dtype=torch.cfloat)
    occupation_inz = torch.zeros(n, batch, dtype=torch.cfloat)
    occupation_inx = torch.zeros(n, batch, dtype=torch.cfloat)
    occupation_iny = torch.zeros(n, batch, dtype=torch.cfloat)

    # (3) Scatter from s to sp (non HF (nhf))
    mask = si_nhf2 & sf_nhf
    psisp_s = psisp_s.T  # (ne,batch)
    h0nloc = tau_k[batch_indices, alphat, alphatnot] * psisp_s  # (ne, batch)
    occupation_inx[:, :] = alfa2 * h0nloc
    occupation_iny[:, :] = alfa2 * tau_ky[batch_indices, alphat, alphatnot] * psisp_s
    occupation_inz[:, :] = alfa2 * tau_kz[batch_indices, alphat, alphatnot] * psisp_s
    occupation_inx[:, :] = torch.where(
        mask, occupation_inx, torch.zeros_like(occupation_inx)
    )
    occupation_iny[:, :] = torch.where(
        mask, occupation_iny, torch.zeros_like(occupation_iny)
    )
    occupation_inz[:, :] = torch.where(
        mask, occupation_inz, torch.zeros_like(occupation_inz)
    )
    result[:, :] = (
        cos_kinds[:] * h0nloc[:, :].T
    ).T  # (ne)*(ne, batch).T) = (ne, batch)
    # First, get all the ones with sf_nhf, this is needed since the dimension of this is
    # n*batch > dim(si_hf) = batch
    result_i2temp[:, :] = torch.where(mask, result, torch.zeros_like(result))
    result_i2[:, :] += result_i2temp[:, :]

    occupation_in = torch.stack(
        [occupation_inx, occupation_iny, occupation_inz], dim=-1
    )
    occupation_in = torch.where(
        mask.unsqueeze(-1), occupation_in, torch.zeros_like(occupation_in)
    )

    occupation_i[:, :, :] += occupation_in[:, :, :]

    occupation_in = torch.zeros(n, batch, 3, dtype=torch.cfloat)
    occupation_inz = torch.zeros(n, batch, dtype=torch.cfloat)
    occupation_inx = torch.zeros(n, batch, dtype=torch.cfloat)
    occupation_iny = torch.zeros(n, batch, dtype=torch.cfloat)

    result_i3temp = torch.zeros(n, batch, dtype=torch.cfloat)
    # # (3.1) Local term for NHF to NHF
    mask = si_nhf2
    h0loc = tau_k[batch_indices, alphat, alphat]  # (ne, batch)
    result[:, :] = (cos_kinds[:] * h0loc[:, :].T).T  # (ne)*(ne, batch).T) = (ne, batch)
    occupation_inx[:, :] = alfa2 * h0loc
    occupation_iny[:, :] = alfa2 * tau_ky[batch_indices, alphat, alphat]
    occupation_inz[:, :] = alfa2 * tau_kz[batch_indices, alphat, alphat]
    occupation_inx[:, :] = torch.where(
        mask, occupation_inx, torch.zeros_like(occupation_inx)
    )
    occupation_iny[:, :] = torch.where(
        mask, occupation_iny, torch.zeros_like(occupation_iny)
    )
    occupation_inz[:, :] = torch.where(
        mask, occupation_inz, torch.zeros_like(occupation_inz)
    )
    # First, get all the ones with sf_nhf, this is needed since the dimension of this is
    result_i2temp[:, :] = torch.where(mask, result, torch.zeros_like(result))
    result_i2[:, :] += result_i2temp[:, :]

    result_i[:, :] += result_i2[:, :] + result_i3[:, :]
    occupation_in = torch.stack(
        [occupation_inx, occupation_iny, occupation_inz], dim=-1
    )
    occupation_in = torch.where(
        mask.unsqueeze(-1), occupation_in, torch.zeros_like(occupation_in)
    )

    occupation_i[:, :, :] += occupation_in[:, :, :]

    results.append(tp * result_i)  # (n_op, batch)
    results2.append(tp * result_i2)
    results3.append(tp * result_i3)
    occupation.append(occupation_i)

    return results, results2, results3, occupation


def compute_grad(model, samples, sample_weight, Eloc, ave, idxns):
    log_amp, log_phase, _ = compute_psi(model, samples, check_duplicate=True)

    E = Eloc
    if len(idxns) == 1:
        # Mask values for HF
        ind2 = int(idxns)
        E[ind2] = 0
        log_amp[ind2] = 0
        log_phase[ind2] = 0
        sample_weight[ind2] = 0

    loss = ((E.real * log_amp + E.imag * log_phase) * sample_weight).sum()
    # loss = (
    #     (E.real * log_amp + E.imag * log_phase) * sample_weight
    # ).sum() - ave * log_amp * sample_weight.sum()

    return loss, log_amp, log_phase


@torch.no_grad()
def compute_observableFMHF(
    model,
    samples,
    Uk,
    tp,
    k,
    q,
    alfa2,
    alfa3,
):
    # Get the probabilities of previous samples
    log_amp, log_phase, _ = compute_psi(
        model, samples, check_duplicate=True
    )  # (batch, ), for each sample in the batch we have a psi
    psifull = log_amp.exp()
    psifull = psifull * psifull

    # ----------------------------------

    n, batch = samples.shape

    # First get ALL the PSIs for the exchanges in one forward in the transformer

    start_time00 = time.perf_counter()
    n, batch = samples.shape

    # Create a tensor of k_indices
    # TO DO: add this to hamiltonian utils
    k_indices = torch.tensor(
        [int(kind[0]) for kind in k], dtype=torch.long, device=samples.device
    )

    sample_int = (samples).detach().clone()

    # Create a mask for flipping the spins
    flip_mask = torch.zeros((len(k), n, batch), dtype=torch.bool)
    flip_mask[torch.arange(len(k)), k_indices, :] = True

    # Flip the selected electrons in the k_indices positions
    sflip = sample_int.unsqueeze(0).expand(len(k), -1, -1).clone().detach()
    sflip[flip_mask] = -(sflip[flip_mask] - 1)  # 0 goes 1 and 1 goes to 0

    sp = sflip.permute(1, 2, 0).reshape(
        n, n * batch
    )  # Note that n*batch refers to how many "flips" we have

    log_amp_2, log_phase_2, _ = compute_psi(
        model,
        sp,
        check_duplicate=True,
    )  # (n_op*batch

    # Create empty list to organize energies into groups (that scatter to HF or not)
    results = []
    results2 = []
    results3 = []
    occupation = []
    n, batch = samples.shape

    # Occupation in the HF band for each k
    # This can also be done once before initializing the program (TO DO)
    _, batch2 = sp.shape
    nsp = torch.zeros(batch2, dtype=torch.uint8)
    ns = torch.zeros(batch, dtype=torch.uint8)

    for i in range(batch):  # over k
        ns[i] = torch.sum(samples[:, i])
    for i in range(batch2):
        nsp[i] = torch.sum(sp[:, i])

    results, results2, results3, occupation = Hloc(
        samples,
        Uk,
        k,
        alfa2,
        alfa3,
        log_amp,
        log_amp_2,
        log_phase,
        log_phase_2,
        tp,
        int(batch),
        n,
        ns,
        nsp,
        sig,
        k_indices,
    )

    # -------------------------------------- Interaction term V(q) --------------------------

    # First get ALL the PSIs for the exchanges
    # to pass them in one forward in the transformer. Much more efficient

    sample_int = torch.zeros(n, dtype=torch.uint8)

    n_electrons, batch_size = samples.shape
    k_indices = torch.tensor([int(kind[0]) for kind in k])  # (n_k = n)
    q_indices = torch.tensor([int(qind[0]) for qind in q])  # (n_q)

    n_k = len(k_indices)
    n_q = len(q_indices)
    n_exch = n_q * n_k

    nt = n_exch * 4  # Total number of possible scattering from V(q) per batch
    nt_batch = 4 * n_exch * batch  # total number of scattering for all batches

    # Create index tensors
    kq_indices = torch.arange(n_exch)
    batch_indices = torch.arange(batch_size)
    # Step 2: Compute all combinations
    # Expand both vectors for dimension (n_k, n_q) for both loops: k_indices is repeated along n_q direction, and
    # q_indices along the n_direction
    k_indices_expanded = k_indices.clone().detach().unsqueeze(1).expand(-1, n_q)
    q_indices_expanded = q_indices.clone().detach().unsqueeze(0).expand(n_k, -1)

    kmq_indices = (
        (k_indices_expanded - q_indices_expanded) % n
    ).long()  # Scattering tensor, %n referes to PBC # (n_k, n_q)
    # Note: The normal loop is appended over n_q for every q : - > concatenate over the first dimension (n_k)
    # This form of concatenation is actually fine, for s_psis_p_d1 just go over 0:63, for s_psis_p_d2 from 64+(0:63) etc (This example is for N=6)

    # Reshape indices to (n_k* n_q, 1)
    k_indices_flat = k_indices_expanded.reshape(-1)  # (n_exch)
    kmq_indices_flat = kmq_indices.reshape(-1)  # (n_exch)

    # Expand indices for broadcasting
    k_indices_expanded = k_indices_flat.unsqueeze(1).expand(-1, batch_size)
    kmq_indices_expanded = kmq_indices_flat.unsqueeze(1).expand(-1, batch_size)
    kq_indices_expanded = kq_indices.unsqueeze(1).expand(-1, batch_size)  # (60, 64)

    # Step 3: Prepare masks
    # What is the None refering to here?
    s_i = samples[k_indices_flat[:, None], batch_indices]
    s_j = samples[kmq_indices_flat[:, None], batch_indices]  # (n_exch, batch_size)

    beta = 1 - s_j  # Opposite of s_j

    # (n_kindices*n_qindices, batch_size)
    mask_d1_int = s_i != 0
    mask_d2_int = s_i != 1
    mask_beta_int = s_j != beta

    # Reshape masks to (n_electrons, n_kindices * n_qindices, batch_size)
    mask_d1 = torch.zeros(
        (n_electrons, n_exch, batch_size),
        dtype=torch.bool,
    )
    mask_d2 = torch.zeros(
        (n_electrons, n_exch, batch_size),
        dtype=torch.bool,
    )
    mask_beta = torch.zeros(
        (n_electrons, n_exch, batch_size),
        dtype=torch.bool,
    )

    mask_d1[
        k_indices_expanded, kq_indices_expanded, batch_indices
    ] = mask_d1_int  # (n_electrons, n_exch, batch_size)
    mask_d2[
        k_indices_expanded, kq_indices_expanded, batch_indices
    ] = mask_d2_int  # (n_electrons, n_exch, batch_size)
    mask_beta[
        kmq_indices_expanded, kq_indices_expanded, batch_indices
    ] = mask_beta_int  # (n_electrons, n_exch, batch_size)

    # Step 4: Apply operations

    # # Scaterrings from descendants from | 1 >
    # s_psis_p_d1 = samples_expanded.clone().detach()
    # s_psis_p_d1[mask_d1] = 1 - s_psis_p_d1[mask_d1]

    # # If spin in positions given by mask is not beta already, flip it
    # # Which delta is this?
    # s_psis_p2_d1 = samples_expanded.clone().detach()
    # s_psis_p2_d1[~mask_d1 + mask_beta] = (
    #     1 - s_psis_p2_d1[~mask_d1 + mask_beta]
    # )  # Group1
    # s_psis_p2_d1[mask_d1 + mask_beta] = 1 - s_psis_p2_d1[mask_d1 + mask_beta]  # Group3
    # s_psis_p2_d1[mask_d1 & ~mask_beta] = (
    #     1 - s_psis_p2_d1[mask_d1 & ~mask_beta]
    # )  # Group2
    # s_psis_p2_d1 = 1 - s_psis_p2_d1

    # # Scaterrings from descendants from | 0 >

    # s_psis_p_d2 = samples_expanded.clone().detach()
    # s_psis_p_d2[mask_d2] = 1 - s_psis_p_d2[mask_d2]

    # s_psis_p2_d2 = samples_expanded.clone().detach()
    # s_psis_p2_d2[~mask_d2 + mask_beta] = (
    #     1 - s_psis_p2_d2[~mask_d2 + mask_beta]
    # )  # Group1
    # s_psis_p2_d2[mask_d2 + mask_beta] = 1 - s_psis_p2_d2[mask_d2 + mask_beta]  # Group3
    # s_psis_p2_d2[mask_d2 & ~mask_beta] = (
    #     1 - s_psis_p2_d2[mask_d2 & ~mask_beta]
    # )  # Group2
    # s_psis_p2_d2 = 1 - s_psis_p2_d2

    # s_psis_p_d1_reshaped = s_psis_p_d1.permute(0, 2, 1).reshape(n_electrons, -1)
    # s_psis_p2_d1_reshaped = s_psis_p2_d1.permute(0, 2, 1).reshape(n_electrons, -1)
    # s_psis_p_d2_reshaped = s_psis_p_d2.permute(0, 2, 1).reshape(n_electrons, -1)
    # s_psis_p2_d2_reshaped = s_psis_p2_d2.permute(0, 2, 1).reshape(n_electrons, -1)

    samples_expanded = (
        samples.clone().detach().unsqueeze(1).expand(-1, n_exch, -1)
    )  # (n_electrons, n_exch, batch_size)
    # s_psis = samples_expanded.clone().detach()

    # Compute common operations once
    s_psis_flipped = 1 - samples_expanded

    # Compute s_psis_p_d1 and s_psis_p_d2
    s_psis_p_d1n = torch.where(mask_d1, s_psis_flipped, samples_expanded)
    s_psis_p_d2n = torch.where(mask_d2, s_psis_flipped, samples_expanded)

    # Check if all elements are equal

    # Compute s_psis_p2_d1 and s_psis_p2_d2
    def compute_s_psis_p2(mask_d, s_psis, s_psis_flipped):
        condition1 = ~mask_d & mask_beta
        condition2 = mask_d & ~mask_beta
        condition3 = mask_d & mask_beta

        result = torch.where(
            condition1 | condition2 | condition3, s_psis, s_psis_flipped
        )
        return 1 - result

    s_psis_p2_d1n = compute_s_psis_p2(mask_d1, samples_expanded, s_psis_flipped)
    s_psis_p2_d2n = compute_s_psis_p2(mask_d2, samples_expanded, s_psis_flipped)

    newsample = torch.zeros(n_electrons, nt_batch, dtype=torch.uint8)

    def reshape_and_assign(tensor, idx):
        newsample[:, idx::4] = tensor.permute(0, 2, 1).reshape(n_electrons, -1)

    reshape_and_assign(s_psis_p_d1n, 0)
    reshape_and_assign(s_psis_p2_d1n, 1)
    reshape_and_assign(s_psis_p_d2n, 2)
    reshape_and_assign(s_psis_p2_d2n, 3)

    # Reshape it from (n_k, 4, n_exch, batch) -> (n_k, 4*n_exch,batch)
    # Here, 4 stands for the "groups" of wavefunctions we have". Do not confuse these groups
    # with the 3 groups of possible scatterings s<->s', s<->hf
    #
    # This change is just a choice of order, such that the samples appended in the interleaved
    # tensor have the same order as in the loop batch -> k -> q in a non-vectorized version of the code.

    # newsample = torch.tensor(np.array(samplest)).T

    log_amp1, log_phase1, _ = compute_psi(
        model, newsample, check_duplicate=True
    )  # (n_electrons, batch*4*n_exch = nt)

    log_amp1 = log_amp1.view(batch, nt)
    log_phase1 = log_phase1.view(batch, nt)

    log_amp_expanded = log_amp.unsqueeze(1).expand(-1, nt)  # (batch, nt)
    log_phase_expanded = log_phase.unsqueeze(1).expand(-1, nt)

    psiloc22 = torch.exp(
        ((log_amp1 - log_amp_expanded) + 1j * (log_phase1 - log_phase_expanded)) / 2
    )
    psilambdasp22 = torch.exp((log_amp1 + 1j * log_phase1) / 2)
    psilambdas22 = torch.exp((log_amp + 1j * log_phase) / 2)

    # Prepare indices
    k_indices = torch.tensor([int(kind[0]) for kind in k], dtype=torch.int16)  # (n)
    q_indices = torch.tensor([int(qind[0]) for qind in q], dtype=torch.int16)  # (n)
    kmq_indices = ((k_indices.unsqueeze(1) - q_indices.unsqueeze(0)) % n).to(
        torch.int32
    )

    # Prepare potentials
    potentials = torch.tensor([qind[2] for qind in q])  # (n_q)

    # Assume k and q are lists of momentum values
    # Add necessary dimensions to sigma matrices for broadcasting
    sigma_0 = torch.eye(2, dtype=torch.complex64)
    sigma_z = torch.tensor([[1.0, 0], [0, -1.0]], dtype=torch.complex64)
    sigma_0 = sigma_0.unsqueeze(0).unsqueeze(
        0
    )  # shape becomes (1, 1, 2, 2) with new indices for loop over k and q
    sigma_z = sigma_z.unsqueeze(0).unsqueeze(0)  # shape becomes (1, 1, 2, 2)

    # Prepare psi values
    psisp_s = psiloc22  # (batch, n_t)
    psisp = psilambdasp22  # (batch, n_t)
    psis = psilambdas22  # (batch)

    # Calculate nsp3
    nsp3 = torch.sum(
        newsample, dim=0, dtype=torch.int8
    )  # (nt_batch = n_exch*n_groups*batch)

    # -----------------------------

    # start_time = time.perf_counter()

    def compute_psis(psisp, psis, psisp_s, nsp3, ns, n):
        batch_size, total_exch = psisp.shape

        # Reshape nsp3 to match psisp shape
        nsp3_reshaped = nsp3.reshape(batch_size, total_exch)

        # Prepare condition masks
        # condition1 = (nsp3_reshaped != n) & (ns.unsqueeze(1) == n)  # alfa3
        condition2 = (nsp3_reshaped == n) & (ns.unsqueeze(1) != n)  # alfa3
        condition3 = (nsp3_reshaped != n) & (ns.unsqueeze(1) != n)  # alfa2

        # Compute psis for HFs and ssp separately
        # psis_HFs = torch.where(
        #     condition1,
        #     psisp,
        #     torch.where(condition2, 1 / psis.unsqueeze(1), torch.zeros_like(psisp)),
        # )

        psis_HFs = torch.where(
            condition2, 1 / psis.unsqueeze(1), torch.zeros_like(psisp)
        )

        psis_ssp = torch.where(condition3, psisp_s, torch.zeros_like(psisp))

        # Reshape and split the results
        psis_HFs = psis_HFs.reshape(batch_size, n_exch, 4)
        psis_ssp = psis_ssp.reshape(batch_size, n_exch, 4)

        return (
            psis_HFs[:, :, 0],  # psis_p_d1HFs
            psis_HFs[:, :, 1],  # psis_p2_d1HFs
            psis_HFs[:, :, 2],  # psis_p_d2HFs
            psis_HFs[:, :, 3],  # psis_p2_d2HFs
            psis_ssp[:, :, 0],  # psis_p_d1ssp
            psis_ssp[:, :, 1],  # psis_p2_d1ssp
            psis_ssp[:, :, 2],  # psis_p_d2ssp
            psis_ssp[:, :, 3],  # psis_p2_d2ssp
        )

    # Usage:
    (
        psis_p_d1HFs,
        psis_p2_d1HFs,
        psis_p_d2HFs,
        psis_p2_d2HFs,
        psis_p_d1ssp,
        psis_p2_d1ssp,
        psis_p_d2ssp,
        psis_p2_d2ssp,
    ) = compute_psis(psisp, psis, psisp_s, nsp3, ns, n)
    # Usage:
    # end_time = time.perf_counter()
    # execution_time = end_time - start_time
    # print(f"4. T for getting psis in HNloc: {execution_time} seconds")

    # Perform the multiplication with automatic broadcasting
    # Calculate form factors
    # Assuming k and q are lists of values from state.chain.k and state.chain.q
    k_values = torch.tensor([k[1] for k in k], dtype=torch.float32)
    q_values = torch.tensor([q[1] for q in q], dtype=torch.float32)

    # Create meshgrid of k and q values
    k_grid, q_grid = torch.meshgrid(k_values, q_values, indexing="ij")  # (n_k, n_q)
    # k_grid = k_values.unsqueeze(1).expand(-1, len(q))  # Shape: (len(k), len(q))
    # q_grid = q_values.unsqueeze(0).expand(len(k), -1)  # Shape: (len(k), len(q))

    # Now we can call form_factors with these grids
    f1_k_mq, f2_k_mq, f3_k_mq, _ = form_factors(k_grid, -q_grid)
    f1_kmq_q, f2_kmq_q, f3_kmq_q, _ = form_factors(
        k_grid - q_grid, q_grid
    )  # (n_k, n_q)

    sigma_0 = (
        torch.eye(2, dtype=torch.complex64).unsqueeze(0).unsqueeze(0)
    )  # (1, 1, 2, 2)
    sigma_z = (
        torch.tensor([[1.0, 0], [0, -1.0]], dtype=torch.complex64)
        .unsqueeze(0)
        .unsqueeze(0)
    )  # (1, 1, 2, 2)
    # Compute bigF matrices for all k, q combinations
    # bigF1 = sigma_0 * f1_k_mq.unsqueeze(-1).unsqueeze(
    #     -1
    # ) + 1j * sigma_z * f2_k_mq.unsqueeze(-1).unsqueeze(-1)
    # bigF2 = sigma_0 * f1_kmq_q.unsqueeze(-1).unsqueeze(
    #     -1
    # ) + 1j * sigma_z * f2_kmq_q.unsqueeze(-1).unsqueeze(
    #     -1
    # )  # (n_k, n_q, 2, 2)
    bigF1 = (
        sigma_0 * f1_k_mq.unsqueeze(-1).unsqueeze(-1)
        + 1j * sigma_z * f2_k_mq.unsqueeze(-1).unsqueeze(-1)
        + sigma_x * f3_k_mq.unsqueeze(-1).unsqueeze(-1)
    )
    bigF2 = (
        sigma_0 * f1_kmq_q.unsqueeze(-1).unsqueeze(-1)
        + 1j * sigma_z * f2_kmq_q.unsqueeze(-1).unsqueeze(-1)
        + sigma_x * f3_kmq_q.unsqueeze(-1).unsqueeze(-1)
    )

    # Expand Uk to include the q dimension
    Uk_expanded = Uk.unsqueeze(1).expand(-1, n_q, -1, -1).to(torch.complex64)
    Uk_kmq_expanded = Uk[kmq_indices].to(torch.complex64)  # (n_k, n_q, 2, 2)

    # Perform einsum operations with the reordered Uk and transpose

    bigF_kmq_q = torch.einsum(
        "kqba,kqbc,kqcd->kqad", torch.conj(Uk_expanded), bigF2, Uk_kmq_expanded
    )
    bigF_k_mq = torch.einsum(
        "kqba,kqbc,kqcd->kqad", torch.conj(Uk_kmq_expanded), bigF1, Uk_expanded
    )

    s_iold = 1 - 2 * samples[k_indices_flat[:, None], batch_indices]
    s_jold = (
        1 - 2 * samples[kmq_indices_flat[:, None], batch_indices]
    )  # (n_exch, batch_size)
    # s_iold = samples[k_indices_flat[:, None], batch_indices]
    # s_jold = samples[kmq_indices_flat[:, None], batch_indices]  # (n_exch, batch_size)

    betaold = -s_jold  # Opposite of s_j
    # betaold = 1 - s_jold
    alphaold = s_iold
    # print(betaoldt)

    n_k, n_q, _, _ = bigF_k_mq.shape
    n_exch = n_k * n_q

    # ---------------------------

    bigF_k_mq_reshaped = (
        bigF_k_mq.reshape(n_exch, 2, 2).unsqueeze(1).expand(-1, batch, -1, -1)
    )

    bigF_kmq_q_reshaped = (
        bigF_kmq_q.reshape(n_exch, 2, 2).unsqueeze(1).expand(-1, batch, -1, -1)
    )

    # conversion
    # betaoldt = (abs((1 - betaold)) / 2).to(torch.uint8)
    # notbetaoldt = abs((1 - betaoldt)).to(torch.uint8)
    # alphaoldt = (abs((1 - alphaold)) / 2).to(torch.uint8)
    betaoldt = (abs((1 - betaold)) / 2).long()
    notbetaoldt = abs((1 - betaoldt)).long()
    alphaoldt = (abs((1 - alphaold)) / 2).long()
    # betaoldt = betaold
    # notbetaoldt = (1 - betaoldt).long()
    # print(betaoldt, notbetaoldt)
    # alphaoldt = alphaold

    # Create index tensors
    i, j = torch.meshgrid(torch.arange(n_exch), torch.arange(batch), indexing="ij")

    # Use advanced indexing to select elements
    bigF_k_mq0 = bigF_k_mq_reshaped[i, j, betaoldt, 0]
    bigF_k_mq0not = bigF_k_mq_reshaped[i, j, notbetaoldt, 0]
    bigF_k_mq1 = bigF_k_mq_reshaped[i, j, betaoldt, 1]
    bigF_k_mq1not = bigF_k_mq_reshaped[i, j, notbetaoldt, 1]
    bigF_kmq_qf = bigF_kmq_q_reshaped[i, j, alphaoldt, betaoldt]
    # Reshape the arrays
    bigF_k_mq_reshaped = bigF_k_mq.reshape(n_exch, 2, 2)
    bigF_kmq_q_reshaped = bigF_kmq_q.reshape(n_exch, 2, 2)

    # ----------------------------

    # Reshape the arrays

    # k goes over batch and l over n_exch here
    # start_time0 = time.perf_counter()
    # Expand potentials to match the shape of other tensors
    potentials_expanded = potentials.repeat(n_exch // n_q)
    potentials_expanded = potentials_expanded.expand(batch, -1)

    # Compute all terms at once
    # enfull = torch.einsum(
    #     "ij,ji,ij->ij",
    #     potentials_expanded,
    #     bigF_kmq_qf,
    #     torch.einsum("ij,ji->ij", psis_p_d1, bigF_k_mq0)
    #     - torch.einsum("ij,ji->ij", psis_p2_d1, bigF_k_mq0not)
    #     + torch.einsum("ij,ji->ij", psis_p_d2, bigF_k_mq1)
    #     - torch.einsum("ij,ji->ij", psis_p2_d2, bigF_k_mq1not),
    # )

    enHFs = torch.einsum(
        "ij,ji,ij->ij",
        potentials_expanded,
        bigF_kmq_qf,
        torch.einsum("ij,ji->ij", psis_p_d1HFs, bigF_k_mq0)
        - torch.einsum("ij,ji->ij", psis_p2_d1HFs, bigF_k_mq0not)
        + torch.einsum("ij,ji->ij", psis_p_d2HFs, bigF_k_mq1)
        - torch.einsum("ij,ji->ij", psis_p2_d2HFs, bigF_k_mq1not),
    )
    enssp = torch.einsum(
        "ij,ji,ij->ij",
        potentials_expanded,
        bigF_kmq_qf,
        torch.einsum("ij,ji->ij", psis_p_d1ssp, bigF_k_mq0)
        - torch.einsum("ij,ji->ij", psis_p2_d1ssp, bigF_k_mq0not)
        + torch.einsum("ij,ji->ij", psis_p_d2ssp, bigF_k_mq1)
        - torch.einsum("ij,ji->ij", psis_p2_d2ssp, bigF_k_mq1not),
    )

    resultsnl = []
    resultsnl2 = []
    resultsnl3 = []

    # If you need to sum over the n_exch dimension to get a result per batch:
    # cp_k0 = enfull.sum(dim=1)
    cp_k1 = enssp.sum(dim=1)
    cp_k2 = enHFs.sum(dim=1)
    cp_k0 = cp_k1 + cp_k2
    resultsnl.append(cp_k0)
    resultsnl2.append(cp_k1)
    resultsnl3.append(cp_k2)

    E = torch.stack(results).sum(dim=(0, 1))
    E2 = torch.stack(results2).sum(dim=(0, 1))
    E3 = torch.stack(results3).sum(dim=(0, 1))

    return (
        E,
        E2,
        E3,
        resultsnl,
        resultsnl2,
        resultsnl3,
        occupation,
    )


@torch.no_grad()
def intnl(
    pot,
    bigF_k_mq,
    bigF_kmq_q,
    a_index,
    b_index,
    psis_p_d1,
    psis_p2_d1,
    psis_p_d2,
    psis_p2_d2,
):
    """
    Calculates addTo1 and addTo2 based on provided parameters.

    Args:
        pot: A numerical value.
        bigF_kmq_q: A 2D list or NumPy array.
        a_index: An integer index.
        b_index: An integer index.
        psis_p_d1: A numerical value.
        psis_p2_d1: A numerical value.
        psis_p_d2: A numerical value.
        psis_p2_d2: A numerical value.

    Returns:
        A tuple containing (addTo1, addTo2).
    """

    # Calculate intermediate values for clarity
    term1 = psis_p_d1 * bigF_k_mq[b_index][0]
    term2 = psis_p2_d1 * bigF_k_mq[int(not b_index)][0]
    intermediate_result1 = term1 - term2

    term3 = psis_p_d2 * bigF_k_mq[b_index][1]
    term4 = psis_p2_d2 * bigF_k_mq[int(not b_index)][1]
    intermediate_result2 = term3 - term4

    # Calculate addTo1 and addTo2
    addTo1 = pot * bigF_kmq_q[a_index][b_index] * intermediate_result1
    addTo2 = pot * bigF_kmq_q[a_index][b_index] * intermediate_result2
    return (
        addTo1 + addTo2,
        addTo1 + addTo2,
        addTo1,
        intermediate_result1,
        intermediate_result2,
    )


def form_factors(k, q):
    """
    Vectorized form factors for interaction term in the Hamiltonian.
    Parameters
    ----------
    k : torch.Tensor
        Tensor of momenta in the BZ
    q : torch.Tensor
        Tensor of momenta in the potential V
    """
    f1 = torch.ones_like(k, dtype=torch.complex64)
    f2 = (0.9 * torch.sin(q) * (torch.sin(k) + torch.sin((k + q)))).to(torch.complex64)
    f3 = torch.zeros_like(k, dtype=torch.complex64)
    f4 = torch.zeros_like(k, dtype=torch.complex64)
    return f1, f2, f3, f4


def compute_phase(spin_pm, phase_idx):
    """


    Parameters
    ----------
    spin_pm : Tensor, (n, batch)
        +-1, sampled spin configurations
    phase_idx : Tensor, (n_op, n_phase)
        indices with either Y or Z acting on it
        additional -i and spin flip for Y are computed outside this function

    Returns
    -------
    O_{x, x'} : (n_op, batch)
        where x is given
        O_loc(x) = O_{x, x'} psi(x') / psi(x)
    """
    n, batch = spin_pm.shape
    # Unsqueeze organizes the (i,j) into 1 index from 1 to n. We just access the corresponding config in spin_pm
    spin_pm_relevant = spin_pm[
        phase_idx.unsqueeze(-1), torch.arange(batch)
    ]  # (n_op, n_phase, batch), +-1

    return spin_pm_relevant.prod(dim=1)  # (n_op, batch)
