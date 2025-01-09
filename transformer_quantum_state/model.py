# -*- coding: utf-8 -*-
"""

Adapted from https://github.com/pytorch/examples/blob/main/word_language_model/model.py

"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pos_encoding import TQSPositionalEncoding1D, TQSPositionalEncoding2D
from torch.nn import TransformerEncoderLayer

pi = np.pi
torch.nn.TransformerEncoder.enable_nested_tensor = True


class TransformerModel(nn.Module):
    """Container module with an encoder transformer module."""

    def __init__(
        self,
        system_sizes,
        param_dim,
        embedding_size,
        n_head,
        n_hid,
        n_layers,
        phys_dim=2,
        dropout=0.5,
        minibatch=None,
    ):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder
        except:
            raise ImportError(
                "TransformerEncoder module does not exist in PyTorch 1.1 or lower."
            )

        self.system_sizes = torch.tensor(
            system_sizes, dtype=torch.int64
        )  # (n_size, n_dim)
        assert (
            len(self.system_sizes.shape) == 2
        )  # Sanity check for desired dimension of system_size
        self.n = self.system_sizes.prod(
            dim=1
        )  # (n_size, ) # Product of all elements in the array
        self.n_size, self.n_dim = self.system_sizes.shape
        # max_system_size, _ = self.system_sizes.max(dim=0)  # (n_dim, )

        self.size_idx = None
        self.system_size = None
        self.param = None
        self.prefix = None
        self.param_dim = param_dim
        self.phys_dim = phys_dim

        # ---------------------------------- Input -------------------------------------------
        # 1. Spins will be one-hot encoded;
        # 2. Parameters Jj (h in this case) with a scaled one-hot vector.
        # input consists of: [phys_dim_0 phys_dim_1 log(system_size[0]) log(system_size[1]) parity(system_size) mask_token params]

        # input_dim = phys_dim + self.n_dim + 2 + param_dim
        input_dim = phys_dim + self.n_dim + param_dim + 2
        self.input_dim = input_dim
        # sequence consists of: [log(system_size[0]) log(system_size[1]) params spins]

        self.seq_prefix_len = self.n_dim + param_dim

        self.param_range = None
        self.n_head = n_head
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.dropout = dropout
        self.minibatch = minibatch
        self.embedding_size = embedding_size
        self.src_mask = None

        # Positional encodding
        pos_encoder = (
            TQSPositionalEncoding1D if self.n_dim == 1 else TQSPositionalEncoding2D
        )

        self.pos_encoder = pos_encoder(
            embedding_size, self.seq_prefix_len, dropout=dropout
        )

        # ------------------------- Passing to Transformer encoder layers ------
        encoder_layers = TransformerEncoderLayer(
            embedding_size, n_head, n_hid, dropout, dtype=torch.float
        )
        self.attention_weights = encoder_layers.state_dict()[
            "self_attn.out_proj.weight"
        ]
        self.attention_weightsout = encoder_layers.state_dict()[
            "self_attn.out_proj.weight"
        ]
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.encoder = nn.Linear(input_dim, embedding_size)
        self.amp_head = nn.Linear(embedding_size, phys_dim)
        # self.amp_head = nn.Linear(embedding_size, phys_dim, dtype=torch.float)
        self.phase_head = nn.Linear(embedding_size, phys_dim)
        # self.phase_head = nn.Linear(embedding_size, phys_dim, dtype=torch.float)
        self.init_weights()

    def set_param(self, system_size=None, param=None):
        self.size_idx = torch.randint(self.n_size, [])
        if system_size is None:
            self.system_size = self.system_sizes[self.size_idx]
        else:
            self.system_size = system_size
            self.size_idx = None
        if param is None:
            self.param = self.param_range[0] + torch.rand(self.param_dim) * (
                self.param_range[1] - self.param_range[0]
            )
        else:
            self.param = param
        self.prefix = self.init_seq()

    def init_seq(self):
        system_size = self.system_size
        param = self.param
        parity = (system_size % 2).to(
            torch.get_default_dtype()
        )  # (n_dim, ) # Parity indicates if system is even or odd
        size_input = torch.diag(system_size.log())  # (n_dim, n_dim)

        init = torch.zeros(self.seq_prefix_len, 1, self.input_dim)

        # sequence consists of: [log(system_size[0]) log(system_size[1]) params spins]
        # input consists of: [phys_dim_0 phys_dim_1 log(system_size[0]) log(system_size[1]) parity(system_size) mask_token params]

        init[
            : self.n_dim, :, self.phys_dim : self.phys_dim + self.n_dim
        ] = size_input.unsqueeze(
            1
        )  # (n_dim, 1, n_dim), unsqueeze transformes the vector into column
        init[: self.n_dim, :, self.phys_dim + self.n_dim] = parity.unsqueeze(
            1
        )  # (n_dim, 1)

        param_offset = self.phys_dim + self.n_dim + 2
        for i in range(self.param_dim):
            init[self.n_dim + i, :, param_offset + i] += param[i]
        return init  # (prefix_len, 1, input_dim)

    def wrap_spins(self, spins):
        """
        prefix: (prefix_len, 1, input_dim)
        spins: (n, batch)
        """
        # print("spins initial", spins.shape)
        prefix = self.prefix
        prefix_len, _, input_dim = prefix.shape
        n, batch = spins.shape
        src = torch.zeros(prefix_len + n, batch, input_dim)
        src[:prefix_len, :, :] = prefix
        src[prefix_len:, :, : self.phys_dim] = F.one_hot(
            spins.to(torch.int64), num_classes=self.phys_dim
        )
        # print("spins final", src.shape)
        # Remove the prefix
        spins_onehot = src[prefix_len:, :, : self.phys_dim]

        # Convert one-hot encoding back to indices
        spins0 = torch.argmax(spins_onehot, dim=-1)
        # print("spins initial0", spins0.shape)
        # print(torch.all(spins == spins0).item())

        return src

    @staticmethod
    def _generate_square_subsequent_mask(sz):
        """Create square mask matrix with upper triangle = -inf and lower triangle with 0
        sz is the length of sequences from spins"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        #
        # # Save the current plot as an image
        # plt.savefig(f'plot_{i}.png')
        #
        # # Clear the plot to create a new one
        # plt.clf()
        #
        # # Read the saved image and append it to the list
        # images.append(imageio.imread(f'plot_{i}.png'))

        return mask

    def init_weights(self):
        # Parameters start with a mean 0 and variance 0.1 (normal distribution, initialization)
        # Try xavier initialization in the future (TODO)
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.encoder.bias)
        nn.init.uniform_(self.amp_head.weight, -initrange, initrange)
        nn.init.zeros_(self.amp_head.bias)
        nn.init.uniform_(self.phase_head.weight, -initrange, initrange)
        nn.init.zeros_(self.phase_head.bias)

    @staticmethod
    def softsign(x):
        """
        Defined in Hibat-Allah, Mohamed, et al.
                    "Recurrent neural network wave functions."
                    Physical Review Research 2.2 (2020): 023358.
        Used as the activation function on the phase output
        range: (-2pi, 2pi)
        NOTE: this function outputs 2\phi, where \phi is the phase
              an additional factor of 2 is included, to ensure \phi\in(-\pi, \pi)
        """
        return 2 * pi * (1 + x / (1 + x.abs()))

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        self.layers = nn.ModuleList(
            [EncoderBlock(**block_args) for _ in range(num_layers)]
        )
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps

    def forward(self, spins, compute_phase=True):
        """
        Get a results vector half-filled with amplitude and the other half with phases
        """
        # src: (seq, batch, input_dim)
        # use_symmetry: has no effect in this function
        # only included to be consistent with the symmetric version

        # One hot incode the spins s_i
        src = self.wrap_spins(spins)

        # Create masked input
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            mask = self._generate_square_subsequent_mask(len(src)).to(src.device)
            self.src_mask = mask

        system_size = src[
            : self.n_dim, 0, self.phys_dim : self.phys_dim + self.n_dim
        ].diag()  # (n_dim, )
        system_size = system_size.exp().round().to(torch.int64)  # (n_dim, )

        result = []
        if self.minibatch is None:
            src = self.encoder(src) * math.sqrt(
                self.embedding_size
            )  # (seq, batch, embedding)
            # src = src + self.pos_embedding[:len(src)]  # (seq, batch, embedding)
            src = self.pos_encoder(src, system_size)  # (seq, batch, embedding)
            output = self.transformer_encoder(
                src, self.src_mask
            )  # (seq, batch, embedding)

            psi_output = output[
                self.seq_prefix_len - 1 :
            ]  # only use the physical degrees of freedom
            amp = F.log_softmax(
                self.amp_head(psi_output), dim=-1
            )  # (seq, batch, phys_dim)

            result.append(amp)
            if compute_phase:
                phase = self.softsign(
                    self.phase_head(psi_output)
                )  # (seq, batch, phys_dim)
                result.append(phase)

        else:
            batch = src.shape[1]
            minibatch = self.minibatch
            repeat = int(np.ceil(batch / minibatch))
            amp = []
            phase = []
            for i in range(repeat):
                src_i = src[:, i * minibatch : (i + 1) * minibatch]
                src_i = self.encoder(src_i) * math.sqrt(
                    self.embedding_size
                )  # (seq, batch, embedding)
                # src_i = src_i + self.pos_embedding[:len(src_i)]  # (seq, batch, embedding)
                src_i = self.pos_encoder(src_i, system_size)  # (seq, batch, embedding)
                output_i = self.transformer_encoder(
                    src_i, self.src_mask
                )  # (seq, batch, embedding)

                psi_output = output_i[
                    self.seq_prefix_len - 1 :
                ]  # only use the physical degrees of freedom
                amp_i = F.log_softmax(
                    self.amp_head(psi_output), dim=-1
                )  # (seq, batch, phys_dim)
                # amp_i = np.log(np.cosh(
                #     self.amp_head(psi_output)
                # ))  # (seq, batch, phys_dim)
                amp.append(amp_i)
                if compute_phase:
                    phase_i = self.softsign(
                        self.phase_head(psi_output)
                    )  # (seq, batch, phys_dim)
                    # phase_i = torch.log(np.cosh(self.phase_head(psi_output)))
                    # phase_i = self.phase_head(psi_output)
                    phase.append(phase_i)
            amp = torch.cat(amp, dim=1)
            result.append(amp)
            if compute_phase:
                phase = torch.cat(phase, dim=1)
                result.append(phase)
        return result, psi_output
