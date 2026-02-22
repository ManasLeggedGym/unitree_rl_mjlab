# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from functools import reduce

from mjlab.rsl_rl.utils import resolve_nn_activation
from mjlab.rsl_rl.networks import MLP


class Teacher_wild(nn.Module):
    def __init__(self,
        input_extero: int,
        input_privi: int,
        output_dim: int | tuple[int] | list[int],
        hidden_dims: tuple[int] | list[int],
        activation: str = "elu",
        last_activation: str | None = None,
        ):
        super().__init__()
        activation_mod = resolve_nn_activation(activation)
        last_activation_mod = resolve_nn_activation(last_activation) if last_activation is not None else None
        #proprio encoder
        self.privil_encoder = nn.Sequential(
            nn.Linear(input_privi,128),
            activation_mod,
        )
        #extero encoder
        self.extero_encoder = nn.Sequential(
            nn.Linear(input_extero,128),
            activation_mod,
        )

        self.network = MLP(256,output_dim,hidden_dims,activation,last_activation)
    
    def forward(self, privi_obs,extero_obs):
        pri = self.privil_encoder(privi_obs)
        ext = self.extero_encoder(extero_obs)
        return self.network.forward(torch.cat((pri,ext),dim=-1))
 