# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from functools import reduce

from mjlab.rsl_rl.networks.belief_encoder import RecurrentAttentionPolicy
from mjlab.rsl_rl.utils import resolve_nn_activation
from mjlab.rsl_rl.networks import MLP


class Teacher_wild(nn.Module):
    def __init__(self,
        input_extero: int,
        input_privi: int,
        output_dim: int | tuple[int] | list[int],
        hidden_dims: tuple[int] | list[int],
        model_cfg
        ):
        super().__init__()

        assert input_extero % 4 == 0
        self.height_dim = input_extero // 4

        # height encoder
        self.height_encoder = MLP(self.height_dim, 24, [80,60], activation="leakyrelu")

        # proprio/priviledged encoder
        self.privil_encoder = MLP(input_privi, 24, [64,32], activation="leakyrelu")

        # extero encoder
        # self.extero_encoder = nn.Sequential(
        #     nn.Linear(input_extero,128),
        #     activation_mod,
        # )

        # belief encoder
        self.belief_encoder = RecurrentAttentionPolicy.BeliefEncoder(model_cfg["belief_encoder"])

        # input = proprio + belief state
        self.network = MLP(model_cfg["policy"]["MLP"]["base_net"]["input"],
                           output_dim, [256,160,128], activation="leakyrelu")

    def forward(self, proprio_obs, privi_obs, extero_obs, hidden_state=None):
        B = extero_obs.shape[0]

        extero_split = extero_obs.view(B, 4, self.height_dim)

        encoded_legs = []
        for i in range(4):
            encoded_legs.append(self.height_encoder(extero_split[:, i]))

        encoded_extero = torch.cat(encoded_legs, dim=-1)

        belief_out = self.belief_encoder(proprio_obs, encoded_extero, hidden_state)

        belief_state = belief_out["belief_state"]
        next_hidden = belief_out["recurrent_hidden"]

        priv_latent = self.privil_encoder(privi_obs)

        fused = torch.cat((belief_state, priv_latent), dim=-1)

        action = self.network(fused)

        return action, next_hidden
