# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ODISE/blob/main/LICENSE
#
# Written by Jiarui Xu
#
# ------------------------------------------------------------------------------
# Modifications for PDDM by Xinhua Xu
# ------------------------------------------------------------------------------

from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import MultiStepParamScheduler

from ..common.models.pddm import model
from ..common.train import train
from ..common.optim import AdamW as optimizer
  
train.max_iter = 60_000
train.grad_clip = 0.01
train.checkpointer.period = 1000




lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1],

        milestones=[40000],
        num_updates=60000,
    ),
    # for warmup length we adopted COCO LSJ setting
    warmup_length=250 / 60000,
    warmup_factor=0.067,
)

optimizer.lr = 1e-4
optimizer.weight_decay = 0.05


