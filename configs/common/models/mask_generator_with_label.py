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
from detectron2.data import MetadataCatalog

from module.modeling.meta_arch.pddm import (
    PseudoDepthDiffusionModel,
    PDDMMultiScaleMaskedTransformerDecoder,
    PooledMaskEmbed,
    PseudoClassEmbed
) 
from Mask2Former.mask2former.modeling.meta_arch.mask_former_head import MaskFormerHead
from Mask2Former.mask2former.modeling.criterion import SetCriterion
from Mask2Former.mask2former.modeling.matcher import HungarianMatcher
from Mask2Former.mask2former.modeling.pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
 

model = L(PseudoDepthDiffusionModel)(
    sem_seg_head=L(MaskFormerHead)(
        ignore_value=255,
        num_classes=40,
        pixel_decoder=L(MSDeformAttnPixelDecoder)(
            conv_dim=256,
            mask_dim=256,
            norm="GN",
            transformer_dropout=0.0,
            transformer_nheads=8,
            transformer_dim_feedforward=1024,
            transformer_enc_layers=6,
            transformer_in_features=["s3", "s4", "s5"],
            common_stride=4,
        ),
        loss_weight=1.0,
        transformer_in_feature="multi_scale_pixel_decoder",
        transformer_predictor=L(PDDMMultiScaleMaskedTransformerDecoder)(
            class_embed=L(PseudoClassEmbed)(num_classes="${..num_classes}"),
            hidden_dim=256,
            post_mask_embed=L(PooledMaskEmbed)(
                hidden_dim="${..hidden_dim}",
                mask_dim="${..mask_dim}",
                projection_dim="${..mask_dim}",
            ),
            in_channels="${..pixel_decoder.conv_dim}",
            mask_classification=True,
            num_classes="${..num_classes}",
            num_queries="${...num_queries}",
            nheads=8,
            dim_feedforward=2048,
            dec_layers=9,
            pre_norm=False,
            enforce_input_project=False,
            mask_dim=256,
        ),
    ),
    criterion=L(SetCriterion)(
        num_layers="${..sem_seg_head.transformer_predictor.dec_layers}",
        class_weight=2.0,
        mask_weight=5.0,
        dice_weight=5.0,
        num_classes="${..sem_seg_head.num_classes}",
        matcher=L(HungarianMatcher)(
            cost_class="${..class_weight}",
            cost_mask="${..mask_weight}",
            cost_dice="${..dice_weight}",
            num_points="${..num_points}",
        ),
        eos_coef=0.1,
        losses=["masks"],
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
    ),
    num_queries=40,
    object_mask_threshold=0.0,
    overlap_threshold=0.8,
    metadata=L(MetadataCatalog.get)(name="nyuv2"),
    size_divisibility=64,
    sem_seg_postprocess_before_inference=True,
    pixel_mean=[0.0, 0.0, 0.0],
    pixel_std=[255.0, 255.0, 255.0],
    semantic_on=True,
    instance_on=True,
    panoptic_on=True,
    test_topk_per_image=40,
)