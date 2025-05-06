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
from module.modeling.meta_arch.ldm import LdmImplicitCaptionerExtractor
from module.modeling.backbone.feature_extractor import FeatureExtractorBackbone
from .mask_generator_with_label import model
 
# 定义一个函数，用于设置模型参数
model.backbone = L(FeatureExtractorBackbone)(
    # 使用LdmImplicitCaptionerExtractor作为特征提取器
    feature_extractor=L(LdmImplicitCaptionerExtractor)(
        # 设置编码器块索引
        encoder_block_indices=(5, 7),
        # 设置UNet块索引
        unet_block_indices=(2, 5, 8, 11),
        # 设置解码器块索引
        decoder_block_indices=(2, 5),
        # 设置步长
        steps=(0,),
        # 设置可学习的时间嵌入
        learnable_time_embed=True,
        # 设置时间步长
        num_timesteps=1,
        # 设置clip模型名称
        clip_model_name="ViT-L-14-336",
    ),
    # 设置输出特征
    out_features=["s2", "s3", "s4", "s5"],
    # 设置使用checkpoint
    use_checkpoint=True,
    # 设置滑动训练
    slide_training=True,
)
# 设置sem_seg_head的像素解码器的输出特征
model.sem_seg_head.pixel_decoder.transformer_in_features = ["s3", "s4", "s5"]
# 设置clip_head的alpha和beta
#model.clip_head.alpha = 0.3
#model.clip_head.beta = 0.7