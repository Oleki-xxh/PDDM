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

import logging
import numpy as np
import operator
from collections import OrderedDict
from typing import Any, Mapping
import diffdist.functional as diff_dist
import torch
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils import comm
from detectron2.utils.memory import retry_if_cuda_oom
from Mask2Former.mask2former.maskformer_model import MaskFormer
from Mask2Former.mask2former.modeling.transformer_decoder.mask2former_transformer_decoder import (
    MLP,
    MultiScaleMaskedTransformerDecoder,
)
from torch import nn
from torch.nn import functional as F

from .helper import ensemble_logits_with_labels

logger = logging.getLogger(__name__)


# Ref:https://stackoverflow.com/questions/27049998/convert-a-mixed-nested-list-to-a-nested-tuple
def to_tuple(lst):
    return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)


@torch.no_grad()
def _concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if comm.get_world_size() == 1:
        return tensor
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def get_world_batch_sizes(batch_size: int, device):
    batch_size = torch.as_tensor([batch_size], dtype=torch.long, device=device)
    global_batch_sizes = _concat_all_gather(batch_size)
    return global_batch_sizes

 
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors, with dynamic batch size.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if comm.get_world_size() == 1:
        return tensor
    global_batch_sizes = get_world_batch_sizes(tensor.shape[0], tensor.device)
    max_batch_size = global_batch_sizes.max().item()
    padded_tensor = torch.zeros(
        max_batch_size, *tensor.shape[1:], device=tensor.device, dtype=tensor.dtype
    )
    padded_tensor[: tensor.shape[0]] = tensor

    tensors_gather = [
        torch.ones((max_batch_size, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
        for _ in range(comm.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, padded_tensor, async_op=False)

    results = []
    for i, batch_size in enumerate(global_batch_sizes):
        results.append(tensors_gather[i][:batch_size])

    output = torch.cat(results, dim=0)
    return output


def dist_collect(tensor):
    """
    Performs all_gather operation on the provided tensors, with dynamic batch size.
    Use diff_dist to get gradient
    """
    if comm.get_world_size() == 1:
        return tensor
    global_batch_sizes = get_world_batch_sizes(tensor.shape[0], tensor.device)
    max_batch_size = global_batch_sizes.max().item()
    padded_tensor = torch.zeros(
        max_batch_size, *tensor.shape[1:], device=tensor.device, dtype=tensor.dtype
    )
    padded_tensor[: tensor.shape[0]] = tensor

    tensors_gather = [
        torch.ones((max_batch_size, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
        for _ in range(comm.get_world_size())
    ]
    tensors_gather = diff_dist.all_gather(tensors_gather, padded_tensor)

    results = []
    for i, batch_size in enumerate(global_batch_sizes):
        results.append(tensors_gather[i][:batch_size])

    output = torch.cat(results, dim=0)
    return output


class PDDM(MaskFormer):
    def ignored_state_dict(self, destination=None, prefix=""):
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        for name, module in self.named_modules(remove_duplicate=True):
            if module is self:
                continue
            if module is not None and hasattr(module, "ignored_state_dict"):
                module.ignored_state_dict(destination, prefix + name + ".")
        return destination

    def _open_state_dict(self):
        return {
            "sem_seg_head.num_classes": self.sem_seg_head.num_classes,
            "metadata": self.metadata,
            "test_topk_per_image": self.test_topk_per_image,
            "semantic_on": self.semantic_on,
            "panoptic_on": self.panoptic_on,
            "instance_on": self.instance_on,
        }

    def _save_open_state_dict(self, destination, prefix):
        for k, v in self._open_state_dict().items():
            destination[prefix + k] = v

    def open_state_dict(self, destination=None, prefix=""):
        if destination is None:
            destination = OrderedDict()
        self._save_open_state_dict(destination, prefix)
        for name, module in self.named_modules(remove_duplicate=True):
            if module is self:
                continue
            if module is not None and hasattr(module, "open_state_dict"):
                module.open_state_dict(destination, prefix + name + ".")
        return destination

    def load_open_state_dict(self, state_dict: Mapping[str, Any]):
        for k, v in state_dict.items():
            # handle nested modules
            if len(k.rsplit(".", 1)) == 2:
                prefix, suffix = k.rsplit(".", 1)
                operator.attrgetter(prefix)(self).__setattr__(suffix, v)
            else:
                self.__setattr__(k, v)
            assert operator.attrgetter(k)(self) == v, f"{k} is not loaded correctly"


class PseudoDepthDiffusionModel(PDDM):
    def __init__(
        self,
        *,
        category_head=None,
        clip_head=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.category_head = category_head
        self.clip_head = clip_head

    def forward(self, batched_inputs):
        images = batched_inputs['rgb'].to(self.device)

        name_list = batched_inputs['identifier']
        depth1 = batched_inputs['depth_anything'].to(self.device)
        depth2 = batched_inputs['depth_meta'].to(self.device)
        depth3 = batched_inputs['depth_mari'].to(self.device)

        denormalized_images = images

        
        features = self.backbone(images,depth1,depth2,depth3,name_list)
        outputs = self.sem_seg_head(features)
        outputs["images"] = denormalized_images

        if self.training:
            targets = batched_inputs["semantic"]
            targets_N_H_W = targets
            targets_N_H_W = targets_N_H_W.cuda()


            N = outputs["pred_masks"].shape[1]
            targets = targets.to(self.device)
            H,W = targets.shape[-2:]
            N_range = torch.arange(1, N+1).view(-1,1,1).to(targets.device)  # shape: (N,1,1)
            targets_expanded = targets.view(targets.shape[0],1,H,W).expand(-1,N,-1,-1)  # shape: (batch_size, N, H, W)
            targets = (targets_expanded == N_range)  # shape: (batch_size, N, H, W)

            losses = self.criterion(outputs, targets,targets_N_H_W)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    losses.pop(k)
            
            return losses
        else:
            mask_pred_results = outputs["pred_masks"]

            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.shape[-2], images.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            del outputs

            mask_pred_results = F.softmax(mask_pred_results, dim=1)
            score, idx = torch.max(mask_pred_results, dim=1)
            processed_results = idx

            return processed_results



class PDDMMultiScaleMaskedTransformerDecoder(MultiScaleMaskedTransformerDecoder):
    def __init__(
        self,
        *,
        class_embed=None,
        mask_embed=None,
        post_mask_embed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert self.mask_classification

        if class_embed is not None:
            self.class_embed = class_embed
        if mask_embed is not None:
            self.mask_embed = mask_embed
        if post_mask_embed is not None:
            assert mask_embed is None
        self.post_mask_embed = post_mask_embed

    def forward(self, x, mask_features, mask=None, *, inputs_dict=None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(
                self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None]
            )

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        # predictions_class = []
        predictions_mask = []
        predictions_extra_results = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask, extra_results = self.forward_prediction_heads(
            output, mask_features, attn_mask_target_size=size_list[0], inputs_dict=inputs_dict
        )
        # predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        predictions_extra_results.append(extra_results)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output,
                src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index],
                query_pos=query_embed,
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](output)

            outputs_class, outputs_mask, attn_mask, extra_results = self.forward_prediction_heads(
                output,
                mask_features,
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels],
                inputs_dict=inputs_dict,
            )
            # predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_extra_results.append(extra_results)

        # assert len(predictions_class) == self.num_layers + 1

        out = {
            "pred_masks": predictions_mask[-1],
            "aux_outputs": self._set_aux_loss(None, predictions_mask
            ),
        }

        # adding extra_results to out and out["aux_outputs"]
        for k in predictions_extra_results[-1].keys():
            out[k] = predictions_extra_results[-1][k]
            for i in range(len(predictions_extra_results) - 1):
                out["aux_outputs"][i][k] = predictions_extra_results[i][k]

        return out

    def forward_prediction_heads(
        self, output, mask_features, attn_mask_target_size, *, inputs_dict=None
    ):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)

        extra_results = dict()

        mask_embed_results = self.mask_embed(decoder_output)
        if isinstance(mask_embed_results, dict):
            mask_embed = mask_embed_results.pop("mask_embed")
            extra_results.update(mask_embed_results)
        # BC
        else:
            mask_embed = mask_embed_results

        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        if self.post_mask_embed is not None:
            post_mask_embed_results = self.post_mask_embed(
                decoder_output, mask_embed, mask_features, None, outputs_mask
            )

            if "outputs_mask" in post_mask_embed_results:
                outputs_mask = post_mask_embed_results.pop("outputs_mask")

            extra_results.update(post_mask_embed_results)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(
            outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False
        )
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend,
        # while ``False`` values will be unchanged.
        attn_mask = (
            attn_mask.sigmoid()
            .flatten(2)
            .unsqueeze(1)
            .repeat(1, self.num_heads, 1, 1)
            .flatten(0, 1)
            < 0.5
        ).bool()
        attn_mask = attn_mask.detach()

        return None, outputs_mask, attn_mask, extra_results




class PseudoClassEmbed(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        # predict as foreground only
        fg_logits = torch.ones((*x.shape[:-1], self.num_classes), dtype=x.dtype, device=x.device)
        bg_logits = torch.zeros((*x.shape[:-1], 1), dtype=x.dtype, device=x.device)
        logits = torch.cat([fg_logits, bg_logits], dim=-1)
        return logits


class MaskPooling(nn.Module):
    def __init__(
        self,
        hard_pooling=True,
        mask_threshold=0.5,
    ):
        super().__init__()
        # if the pooling is hard, it's not differentiable
        self.hard_pooling = hard_pooling
        self.mask_threshold = mask_threshold

    def extra_repr(self) -> str:
        return f"hard_pooling={self.hard_pooling}\n" f"mask_threshold={self.mask_threshold}\n"

    def forward(self, x, mask):
        """
        Args:
            x: [B, C, H, W]
            mask: [B, Q, H, W]
        """

        assert x.shape[-2:] == mask.shape[-2:]

        mask = mask.detach()

        mask = mask.sigmoid()

        if self.hard_pooling:
            mask = (mask > self.mask_threshold).to(mask.dtype)

        denorm = mask.sum(dim=(-1, -2), keepdim=True) + 1e-8

        mask_pooled_x = torch.einsum(
            "bchw,bqhw->bqc",
            x,
            mask / denorm,
        )

        output = {"mask_pooled_features": mask_pooled_x}

        return output


class PooledMaskEmbed(nn.Module):
    def __init__(
        self,
        hidden_dim,
        mask_dim,
        projection_dim,
        temperature=0.07,
    ):
        super().__init__()
        self.pool_proj = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))
        self.mask_embed = nn.Sequential(
            nn.LayerNorm(mask_dim), MLP(mask_dim, hidden_dim, projection_dim, 3)
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

        self.mask_pooling = MaskPooling()

    def forward(self, decoder_output, input_mask_embed, mask_features, pred_logits, pred_masks):
        """
        Args:
            decoder_output: [B, Q, C]
            input_mask_embed: [B, Q, C]
            mask_features: [B, C, H, W]
            pred_logits: [B, Q, K+1]
            pred_masks: [B, Q, H, W]
        """
        mask_pooled_x = self.mask_pooling(mask_features, pred_masks)
        mask_pooled_results = self.mask_pooling(mask_features, pred_masks)
        mask_pooled_x = mask_pooled_results["mask_pooled_features"]
        outputs_mask = mask_pooled_results.get("outputs_mask", None)

        mask_pooled_x = self.pool_proj(mask_pooled_x)

        mask_pooled_x += decoder_output

        mask_embed = self.mask_embed(mask_pooled_x)

        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)

        output = {
            "mask_embed": mask_embed,
            "mask_pooled_features": mask_pooled_x,
            "logit_scale": logit_scale,
        }

        if outputs_mask is not None:
            output["outputs_mask"] = outputs_mask

        return output


