#!/usr/bin/env python
#
# ------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates.
# To view a copy of this license, visit
# https://github.com/facebookresearch/detectron2/blob/main/LICENSE
# ------------------------------------------------------------------------------
#
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
"""
Training script using the new "LazyConfig" python config files.
 
This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import argparse
import logging
import os.path as osp
from contextlib import ExitStack
from typing import MutableSequence
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import create_ddp_model, default_argument_parser, hooks, launch
from detectron2.evaluation import print_csv_format
from detectron2.utils import comm
from detectron2.utils.events import JSONWriter
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from iopath.common.s3 import S3PathHandler
from omegaconf import OmegaConf

from module.checkpoint import PDDMCheckpointer
from module.config import auto_scale_workers, instantiate_PDDM
from module.engine.defaults import default_setup, get_dataset_from_loader, get_model_from_module
from module.engine.hooks import EvalHook
from module.engine.train_loop import AMPTrainer, SimpleTrainer
from module.evaluation import inference_on_dataset
from module.utils.events import CommonMetricPrinter, WandbWriter, WriterStack
from torch.optim.lr_scheduler import OneCycleLR
import torch
torch.backends.cudnn.benchmark = True

import os
PathManager.register_handler(S3PathHandler())

logger = logging.getLogger("PDDM")

def default_writers(cfg):
    """
    Build a list of :class:`EventWriter` to be used.
    It now consists of a :class:`CommonMetricPrinter`,
    :class:`TensorboardXWriter` and :class:`JSONWriter`.

    Args:
        output_dir: directory to store JSON metrics and tensorboard events
        max_iter: the total number of iterations

    Returns:
        list[EventWriter]: a list of :class:`EventWriter` objects.
    """
    if "log_dir" in cfg.train:
        log_dir = cfg.train.log_dir
    else:
        log_dir = cfg.train.output_dir
    PathManager.mkdirs(log_dir)
    ret = [
        # It may not always print what you want to see, since it prints "common" metrics only.
        CommonMetricPrinter(
            cfg.train.max_iter, run_name=osp.join(cfg.train.run_name, cfg.train.run_tag)
        ),
        JSONWriter(osp.join(log_dir, "metrics.json")),
    ]
    if cfg.train.wandb.enable_writer:
        ret.append(
            WandbWriter(
                max_iter=cfg.train.max_iter,
                run_name=osp.join(cfg.train.run_name, cfg.train.run_tag),
                output_dir=log_dir,
                project=cfg.train.wandb.project,
                config=OmegaConf.to_container(cfg, resolve=False),
                resume=cfg.train.wandb.resume,
            )
        )

    return ret


class InferenceRunner:
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model

    def __call__(self, final_iter=False, next_iter=0):
        return do_test(self.cfg, self.model, final_iter=final_iter, next_iter=next_iter)


def do_test(cfg, model, *, final_iter=False, next_iter=0):
    all_ret = dict()
    # make a copy incase we modify it every time calling do_test
    cfg = OmegaConf.create(cfg)

    # BC for detectron

    import sys
    sys.path.append("./EMSANet/lib/nicr-multitask-scene-analysis")
    sys.path.append("./EMSANet/lib/nicr-scene-analysis-datasets")


    from EMSANet.emsanet.data import get_datahelper_pddm
    from EMSANet.emsanet.preprocessing import get_preprocessor_pddm
    datahelper = get_datahelper_pddm()
    datahelper.set_valid_preprocessor(
        get_preprocessor_pddm(
            dataset=datahelper.datasets_valid,
            phase='test'
        )
    )
    loader = datahelper.valid_dataloaders


    inference_model = create_ddp_model(model)

    ret = inference_on_dataset(
        inference_model,
        loader,
        None,
        use_amp=cfg.train.amp.enabled,
    )
    print_csv_format(ret)
    all_ret.update(ret)

    logger.info("Evaluation results for all tasks:")
    print_csv_format(all_ret)


    return all_ret


def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    logger = logging.getLogger("PDDM")
    cfg.train.wandb.resume = args.resume and PDDMCheckpointer.has_checkpoint_in_dir(
        cfg.train.output_dir
    )
    if comm.is_main_process():
        writers = default_writers(cfg)
    comm.synchronize()
    with ExitStack() as stack:
        stack.enter_context(
            WriterStack(
                logger=logger,
                writers=writers if comm.is_main_process() else None,
            )
        )
        logger.info(f"Wandb resume: {cfg.train.wandb.resume}")
        logger.info(f"Config:\n{LazyConfig.to_py(cfg)}")

        model = instantiate_PDDM(cfg.model) 
        model.to(cfg.train.device)

        cfg.optimizer.params.model = model

        control_model_param = []
        model_param = []
        for name, prm in model.named_parameters():
            if not prm.requires_grad:
                continue
            if "ldm_extractor" in name:
                control_model_param.append(prm)
            else:
                model_param.append(prm)
        
        optim = torch.optim.AdamW(
                                params=[{'params':model_param},
                                        {'params':control_model_param,'lr':6e-6}
                                        ],
                                lr=1e-4,
                                weight_decay=0.05,
                            )

        import sys
        sys.path.append("./EMSANet/lib/nicr-multitask-scene-analysis")
        sys.path.append("./EMSANet/lib/nicr-scene-analysis-datasets")
        from EMSANet.emsanet.data import get_datahelper_pddm
        from EMSANet.emsanet.preprocessing import get_preprocessor_pddm
        datahelper = get_datahelper_pddm()
        datahelper.set_train_preprocessor(
            get_preprocessor_pddm(
                dataset=datahelper.dataset_train,
                phase='train'
            )
        )
        train_loader = datahelper.train_dataloader


        if cfg.train.amp.enabled:
            model = create_ddp_model(model, **cfg.train.ddp)
            trainer = AMPTrainer(model, train_loader, optim, grad_clip=cfg.train.grad_clip)
        else:
            model = create_ddp_model(model, **cfg.train.ddp)
            trainer = SimpleTrainer(model, train_loader, optim, grad_clip=cfg.train.grad_clip)
        checkpointer = PDDMCheckpointer(
            model,
            cfg.train.output_dir,
            trainer=trainer,
        )
        # set wandb resume before create writer
        cfg.train.wandb.resume = args.resume and checkpointer.has_checkpoint()
        logger.info(f"Wandb resume: {cfg.train.wandb.resume}")



        trainer.register_hooks(
            [
                hooks.IterationTimer(),
                hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
                hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
                if comm.is_main_process()
                else None,
                EvalHook(cfg.train.eval_period, InferenceRunner(cfg, model)),
                hooks.BestCheckpointer(checkpointer=checkpointer, **cfg.train.best_checkpointer)
                if comm.is_main_process() and "best_checkpointer" in cfg.train
                else None,
                hooks.PeriodicWriter(
                    writers=writers,
                    period=cfg.train.log_period,
                )
                if comm.is_main_process()
                else None,
            ]
        )
        comm.synchronize()
        checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
        if args.resume and checkpointer.has_checkpoint():
            start_iter = trainer.iter + 1
        else:
            start_iter = 0
    comm.synchronize()
    trainer.train(start_iter, cfg.train.max_iter)

def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg.train.run_name = (
        "${train.cfg_name}_bs" + f"x{comm.get_world_size()}"
    )
    if hasattr(args, "reference_world_size") and args.reference_world_size:
        cfg.train.reference_world_size = args.reference_world_size
    cfg = auto_scale_workers(cfg, comm.get_world_size())
    cfg.train.cfg_name = osp.splitext(osp.basename(args.config_file))[0]
    if hasattr(args, "output") and args.output:
        cfg.train.output_dir = args.output
    else:
        cfg.train.output_dir = osp.join("output", cfg.train.run_name)
    if hasattr(args, "tag") and args.tag:
        cfg.train.run_tag = args.tag
        cfg.train.output_dir = osp.join(cfg.train.output_dir, cfg.train.run_tag)
    if hasattr(args, "wandb") and args.wandb:
        cfg.train.wandb.enable_writer = args.wandb
        cfg.train.wandb.enable_visualizer = args.wandb
    if hasattr(args, "amp") and args.amp:
        cfg.train.amp.enabled = args.amp
    if hasattr(args, "init_from") and args.init_from:
        cfg.train.init_checkpoint = args.init_from
    cfg.train.log_dir = cfg.train.output_dir
    if hasattr(args, "log_tag") and args.log_tag:
        cfg.train.log_dir = osp.join(cfg.train.log_dir, args.log_tag)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)
    logger = setup_logger(cfg.train.log_dir, distributed_rank=comm.get_rank(), name="PDDM")

    logger.info(f"Running with config:\n{LazyConfig.to_py(cfg)}")

    if args.eval_only:
        model = instantiate_PDDM(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        PDDMCheckpointer(model, cfg.train.output_dir).resume_or_load(
            cfg.train.init_checkpoint, resume=args.resume
        )
        with ExitStack() as stack:
            stack.enter_context(
                WriterStack(
                    logger=logger,
                    writers=default_writers(cfg) if comm.is_main_process() else None,
                )
            )
            logger.info(do_test(cfg, model, final_iter=True))
        comm.synchronize()
    else:
        do_train(args, cfg)


def parse_args():
    parser = argparse.ArgumentParser(
        parents=[default_argument_parser()],
        add_help=False,
    )

    parser.add_argument(
        "--output",
        type=str,
        help="root of output folder, " "the full path is <output>/<model_name>/<tag>",
    )
    parser.add_argument("--init-from", type=str, help="init from the given checkpoint")
    parser.add_argument("--tag", default="default", type=str, help="tag of experiment")
    parser.add_argument("--log-tag", type=str, help="tag of experiment")
    parser.add_argument("--wandb", action="store_true", help="Use W&B to log experiments")
    parser.add_argument("--amp", action="store_true", help="Use AMP for mixed precision training")
    parser.add_argument("--reference-world-size", "--ref", type=int)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
