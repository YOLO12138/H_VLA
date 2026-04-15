# Copyright 2025 hvla community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].


"""
HVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).  
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.  
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).  
"""

# Standard Library
import argparse
import json
import os
from pathlib import Path
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import re

# Third-Party Libraries
import torch
import torch.distributed as dist
import wandb
import yaml
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from hvla.training.trainer_utils.trainer_tools import normalize_dotlist_args
from hvla.model.framework import build_framework
from hvla.training.trainer_utils.trainer_tools import TrainerUtils
from hvla.training.trainer_utils.trainer_tools import build_param_lr_groups
from hvla.training.trainer_utils.config_tracker import wrap_config, AccessTrackedConfig

deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
accelerator.print(accelerator.state)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Overwatch =>> Wraps `logging.Logger`
from accelerate.logging import get_logger

logger = get_logger(__name__)


def load_fast_tokenizer():
    fast_tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)
    return fast_tokenizer


def setup_directories(cfg) -> Path:
    """create output directory and save config"""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        # create output directory and checkpoint directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

        # # save config
        # OmegaConf.save(cfg, output_dir / "config.yaml")
        # with open(output_dir / "config.yaml", "r") as f_yaml, open(output_dir / "config.json", "w") as f_json:
        #     yaml_cfg = yaml.safe_load(f_yaml)
        #     json.dump(yaml_cfg, f_json, indent=2)

    return output_dir


def build_model(cfg) -> torch.nn.Module:
    """build model framework"""
    logger.info(f"Loading Base VLM `{cfg.framework.qwenvl.base_vlm}` from ID/Path")
    model = build_framework(cfg)

    return model


# here changes need to 📦 encapsulate Dataloader
from hvla.dataloader import build_dataloader


def prepare_data(cfg, accelerator, output_dir) -> Tuple[DataLoader, DataLoader]:
    """prepare training data"""
    # VLA data loader
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader, vla_val_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py, accelerator=accelerator)

    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()

    return vla_train_dataloader, vla_val_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """set optimizer and scheduler"""
    # initialize optimizer
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    # print optimizer group info
    if dist.is_initialized() and dist.get_rank() == 0:
        for i, group in enumerate(optimizer.param_groups):
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # initialize learning rate scheduler
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,  # minimum learning rate
    )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator, vla_val_dataloader=None):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.vla_val_dataloader = vla_val_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        # training status tracking
        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()
    
    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # load pretrained weights (optimizer/scheduler/EMA deferred to after accelerator.prepare)
        self._init_checkpointing() # TODO merge with load pretrained weights

        # freeze parameters
        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)

        #  print model trainable parameters:
        self.print_trainable_parameters(self.model)

        # initialize distributed training components
        if self.vla_val_dataloader is not None:
            self.model, self.optimizer, self.vla_train_dataloader, self.vla_val_dataloader = self.setup_distributed_training(
                self.accelerator,
                self.model,
                self.optimizer,
                self.vla_train_dataloader,
                self.vla_val_dataloader,
            )
        else:
            self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
                self.accelerator,
                self.model,
                self.optimizer,
                self.vla_train_dataloader,
            )

        # Restore optimizer/scheduler/EMA now that the optimizer is DeepSpeed-wrapped
        self._restore_pending_training_state()
        if not getattr(self, "_training_state_restored", False):
            self._adjust_lr_scheduler_for_resume()

        self._init_wandb()


    def _adjust_lr_scheduler_for_resume(self):
        """根据已完成的步数调整学习率调度器状态"""
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            
            # 方法1: 直接模拟已完成的步数（适用于大多数调度器）
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            
            # 或者方法2: 对于某些调度器，可以直接设置最后步数
            # if hasattr(self.lr_scheduler, '_step_count'):
            #     self.lr_scheduler._step_count = self.completed_steps
            
            logger.info(f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}")

    def _calculate_total_batch_size(self):
        """calculate global batch size"""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """initialize Weights & Biases"""
        if self.accelerator.is_main_process:
            wandb_id_file = os.path.join(self.config.output_dir, "wandb_run_id.txt")
            is_resume = getattr(self.config.trainer, "is_resume", False)
            run_id = None
            if is_resume and os.path.exists(wandb_id_file):
                with open(wandb_id_file) as f:
                    run_id = f.read().strip()
            run = wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
                id=run_id,
                resume="allow" if run_id else None,
            )
            with open(wandb_id_file, "w") as f:
                f.write(run.id)

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # 获取预训练检查点和是否恢复训练的标志
        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint
        # TODO retinking resume and load from pretrained_checkpoint
        if is_resume:
            # 恢复训练状态
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                logger.info(f"Resuming training from checkpoint: {self.resume_from_checkpoint}, steps: {self.completed_steps}")

                # restore optimizer, scheduler, and EMA state if available
                training_state_path = resume_from_checkpoint.replace(
                    "_pytorch_model.pt", "_training_state.pt"
                ).replace(
                    "_model.safetensors", "_training_state.pt"
                )
                if os.path.exists(training_state_path):
                    # Defer optimizer/scheduler/EMA restore to after accelerator.prepare()
                    # because the checkpoint was saved from a DeepSpeed-wrapped optimizer whose
                    # state_dict format differs from plain AdamW.  We only load the file here;
                    # _restore_pending_training_state() applies it post-wrapping.
                    self._pending_training_state = torch.load(training_state_path, map_location="cpu", weights_only=False)
                    self._training_state_restored = False  # will be set True after apply
                    logger.info("Training state file loaded; will restore optimizer/scheduler/EMA after distributed setup.")
                else:
                    self._pending_training_state = None
                    self._training_state_restored = False
                    logger.warning("No _training_state.pt found; optimizer/EMA will start fresh.")

                return None
            else:
                logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
                self.completed_steps = 0

        # 加载预训练权重
        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0
    

    def _load_checkpoint(self, checkpoint_path):
        """load checkpoint"""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _restore_pending_training_state(self):
        """Apply scheduler/EMA state loaded in _init_checkpointing().

        Must be called AFTER accelerator.prepare() so self.optimizer is already
        DeepSpeed-wrapped.

        NOTE: DeepSpeed ZeRO optimizer state is NOT restored here.  The checkpoint
        saves only rank-0's optimizer shard (save is guarded by is_main_process),
        but ZeRO load_state_dict expects every rank to supply its own shard via
        state_dict_list[dp_rank].  Attempting to load causes KeyError on all
        non-zero ranks.  The optimizer will restart from its initial state; the
        first ~100 steps may show slightly elevated loss while momentum/variance
        warm up, but training will continue correctly from the saved step count.
        """
        training_state = getattr(self, "_pending_training_state", None)
        if training_state is None:
            return
        # --- optimizer: skip (ZeRO rank-shard mismatch, see docstring) ---
        logger.warning(
            "Skipping DeepSpeed optimizer state restore (ZeRO shard format incompatible "
            "with single-rank save).  Optimizer will restart from initial state."
        )
        # --- lr_scheduler ---
        if training_state.get("lr_scheduler"):
            self.lr_scheduler.load_state_dict(training_state["lr_scheduler"])
            logger.info("LR scheduler state restored.")
        # --- EMA ---
        framework = self.accelerator.unwrap_model(self.model)
        if hasattr(framework, "load_ema_state") and training_state.get("ema"):
            framework.load_ema_state(training_state["ema"])
            logger.info("EMA state restored.")
        self._training_state_restored = True
        self._pending_training_state = None  # free memory
        logger.info("Training state (scheduler/EMA) restored; optimizer restarted fresh.")

    def _save_checkpoint(self):
        """save current training state"""

        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")

            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
            # save model state
            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, checkpoint_path + "_model.safetensors")
            elif save_format == "pt":
                torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

            # save training metadata
            summary_data = {
                "steps": self.completed_steps,
            }
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")

            # save full training state (optimizer, scheduler, EMA)
            framework = self.accelerator.unwrap_model(self.model)
            training_state = {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "ema": framework.get_ema_state() if hasattr(framework, "get_ema_state") else None,
                "completed_steps": self.completed_steps,
            }
            torch.save(training_state, checkpoint_path + "_training_state.pt")

            # ✅ Save accessed configuration only
            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                # self.config.save_accessed_config(
                #     output_dir / "config.json", 
                #     use_original_values=False
                # )
                self.config.save_accessed_config(
                    output_dir / "config.yaml", 
                    use_original_values=False 
                )
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """record training metrics"""
        if self.completed_steps % self.config.trainer.logging_frequency == 0:
            if dist.get_rank() == 0:
                # add learning rate 
                metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0] # see lr group in yaml.trainer.learning_rate

                # add epoch info
                metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)

                # record to W&B
                wandb.log(metrics, step=self.completed_steps)
                # debug output
                logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """create data iterators"""
        self.vla_iter = iter(self.vla_train_dataloader)
        # self.vlm_iter = iter(self.vlm_train_dataloader)

    def _get_next_batch(self):
        """get next batch (automatically handle data loop)"""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def train(self):
        """execute training loop"""
        # print training config
        self._log_training_config()

        # prepare data iterators
        self._create_data_iterators()

        # create progress bar (initial=completed_steps so bar shows correct position on resume)
        progress_bar = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        # main training loop
        while self.completed_steps < self.config.trainer.max_train_steps:
            # get data batch
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            # execute training step
            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            # update progress
            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1
            
            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                        {
                            "data_times": f"{t_end_data - t_start_data:.3f}",
                            "model_times": f"{t_end_model - t_start_model:.3f}",
                        }
                    )

            # evaluate model
            if self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics)

            # record metrics
            step_metrics["data_time"] = t_end_data - t_start_data
            step_metrics["model_time"] = t_end_model - t_start_model
            self._log_metrics(step_metrics)

            # save checkpoint
            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()

            # check termination condition
            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        # training end processing
        self._finalize_training()

        # execute evaluation step

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """
        Evaluate the model on the given dataset using the specified metric function.

        :param eval_dataset: List of evaluation samples, each containing 'image', 'instruction', and 'action'.
        :param metric_fn: Function to compute the distance between predicted and ground truth actions.
        :return: Average metric score across the evaluation dataset.
        """

        # Use val dataloader if available, otherwise fall back to a train batch
        if self.vla_val_dataloader is not None:
            if not hasattr(self, "_val_iter") or self._val_iter is None:
                self._val_iter = iter(self.vla_val_dataloader)
            try:
                examples = next(self._val_iter)
            except StopIteration:
                self._val_iter = iter(self.vla_val_dataloader)
                examples = next(self._val_iter)
        else:
            examples = self._get_next_batch()
        score = 0.0
        actions = examples["action"].cpu().numpy()  # ground truth [B, T, D]
        # Predict actions using the model
        framework = self.accelerator.unwrap_model(self.model)
        framework.eval()
        with torch.no_grad():
            output_dict = self.model.predict_action(
                examples=examples, use_ddim=True, num_ddim_steps=20
            )
        framework.train()

        if self.accelerator.is_main_process:
            normalized_actions = output_dict["normalized_actions"]  # B, T, D
            # B, Chunk, dim = actions.shape
            num_pots = np.prod(actions.shape)
            # Compute the metric score
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            average_score = score / num_pots
            step_metrics["loss/mse_score"] = average_score

        del examples
        dist.barrier()  # ensure all processes are synchronized
        return step_metrics

    def _log_training_config(self):
        """record training config"""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.config.trainer.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_vla, batch_vlm=None):
        """execute single training step"""
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()

            # autocast is applied inside model.forward() to avoid DeepSpeed stripping it
            output_dict = self.model.forward(batch_vla)
            action_loss = output_dict["action_loss"]
            total_loss = action_loss

            # VLA backward propagation
            self.accelerator.backward(total_loss)

            # gradient clipping is handled by DeepSpeed (ds_config.yaml: gradient_clipping: 1.0)

            # optimizer step
            self.optimizer.step()
            self.lr_scheduler.step()

        # update EMA after optimizer step (only when gradients actually synced)
        if self.accelerator.sync_gradients:
            framework = self.accelerator.unwrap_model(self.model)
            if hasattr(framework, "ema"):
                framework.ema.step(framework.policy)

        step_metrics = {"loss/action_dit": action_loss.item()}

        framework_name = getattr(self.config.framework, "name", "")
        if framework_name.startswith("ManiFlow"):
            _mf_key_map = {
                "mf_loss_flow":             "loss/flow_matching",
                "mf_loss_ct":               "loss/consistency_training",
                "mf_v_flow_pred_magnitude": "debug/flow_pred_magnitude",
                "mf_v_ct_pred_magnitude":   "debug/consistency_training_pred_magnitude",
                "mf_bc_loss":               "loss/total",
                "mf_rgb_cond_token_std":    "debug/rgb_token_std",
                "mf_ptmap_cond_token_std":  "debug/ptmap_token_std",
            }
            step_metrics.update({
                _mf_key_map[k]: v
                for k, v in output_dict.items()
                if k in _mf_key_map
            })

        return step_metrics

    def _finalize_training(self):
        """training end processing"""
        # save final model
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, os.path.join(final_checkpoint, "model.safetensors"))
            elif save_format == "pt":
                torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

            # save full in-memory config to final_model metadata
            import json 
            import yaml
            cfg_obj = self.config.unwrap() if isinstance(self.config, AccessTrackedConfig) else self.config
            OmegaConf.save(cfg_obj, os.path.join(final_checkpoint, "config.yaml"))
            with open(os.path.join(final_checkpoint, "config.yaml"), "r") as f_yaml, \
                 open(os.path.join(final_checkpoint, "config.json"), "w") as f_json:
                json.dump(yaml.safe_load(f_yaml), f_json, indent=2)
            logger.info(f"Config saved to {final_checkpoint}/config.yaml and config.json")


        # close W&B
        if self.accelerator.is_main_process:
            wandb.finish()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")

    #  Wrap config to enable access tracking
    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    # create output directory and save config
    output_dir = setup_directories(cfg=cfg)
    # build model
    vla = build_framework(cfg)
    # prepare data
    vla_train_dataloader, vla_val_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)

    # set optimizer and scheduler
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    # create trainer
    # Run VLA Training
    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        vla_val_dataloader=vla_val_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    # execute training preparation
    trainer.prepare_training()
    # execute training
    trainer.train()

    # And... we're done!
    logger.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="hvla/config/training/hvla_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    # Load YAML config & Convert CLI overrides to dotlist config
    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)  # Normalize CLI args to dotlist format
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # if cfg.is_debug:
    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
