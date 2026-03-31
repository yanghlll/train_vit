"""
Quadtree CU Multi-Scale Training (LLaVA-ViT framework)
========================================================
Based on train_univit_11_08_ip_three_input_residual_mv_tmp.py.
Uses MultiGranularityOVEncoder with pre-extracted cu_visidx.npz.

Key differences from MV+Res training:
- Multi-scale patches (16/32/64) instead of uniform 16x16
- PyTorch DataLoader (variable-length) instead of DALI (fixed-size)
- forward_packed() with FlashAttention varlen instead of gather-based forward

Usage:
  torchrun --nproc_per_node=8 -m training.train_quadtree \
    --list_datasets hevc_gop32_quadtree \
    --list_batch_sizes 4 \
    --lr 1e-3
"""

import argparse
import logging
import os
import sys
import time
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch import distributed
from torch.nn.utils import clip_grad_norm_
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import model_factory
from dataset import DATASET_REGISTRY, Property
from training.checkpoint_utils import load_checkpoint, save_checkpoint
from training.fused_partial_fc_v2 import CombinedMarginLoss, PartialFC_V2
from training.lr_scheduler import PolynomialLRWarmup

# Import quadtree components
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "model_factory"))
from quadtree_dataset import QuadtreeVideoDataset
from multigranularity_dataset import multigranularity_collate_fn
from model_factory.vit_quadtree import QuadtreeViTEncoder

torch._dynamo.config.optimize_ddp = False


# ---------------------------------------------------------------------------
# Args (aligned with LLaVA-ViT defaults)
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Quadtree CU multi-scale training")

# General
parser.add_argument("--debug", type=int, default=0)
parser.add_argument("--output", default="output")
parser.add_argument("--local_rank", type=int, default=0)

# Data
parser.add_argument("--list_datasets", nargs='+', type=str, default=["hevc_gop32_quadtree"])
parser.add_argument("--list_batch_sizes", nargs='+', type=int, default=[4])
parser.add_argument("--list_sample_rates", nargs='+', type=float, default=[0.1])
parser.add_argument("--list_margins", nargs='+', type=float, default=[0.3])
parser.add_argument("--list_filters", nargs='+', type=float, default=[0.75])
parser.add_argument("--list_lr_pfc_weights", nargs='+', type=float, default=[1.0])
parser.add_argument("--list_loss_weights", nargs='+', type=float, default=[1.0])
parser.add_argument("--list_init_partial_fc_paths", nargs='+', type=str, default=["NULL"])
parser.add_argument("--num_frames", type=int, default=64)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--target_num", type=int, default=1960,
                    help="Fixed total tokens per sample (I-frame all + P/B sampled)")
parser.add_argument("--i_frame_ids", nargs='+', type=int, default=[0, 32],
                    help="I-frame indices to always fully select (GOP=32)")
parser.add_argument("--random_diff", type=int, default=10)

# Model
parser.add_argument("--hidden_size", type=int, default=384)
parser.add_argument("--num_layers", type=int, default=12)
parser.add_argument("--num_heads", type=int, default=6)
parser.add_argument("--intermediate_size", type=int, default=1536)
parser.add_argument("--image_size", type=int, default=224)
parser.add_argument("--patch_size", type=int, default=16)
parser.add_argument("--embedding_size", type=int, default=384)
parser.add_argument("--gradient_checkpointing", action="store_true")
parser.add_argument("--finetune_backbone", type=int, default=1)

# Optimization
parser.add_argument("--opt", default="adamw")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=0.05)
parser.add_argument("--weight_decay_pfc", type=float, default=0.05)
parser.add_argument("--warmup_ratio", type=float, default=0.1)
parser.add_argument("--backward_passes_per_step", type=int, default=1)
parser.add_argument("--save_pfc", type=int, default=1)

# Initialization / Resume
parser.add_argument("--init_backbone", default="NULL")

# Logging & Checkpoint
parser.add_argument("--frequent", type=int, default=10)
parser.add_argument("--ckpt_interval", type=int, default=2000)

# Training schedule
parser.add_argument("--num_sampled_data", type=int, default=60000000)

args = parser.parse_args()

rank = int(os.getenv("RANK", "0"))
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
distributed.init_process_group(backend="nccl")
torch.cuda.set_device(local_rank)
torch.backends.cudnn.benchmark = True

os.makedirs(args.output, exist_ok=True)

if rank == 0:
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter(f"rank-id:{rank:03d}:%(asctime)s-%(message)s")
    file_handler = logging.FileHandler(os.path.join(args.output, f"training_{rank:03d}.logger"))
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)
else:
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter(f"rank-id:{rank:03d}:%(asctime)s-%(message)s")
    file_handler = logging.FileHandler(os.path.join(args.output, f"training_{rank:03d}.logger"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class ScalaMetric:
    def __init__(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n
        self.avg = self.sum / self.count
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0


class BatchEndCallBack:
    """Logging callback aligned with LLaVA-ViT style."""
    def __init__(self, frequent, list_head_names, output, total_steps, tb_writer=None):
        self.frequent = frequent
        self.list_head_names = list_head_names
        self.output = output
        self.total_steps = total_steps
        self.num_head = len(list_head_names)
        self.time_start = time.time()
        self.list_loss_metric = [ScalaMetric() for _ in list_head_names]
        self.tokens_metric = ScalaMetric()
        self.scale_metrics = {16: ScalaMetric(), 32: ScalaMetric(), 64: ScalaMetric()}
        self.init = False
        self.tic = 0
        self.step_times = []
        self.max_time_history = 100
        self.total_examples = 0
        self.tb_writer = tb_writer if rank == 0 else None
        self.logger = logging.getLogger(__name__)

    def __call__(self, global_step, lr_scheduler, list_loss_float,
                 batch_size, num_samples=None,
                 total_tokens=0, scale_counts=None,
                 grad_norm_backbone=0.0, grad_norm_pfc=0.0, embed_norm=0.0):
        for i in range(self.num_head):
            self.list_loss_metric[i].update(list_loss_float[i])
        if total_tokens > 0:
            self.tokens_metric.update(total_tokens)
        if scale_counts:
            for s, n in scale_counts.items():
                if s in self.scale_metrics:
                    self.scale_metrics[s].update(n)

        if global_step > 0 and global_step % self.frequent == 0:
            if self.init:
                current_time = time.time()
                time_elapsed = current_time - self.tic
                self.tic = current_time

                time_per_step = time_elapsed / self.frequent
                self.step_times.append(time_per_step)
                if len(self.step_times) > self.max_time_history:
                    self.step_times.pop(0)

                avg_time_per_step = sum(self.step_times) / len(self.step_times)
                remaining_steps = self.total_steps - global_step
                remaining_hours = (avg_time_per_step * remaining_steps) / 3600

                try:
                    speed = self.frequent * batch_size / time_elapsed
                    speed_total = speed * world_size
                except ZeroDivisionError:
                    speed = speed_total = float("inf")

                # Header
                header = (f"rank {speed:.2f} total {speed_total:.2f} its/s "
                          f"lr: {lr_scheduler.get_last_lr()[0]:.8f} ")
                progress = (f"step: {global_step}/{self.total_steps} "
                            f"({global_step/self.total_steps*100:.2f}%) ")
                time_info = f"remain: {remaining_hours:.2f}h "

                # Per-head loss + lr
                loss_str = ""
                for head_id, name in enumerate(self.list_head_names):
                    if self.tb_writer:
                        self.tb_writer.add_scalar(f"loss/{name}", self.list_loss_metric[head_id].avg, global_step)
                        lr_idx = min(head_id + 1, len(lr_scheduler.get_last_lr()) - 1)
                        self.tb_writer.add_scalar(f"lr/{name}", lr_scheduler.get_last_lr()[lr_idx], global_step)
                        if num_samples:
                            self.tb_writer.add_scalar(f"samples_vs_loss/{name}", self.list_loss_metric[head_id].avg, num_samples)
                    loss_str += (f"\n  {f'head: {name}':<40}"
                                 f"{f'loss: {self.list_loss_metric[head_id].avg:.4f}':<20}")
                    self.list_loss_metric[head_id].reset()

                # Token stats
                tok_str = ""
                if self.tokens_metric.count > 0:
                    tok_str = f"tokens/batch: {self.tokens_metric.avg:.0f} "
                    s_parts = []
                    for s in [16, 32, 64]:
                        if self.scale_metrics[s].count > 0:
                            s_parts.append(f"{s}px={self.scale_metrics[s].avg:.0f}")
                    if s_parts:
                        tok_str += f"({', '.join(s_parts)}) "
                    if self.tb_writer:
                        self.tb_writer.add_scalar("tokens/batch_avg", self.tokens_metric.avg, global_step)
                        for s in [16, 32, 64]:
                            if self.scale_metrics[s].count > 0:
                                self.tb_writer.add_scalar(f"tokens/scale_{s}px", self.scale_metrics[s].avg, global_step)
                    self.tokens_metric.reset()
                    for m in self.scale_metrics.values():
                        m.reset()

                samples_info = f"samples: {num_samples}" if num_samples else ""

                # Gradient norm + embedding norm
                norm_str = ""
                if grad_norm_backbone > 0 or embed_norm > 0:
                    norm_str = f"grad_norm: bb={grad_norm_backbone:.4f} pfc={grad_norm_pfc:.4f} embed_norm: {embed_norm:.4f} "
                    if self.tb_writer:
                        self.tb_writer.add_scalar("grad_norm/backbone", grad_norm_backbone, global_step)
                        self.tb_writer.add_scalar("grad_norm/pfc", grad_norm_pfc, global_step)
                        self.tb_writer.add_scalar("embed_norm", embed_norm, global_step)

                gpu_mem = ""
                try:
                    mem_used = torch.cuda.max_memory_allocated() / 1024**3
                    mem_total = torch.cuda.get_device_properties(0).total_mem / 1024**3
                    gpu_mem = f"gpu_mem: {mem_used:.1f}/{mem_total:.1f}GB "
                    if self.tb_writer:
                        self.tb_writer.add_scalar("gpu_mem_gb", mem_used, global_step)
                except Exception:
                    pass

                msg = f"{header}{progress}{time_info}{tok_str}{norm_str}{gpu_mem}{samples_info}{loss_str}"

                if rank == 0:
                    self.logger.info(msg)
                    if self.tb_writer:
                        self.tb_writer.flush()
            else:
                self.init = True
                self.tic = time.time()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global_step = 0

    # Dataset config
    args.list_datasets = [DATASET_REGISTRY.get(x)() for x in args.list_datasets]
    args.num_dataset_heads = len(args.list_datasets)

    def _expand(name, v):
        if len(v) == 1:
            return v * args.num_dataset_heads
        assert len(v) == args.num_dataset_heads
        return v

    args.list_batch_sizes = _expand("list_batch_sizes", args.list_batch_sizes)
    args.list_sample_rates = _expand("list_sample_rates", args.list_sample_rates)
    args.list_margins = _expand("list_margins", args.list_margins)
    args.list_filters = _expand("list_filters", args.list_filters)
    args.list_lr_pfc_weights = _expand("list_lr_pfc_weights", args.list_lr_pfc_weights)
    args.list_loss_weights = _expand("list_loss_weights", args.list_loss_weights)
    args.list_init_partial_fc_paths = _expand("list_init_partial_fc_paths", args.list_init_partial_fc_paths)

    args.batch_size = sum(args.list_batch_sizes)
    args.list_head_names = [x.name for x in args.list_datasets]
    args.total_steps = int(args.num_sampled_data / args.batch_size / world_size)

    for arg in vars(args):
        logger.info(f"{format(arg, '<30')}  {format(str(getattr(args, arg)))}")

    # ---- Model: QuadtreeViTEncoder ----
    backbone = QuadtreeViTEncoder(
        patch_size=args.patch_size,
        hidden_size=args.hidden_size,
        head_dim=64,
        num_hidden_layers=args.num_layers,
        intermediate_size=args.intermediate_size,
        act_layer=nn.GELU,
        use_gradient_checkpointing=args.gradient_checkpointing,
        norm_cls=nn.LayerNorm,
        use_head=True,
    ).cuda().train()

    if args.init_backbone != "NULL":
        assert os.path.exists(args.init_backbone)
        state_dict = torch.load(args.init_backbone, "cpu")
        state_dict = {k.replace("_orig_mod.", "").replace("module.", ""): v
                      for k, v in state_dict.items()}
        missing, unexpected = backbone.load_pretrained_base(state_dict)
        logger.info(f"Loaded backbone from {args.init_backbone}, "
                     f"missing={len(missing)}, unexpected={len(unexpected)}")

    backbone.requires_grad_(bool(args.finetune_backbone))

    # Separate parameter groups: base (pretrained) vs new (multi-scale heads)
    new_param_names = {"scale_embed", "aggregate_32", "aggregate_64", "zero_mlp_32", "zero_mlp_64"}
    base_params = []
    new_params = []
    for name, param in backbone.named_parameters():
        if not param.requires_grad:
            continue
        if any(n in name for n in new_param_names):
            new_params.append(param)
        else:
            base_params.append(param)

    parameters: List[dict] = [
        {"params": base_params, "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": new_params, "lr": args.lr * 10, "weight_decay": args.weight_decay},
    ]

    if rank == 0:
        n_base = sum(p.numel() for p in base_params)
        n_new = sum(p.numel() for p in new_params)
        logger.info(f"Backbone params: base={n_base/1e6:.1f}M (lr={args.lr}), "
                     f"new={n_new/1e6:.1f}M (lr={args.lr*10})")

    # ---- PartialFC ----
    dict_pfc_modules = {}
    list_module_pfc = []

    for head_id in range(args.num_dataset_heads):
        head_name = args.list_head_names[head_id]
        dataset_config = args.list_datasets[head_id]

        margin_loss = CombinedMarginLoss(64, 1, 0, args.list_margins[head_id], args.list_filters[head_id])
        partial_fc = PartialFC_V2(
            margin_loss,
            args.embedding_size,
            dataset_config.num_classes,
            args.list_sample_rates[head_id],
            fp16=False,
        )
        partial_fc.train().cuda()
        list_module_pfc.append(partial_fc)
        dict_pfc_modules[head_name] = partial_fc

        lr_pfc = args.lr * args.list_lr_pfc_weights[head_id]
        parameters.append({
            "params": partial_fc.parameters(),
            "lr": lr_pfc,
            "weight_decay": args.weight_decay_pfc,
        })

        init_pfc = args.list_init_partial_fc_paths[head_id]
        if init_pfc != "NULL":
            init_pfc = init_pfc % rank
            if os.path.exists(init_pfc):
                if init_pfc.endswith(".npy"):
                    partial_fc.weight = torch.nn.Parameter(torch.from_numpy(np.load(init_pfc)).cuda())
                elif init_pfc.endswith(".pt"):
                    partial_fc.load_state_dict(torch.load(init_pfc, "cpu"), strict=True)
                logger.info(f"Loaded PFC from {init_pfc}")

    # ---- Optimizer + Scheduler ----
    opt = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialLRWarmup(
        opt, int(args.total_steps * args.warmup_ratio), args.total_steps, power=2
    )

    # ---- Resume ----
    result = load_checkpoint(args.output, None, backbone, dict_pfc_modules,
                             lr_scheduler, None, args.list_head_names)
    if result is not None:
        global_step = result["global_step"]
        logger.info(f"Resuming from step {global_step}")

    # ---- DDP + compile ----
    def wrap_ddp(model):
        return torch.nn.parallel.DistributedDataParallel(
            module=model, broadcast_buffers=False, device_ids=[local_rank],
            bucket_cap_mb=32, find_unused_parameters=True, static_graph=True)

    backbone_ddp = wrap_ddp(backbone)
    # Note: torch.compile disabled for quadtree — padded attention with variable
    # max_seqlen causes inductor to allocate extra buffers leading to OOM.
    backbone_ddp_compiled = backbone_ddp

    # ---- DataLoaders (PyTorch, not DALI) ----
    list_dataloader = []
    for head_id, dataset_config in enumerate(args.list_datasets):
        ds = QuadtreeVideoDataset(
            video_list=dataset_config.mp4_list_path,
            cu_visidx_src=dataset_config.cu_visidx_src,
            cu_visidx_dst=dataset_config.cu_visidx_dst,
            num_frames=args.num_frames,
            label_path=dataset_config.label_list_path,
            target_num=args.target_num,
            i_frame_ids=tuple(args.i_frame_ids),
        )
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
        dl = DataLoader(
            ds,
            batch_size=args.list_batch_sizes[head_id],
            sampler=sampler,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=multigranularity_collate_fn,
            pin_memory=True,
            drop_last=True,
            persistent_workers=(args.num_workers > 0),
            prefetch_factor=2 if args.num_workers > 0 else None,
        )
        list_dataloader.append((dl, sampler))
        logger.info(f"[head {head_id}] {dataset_config.name}: {len(ds)} videos, "
                     f"bs={args.list_batch_sizes[head_id]}")

    if rank == 0 and SummaryWriter is not None:
        tb_writer = SummaryWriter(log_dir=f"{args.output}/tensorboard")
    else:
        tb_writer = None

    batch_end_callback = BatchEndCallBack(
        frequent=args.frequent,
        list_head_names=[x.name for x in args.list_datasets],
        output=args.output,
        total_steps=args.total_steps,
        tb_writer=tb_writer,
    )

    # ---- Training loop ----
    list_iter = [iter(dl) for dl, _ in list_dataloader]
    list_loss_metric = [ScalaMetric() for _ in range(args.num_dataset_heads)]
    epoch = 0
    num_samples = 0

    if global_step > args.total_steps:
        logger.info("global_step > total_steps")
        return

    while global_step <= args.total_steps:
        list_embedding = []
        list_labels = []
        num_samples += args.batch_size * world_size

        for head_id, dataset_config in enumerate(args.list_datasets):
            # Get batch
            try:
                batch = next(list_iter[head_id])
            except StopIteration:
                epoch += 1
                _, sampler = list_dataloader[head_id]
                sampler.set_epoch(epoch)
                list_iter[head_id] = iter(list_dataloader[head_id][0])
                batch = next(list_iter[head_id])

            # Move to GPU
            patches_by_scale = {s: p.cuda(non_blocking=True)
                                for s, p in batch["patches_by_scale"].items()}
            positions = batch["positions"].cuda(non_blocking=True)
            scale_indices = batch["scale_indices"].cuda(non_blocking=True)
            cu_seqlens = batch["cu_seqlens"].cuda(non_blocking=True)
            max_seqlen = batch["max_seqlen"]
            labels = batch["labels"].cuda(non_blocking=True)

            # Forward (unwrap DDP module for forward_packed)
            fwd_module = backbone_ddp_compiled.module if hasattr(backbone_ddp_compiled, 'module') else backbone_ddp_compiled
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = fwd_module.forward_packed(
                    patches_by_scale=patches_by_scale,
                    positions_thw=positions,
                    scale_indices=scale_indices,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                )

            pooled = output["head_output"]
            if pooled is None:
                logger.warning(f"[step {global_step}] pooled is None, skipping")
                continue

            head_embedding = pooled.float()
            list_embedding.append(head_embedding)
            list_labels.append(labels)

        # ---- Loss ----
        list_loss = []
        for head_id, pfc in enumerate(list_module_pfc):
            dataset_config = args.list_datasets[head_id]
            head_embedding = list_embedding[head_id]
            head_label = list_labels[head_id].long()
            label_select = dataset_config.label_select
            random_diff = dataset_config.random_diff
            loss_weight = args.list_loss_weights[head_id]
            head_label = head_label[:, label_select:label_select + random_diff]
            head_loss = pfc(head_embedding, head_label, random_diff) * loss_weight
            list_loss.append(head_loss)

        # Embedding norm (before loss backward)
        embed_norm = head_embedding.detach().norm(dim=-1).mean().item()

        is_accumulation_step = (global_step % args.backward_passes_per_step != 0)
        scaled_loss = sum(list_loss) / args.backward_passes_per_step

        grad_norm_backbone = 0.0
        grad_norm_pfc = 0.0
        if is_accumulation_step:
            with backbone_ddp_compiled.no_sync():
                scaled_loss.backward()
        else:
            scaled_loss.backward()
            if global_step % args.backward_passes_per_step == 0:
                grad_norm_backbone = clip_grad_norm_(backbone_ddp_compiled.parameters(), max_norm=5, norm_type=2).item()
                for pfc in list_module_pfc:
                    grad_norm_pfc = max(grad_norm_pfc, clip_grad_norm_(pfc.parameters(), max_norm=5, norm_type=2).item())
                opt.step()
                opt.zero_grad(set_to_none=True)

        lr_scheduler.step()

        # ---- Logging via callback ----
        total_tokens = sum(batch.get("seq_lengths", [0]))
        scale_counts = {}
        for s, p in patches_by_scale.items():
            scale_counts[s] = p.shape[0]

        batch_end_callback(
            global_step=global_step,
            lr_scheduler=lr_scheduler,
            list_loss_float=[l.item() for l in list_loss],
            batch_size=args.batch_size,
            num_samples=num_samples,
            total_tokens=total_tokens,
            scale_counts=scale_counts,
            grad_norm_backbone=grad_norm_backbone,
            grad_norm_pfc=grad_norm_pfc,
            embed_norm=embed_norm,
        )

        global_step += 1

        # ---- Checkpoint ----
        if global_step % args.ckpt_interval == 0:
            save_checkpoint(args.output, backbone, pfc_modules=dict_pfc_modules,
                            lr_scheduler=lr_scheduler, amp=None, global_step=global_step,
                            list_head_names=args.list_head_names, keep_num=20)

        if global_step > args.total_steps:
            save_checkpoint(args.output, backbone, pfc_modules=dict_pfc_modules,
                            lr_scheduler=lr_scheduler, amp=None, global_step=global_step,
                            list_head_names=args.list_head_names, keep_num=20)
            logger.info(f"Training completed at step {global_step}")
            break


if __name__ == "__main__":
    main()
