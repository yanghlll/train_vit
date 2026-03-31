"""
Quadtree CU Selection Video Dataset (Optimized v3)
====================================================
I-frame must-select + P/B sample → fixed target_num tokens per sample.

Optimizations:
1. cu_visidx as .npy (K,4) int16 mmap — ~0.1ms load
2. Vectorized 16px crop via stride_tricks
3. decord num_threads=4
4. Fixed output token count → no padding needed in collate → no OOM
"""

import os
import numpy as np
from typing import Optional, Tuple, List, Dict

import torch
from torch.utils.data import Dataset

try:
    import decord
    decord.bridge.set_bridge("native")
    _HAS_DECORD = True
except ImportError:
    _HAS_DECORD = False


SCALE_TO_IDX = {16: 1, 32: 2, 64: 3}
CELL = 8


class QuadtreeVideoDataset(Dataset):
    def __init__(
        self,
        video_list: str,
        cu_visidx_src: str,
        cu_visidx_dst: str,
        num_frames: int = 64,
        label_path: Optional[str] = None,
        target_num: int = 1960,
        i_frame_ids: Tuple[int, ...] = (0, 32),
        mean: Tuple[float, ...] = (0.48145466, 0.4578275, 0.40821073),
        std: Tuple[float, ...] = (0.26862954, 0.26130258, 0.27577711),
        decord_threads: int = 4,
    ):
        """
        Args:
            target_num: fixed total tokens per sample (I + P/B).
                        I-frame tokens are always included; P/B tokens are sampled
                        to fill the remaining budget. If total available < target_num,
                        all tokens are used (output may be shorter).
            i_frame_ids: frame indices that are I-frames (GOP=32 → (0, 32)).
        """
        super().__init__()
        assert _HAS_DECORD, "decord is required"

        self.cu_visidx_src = cu_visidx_src
        self.cu_visidx_dst = cu_visidx_dst
        self.num_frames = num_frames
        self.target_num = target_num
        self.i_frame_ids = set(i_frame_ids)
        self.mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(3, 1, 1)
        self.decord_threads = decord_threads

        with open(video_list, "r") as f:
            self.video_paths = [l.strip() for l in f if l.strip() and not l.startswith("#")]

        if label_path and os.path.exists(label_path):
            self.all_labels = np.load(label_path, mmap_mode="r").astype(np.int64)
            assert len(self.all_labels) == len(self.video_paths)
        else:
            self.all_labels = np.zeros((len(self.video_paths), 10), dtype=np.int64)

        self._cu_npy_paths = []
        self._cu_npz_paths = []
        for vp in self.video_paths:
            replaced = vp.replace(self.cu_visidx_src, self.cu_visidx_dst)
            stem, _ = os.path.splitext(replaced)
            self._cu_npy_paths.append(stem + ".cu_visidx.npy")
            self._cu_npz_paths.append(stem + ".cu_visidx.npz")

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> dict:
        try:
            return self._load_sample(idx)
        except Exception:
            return self._dummy_sample(idx)

    def _load_cu_visidx(self, idx: int):
        npy_path = self._cu_npy_paths[idx]
        if os.path.exists(npy_path):
            arr = np.load(npy_path, mmap_mode="r")
            t_all = arr[:, 0].astype(np.int64)
            y_all = arr[:, 1].astype(np.int64)
            x_all = arr[:, 2].astype(np.int64)
            scale_px = arr[:, 3].astype(np.int32)
            return t_all, y_all, x_all, scale_px
        else:
            data = np.load(self._cu_npz_paths[idx])
            counts = data["counts"]
            t_all = data["t"].astype(np.int64)
            y_all = data["y"].astype(np.int64)
            x_all = data["x"].astype(np.int64)
            n16, n32, n64 = int(counts[0]), int(counts[1]), int(counts[2])
            K = n16 + n32 + n64
            scale_px = np.empty(K, dtype=np.int32)
            scale_px[:n16] = 16
            scale_px[n16:n16 + n32] = 32
            scale_px[n16 + n32:] = 64
            return t_all, y_all, x_all, scale_px

    def _select_tokens(self, t_all, y_all, x_all, scale_px):
        """I-frame must-select + P/B sample → fixed target_num tokens."""
        K = len(t_all)

        # Split I-frame vs P/B
        i_mask = np.isin(t_all, list(self.i_frame_ids))
        i_idx = np.where(i_mask)[0]
        pb_idx = np.where(~i_mask)[0]

        n_i = len(i_idx)
        pb_budget = max(0, self.target_num - n_i)

        # Sample P/B tokens
        if len(pb_idx) <= pb_budget:
            # Not enough P/B → use all
            pb_sel = pb_idx
        else:
            # Random sample without replacement
            pb_sel = np.sort(np.random.choice(pb_idx, pb_budget, replace=False))

        # Concatenate and sort
        sel_idx = np.sort(np.concatenate([i_idx, pb_sel]))

        return (t_all[sel_idx], y_all[sel_idx], x_all[sel_idx], scale_px[sel_idx])

    def _load_sample(self, idx: int) -> dict:
        video_path = self.video_paths[idx]
        label = self.all_labels[idx].copy()

        # ---- 1. Load cu_visidx + select tokens ----
        t_raw, y_raw, x_raw, scale_raw = self._load_cu_visidx(idx)
        t_all, y_all, x_all, scale_px = self._select_tokens(t_raw, y_raw, x_raw, scale_raw)
        K = len(t_all)

        # Count per scale (after selection)
        n16 = int((scale_px == 16).sum())
        n32 = int((scale_px == 32).sum())
        n64 = K - n16 - n32

        # Re-sort by scale: 16 first, then 32, then 64
        order = np.argsort(scale_px, kind='stable')
        t_all = t_all[order]
        y_all = y_all[order]
        x_all = x_all[order]
        scale_px = scale_px[order]

        # ---- 2. Decode needed frames ----
        needed_frames = np.unique(t_all)
        vr = decord.VideoReader(video_path, num_threads=self.decord_threads)
        frames_batch = vr.get_batch(needed_frames.tolist())
        frames_np = frames_batch.asnumpy() if hasattr(frames_batch, 'asnumpy') else np.asarray(frames_batch)
        fi_to_idx = {int(fi): i for i, fi in enumerate(needed_frames)}
        H, W = frames_np.shape[1], frames_np.shape[2]

        # ---- 3. Crop per scale ----
        patches_by_scale = {}

        if n16 > 0:
            patches_by_scale[16] = self._crop_batch_16(
                frames_np, fi_to_idx, t_all[:n16], y_all[:n16], x_all[:n16], H, W, n16)
        if n32 > 0:
            patches_by_scale[32] = self._crop_batch_loop(
                frames_np, fi_to_idx, t_all[n16:n16+n32], y_all[n16:n16+n32], x_all[n16:n16+n32],
                H, W, n32, 32)
        if n64 > 0:
            patches_by_scale[64] = self._crop_batch_loop(
                frames_np, fi_to_idx, t_all[n16+n32:], y_all[n16+n32:], x_all[n16+n32:],
                H, W, n64, 64)

        # ---- 4. Positions ----
        positions = np.empty((K, 3), dtype=np.float32)
        positions[:, 0] = t_all.astype(np.float32)
        positions[:, 1] = (y_all + scale_px * 0.5).astype(np.float32) / 16.0
        positions[:, 2] = (x_all + scale_px * 0.5).astype(np.float32) / 16.0

        # ---- 5. Scale indices ----
        scale_indices = np.empty(K, dtype=np.int64)
        scale_indices[:n16] = 1
        scale_indices[n16:n16 + n32] = 2
        scale_indices[n16 + n32:] = 3

        return {
            "patches_by_scale": patches_by_scale,
            "positions": torch.from_numpy(positions),
            "scale_indices": torch.from_numpy(scale_indices),
            "label": torch.from_numpy(label),
            "num_tokens": K,
        }

    def _crop_batch_16(self, frames_np, fi_to_idx, t, y, x, H, W, n):
        S = 16
        ai = np.array([fi_to_idx[int(t[j])] for j in range(n)], dtype=np.int64)
        yi = y.astype(np.int64)
        xi = x.astype(np.int64)

        if np.all(yi + S <= H) and np.all(xi + S <= W):
            from numpy.lib.stride_tricks import as_strided
            sh = frames_np.strides
            n_frames = frames_np.shape[0]
            window_view = as_strided(
                frames_np,
                shape=(n_frames, H - S + 1, W - S + 1, S, S, 3),
                strides=(sh[0], sh[1], sh[2], sh[1], sh[2], sh[3]),
            )
            buf = window_view[ai, yi, xi]
        else:
            buf = np.empty((n, S, S, 3), dtype=np.uint8)
            for j in range(n):
                yj, xj = int(yi[j]), int(xi[j])
                ye, xe = min(yj + S, H), min(xj + S, W)
                ph, pw = ye - yj, xe - xj
                if ph == S and pw == S:
                    buf[j] = frames_np[int(ai[j]), yj:ye, xj:xe]
                else:
                    buf[j] = 0
                    buf[j, :ph, :pw] = frames_np[int(ai[j]), yj:ye, xj:xe]

        out = buf.transpose(0, 3, 1, 2).astype(np.float32) * np.float32(1.0 / 255.0)
        out = (out - self.mean) / self.std
        return torch.from_numpy(out)

    def _crop_batch_loop(self, frames_np, fi_to_idx, t, y, x, H, W, n, scale):
        out = np.empty((n, 3, scale, scale), dtype=np.float32)
        inv = np.float32(1.0 / 255.0)
        for j in range(n):
            ai = fi_to_idx[int(t[j])]
            yj, xj = int(y[j]), int(x[j])
            ye, xe = min(yj + scale, H), min(xj + scale, W)
            p = frames_np[ai, yj:ye, xj:xe]
            ph, pw = p.shape[0], p.shape[1]
            if ph == scale and pw == scale:
                out[j] = p.transpose(2, 0, 1) * inv
            else:
                out[j] = 0.0
                out[j, :, :ph, :pw] = p.transpose(2, 0, 1) * inv
        out = (out - self.mean) / self.std
        return torch.from_numpy(out)

    def _dummy_sample(self, idx: int) -> dict:
        label = self.all_labels[idx].copy() if idx < len(self.all_labels) else np.zeros(10, dtype=np.int64)
        return {
            "patches_by_scale": {16: torch.zeros(1, 3, 16, 16)},
            "positions": torch.zeros(1, 3),
            "scale_indices": torch.ones(1, dtype=torch.long),
            "label": torch.from_numpy(label),
            "num_tokens": 1,
        }
