"""
Multi-Granularity Video Dataset
================================
Loads pre-extracted HEVC features (depth maps, MV, residual) and raw video frames,
applies multi-granularity patch assignment, and returns variable-length token sequences
ready for the MultiGranularityOVEncoder.
"""

import os
import math
import numpy as np
from typing import Optional, Tuple, List, Dict

import torch
from torch.utils.data import Dataset, DataLoader

try:
    import decord
    decord.bridge.set_bridge("torch")
    _HAS_DECORD = True
except ImportError:
    _HAS_DECORD = False

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


# ---------------------------------------------------------------------------
# Granularity Assignment (numpy-based, runs in DataLoader workers)
# ---------------------------------------------------------------------------

SCALE_TO_IDX = {8: 0, 16: 1, 32: 2, 64: 3}
IDX_TO_SCALE = {0: 8, 1: 16, 2: 32, 3: 64}


def _aggregate_depth_to_patch_grid(
    depth_map: np.ndarray, patch_grid_h: int, patch_grid_w: int
) -> np.ndarray:
    """Aggregate 1/8-resolution depth map to patch grid resolution.

    Args:
        depth_map: (H//8, W//8) uint8 depth values 0-3
        patch_grid_h: number of patch rows
        patch_grid_w: number of patch columns

    Returns:
        (patch_grid_h, patch_grid_w) float32 mean depth per patch
    """
    dh, dw = depth_map.shape
    result = np.zeros((patch_grid_h, patch_grid_w), dtype=np.float32)
    # Map each patch cell to its corresponding depth map region
    for ph in range(patch_grid_h):
        for pw in range(patch_grid_w):
            # Patch covers pixels [ph*16, (ph+1)*16) x [pw*16, (pw+1)*16)
            # Depth map is at 1/8 resolution
            r0 = int(ph * 16 / 8)
            r1 = int(min((ph + 1) * 16 / 8, dh))
            c0 = int(pw * 16 / 8)
            c1 = int(min((pw + 1) * 16 / 8, dw))
            if r1 > r0 and c1 > c0:
                result[ph, pw] = depth_map[r0:r1, c0:c1].mean()
    return result


def _aggregate_energy_to_patch_grid(
    energy_map: np.ndarray, patch_grid_h: int, patch_grid_w: int, patch_size: int = 16
) -> np.ndarray:
    """Aggregate full-resolution energy map to patch grid."""
    eh, ew = energy_map.shape
    result = np.zeros((patch_grid_h, patch_grid_w), dtype=np.float32)
    for ph in range(patch_grid_h):
        for pw in range(patch_grid_w):
            r0 = ph * patch_size
            r1 = min((ph + 1) * patch_size, eh)
            c0 = pw * patch_size
            c1 = min((pw + 1) * patch_size, ew)
            if r1 > r0 and c1 > c0:
                result[ph, pw] = energy_map[r0:r1, c0:c1].mean()
    return result


def assign_patch_scales(
    depth_map: np.ndarray,
    mv_magnitude: Optional[np.ndarray],
    residual_energy: Optional[np.ndarray],
    frame_h: int,
    frame_w: int,
    patch_size: int = 16,
    thresh_depth_low: float = 0.5,
    thresh_depth_med: float = 1.5,
    thresh_depth_high: float = 2.5,
    thresh_mv_low: float = 0.1,
    thresh_mv_med: float = 0.3,
    thresh_res_high: float = 0.5,
) -> np.ndarray:
    """Assign patch scale to each spatial location based on CTU depth + MV + residual.

    Args:
        depth_map: (H//8, W//8) uint8
        mv_magnitude: (H, W) float32 normalized [0,1], or None for I-frames
        residual_energy: (H, W) float32 normalized [0,1], or None for I-frames
        frame_h, frame_w: frame dimensions

    Returns:
        scale_map: (patch_grid_h, patch_grid_w) int32 with values in {8, 16, 32, 64}
    """
    patch_grid_h = frame_h // patch_size
    patch_grid_w = frame_w // patch_size

    # Aggregate depth to patch grid
    depth_grid = _aggregate_depth_to_patch_grid(depth_map, patch_grid_h, patch_grid_w)

    # Aggregate MV and residual if available
    if mv_magnitude is not None:
        mv_grid = _aggregate_energy_to_patch_grid(mv_magnitude, patch_grid_h, patch_grid_w, patch_size)
    else:
        mv_grid = np.zeros((patch_grid_h, patch_grid_w), dtype=np.float32)

    if residual_energy is not None:
        res_grid = _aggregate_energy_to_patch_grid(residual_energy, patch_grid_h, patch_grid_w, patch_size)
    else:
        res_grid = np.zeros((patch_grid_h, patch_grid_w), dtype=np.float32)

    # Decision logic
    scale_map = np.full((patch_grid_h, patch_grid_w), 16, dtype=np.int32)

    # Very homogeneous + static → large patch
    mask_64 = (depth_grid <= thresh_depth_low) & (mv_grid < thresh_mv_low)
    scale_map[mask_64] = 64

    # Moderately simple
    mask_32 = (~mask_64) & (depth_grid <= thresh_depth_med) & (mv_grid < thresh_mv_med)
    scale_map[mask_32] = 32

    # Very complex → fine patch
    mask_8 = (depth_grid >= thresh_depth_high) | (res_grid > thresh_res_high)
    scale_map[mask_8] = 8

    return scale_map


def enforce_token_budget(
    scale_maps: List[np.ndarray],
    energy_maps: List[np.ndarray],
    token_budget: int,
) -> List[np.ndarray]:
    """Adjust scale maps to meet token budget.

    Token cost per scale:
        64x64 → covers 4x4=16 base patches, costs 1 token
        32x32 → covers 2x2=4 base patches, costs 1 token
        16x16 → covers 1 base patch, costs 1 token
        8x8  → covers 0.25 base patches, costs 1 token (4 needed per base patch)

    But for budget, we count actual tokens produced:
        One 64x64 region that would be 16 base patches → 1 token
        One 32x32 region that would be 4 base patches → 1 token
        One 16x16 region → 1 token
        One 8x8 region → 4 tokens (split the base 16x16 into four 8x8)
    """
    # Count current tokens
    def count_tokens(scale_map):
        total = 0
        for ph in range(scale_map.shape[0]):
            for pw in range(scale_map.shape[1]):
                s = scale_map[ph, pw]
                if s == 64:
                    # This covers 4x4 = 16 base patch positions, producing 1 token
                    total += 1
                elif s == 32:
                    total += 1
                elif s == 16:
                    total += 1
                elif s == 8:
                    total += 4  # 4 sub-patches per base patch
        return total

    # For simplicity, we handle budget per-frame proportionally
    num_frames = len(scale_maps)
    budget_per_frame = token_budget // max(num_frames, 1)

    adjusted = []
    for frame_idx, (sm, em) in enumerate(zip(scale_maps, energy_maps)):
        sm = sm.copy()
        current = count_tokens(sm)

        # If over budget, promote small patches to larger (low energy first)
        if current > budget_per_frame:
            # Find 8x8 patches sorted by energy (ascending)
            fine_patches = []
            for ph in range(sm.shape[0]):
                for pw in range(sm.shape[1]):
                    if sm[ph, pw] == 8:
                        fine_patches.append((em[ph, pw], ph, pw))
            fine_patches.sort()  # lowest energy first

            for _, ph, pw in fine_patches:
                if current <= budget_per_frame:
                    break
                sm[ph, pw] = 16  # promote 8→16, saving 3 tokens
                current -= 3

            # If still over, promote 16→32
            if current > budget_per_frame:
                base_patches = []
                for ph in range(sm.shape[0]):
                    for pw in range(sm.shape[1]):
                        if sm[ph, pw] == 16:
                            base_patches.append((em[ph, pw], ph, pw))
                base_patches.sort()
                # Group into 2x2 blocks for promotion
                promoted = set()
                for _, ph, pw in base_patches:
                    if current <= budget_per_frame:
                        break
                    if (ph, pw) in promoted:
                        continue
                    # Check if 2x2 block is all scale=16
                    ph0 = (ph // 2) * 2
                    pw0 = (pw // 2) * 2
                    if (ph0 + 1 < sm.shape[0] and pw0 + 1 < sm.shape[1] and
                        sm[ph0, pw0] == 16 and sm[ph0, pw0 + 1] == 16 and
                        sm[ph0 + 1, pw0] == 16 and sm[ph0 + 1, pw0 + 1] == 16):
                        sm[ph0, pw0] = 32
                        sm[ph0, pw0 + 1] = 0  # mark as absorbed
                        sm[ph0 + 1, pw0] = 0
                        sm[ph0 + 1, pw0 + 1] = 0
                        promoted.update([(ph0, pw0), (ph0, pw0 + 1), (ph0 + 1, pw0), (ph0 + 1, pw0 + 1)])
                        current -= 3  # 4 tokens → 1 token

        # If under budget, split large patches (high energy first)
        elif current < budget_per_frame:
            large_patches = []
            for ph in range(sm.shape[0]):
                for pw in range(sm.shape[1]):
                    if sm[ph, pw] in (32, 64):
                        large_patches.append((-em[ph, pw], ph, pw))  # negative for descending
            large_patches.sort()

            for _, ph, pw in large_patches:
                if current >= budget_per_frame:
                    break
                if sm[ph, pw] == 64:
                    sm[ph, pw] = 32
                    current += 3  # 1 token → 4 tokens (but 32 is still 1 token)
                elif sm[ph, pw] == 32:
                    sm[ph, pw] = 16
                    # This is already 1 token, splitting doesn't help directly
                    # We'd need to "unsplit" the absorbed patches

        adjusted.append(sm)
    return adjusted


def build_patch_descriptors(
    scale_maps: List[np.ndarray],
    frame_indices: List[int],
    image_h: int,
    image_w: int,
    patch_size: int = 16,
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """Convert scale maps to patch descriptors with continuous positions.

    Returns:
        positions_thw: (N, 3) float32 continuous positions
        scale_indices: (N,) int32 scale indices
        patch_info: list of dicts with crop info for extracting pixel patches
    """
    ref_grid_h = image_h / patch_size
    ref_grid_w = image_w / patch_size

    positions = []
    scales = []
    patches_info = []

    for frame_idx, scale_map in zip(frame_indices, scale_maps):
        for ph in range(scale_map.shape[0]):
            for pw in range(scale_map.shape[1]):
                s = int(scale_map[ph, pw])
                if s == 0:
                    continue  # absorbed by larger patch

                # Pixel coordinates of patch top-left
                y = ph * patch_size
                x = pw * patch_size

                if s == 8:
                    # Split into 4 sub-patches
                    for si in range(2):
                        for sj in range(2):
                            sy = y + si * 8
                            sx = x + sj * 8
                            center_h = (sy + 4) / image_h * ref_grid_h
                            center_w = (sx + 4) / image_w * ref_grid_w
                            positions.append([float(frame_idx), center_h, center_w])
                            scales.append(SCALE_TO_IDX[8])
                            patches_info.append({
                                'frame': frame_idx, 'y': sy, 'x': sx,
                                'h': 8, 'w': 8, 'scale': 8,
                            })
                elif s in (16, 32, 64):
                    center_h = (y + s / 2) / image_h * ref_grid_h
                    center_w = (x + s / 2) / image_w * ref_grid_w
                    positions.append([float(frame_idx), center_h, center_w])
                    scales.append(SCALE_TO_IDX[s])
                    patches_info.append({
                        'frame': frame_idx, 'y': y, 'x': x,
                        'h': s, 'w': s, 'scale': s,
                    })

    if not positions:
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
                [])

    return (
        np.array(positions, dtype=np.float32),
        np.array(scales, dtype=np.int32),
        patches_info,
    )


def extract_pixel_patches(
    frames: np.ndarray,
    patches_info: List[dict],
    frame_indices: List[int],
) -> Dict[int, np.ndarray]:
    """Extract pixel patches from video frames grouped by scale.

    Args:
        frames: (T, H, W, C) uint8 or (T, C, H, W) float
        patches_info: list of patch descriptors from build_patch_descriptors
        frame_indices: mapping from patch frame index to frames array index

    Returns:
        {scale: (N_scale, C, scale, scale)} float32 patches
    """
    frame_idx_map = {fi: i for i, fi in enumerate(frame_indices)}
    patches_by_scale: Dict[int, list] = {8: [], 16: [], 32: [], 64: []}

    is_chw = frames.ndim == 4 and frames.shape[1] in (1, 3)

    for info in patches_info:
        fi = info['frame']
        arr_idx = frame_idx_map.get(fi, 0)
        y, x, h, w = info['y'], info['x'], info['h'], info['w']
        scale = info['scale']

        if is_chw:
            patch = frames[arr_idx, :, y:y + h, x:x + w]
        else:
            patch = frames[arr_idx, y:y + h, x:x + w]
            if patch.ndim == 3:
                patch = patch.transpose(2, 0, 1)  # HWC → CHW

        # Ensure correct size (handle boundary)
        C = patch.shape[0]
        if patch.shape[1] != h or patch.shape[2] != w:
            padded = np.zeros((C, h, w), dtype=patch.dtype)
            ph = min(patch.shape[1], h)
            pw = min(patch.shape[2], w)
            padded[:, :ph, :pw] = patch[:, :ph, :pw]
            patch = padded

        patches_by_scale[scale].append(patch.astype(np.float32))

    result = {}
    for scale, patches in patches_by_scale.items():
        if patches:
            result[scale] = np.stack(patches, axis=0)
    return result


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MultiGranularityVideoDataset(Dataset):
    """Video dataset with multi-granularity patch assignment.

    Each sample loads:
    1. Video frames via decord
    2. Pre-extracted depth maps (from HEVC CTU)
    3. Pre-extracted MV/residual energy (optional)

    Returns variable-length token descriptors.
    """

    def __init__(
        self,
        video_list: str,
        depth_map_dir: Optional[str] = None,
        mv_res_dir: Optional[str] = None,
        num_frames: int = 64,
        image_size: int = 224,
        patch_size: int = 16,
        token_budget: int = 2048,
        mode: str = "multigranularity",  # "multigranularity", "uniform", "random_multiscale", "entropy"
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        src_replace: str = "",
        dst_replace: str = "",
    ):
        super().__init__()
        self.num_frames = num_frames
        self.image_size = image_size
        self.patch_size = patch_size
        self.token_budget = token_budget
        self.mode = mode
        self.mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(3, 1, 1)
        self.depth_map_dir = depth_map_dir
        self.mv_res_dir = mv_res_dir
        self.src_replace = src_replace
        self.dst_replace = dst_replace

        # Load video list
        with open(video_list, 'r') as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]

        self.samples = []
        for line in lines:
            parts = line.split()
            video_path = parts[0]
            label = int(parts[1]) if len(parts) > 1 else 0
            self.samples.append((video_path, label))

    def __len__(self):
        return len(self.samples)

    def _get_depth_path(self, video_path: str) -> Optional[str]:
        if self.depth_map_dir is None:
            return None
        if self.src_replace and self.dst_replace:
            p = video_path.replace(self.src_replace, self.dst_replace)
        else:
            p = video_path
        stem = os.path.splitext(p)[0]
        path = stem + ".depth.npy"
        # Also try in depth_map_dir
        if not os.path.exists(path):
            basename = os.path.basename(stem) + ".depth.npy"
            path = os.path.join(self.depth_map_dir, basename)
        return path if os.path.exists(path) else None

    def _load_frames(self, video_path: str) -> np.ndarray:
        """Load and preprocess video frames. Returns (T, C, H, W) float32."""
        if not _HAS_DECORD:
            raise RuntimeError("decord required for video loading")

        vr = decord.VideoReader(video_path, num_threads=4)
        total = len(vr)
        T = self.num_frames

        if total >= T:
            indices = list(range(T))
        else:
            indices = list(range(total)) + [total - 1] * (T - total)

        frames = vr.get_batch(indices).numpy()  # (T, H, W, C)

        # Resize to image_size
        if _HAS_CV2 and (frames.shape[1] != self.image_size or frames.shape[2] != self.image_size):
            resized = np.empty((T, self.image_size, self.image_size, 3), dtype=np.uint8)
            for i in range(T):
                resized[i] = cv2.resize(frames[i], (self.image_size, self.image_size))
            frames = resized

        # HWC → CHW, normalize
        frames = frames.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        frames = (frames - self.mean[None]) / self.std[None]
        return frames

    def __getitem__(self, idx: int) -> dict:
        video_path, label = self.samples[idx]

        # Load frames
        try:
            frames = self._load_frames(video_path)
        except Exception:
            # Fallback: zero frames
            frames = np.zeros((self.num_frames, 3, self.image_size, self.image_size), dtype=np.float32)

        T = frames.shape[0]
        H = frames.shape[2]
        W = frames.shape[3]
        frame_indices = list(range(T))

        if self.mode == "multigranularity":
            return self._getitem_multigranularity(frames, frame_indices, H, W, label, video_path)
        elif self.mode == "uniform":
            return self._getitem_uniform(frames, frame_indices, H, W, label)
        elif self.mode == "random_multiscale":
            return self._getitem_random_multiscale(frames, frame_indices, H, W, label)
        else:
            return self._getitem_uniform(frames, frame_indices, H, W, label)

    def _getitem_multigranularity(self, frames, frame_indices, H, W, label, video_path):
        """Full multi-granularity pipeline with CTU depth."""
        T = len(frame_indices)
        pg_h = H // self.patch_size
        pg_w = W // self.patch_size

        # Load depth maps
        depth_path = self._get_depth_path(video_path)
        if depth_path is not None:
            try:
                depth_maps_all = np.load(depth_path)  # (T_all, H//8, W//8)
                depth_maps = depth_maps_all[:T]
            except Exception:
                depth_maps = [np.ones((H // 8, W // 8), dtype=np.uint8) for _ in range(T)]
        else:
            depth_maps = [np.ones((H // 8, W // 8), dtype=np.uint8) for _ in range(T)]

        if not isinstance(depth_maps, list):
            depth_maps = [depth_maps[i] for i in range(min(T, len(depth_maps)))]
            while len(depth_maps) < T:
                depth_maps.append(depth_maps[-1] if depth_maps else np.ones((H // 8, W // 8), dtype=np.uint8))

        # Assign scales per frame
        scale_maps = []
        energy_maps = []
        for i in range(T):
            # Simple energy: use pixel variance as proxy when no MV/residual
            frame_data = frames[i]  # (C, H, W)
            local_var = np.var(frame_data, axis=0)  # (H, W)
            pct = np.percentile(local_var, 95)
            energy = np.clip(local_var / max(pct, 1e-6), 0, 1).astype(np.float32)

            sm = assign_patch_scales(
                depth_maps[i], None, energy, H, W, self.patch_size,
            )
            scale_maps.append(sm)
            energy_maps.append(_aggregate_energy_to_patch_grid(energy, pg_h, pg_w, self.patch_size))

        # Enforce token budget
        scale_maps = enforce_token_budget(scale_maps, energy_maps, self.token_budget)

        # Build descriptors
        positions, scale_idx, patches_info = build_patch_descriptors(
            scale_maps, frame_indices, H, W, self.patch_size,
        )

        # Extract pixel patches
        patches_by_scale = extract_pixel_patches(frames, patches_info, frame_indices)

        return {
            'patches_by_scale': {s: torch.from_numpy(p) for s, p in patches_by_scale.items()},
            'positions': torch.from_numpy(positions),
            'scale_indices': torch.from_numpy(scale_idx).long(),
            'label': torch.tensor(label, dtype=torch.long),
            'num_tokens': len(positions),
        }

    def _getitem_uniform(self, frames, frame_indices, H, W, label):
        """Uniform 16x16 baseline (B2-style)."""
        T = len(frame_indices)
        pg_h = H // self.patch_size
        pg_w = W // self.patch_size
        total_patches = T * pg_h * pg_w

        # Build positions
        positions = []
        ref_grid_h = H / self.patch_size
        ref_grid_w = W / self.patch_size
        for t in frame_indices:
            for ph in range(pg_h):
                for pw in range(pg_w):
                    center_h = (ph * self.patch_size + self.patch_size / 2) / H * ref_grid_h
                    center_w = (pw * self.patch_size + self.patch_size / 2) / W * ref_grid_w
                    positions.append([float(t), center_h, center_w])

        positions = np.array(positions, dtype=np.float32)
        scale_idx = np.ones(total_patches, dtype=np.int32)  # all 16x16

        # Select top-K by energy
        K = min(self.token_budget, total_patches)
        energies = []
        for t in range(T):
            frame_var = np.var(frames[t], axis=0)
            for ph in range(pg_h):
                for pw in range(pg_w):
                    y0, x0 = ph * self.patch_size, pw * self.patch_size
                    e = frame_var[y0:y0 + self.patch_size, x0:x0 + self.patch_size].sum()
                    energies.append(e)
        energies = np.array(energies)
        topk_idx = np.argsort(energies)[-K:]
        topk_idx.sort()

        positions = positions[topk_idx]
        scale_idx = scale_idx[topk_idx]

        # Extract 16x16 patches
        patches_16 = []
        all_patch_coords = []
        for t in frame_indices:
            for ph in range(pg_h):
                for pw in range(pg_w):
                    all_patch_coords.append((t, ph * self.patch_size, pw * self.patch_size))

        for idx in topk_idx:
            t, y, x = all_patch_coords[idx]
            patches_16.append(frames[t, :, y:y + 16, x:x + 16])

        patches_16 = np.stack(patches_16, axis=0) if patches_16 else np.zeros((0, 3, 16, 16), dtype=np.float32)

        return {
            'patches_by_scale': {16: torch.from_numpy(patches_16)},
            'positions': torch.from_numpy(positions),
            'scale_indices': torch.from_numpy(scale_idx).long(),
            'label': torch.tensor(label, dtype=torch.long),
            'num_tokens': len(positions),
        }

    def _getitem_random_multiscale(self, frames, frame_indices, H, W, label):
        """Random multi-scale assignment (B3 ablation)."""
        T = len(frame_indices)
        pg_h = H // self.patch_size
        pg_w = W // self.patch_size

        rng = np.random.RandomState()
        scale_maps = []
        energy_maps = []
        for i in range(T):
            sm = np.random.choice([8, 16, 32, 64], size=(pg_h, pg_w)).astype(np.int32)
            scale_maps.append(sm)
            energy_maps.append(np.random.rand(pg_h, pg_w).astype(np.float32))

        scale_maps = enforce_token_budget(scale_maps, energy_maps, self.token_budget)

        positions, scale_idx, patches_info = build_patch_descriptors(
            scale_maps, frame_indices, H, W, self.patch_size,
        )
        patches_by_scale = extract_pixel_patches(frames, patches_info, frame_indices)

        return {
            'patches_by_scale': {s: torch.from_numpy(p) for s, p in patches_by_scale.items()},
            'positions': torch.from_numpy(positions),
            'scale_indices': torch.from_numpy(scale_idx).long(),
            'label': torch.tensor(label, dtype=torch.long),
            'num_tokens': len(positions),
        }


# ---------------------------------------------------------------------------
# Collator for variable-length samples
# ---------------------------------------------------------------------------


def multigranularity_collate_fn(samples: List[dict]) -> dict:
    """Collate variable-length multi-granularity samples into packed batch."""
    all_patches_by_scale: Dict[int, list] = {8: [], 16: [], 32: [], 64: []}
    all_positions = []
    all_scale_indices = []
    all_labels = []
    seq_lengths = []

    # Track per-scale counts for ordering reconstruction
    ordering_parts = []
    global_offset = 0

    for sample in samples:
        n = sample['num_tokens']
        seq_lengths.append(n)
        all_positions.append(sample['positions'])
        all_scale_indices.append(sample['scale_indices'])
        all_labels.append(sample['label'])

        for scale in [8, 16, 32, 64]:
            if scale in sample['patches_by_scale']:
                all_patches_by_scale[scale].append(sample['patches_by_scale'][scale])

    # Concatenate
    packed_positions = torch.cat(all_positions, dim=0) if all_positions else torch.zeros(0, 3)
    packed_scales = torch.cat(all_scale_indices, dim=0) if all_scale_indices else torch.zeros(0, dtype=torch.long)

    patches_by_scale = {}
    for scale in [8, 16, 32, 64]:
        if all_patches_by_scale[scale]:
            patches_by_scale[scale] = torch.cat(all_patches_by_scale[scale], dim=0)

    total_L = packed_positions.shape[0]
    cu_seqlens = torch.zeros(len(seq_lengths) + 1, dtype=torch.int32)
    for i, sl in enumerate(seq_lengths):
        cu_seqlens[i + 1] = cu_seqlens[i] + sl
    max_seqlen = max(seq_lengths) if seq_lengths else 0

    labels = torch.stack(all_labels) if all_labels else torch.zeros(0, dtype=torch.long)

    return {
        'patches_by_scale': patches_by_scale,
        'positions': packed_positions,
        'scale_indices': packed_scales,
        'cu_seqlens': cu_seqlens,
        'max_seqlen': max_seqlen,
        'seq_lengths': seq_lengths,
        'labels': labels,
    }
