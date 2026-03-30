#!/bin/bash
set -e

source /home/haolin.yang/.bashrc
conda activate vit

export CUDA_VISIBLE_DEVICES=0
export HEVC_FEAT_DECODER=/nfs-stor/haolin.yang/Code/VFM/Video_MLCD/codec-infra/hevc-quadtree/hevc

cd /nfs-stor/haolin.yang/Code/VFM/Video_MLCD/LLava-ViT

python -c "
import os, sys, json, subprocess as sp
import numpy as np
import cv2
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- Import both dataloaders ----
sys.path.insert(0, '/nfs-stor/haolin.yang/Code/VFM/Video_MLCD/model_factory')
from quadtree_dataset import QuadtreeVideoDataset
from multigranularity_dataset import multigranularity_collate_fn

# ---- Config ----
CELL = 8
PS = 16
PPF = 14 * 14  # 196
LINE_COLOR = (0, 255, 255)  # yellow
VIZ_ROOT = '/nfs-stor/haolin.yang/Code/VFM/Video_MLCD/viz_results'

# Pick a video that has both cu_visidx and is accessible
test_video = '/nfs-stor/haolin.yang/video_data/K710_train_hevc_gop32/2/6/rank_000_sample_0000018826.mp4'
name = 'rank_000_sample_0000018826'
out_dir = os.path.join(VIZ_ROOT, f'dataloader_viz_{name}')
os.makedirs(out_dir, exist_ok=True)

# ---- Decode video frames for visualization ----
def decode_frames(path, n):
    info = json.loads(sp.run(
        ['ffprobe', '-v', 'error', '-show_streams', '-select_streams', 'v:0',
         '-print_format', 'json', path], capture_output=True, text=True).stdout)
    st = info['streams'][0]
    W, H = int(st['width']), int(st['height'])
    buf = sp.run(['ffmpeg', '-i', path, '-vframes', str(n),
                  '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-v', 'error', '-'],
                 capture_output=True).stdout
    fsz = H * W * 3
    k = min(n, len(buf) // fsz)
    return [np.frombuffer(buf, np.uint8, fsz, i * fsz).reshape(H, W, 3).copy() for i in range(k)], H, W

frames_bgr, H, W = decode_frames(test_video, 64)
actual_T = len(frames_bgr)
print(f'Video: {name} | {W}x{H} | {actual_T} frames')

# ===========================================================================
# Part 1: QuadtreeVideoDataset (CU selection, multi-scale)
# ===========================================================================
print('\\n=== QuadtreeVideoDataset (CU multi-scale) ===')
list_path = '/tmp/viz_dl_test.txt'
with open(list_path, 'w') as f:
    f.write(f'{test_video} 0\\n')

qt_ds = QuadtreeVideoDataset(
    video_list=list_path,
    cu_visidx_src='_hevc_gop32',
    cu_visidx_dst='_cu_selection_gop32',
    num_frames=64,
)
qt_sample = qt_ds[0]
qt_n = qt_sample['num_tokens']
qt_pos = qt_sample['positions'].numpy()   # (K, 3)
qt_scales = qt_sample['scale_indices'].numpy()  # (K,)

# Decode scale from scale_indices: 1->16, 2->32, 3->64
idx_to_px = {1: 16, 2: 32, 3: 64}
qt_scale_px = np.array([idx_to_px[int(s)] for s in qt_scales])

# Count per scale
n16 = int((qt_scale_px == 16).sum())
n32 = int((qt_scale_px == 32).sum())
n64 = int((qt_scale_px == 64).sum())
print(f'  Tokens: {qt_n} (16px={n16}, 32px={n32}, 64px={n64})')

# Load cu_visidx directly for pixel coords
cu_path = test_video.replace('_hevc_gop32', '_cu_selection_gop32').replace('.mp4', '.cu_visidx.npz')
cu_data = np.load(cu_path)
cu_t = cu_data['t'].astype(np.int64)
cu_y = cu_data['y'].astype(np.int64)
cu_x = cu_data['x'].astype(np.int64)
cu_counts = cu_data['counts']
cu_n16, cu_n32, cu_n64 = int(cu_counts[0]), int(cu_counts[1]), int(cu_counts[2])
cu_scale = np.empty(len(cu_t), dtype=np.int32)
cu_scale[:cu_n16] = 16
cu_scale[cu_n16:cu_n16+cu_n32] = 32
cu_scale[cu_n16+cu_n32:] = 64

# Per-frame counts
qt_pf = np.zeros(actual_T, dtype=int)
for i in range(len(cu_t)):
    fi = int(cu_t[i])
    if 0 <= fi < actual_T:
        qt_pf[fi] += 1

# ===========================================================================
# Part 2: Original MV+Res visidx (uniform 16x16)
# ===========================================================================
print('\\n=== MV+Res visidx (uniform 16x16) ===')
mv_path = test_video.replace('_hevc_gop32', '_residual_mv_gop32').replace('.mp4', '.visidx.npy')
mv_idx = np.load(mv_path)
print(f'  Tokens: {len(mv_idx)} (all 16x16)')

mv_pf = np.zeros(actual_T, dtype=int)
for idx in mv_idx:
    fi = idx // PPF
    if 0 <= fi < actual_T:
        mv_pf[fi] += 1

# ===========================================================================
# Draw functions
# ===========================================================================
def draw_cu_on_frame(bgr, frame_idx, cu_t, cu_y, cu_x, cu_scale):
    out = bgr.copy()
    mask = (cu_t == frame_idx)
    for i in np.where(mask)[0]:
        y, x, s = int(cu_y[i]), int(cu_x[i]), int(cu_scale[i])
        y1 = min(y + s, H) - 1
        x1 = min(x + s, W) - 1
        cv2.rectangle(out, (x, y), (x1, y1), LINE_COLOR, 1, cv2.LINE_AA)
    return out

def draw_mv_on_frame(bgr224, frame_idx, mv_idx):
    out = bgr224.copy()
    gh, gw = 14, 14
    for idx in mv_idx:
        if idx // PPF != frame_idx:
            continue
        rem = idx % PPF
        r, c = rem // gw, rem % gw
        y0, x0 = r * PS, c * PS
        y1, x1 = min((r+1)*PS, 224)-1, min((c+1)*PS, 224)-1
        cv2.rectangle(out, (x0, y0), (x1, y1), LINE_COLOR, 1, cv2.LINE_AA)
    return out

# ===========================================================================
# Visualization: 8 frames x 3 columns
# ===========================================================================
sample_idx = [0, 1, 9, 16, 32, 33, 45, 63]
sample_idx = [i for i in sample_idx if i < actual_T]

# Determine I-frames from cu_visidx (frames with many tokens = I-frame)
i_frame_set = set()
for fi in range(actual_T):
    if qt_pf[fi] > 100:  # I-frames have ~196+ tokens
        i_frame_set.add(fi)

fig, axes = plt.subplots(len(sample_idx), 3, figsize=(18, len(sample_idx) * 5.5), dpi=150)
fig.patch.set_facecolor('white')
fig.suptitle(
    f'{name} | {W}x{H} | Dataloader Comparison\\n'
    f'CU Quadtree: {qt_n} tok (16={n16}, 32={n32}, 64={n64}) | '
    f'MV+Res: {len(mv_idx)} tok (uniform 16x16)',
    fontsize=12, fontweight='bold', y=0.99,
)

cols = ['Original',
        f'CU Quadtree (QuadtreeDataset)\\n{qt_n} tok, multi-scale',
        f'MV+Res visidx\\n{len(mv_idx)} tok, uniform 16x16']
for col, lb in enumerate(cols):
    axes[0, col].set_title(lb, fontsize=10, fontweight='bold', pad=10)

for row, fi in enumerate(sample_idx):
    bgr = frames_bgr[fi]
    ft_str = 'I' if fi in i_frame_set else 'P/B'

    # Col 0: Original
    axes[row, 0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    axes[row, 0].set_ylabel(f'F{fi}({ft_str})', fontsize=10, fontweight='bold',
                             rotation=0, labelpad=35)
    axes[row, 0].axis('off')

    # Col 1: CU Quadtree
    img_cu = draw_cu_on_frame(bgr, fi, cu_t, cu_y, cu_x, cu_scale)
    axes[row, 1].imshow(cv2.cvtColor(img_cu, cv2.COLOR_BGR2RGB))
    axes[row, 1].set_xlabel(f'{int(qt_pf[fi])} tok', fontsize=9)
    axes[row, 1].axis('off')

    # Col 2: MV+Res (resize to 224 for drawing)
    bgr224 = cv2.resize(bgr, (224, 224))
    img_mv = draw_mv_on_frame(bgr224, fi, mv_idx)
    axes[row, 2].imshow(cv2.cvtColor(img_mv, cv2.COLOR_BGR2RGB))
    axes[row, 2].set_xlabel(f'{int(mv_pf[fi])} tok', fontsize=9)
    axes[row, 2].axis('off')

plt.tight_layout(rect=[0.04, 0, 1, 0.95])
fig.savefig(os.path.join(out_dir, 'compare.png'), bbox_inches='tight', facecolor='white')
plt.close(fig)

# Token bar chart
fig2, ax = plt.subplots(figsize=(16, 4), dpi=150)
xs = np.arange(actual_T)
bw = 0.35
ax.bar(xs - bw/2, qt_pf, bw, label=f'CU Quadtree ({qt_n})', color='#E67E22', alpha=0.85)
ax.bar(xs + bw/2, mv_pf, bw, label=f'MV+Res ({len(mv_idx)})', color='#5B9BD5', alpha=0.85)
for fi in i_frame_set:
    ax.axvline(fi, color='red', ls='--', alpha=0.5, lw=1)
ax.set_xlabel('Frame')
ax.set_ylabel('Tokens')
ax.set_title(f'CU Quadtree vs MV+Res per-frame tokens — {name}', fontweight='bold')
ax.legend(fontsize=9)
fig2.tight_layout()
fig2.savefig(os.path.join(out_dir, 'token_compare.png'), bbox_inches='tight', facecolor='white')
plt.close(fig2)

# Verify patches_by_scale shapes from dataset
print(f'\\n=== QuadtreeVideoDataset output shapes ===')
for s, p in qt_sample['patches_by_scale'].items():
    print(f'  scale {s}px: {p.shape} (dtype={p.dtype})')
print(f'  positions: {qt_sample[\"positions\"].shape}')
print(f'  scale_indices: {qt_sample[\"scale_indices\"].shape}')

print(f'\\nSaved to {out_dir}/')
"
