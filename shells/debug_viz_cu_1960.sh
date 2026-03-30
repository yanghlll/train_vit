#!/bin/bash
set -e

source /home/haolin.yang/.bashrc
conda activate vit

export CUDA_VISIBLE_DEVICES=0
export HEVC_FEAT_DECODER=/nfs-stor/haolin.yang/Code/VFM/Video_MLCD/codec-infra/hevc-quadtree/hevc
export PYTHONPATH="/nfs-stor/haolin.yang/Code/VFM/Video_MLCD/codec-infra/hevc-quadtree:${PYTHONPATH}"

cd /nfs-stor/haolin.yang/Code/VFM/Video_MLCD/LLava-ViT

python -c "
import os, sys, json, subprocess as sp
import numpy as np
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/nfs-stor/haolin.yang/Code/VFM/Video_MLCD/codec-infra/hevc-quadtree')
sys.path.insert(0, '/nfs-stor/haolin.yang/Code/VFM/Video_MLCD/model_factory')
os.environ['HEVC_FEAT_DECODER'] = '/nfs-stor/haolin.yang/Code/VFM/Video_MLCD/codec-infra/hevc-quadtree/hevc'

from hevc_feature_decoder_quadtree import HevcFeatureReader
from step3_extract_cu_selection import (
    classify_modes, build_cu_grid, enumerate_cu_blocks, select_D, CELL, SKIP, INTRA, MERGE, AMVP
)

LINE_COLOR = (0, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
CYAN = (255, 255, 0)
VIZ_ROOT = '/nfs-stor/haolin.yang/Code/VFM/Video_MLCD/viz_results'

test_videos = [
    '/nfs-stor/haolin.yang/video_data/K710_train_hevc_gop32/2/6/rank_000_sample_0000018826.mp4',
    '/nfs-stor/haolin.yang/video_data/K710_train_hevc_gop32/7/3/rank_001_sample_0000067773.mp4',
    '/nfs-stor/haolin.yang/video_data/K710_train_hevc_gop32/0/5/rank_001_sample_0000062005.mp4',
]

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


def online_select(video_path, T=64, K_pb=2000):
    \"\"\"Run select_D online, return per-token (t, y_px, x_px, scale_px).\"\"\"
    rdr = HevcFeatureReader(video_path, nb_frames=T, n_parallel=1)
    H, W = rdr.height, rdr.width
    modes_list, frame_types, size_maps = [], [], []
    for i, (ft, meta) in enumerate(rdr.nextFrameEx()):
        if i >= T: break
        (ftype, qt, rgb, mvx0, mvy0, mvx1, mvy1, r0, r1, sm, resid) = ft
        ft_val = 0 if meta.get('is_i_frame', False) else 1
        frame_types.append(ft_val)
        size_maps.append(sm.copy())
        modes_list.append(classify_modes(ft_val, sm, mvx0, mvy0, r0, H, W))
    rdr.close()

    frame_types_arr = np.array(frame_types, dtype=np.uint8)
    actual_T = len(frame_types)
    selected, all_blocks, n_i, n_pb = select_D(
        modes_list, size_maps, frame_types_arr, actual_T, K_pb
    )
    # selected: list of (fi, y_px, x_px, bs_cells)
    t_arr = np.array([s[0] for s in selected], dtype=np.int64)
    y_arr = np.array([s[1] for s in selected], dtype=np.int64)
    x_arr = np.array([s[2] for s in selected], dtype=np.int64)
    bs_arr = np.array([s[3] for s in selected], dtype=np.int32)
    # bs is in cells, convert to pixels
    scale_arr = bs_arr * CELL
    return t_arr, y_arr, x_arr, scale_arr, n_i, n_pb, frame_types


def draw_cu(bgr, t_arr, y_arr, x_arr, scale_arr, frame_idx, color=LINE_COLOR):
    out = bgr.copy()
    H, W = bgr.shape[:2]
    mask = (t_arr == frame_idx)
    for i in np.where(mask)[0]:
        y, x, s = int(y_arr[i]), int(x_arr[i]), int(scale_arr[i])
        cv2.rectangle(out, (x, y), (min(x+s, W)-1, min(y+s, H)-1), color, 1, cv2.LINE_AA)
    return out


def draw_diff(bgr, H, W, t1, y1, x1, s1, t2, y2, x2, s2, fi):
    \"\"\"green=both, red=stored only, cyan=online only\"\"\"
    out = bgr.copy()
    set1 = set()
    set2 = set()
    for i in np.where(t1 == fi)[0]:
        set1.add((int(y1[i]), int(x1[i]), int(s1[i])))
    for i in np.where(t2 == fi)[0]:
        set2.add((int(y2[i]), int(x2[i]), int(s2[i])))
    for (y, x, s) in set1 | set2:
        in1 = (y, x, s) in set1
        in2 = (y, x, s) in set2
        if in1 and in2:
            color = GREEN
        elif in1 and not in2:
            color = RED       # stored only
        else:
            color = CYAN      # online only
        cv2.rectangle(out, (x, y), (min(x+s, W)-1, min(y+s, H)-1), color, 1, cv2.LINE_AA)
    return out


for vi, test_video in enumerate(test_videos):
    name = os.path.splitext(os.path.basename(test_video))[0]
    out_dir = os.path.join(VIZ_ROOT, f'cu_verify_{vi}_{name}')
    os.makedirs(out_dir, exist_ok=True)
    print(f'\\n[{vi}] {name}')

    frames_bgr, H, W = decode_frames(test_video, 64)
    actual_T = len(frames_bgr)

    # ---- Stored cu_visidx.npz ----
    cu_path = test_video.replace('_hevc_gop32', '_cu_selection_gop32').replace('.mp4', '.cu_visidx.npz')
    cu_data = np.load(cu_path)
    st_t = cu_data['t'].astype(np.int64)
    st_y = cu_data['y'].astype(np.int64)
    st_x = cu_data['x'].astype(np.int64)
    st_counts = cu_data['counts']
    sn16, sn32, sn64 = int(st_counts[0]), int(st_counts[1]), int(st_counts[2])
    st_K = sn16 + sn32 + sn64
    st_scale = np.empty(st_K, dtype=np.int32)
    st_scale[:sn16] = 16
    st_scale[sn16:sn16+sn32] = 32
    st_scale[sn16+sn32:] = 64

    # ---- Online select_D ----
    on_t, on_y, on_x, on_scale, on_ni, on_npb, frame_types = online_select(test_video, 64, 2000)
    on_K = len(on_t)
    on_n16 = int((on_scale == 16).sum())
    on_n32 = int((on_scale == 32).sum())
    on_n64 = int((on_scale == 64).sum())

    # I-frame detection
    i_frame_ids = [fi for fi in range(len(frame_types)) if frame_types[fi] == 0]

    # Overlap check
    st_set = set(zip(st_t.tolist(), st_y.tolist(), st_x.tolist(), st_scale.tolist()))
    on_set = set(zip(on_t.tolist(), on_y.tolist(), on_x.tolist(), on_scale.tolist()))
    overlap = len(st_set & on_set)

    print(f'  Stored:  {st_K} tok (16={sn16}, 32={sn32}, 64={sn64})')
    print(f'  Online:  {on_K} tok (16={on_n16}, 32={on_n32}, 64={on_n64}, I={on_ni}, P/B={on_npb})')
    print(f'  Overlap: {overlap}/{st_K} ({overlap/max(st_K,1)*100:.1f}%)')
    print(f'  I-frames: {i_frame_ids}')

    # Per-frame counts
    st_pf = np.zeros(actual_T, dtype=int)
    on_pf = np.zeros(actual_T, dtype=int)
    for i in range(st_K):
        fi = int(st_t[i])
        if 0 <= fi < actual_T: st_pf[fi] += 1
    for i in range(on_K):
        fi = int(on_t[i])
        if 0 <= fi < actual_T: on_pf[fi] += 1

    # ---- Visualization: 4 columns ----
    sample_idx = [0, 1, 9, 16, 32, 33, 45, 63]
    sample_idx = [i for i in sample_idx if i < actual_T]

    fig, axes = plt.subplots(len(sample_idx), 4, figsize=(24, len(sample_idx) * 5.5), dpi=150)
    fig.patch.set_facecolor('white')
    fig.suptitle(
        f'{name} | {W}x{H} | I-frames={i_frame_ids}\\n'
        f'Stored: {st_K} tok | Online: {on_K} tok | Overlap: {overlap} ({overlap/max(st_K,1)*100:.1f}%)',
        fontsize=12, fontweight='bold', y=0.99,
    )
    cols = ['Original',
            f'Stored cu_visidx\\n({st_K} tok)',
            f'Online select_D\\n({on_K} tok)',
            'Diff\\n(green=both, red=stored, cyan=online)']
    for col, lb in enumerate(cols):
        axes[0, col].set_title(lb, fontsize=10, fontweight='bold', pad=10)

    for row, fi in enumerate(sample_idx):
        bgr = frames_bgr[fi]
        ft_str = 'I' if fi in i_frame_ids else 'P/B'

        axes[row, 0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        axes[row, 0].set_ylabel(f'F{fi}({ft_str})', fontsize=10, fontweight='bold',
                                 rotation=0, labelpad=35)
        axes[row, 0].axis('off')

        img_st = draw_cu(bgr, st_t, st_y, st_x, st_scale, fi)
        axes[row, 1].imshow(cv2.cvtColor(img_st, cv2.COLOR_BGR2RGB))
        axes[row, 1].set_xlabel(f'{int(st_pf[fi])} tok', fontsize=9)
        axes[row, 1].axis('off')

        img_on = draw_cu(bgr, on_t, on_y, on_x, on_scale, fi)
        axes[row, 2].imshow(cv2.cvtColor(img_on, cv2.COLOR_BGR2RGB))
        axes[row, 2].set_xlabel(f'{int(on_pf[fi])} tok', fontsize=9)
        axes[row, 2].axis('off')

        img_diff = draw_diff(bgr, H, W, st_t, st_y, st_x, st_scale,
                             on_t, on_y, on_x, on_scale, fi)
        axes[row, 3].imshow(cv2.cvtColor(img_diff, cv2.COLOR_BGR2RGB))
        axes[row, 3].axis('off')

    plt.tight_layout(rect=[0.04, 0, 1, 0.95])
    fig.savefig(os.path.join(out_dir, 'compare.png'), bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # Token bar chart
    fig2, ax = plt.subplots(figsize=(16, 4), dpi=150)
    xs = np.arange(actual_T)
    bw = 0.35
    ax.bar(xs - bw/2, st_pf, bw, label=f'Stored ({st_K})', color='#E67E22', alpha=0.85)
    ax.bar(xs + bw/2, on_pf, bw, label=f'Online ({on_K})', color='#5B9BD5', alpha=0.85)
    for fi in i_frame_ids:
        ax.axvline(fi, color='red', ls='--', alpha=0.5, lw=1)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Fused CU Tokens')
    ax.set_title(f'Stored vs Online CU Selection — {name} | Overlap={overlap/max(st_K,1)*100:.1f}%', fontweight='bold')
    ax.legend(fontsize=9)
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, 'token_compare.png'), bbox_inches='tight', facecolor='white')
    plt.close(fig2)

    print(f'  Saved to {out_dir}/')

print('\\nDone!')
"
