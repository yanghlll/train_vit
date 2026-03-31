#!/bin/bash
set -e
source /home/haolin.yang/.bashrc
conda activate vit

export HEVC_FEAT_DECODER=/nfs-stor/haolin.yang/Code/VFM/Video_MLCD/codec-infra/hevc-quadtree/hevc

python -c "
import os, sys, json, subprocess as sp
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LINE_COLOR = (0, 255, 255)
VIZ_ROOT = '/nfs-stor/haolin.yang/Code/VFM/Video_MLCD/viz_results'

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

def draw_cu(bgr, t_arr, y_arr, x_arr, scale_arr, frame_idx):
    out = bgr.copy()
    H, W = bgr.shape[:2]
    mask = (t_arr == frame_idx)
    for i in np.where(mask)[0]:
        y, x, s = int(y_arr[i]), int(x_arr[i]), int(scale_arr[i])
        cv2.rectangle(out, (x, y), (min(x+s, W)-1, min(y+s, H)-1), LINE_COLOR, 1, cv2.LINE_AA)
    return out

# Load low-token video list
low_list = '/nfs-stor/haolin.yang/video_data/cu_selection_low_token_videos.txt'
videos = []
with open(low_list) as f:
    for line in f:
        parts = line.strip().split('\t')
        vp, k = parts[0], int(parts[1])
        videos.append((k, vp))

# Sort by K, pick diverse samples: worst 3 + some medium
videos.sort()
selected = videos[:3]  # K < 100
selected += [v for v in videos if 200 < v[0] < 300][:2]
selected += [v for v in videos if 800 < v[0] < 1000][:2]
selected += [v for v in videos if 1500 < v[0] < 1700][:2]

print(f'Total low-token videos: {len(videos)}')
print(f'Visualizing {len(selected)} samples')

for vi, (K_total, vp) in enumerate(selected):
    name = os.path.splitext(os.path.basename(vp))[0]
    out_dir = os.path.join(VIZ_ROOT, f'low_token_{vi}_{name}_K{K_total}')
    os.makedirs(out_dir, exist_ok=True)

    # Load cu_visidx
    cu_npy = vp.replace('_hevc_gop32', '_cu_selection_gop32').replace('.mp4', '.cu_visidx.npy')
    cu_npz = cu_npy.replace('.npy', '.npz')
    if os.path.exists(cu_npy):
        arr = np.load(cu_npy)
        t_all = arr[:, 0].astype(np.int64)
        y_all = arr[:, 1].astype(np.int64)
        x_all = arr[:, 2].astype(np.int64)
        scale_px = arr[:, 3].astype(np.int32)
    else:
        data = np.load(cu_npz)
        counts = data['counts']
        t_all = data['t'].astype(np.int64)
        y_all = data['y'].astype(np.int64)
        x_all = data['x'].astype(np.int64)
        n16, n32, n64 = int(counts[0]), int(counts[1]), int(counts[2])
        K = n16 + n32 + n64
        scale_px = np.empty(K, dtype=np.int32)
        scale_px[:n16] = 16
        scale_px[n16:n16+n32] = 32
        scale_px[n16+n32:] = 64

    K = len(t_all)
    n16 = int((scale_px == 16).sum())
    n32 = int((scale_px == 32).sum())
    n64 = K - n16 - n32

    # Per-frame counts
    frames_bgr, H, W = decode_frames(vp, 64)
    actual_T = len(frames_bgr)
    pf = np.zeros(actual_T, dtype=int)
    for i in range(K):
        fi = int(t_all[i])
        if 0 <= fi < actual_T:
            pf[fi] += 1

    i_frame_ids = [fi for fi in range(actual_T) if pf[fi] > 50]

    print(f'  [{vi}] {name}: K={K} (16={n16}, 32={n32}, 64={n64}), I-frames={i_frame_ids}')

    # 8 frames
    sample_idx = [0, 1, 9, 16, 32, 33, 45, 63]
    sample_idx = [i for i in sample_idx if i < actual_T]

    fig, axes = plt.subplots(len(sample_idx), 2, figsize=(14, len(sample_idx) * 5.5), dpi=150)
    fig.patch.set_facecolor('white')
    fig.suptitle(
        f'{name} | K={K} tokens (16={n16}, 32={n32}, 64={n64})\\n'
        f'LOW TOKEN VIDEO — I-frames={i_frame_ids}',
        fontsize=12, fontweight='bold', y=0.99, color='red',
    )

    cols = ['Original', f'CU Selection (K={K} tokens)']
    for col, lb in enumerate(cols):
        axes[0, col].set_title(lb, fontsize=11, fontweight='bold', pad=10)

    for row, fi in enumerate(sample_idx):
        bgr = frames_bgr[fi]
        ft_str = 'I' if fi in i_frame_ids else 'P/B'

        axes[row, 0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        axes[row, 0].set_ylabel(f'F{fi}({ft_str})', fontsize=10, fontweight='bold',
                                 rotation=0, labelpad=35)
        axes[row, 0].axis('off')

        img_cu = draw_cu(bgr, t_all, y_all, x_all, scale_px, fi)
        axes[row, 1].imshow(cv2.cvtColor(img_cu, cv2.COLOR_BGR2RGB))
        axes[row, 1].set_xlabel(f'{int(pf[fi])} tok', fontsize=9)
        axes[row, 1].axis('off')

    plt.tight_layout(rect=[0.04, 0, 1, 0.94])
    fig.savefig(os.path.join(out_dir, 'compare.png'), bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # Token bar chart
    fig2, ax = plt.subplots(figsize=(16, 4), dpi=150)
    colors = ['#E74C3C' if fi in i_frame_ids else '#3498DB' for fi in range(actual_T)]
    ax.bar(range(actual_T), pf, color=colors, alpha=0.85)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Tokens')
    ax.set_title(f'LOW TOKEN: {name} | K={K} (red=I, blue=P/B)', fontweight='bold', color='red')
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, 'token_dist.png'), bbox_inches='tight', facecolor='white')
    plt.close(fig2)

    print(f'    Saved to {out_dir}/')

print('\\nDone!')
"
