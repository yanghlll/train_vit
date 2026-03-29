#!/usr/bin/env python3
"""
Build inverted index from step4 collision labels + video lists.
Produces:
  - inverted_index.pkl: {cluster_id: [global_video_idx, ...]}
  - video_list_all.txt: one video absolute path per line, order matches label rows
"""
import argparse
import glob
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_dir", required=True, help="Directory with *_label_*.npy files")
    parser.add_argument("--label_suffix", required=True, help="Label file suffix, e.g. '_label_merged_centers_500k'")
    parser.add_argument("--video_lists", nargs="+", required=True,
                        help="Ordered video list files (abs paths), matching feature file order")
    parser.add_argument("--video_roots", nargs="+", required=True,
                        help="Root dirs for each video list (to resolve relative paths)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--topk", type=int, default=10, help="Number of label columns to use (default: all)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build ordered video path list
    print("Building video list...")
    video_paths = []
    for vlist, vroot in zip(args.video_lists, args.video_roots):
        with open(vlist) as f:
            for line in f:
                rel = line.strip()
                if rel:
                    video_paths.append(str(Path(vroot) / rel))
    print(f"Total videos: {len(video_paths)}")

    # Save video list
    video_list_path = output_dir / "video_list_all.txt"
    with open(video_list_path, "w") as f:
        f.write("\n".join(video_paths) + "\n")
    print(f"Saved video list: {video_list_path}")

    # 2. Load all label files in sorted order and concatenate
    print("Loading labels...")
    label_files = sorted(glob.glob(str(Path(args.label_dir) / f"*{args.label_suffix}.npy")))
    print(f"Found {len(label_files)} label files")

    all_labels = np.concatenate([np.load(f) for f in label_files], axis=0)
    print(f"Labels shape: {all_labels.shape}")  # (N, topk)

    if all_labels.shape[0] > len(video_paths):
        print(f"Labels ({all_labels.shape[0]}) > videos ({len(video_paths)}), "
              f"truncating {all_labels.shape[0] - len(video_paths)} padding rows")
        all_labels = all_labels[:len(video_paths)]
    elif all_labels.shape[0] < len(video_paths):
        print(f"WARNING: Labels ({all_labels.shape[0]}) < videos ({len(video_paths)}), "
              f"truncating video list")
        video_paths = video_paths[:all_labels.shape[0]]

    # Limit topk columns
    if args.topk < all_labels.shape[1]:
        all_labels = all_labels[:, :args.topk]

    # 3. Build inverted index
    print("Building inverted index...")
    inverted_index = defaultdict(list)
    for row_idx in range(all_labels.shape[0]):
        for label in np.unique(all_labels[row_idx]):
            inverted_index[int(label)].append(row_idx)
        if (row_idx + 1) % 100000 == 0:
            print(f"  {row_idx + 1}/{all_labels.shape[0]}")

    inverted_index = dict(inverted_index)
    print(f"Unique clusters: {len(inverted_index)}")

    # Stats
    sizes = [len(v) for v in inverted_index.values()]
    print(f"Cluster size: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f}, median={np.median(sizes):.1f}")

    # Save
    idx_path = output_dir / "inverted_index.pkl"
    with open(idx_path, "wb") as f:
        pickle.dump(inverted_index, f)
    print(f"Saved inverted index: {idx_path}")

    # Save all_labels as single file too
    np.save(output_dir / "all_labels.npy", all_labels)
    print(f"Saved concatenated labels: {output_dir / 'all_labels.npy'}")


if __name__ == "__main__":
    main()
