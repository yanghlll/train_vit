#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#! conda create -n faiss python=3.9
#! conda install -c pytorch -c conda-forge faiss-gpu=1.7.3 cudatoolkit=11.8 numpy scipy

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function

import argparse
import os
import sys
import time

import faiss
import numpy as np


def iter_npy_paths(src):
    """
    Generator: Yields .npy file paths to load from a directory, list file, or single .npy path.
    - Directory: Collects all .npy files in the directory (non-recursive), sorted by filename
    - List file: Reads line by line, relative paths are relative to the list file's directory, ignores empty lines and non-existent paths
    - Single .npy file: Returns that path directly
    """
    if os.path.isdir(src):
        for fn in sorted(os.listdir(src)):
            if fn.lower().endswith(".npy"):
                yield os.path.join(src, fn)
        return

    if os.path.isfile(src):
        if src.lower().endswith(".npy"):
            yield src
            return

        # Treat as list file
        base = os.path.dirname(os.path.abspath(src))
        collected = []
        with open(src, "r", encoding="utf-8") as f:
            for line in f:
                p = line.strip()
                if not p or p.startswith("#"):
                    continue
                if not os.path.isabs(p):
                    p = os.path.normpath(os.path.join(base, p))
                if os.path.isfile(p) and p.lower().endswith(".npy"):
                    collected.append(p)
                else:
                    print(f"[warn] Path in list does not exist or is not .npy, ignored: {p}", file=sys.stderr)
        for p in sorted(collected):
            yield p
        return

    print(f"[error] --input path is invalid: {src}", file=sys.stderr)


def load_and_concat(paths, drop_last=False):
    """
    Load multiple .npy files and concatenate along the sample dimension.
    - Arrays are reshaped to [num_samples, -1] to ensure consistent concatenation
    - When drop_last=True, each array has arr = arr[:, :-1] applied
    - File paths are sorted to ensure stable order
    """
    paths = sorted(list(paths))
    if not paths:
        raise FileNotFoundError("No .npy files were collected")

    arrays = []
    for idx, p in enumerate(paths):
        print(f"[load] ({idx+1}/{len(paths)}) {p}")
        arr = np.load(p)
        if arr.ndim == 0:
            raise ValueError(f"File {p} loaded as scalar, cannot use")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        else:
            # Flatten subsequent dimensions to make it 2D [num_samples, feat_dim]
            arr = arr.reshape(arr.shape[0], -1)

        if drop_last and arr.shape[1] > 0:
            arr = arr[:, :-1]

        arrays.append(arr.astype(np.float32, copy=False))

    if len(arrays) == 1:
        return arrays[0]
    return np.concatenate(arrays, axis=0)


def read_feat(input_path, drop_last=False):
    """
    Feature reading entry point: supports directory, list file, or single .npy file.
    Does not perform any normalization (normalization is controlled by main process based on args).
    """
    paths = iter_npy_paths(input_path)
    x = load_and_concat(paths, drop_last=bool(drop_last))
    return x


def l2_row_normalize(a, eps=1e-12):
    """
    L2 row-wise normalization, returns a new array (float32).
    Uses eps to prevent division by zero.
    """
    # a: (n, d)
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    return (a / (norms + eps)).astype(np.float32, copy=False)


def train_kmeans(x, k, ngpu, niter=20, use_fp16=True):
    """
    Run KMeans on one or more GPUs (Faiss)
    x: float32 [n, d]
    k: number of clusters
    use_fp16: use float16 on GPU to halve memory (recommended for large k or d)
    """
    d = x.shape[1]
    clus = faiss.Clustering(d, k)
    clus.verbose = True
    clus.niter = niter

    # No subsampling
    clus.max_points_per_centroid = 10_000_000

    res = [faiss.StandardGpuResources() for _ in range(ngpu)]
    # Limit temp memory per GPU to avoid OOM
    for r in res:
        r.setTempMemory(1024 * 1024 * 1024)  # 1GB temp memory

    flat_config = []
    for i in range(ngpu):
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = use_fp16
        cfg.device = i
        flat_config.append(cfg)

    print(f"[kmeans] k={k}, d={d}, ngpu={ngpu}, fp16={use_fp16}")
    mem_per_gpu_mb = k * d * (2 if use_fp16 else 4) / 1024 / 1024
    print(f"[kmeans] Estimated centroid memory (full): {mem_per_gpu_mb:.0f} MB")
    print(f"[kmeans] Per GPU with sharding: {mem_per_gpu_mb / ngpu:.0f} MB")

    if ngpu == 1:
        index = faiss.GpuIndexFlatL2(res[0], d, flat_config[0])
    else:
        # Use IndexShards to split centroids across GPUs (reduces per-GPU memory)
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = use_fp16
        cpu_index = faiss.IndexFlatL2(d)
        index = faiss.index_cpu_to_all_gpus(cpu_index, co=co, ngpu=ngpu)

    # Train
    clus.train(x, index)
    centroids = faiss.vector_float_to_array(clus.centroids)
    return centroids.reshape(k, d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Numpy input: directory / list file / single .npy file")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of clusters (k)")
    parser.add_argument("--output", required=True, help="Output .npy file path (saves centroids, no normalization unless --l2norm is specified)")
    parser.add_argument("--drop_last", type=int, default=0, help="Whether to drop the last dimension (1 drop, 0 keep)")
    parser.add_argument("--ngpu", type=int, default=8, help="Number of GPUs to use")
    parser.add_argument(
        "--l2norm",
        action="store_true",
        help=(
            "Whether to L2 normalize input features (row normalization). "
            "If specified, input features will be unit-normalized before clustering, and centroids will also be unit-normalized before saving. "
            "Suitable for scenarios targeting cosine/inner product similarity."
        ),
    )
    args = parser.parse_args()

    ngpu = args.ngpu

    print("[info] Reading features...")
    x = read_feat(args.input, args.drop_last)
    x = x.reshape(x.shape[0], -1).astype("float32", copy=False)

    print(f"[info] Feature dimensions: n={x.shape[0]}, d={x.shape[1]}")

    if args.l2norm:
        print("[info] Performing L2 normalization on input features (unit-norm)...")
        x = l2_row_normalize(x)

    print("[info] Running KMeans...")
    t0 = time.time()
    centroids = train_kmeans(x, args.num_classes, ngpu)
    t1 = time.time()
    print("total runtime: %.3f s" % (t1 - t0))

    # If input was normalized, we also normalize centroids before saving, so saved centroids can be used directly for inner product/cosine-based retrieval.
    if args.l2norm:
        print("[info] L2 normalizing centroids before saving (because --l2norm was specified)...")
        centroids = l2_row_normalize(centroids)

    print(f"[info] Saving centroids to: {args.output}")
    np.save(args.output, centroids)
    print("[done]")
