#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit visualization for video clustering results.
Usage:
    streamlit run visualize_clusters.py -- \
        --index_file /path/to/inverted_index.pkl \
        --video_list /path/to/video_list_all.txt
"""
import argparse
import base64
import multiprocessing
import os
import pickle
import random
import tempfile
import time
from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional, Tuple

import cv2
import imageio.v2 as imageio
import streamlit as st


def parse_arguments():
    parser = argparse.ArgumentParser(description="视频聚类可视化")
    parser.add_argument("--index_file", type=str, required=True, help="倒排索引文件 (inverted_index.pkl)")
    parser.add_argument("--video_list", type=str, required=True, help="视频列表文件")
    return parser.parse_args()


st.set_page_config(
    page_title="Video Cluster Visualization",
    layout="wide",
)


@st.cache_resource
def load_video_list(list_path: str) -> List[str]:
    with open(list_path, 'r') as f:
        paths = [line.strip() for line in f if line.strip()]
    return paths


@st.cache_resource
def load_inverted_index(index_path: str) -> Dict:
    with open(index_path, 'rb') as f:
        return pickle.load(f)


def get_paths_for_cluster(cluster_indices: List[int], video_list: List[str]) -> List[str]:
    return [video_list[idx] for idx in cluster_indices if 0 <= idx < len(video_list)]


def _convert_video_to_gif(video_path: str, max_frames: int = 50, fps: int = 10,
                          resize_factor: float = 0.5) -> Optional[Tuple[str, str]]:
    try:
        if not os.path.exists(video_path):
            return (video_path, None)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return (video_path, None)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        new_w = int(width * resize_factor)
        new_h = int(height * resize_factor)
        sampling_interval = max(1, total_frames // max_frames)

        frames = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % sampling_interval == 0:
                resized = cv2.resize(frame, (new_w, new_h))
                frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
            frame_count += 1
            if len(frames) >= max_frames:
                break
        cap.release()

        if not frames:
            return (video_path, None)

        with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmp:
            gif_path = tmp.name
        imageio.mimsave(gif_path, frames, fps=fps, format='GIF')
        with open(gif_path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')
        os.unlink(gif_path)
        return (video_path, f"data:image/gif;base64,{b64}")
    except Exception as e:
        return (video_path, None)


def batch_convert(video_paths, max_frames=50, fps=10, resize_factor=0.5):
    if not video_paths:
        return {}
    num_proc = min(multiprocessing.cpu_count(), len(video_paths))
    func = partial(_convert_video_to_gif, max_frames=max_frames, fps=fps, resize_factor=resize_factor)
    results = {}
    with multiprocessing.Pool(processes=num_proc) as pool:
        for vpath, gif_data in pool.starmap(func, [(p,) for p in video_paths]):
            if gif_data:
                results[vpath] = gif_data
    return results


def main():
    try:
        args = parse_arguments()
        index_file = args.index_file
        video_list_file = args.video_list
    except SystemExit:
        if 'index_file' in st.session_state and 'video_list_file' in st.session_state:
            index_file = st.session_state.index_file
            video_list_file = st.session_state.video_list_file
        else:
            st.title("Video Cluster Visualization")
            index_file = st.text_input("倒排索引文件:", "/nfs-stor/haolin.yang/video_data/cluster_viz/inverted_index.pkl")
            video_list_file = st.text_input("视频列表文件:", "/nfs-stor/haolin.yang/video_data/cluster_viz/video_list_all.txt")
            if st.button("加载"):
                if index_file and video_list_file:
                    st.session_state.index_file = index_file
                    st.session_state.video_list_file = video_list_file
                    st.rerun()
            return

    st.title("Video Cluster Visualization")

    # Sidebar
    st.sidebar.header("配置")
    st.sidebar.text(f"索引: {os.path.basename(index_file)}")
    st.sidebar.text(f"视频: {os.path.basename(video_list_file)}")

    samples_per_page = st.sidebar.number_input("每页样本数", 1, 20, 8)
    samples_per_row = st.sidebar.number_input("每行样本数", 1, 4, 2)
    gif_fps = st.sidebar.slider("GIF帧率", 1, 30, 10)
    gif_max_frames = st.sidebar.slider("GIF最大帧数", 10, 100, 50)
    gif_resize = st.sidebar.slider("GIF缩放", 0.1, 1.0, 0.5, 0.1)

    # Load data
    video_list = load_video_list(video_list_file)
    inverted_index = load_inverted_index(index_file)

    if not inverted_index:
        st.error("无法加载倒排索引")
        return

    # Sort clusters by size
    clusters = sorted(inverted_index.items(), key=lambda x: len(x[1]), reverse=True)

    st.write(f"总共 **{len(clusters)}** 个簇, **{len(video_list)}** 个视频")

    # Cluster stats
    sizes = [len(v) for _, v in clusters]
    st.write(f"簇大小: 最大={sizes[0]}, 最小={sizes[-1]}, 平均={sum(sizes)/len(sizes):.1f}")

    # Navigation
    col1, col2, col3 = st.columns(3)
    with col1:
        sort_mode = st.selectbox("排序方式", ["按大小降序", "按大小升序", "按簇ID"])
    with col2:
        if sort_mode == "按大小升序":
            clusters = list(reversed(clusters))
        elif sort_mode == "按簇ID":
            clusters = sorted(clusters, key=lambda x: x[0])
    with col3:
        if st.button("随机跳转"):
            st.session_state.cluster_idx = random.randint(0, len(clusters) - 1)

    if 'cluster_idx' not in st.session_state:
        st.session_state.cluster_idx = 0

    cluster_idx = st.number_input(
        f"簇索引 (0 ~ {len(clusters)-1})",
        min_value=0, max_value=len(clusters)-1,
        value=st.session_state.cluster_idx
    )

    label, indices = clusters[cluster_idx]
    paths = get_paths_for_cluster(indices, video_list)
    st.write(f"**簇 {label}** — {len(paths)} 个视频")

    # Pagination
    total_pages = max(1, (len(paths) + samples_per_page - 1) // samples_per_page)
    page = st.number_input("页码", 1, total_pages, 1)
    start = (page - 1) * samples_per_page
    end = min(start + samples_per_page, len(paths))
    page_paths = paths[start:end]

    st.write(f"显示 {start+1} ~ {end} / {len(paths)}")

    # Generate GIFs
    with st.spinner("生成 GIF 预览..."):
        gif_dict = batch_convert(page_paths, gif_max_frames, gif_fps, gif_resize)

    # Display grid
    num_rows = (len(page_paths) + samples_per_row - 1) // samples_per_row
    for row in range(num_rows):
        cols = st.columns(samples_per_row)
        for col_idx in range(samples_per_row):
            idx = row * samples_per_row + col_idx
            if idx < len(page_paths):
                vpath = page_paths[idx]
                with cols[col_idx]:
                    st.caption(f"#{start+idx+1} — {os.path.basename(vpath)}")
                    if vpath in gif_dict:
                        st.markdown(
                            f'<img src="{gif_dict[vpath]}" width="100%" style="display:block;margin:auto;">',
                            unsafe_allow_html=True
                        )
                    elif os.path.exists(vpath):
                        st.warning("GIF 生成失败")
                    else:
                        st.error("文件不存在")


if __name__ == "__main__":
    main()
