import os
import numpy as np
import mmap

from dataset.registry import DATASET_REGISTRY

from .properties import Property

rank = int(os.getenv("RANK", "0"))              # 全局进程排名 (Global process rank)
local_rank = int(os.getenv("LOCAL_RANK", "0"))  # 本地进程排名 (Local process rank)
world_size = int(os.getenv("WORLD_SIZE", "1"))  # 总进程数 (Total number of processes)


# name: str,                      # Dataset name / 数据集名称
# prefixes: List[str],            # Data path prefixes / 数据路径前缀（多个）
# num_classes: int,               # Total number of classes / 类别总数
# num_examples: int,              # Total number of examples / 样本总数
# label_start: int = 0,           # Label starting offset / 标签起始偏移
# label_select: int = 0,          # Label selection index / 标签选择索引
# num_shards: int = world_size,   # Number of data shards / 数据分片数量
# shard_id: int = rank,           # Current process shard ID / 当前进程的分片ID
# dali_type: str = "origin",      # Data loader type / 数据加载器类型
# random_diff: int = 10,          # Random difference parameter / 随机差异参数
# pfc_types: tuple = ("partial_fc",),  # PFC variants / PFC 类型（可多选）
# mp4_list_path: Optional[str] = None,    # Path to mp4 list file / mp4列表文件路径
# label_list_path: Optional[str] = None,  # Path to label list file / 标签列表文件路径


@DATASET_REGISTRY.register()
def k710_ssv2_univit_pfs_ip_path():
    """
    """
    with open("/video_vit/train_UniViT/mp4_list.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = [x.strip().split(",")[0] for x in lines]
    return Property(
        name="k710_ssv2_diving48_pfs",
        prefixes=lines,
        num_classes=200000,
        num_examples=6089886,
        num_shards=world_size,
        shard_id=rank,
        dali_type="decord",
        label=["/video_vit/train_UniViT/list_merged.npy", "/video_vit/train_UniViT/merged_visible_indices_uint16.npy"]
    )


@DATASET_REGISTRY.register()
def k710_ssv2_univit_pfs_fix_ip_fix_size():
    """"""
    with open("/video_vit/dataset/clips_square_aug_k710_ssv2_hevc_v2/shuf_merged_list", "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = [x.strip().split(",")[0] for x in lines]
    return Property(
        name="k710_ssv2_diving48_pfs",
        prefixes=lines,
        num_classes=200000,
        num_examples=0,
        num_shards=world_size,
        shard_id=rank,
        dali_type="decord"
    )


@DATASET_REGISTRY.register()
def hevc_gop32_quadtree():
    """K710+SSV2 HEVC GOP=32 with quadtree CU selection (762K videos, 500K classes).

    Uses pre-extracted cu_visidx.npz for multi-scale patch selection (16/32/64px).
    QuadtreeVideoDataset loads cu_visidx.npz + video frames at native resolution.
    """
    return Property(
        name="hevc_gop32_quadtree",
        prefixes=[],
        num_classes=500000,
        num_examples=762624,
        num_shards=world_size,
        shard_id=rank,
        dali_type="quadtree",
        random_diff=10,
        mp4_list_path="/nfs-stor/haolin.yang/video_data/hevc_gop32_video_list_all.txt",
        label_list_path="/nfs-stor/haolin.yang/video_data/cluster_viz/all_labels.npy",
        cu_visidx_src="_hevc_gop32",
        cu_visidx_dst="_cu_selection_gop32",
    )


@DATASET_REGISTRY.register()
def hevc_gop32_video_codec():
    """K710+SSV2 HEVC GOP=32 with MV+Res visidx (762K videos, 500K pseudo-label classes).

    file_list: video paths from hevc_gop32_video_list_all.txt
    all_labels: (762624, 10) int64 from all_labels.npy
    visidx: per-video .visidx.npy via path replacement _hevc_gop32 -> _residual_mv_gop32
    """
    video_list_path = "/nfs-stor/haolin.yang/video_data/hevc_gop32_video_list_all.txt"
    labels_path = "/nfs-stor/haolin.yang/video_data/cluster_viz/all_labels.npy"

    with open(video_list_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    return Property(
        name="hevc_gop32_video_codec",
        prefixes=lines,
        num_classes=500000,
        num_examples=len(lines),
        num_shards=world_size,
        shard_id=rank,
        dali_type="decord_hevc_gop32",
        random_diff=10,
        label_list_path=labels_path,
    )
    