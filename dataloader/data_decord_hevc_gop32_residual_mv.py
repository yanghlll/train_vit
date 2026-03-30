"""
DALI video dataloader for HEVC GOP=32 videos with pre-extracted MV+Res visidx.

Adapted from data_decord_video_fix_ip_fix_size_residual_mv.py.
Key change: labels are loaded from a global all_labels.npy (indexed by video order),
and visidx paths use _hevc_gop32 -> _residual_mv_gop32 replacement.
"""

import os
import random

import decord
import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

rank = int(os.environ.get("RANK", "0"))
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))


class DALIWarper(object):
    def __init__(self, dali_iter, step_data_num, mode="train"):
        self.iter = dali_iter
        self.step_data_num = step_data_num
        assert mode in ["train", "val", "test"]
        self.mode = mode

    def __next__(self):
        try:
            data_dict = self.iter.__next__()[0]
            return {
                "pixel_values": data_dict["videos"],
                "visible_indices": data_dict["visible_indices"],
                "labels": data_dict["labels"],
            }
        except StopIteration:
            self.iter.reset()
            return self.__next__()

    def __iter__(self):
        return self

    def __len__(self):
        return self.step_data_num

    def reset(self):
        self.iter.reset()


class ExternalInputCallable:
    def __init__(self, source_params):
        self.file_list = source_params["file_list"]
        self.all_labels = source_params["all_labels"]  # (N, 10) int64
        self.visidx_src = source_params.get("visidx_src", "_hevc_gop32")
        self.visidx_dst = source_params.get("visidx_dst", "_residual_mv_gop32")

        self.num_shards = source_params.get("num_shards", world_size)
        self.shard_id = source_params.get("shard_id", rank)
        self.batch_size = source_params.get("batch_size", 32)
        self.sequence_length = source_params.get("sequence_length", 64)
        self.seed = source_params.get("seed", 0)
        self.perm = None
        self.last_seen_epoch = None
        self.mode = "train"

        self.shard_size = len(self.file_list) // self.num_shards
        self.shard_offset = self.shard_size * self.shard_id
        self.full_iterations = self.shard_size // self.batch_size

    def _load_video(self, video_path):
        vr = decord.VideoReader(video_path, num_threads=1, ctx=decord.cpu(0))
        duration = len(vr)
        if duration >= self.sequence_length:
            frame_ids = list(range(self.sequence_length))
        else:
            frame_ids = list(range(duration)) + [duration - 1] * (self.sequence_length - duration)
        vr.seek(0)
        return vr.get_batch(frame_ids).asnumpy()

    def _get_label_and_visidx(self, sample_idx, video_path):
        label = self.all_labels[sample_idx]  # (10,) int64
        visidx_path = video_path.replace(self.visidx_src, self.visidx_dst).replace(".mp4", ".visidx.npy")
        visidx = np.load(visidx_path, mmap_mode="r")
        return label, visidx

    def __call__(self, sample_info):
        if sample_info.iteration >= self.full_iterations:
            raise StopIteration

        if self.last_seen_epoch != sample_info.epoch_idx:
            self.last_seen_epoch = sample_info.epoch_idx
            self.perm = np.random.default_rng(
                seed=self.seed + sample_info.epoch_idx
            ).permutation(len(self.file_list))

        sample_idx = self.perm[sample_info.idx_in_epoch + self.shard_offset]
        video_path = self.file_list[sample_idx]

        try:
            video_data = self._load_video(video_path)
            label, visidx = self._get_label_and_visidx(sample_idx, video_path)
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            # Random fallback
            for _ in range(5):
                try:
                    fallback_idx = random.randint(0, len(self.file_list) - 1)
                    video_path = self.file_list[fallback_idx]
                    video_data = self._load_video(video_path)
                    label, visidx = self._get_label_and_visidx(fallback_idx, video_path)
                    break
                except Exception:
                    continue
            else:
                # Last resort: zeros
                video_data = np.zeros((self.sequence_length, 224, 224, 3), dtype=np.uint8)
                label = np.zeros(10, dtype=np.int64)
                visidx = np.zeros(2000, dtype=np.int16)

        label = label.astype(np.int64)
        visidx = visidx.astype(np.int16)
        return video_data, visidx, label


@pipeline_def()
def dali_pipeline(mode, source_params):
    videos, visible_indices, labels = fn.external_source(
        source=ExternalInputCallable(source_params),
        num_outputs=3,
        batch=False,
        parallel=True,
        dtype=[types.UINT8, types.INT16, types.INT64],
        layout=["FHWC", "C", "C"],
    )

    videos = videos.gpu()
    videos = fn.crop_mirror_normalize(
        videos,
        device="gpu",
        dtype=types.FLOAT,
        output_layout="CFHW",
        mean=source_params["mean"],
        std=source_params["std"],
    )
    visible_indices = visible_indices.gpu()
    labels = labels.gpu()
    return videos, visible_indices, labels


def dali_dataloader(
    file_list,
    all_labels,
    dali_num_threads=4,
    dali_py_num_workers=8,
    batch_size=8,
    sequence_length=64,
    mode="train",
    seed=0,
    num_shards=None,
    shard_id=None,
    visidx_src="_hevc_gop32",
    visidx_dst="_residual_mv_gop32",
):
    mean = [x * 255 for x in [0.48145466, 0.4578275, 0.40821073]]
    std = [x * 255 for x in [0.26862954, 0.26130258, 0.27577711]]

    source_params = {
        "batch_size": batch_size,
        "seed": seed + rank,
        "num_shards": num_shards,
        "shard_id": shard_id,
        "file_list": file_list,
        "all_labels": all_labels,
        "sequence_length": sequence_length,
        "visidx_src": visidx_src,
        "visidx_dst": visidx_dst,
        "mean": mean,
        "std": std,
    }

    pipe = dali_pipeline(
        batch_size=batch_size,
        num_threads=dali_num_threads,
        device_id=local_rank,
        seed=seed + rank,
        py_num_workers=dali_py_num_workers,
        py_start_method="spawn",
        prefetch_queue_depth=2,
        mode=mode,
        source_params=source_params,
    )
    pipe.build()

    dataloader = DALIWarper(
        dali_iter=DALIGenericIterator(
            pipelines=pipe,
            output_map=["videos", "visible_indices", "labels"],
            auto_reset=True,
            size=-1,
            last_batch_padded=False,
            last_batch_policy=LastBatchPolicy.FILL,
            prepare_first_batch=False,
        ),
        step_data_num=len(file_list) // max(world_size, 1) // batch_size,
        mode=mode,
    )
    return dataloader
