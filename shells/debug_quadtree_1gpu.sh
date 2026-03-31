#!/bin/bash
set -e

source /home/haolin.yang/.bashrc
conda activate vit

export CUDA_VISIBLE_DEVICES=2
export HEVC_FEAT_DECODER=/nfs-stor/haolin.yang/Code/VFM/Video_MLCD/codec-infra/hevc-quadtree/hevc

cd /nfs-stor/haolin.yang/Code/VFM/Video_MLCD/LLava-ViT

torchrun --nproc_per_node 1 --master_port 29501 \
    -m training.train_quadtree \
    --list_datasets hevc_gop32_quadtree \
    --list_batch_sizes 32 \
    --hidden_size 384 \
    --num_layers 12 \
    --num_heads 6 \
    --intermediate_size 1536 \
    --embedding_size 384 \
    --patch_size 16 \
    --num_frames 64 \
    --lr 1e-3 \
    --num_sampled_data 2000 \
    --output /tmp/test_quadtree_train \
    --init_backbone NULL \
    --list_init_partial_fc_paths NULL \
    --frequent 1 \
    --num_workers 8 \
    --backward_passes_per_step 2 \
    --target_num 1960 \
    --i_frame_ids 0 32
