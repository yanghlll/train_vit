#!/bin/bash
set -e

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ---- HEVC decoder ----
export HEVC_FEAT_DECODER=/nfs-stor/haolin.yang/Code/VFM/Video_MLCD/codec-infra/hevc-quadtree/hevc

# ---- Distributed config ----
master_addr="${MASTER_ADDR:-127.0.0.1}"
master_port="${MASTER_PORT:-29500}"
nnode=1
node_rank=0

OUTPUT_DIR="ckpts/quadtree_hevc_gop32"
mkdir -p "$OUTPUT_DIR"

echo "=========================="
echo "Quadtree CU Multi-Scale Training"
echo "=========================="
echo "Dataset: hevc_gop32_quadtree (762K videos, 500K classes)"
echo "Model: MultiGranularityOVEncoder (small: 384d, 12L)"
echo "Scales: 16/32/64px CU blocks"
echo "Output: $OUTPUT_DIR"
echo "=========================="

# Optional: pretrained checkpoint
init_backbone="${INIT_BACKBONE:-NULL}"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --master_addr $master_addr --master_port $master_port \
  --nnode $nnode --node_rank $node_rank --nproc_per_node 8 \
  -m \
  training.train_quadtree \
  --hidden_size 384 \
  --num_layers 12 \
  --num_heads 6 \
  --intermediate_size 1536 \
  --image_size 224 \
  --patch_size 16 \
  --embedding_size 384 \
  --num_frames 64 \
  --list_batch_sizes 32 \
  --lr 1e-3 \
  --weight_decay 0.05 \
  --warmup_ratio 0.1 \
  --list_datasets hevc_gop32_quadtree \
  --output "$OUTPUT_DIR" \
  --init_backbone $init_backbone \
  --list_init_partial_fc_paths NULL \
  --list_sample_rates 0.1 \
  --list_lr_pfc_weights 1 \
  --list_loss_weights 1 \
  --list_margins 0.3 \
  --list_filters 0.75 \
  --num_sampled_data 60000000 \
  --finetune_backbone 1 \
  --backward_passes_per_step 1 \
  --ckpt_interval 2000 \
  --frequent 10 \
  --num_workers 4
