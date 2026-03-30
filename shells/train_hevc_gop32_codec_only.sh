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

OUTPUT_DIR="ckpts/llava_vit_hevc_gop32_codec_only"
mkdir -p "$OUTPUT_DIR"

echo "=========================="
echo "LLaVA-ViT HEVC GOP32 Codec-Only Training"
echo "=========================="
echo "Dataset: hevc_gop32_video_codec (K710+SSV2, 762K videos, 500K classes)"
echo "Training: train_univit_11_08_ip_three_input_residual_mv_tmp.py"
echo "Output: $OUTPUT_DIR"
echo "=========================="

# ---- Optional: pretrained checkpoint ----
init_backbone="${INIT_BACKBONE:-NULL}"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --master_addr $master_addr --master_port $master_port \
  --nnode $nnode --node_rank $node_rank --nproc_per_node 8 \
  -m \
  training.train_univit_11_08_ip_three_input_residual_mv_tmp \
  --model_name pretrain_encoder_small_patch16_224_v10_12_rms_unmask_with_head \
  --image_size 224 \
  --num_frames 64 \
  --embedding_size 384 \
  --list_batch_sizes 8 \
  --lr 1e-3 \
  --weight_decay 0.05 \
  --warmup_ratio 0.1 \
  --list_datasets hevc_gop32_video_codec \
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
  --total_indices 2000 \
  --target_num 2000 \
  --must_num 196 \
  --ckpt_interval 2000 \
  --frequent 10
