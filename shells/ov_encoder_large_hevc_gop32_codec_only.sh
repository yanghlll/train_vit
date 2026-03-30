#!/bin/bash
set -e

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ---- HEVC decoder ----
export HEVC_FEAT_DECODER=/nfs-stor/haolin.yang/Code/VFM/Video_MLCD/codec-infra/hevc-quadtree/hevc
export PYTHONPATH="/nfs-stor/haolin.yang/Code/VFM/Video_MLCD/OneVision-Encoder/tools/tools_for_hevc:${PYTHONPATH}"

# ---- Distributed config ----
master_addr="${MASTER_ADDR:-127.0.0.1}"
master_port="${MASTER_PORT:-29500}"
nnode=1
node_rank=0

OUTPUT_DIR="ckpts/ov_encoder_large_hevc_gop32_codec_only"
mkdir -p "$OUTPUT_DIR"

echo "=========================="
echo "HEVC GOP32 Codec-Only Training"
echo "=========================="
echo "Dataset: hevc_gop32_video_codec (K710+SSV2, 762K videos)"
echo "Model: ov_encoder_large"
echo "Output: $OUTPUT_DIR"
echo "=========================="

# ---- Optional: pretrained checkpoint ----
# Set to your pretrained model path, or "NULL" to train from scratch
init_backbone="${INIT_BACKBONE:-NULL}"

# ---- Build init_partial_fc arg ----
if [[ "$init_backbone" == "NULL" ]]; then
    init_pfc="NULL"
else
    init_pfc="${init_backbone}/ov_encoder_si_8gpus/ov_encoder_si_%03d.npy"
    # If PFC init doesn't exist, use NULL
    if [[ ! -f "$(printf "$init_pfc" 0)" ]]; then
        init_pfc="NULL"
    fi
fi

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --master_addr $master_addr --master_port $master_port \
  --nnode $nnode --node_rank $node_rank --nproc_per_node 8 \
  -m \
  training.train \
  --model_name ov_encoder_large \
  --image_size 224 \
  --image_size_video 224 \
  --num_frames 8 \
  --embedding_size 1024 \
  --list_batch_sizes 8 \
  --lr 1e-3 \
  --weight_decay 0.05 \
  --warmup_ratio 0.1 \
  --list_datasets hevc_gop32_video_codec \
  --output "$OUTPUT_DIR" \
  --init_backbone $init_backbone \
  --list_init_partial_fc_paths $init_pfc \
  --list_sample_rates 0.1 \
  --list_lr_pfc_weights 1 \
  --list_loss_weights 1 \
  --list_margins 0.3 \
  --list_filters 0.75 \
  --num_sampled_data 60000000 \
  --finetune_backbone 1 \
  --backward_passes_per_step 1 \
  --num_tokens_per_frame 256 \
  --target_num 2000 \
  --total_indices 2000 \
  --residual_ratio 1.0 \
  --frame_sampling_ratio 0.0 \
  --ckpt_interval 2000 \
  --frequent 10
