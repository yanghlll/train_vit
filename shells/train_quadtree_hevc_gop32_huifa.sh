#!/bin/bash
#SBATCH --job-name=qt_train
#SBATCH --output=/nfs-stor/haolin.yang/logs/%x_%j.out
#SBATCH --error=/nfs-stor/haolin.yang/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --gres=gpu:4
#SBATCH -p cscc-gpu-p
#SBATCH --time=48:00:00
#SBATCH --qos=cscc-gpu-qos

set -e

source /home/haolin.yang/.bashrc
conda activate /home/haolin.yang/.conda/envs/vit

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HEVC_FEAT_DECODER=/nfs-stor/haolin.yang/Code/VFM/Video_MLCD/codec-infra/hevc-quadtree/hevc

cd /nfs-stor/haolin.yang/Code/VFM/Video_MLCD/LLava-ViT

NNODES=${SLURM_NNODES:-1}
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-4}
NODE_RANK=${SLURM_NODEID:-0}
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=${MASTER_PORT:-29500}

OUTPUT_DIR="/nfs-stor/haolin.yang/Code/VFM/Video_MLCD/LLava-ViT/ckpts/quadtree_hevc_gop32"
mkdir -p "$OUTPUT_DIR"

echo "=========================="
echo "Quadtree CU Multi-Scale Training (huifa)"
echo "Nodes: $NNODES, GPUs/node: $GPUS_PER_NODE"
echo "Effective batch: 32 x 2 x $GPUS_PER_NODE = $((32 * 2 * GPUS_PER_NODE))"
echo "=========================="

init_backbone="${INIT_BACKBONE:-NULL}"

torchrun \
  --nnodes $NNODES \
  --nproc_per_node $GPUS_PER_NODE \
  --node_rank $NODE_RANK \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
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
  --backward_passes_per_step 2 \
  --ckpt_interval 2000 \
  --frequent 10 \
  --num_workers 8 \
  --target_num 1960 \
  --i_frame_ids 0 32
