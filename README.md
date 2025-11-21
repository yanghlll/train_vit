<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="asset/llava_vit_white.png">
    <source media="(prefers-color-scheme: light)" srcset="asset/llava_vit_white.png">
    <img alt="LLaVA-OneVision 1.5" src="output/llava_onevision_white.png" width="600" style="max-width: 100%;">
  </picture>
</p>

---

## 预训练建议

1. 上规模 是最后一步，应该想尽一切办法在 scaling 前提升模型能力，而且必须有够泛化的现象出现
2. 模型监督尽可能不要直接利用现有模型（如直接蒸馏现有模型），可以相对间接的利用，否则scaling 能力会受限
3. 资源受限时，训练需要渐进，例如先训练低分辨率，低帧率，再逐步微调提升，参考 CLIPA


## 🔧 Setup

```shell
# Mount NFS

mkdir -p /video_vit
mount -t nfs4 -o minorversion=1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport cfs-iyHiNUmePn.lb-0a25b0a7.cfs.bj.baidubce.com:/ /video_vit

mkdir -p /vlm
mount -t nfs4 -o minorversion=1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport cfs-xvbkSb1zPT.lb-563926be.cfs.bj.baidubce.com:/ /vlm
```

### 1. Docker Build


> #### Option 1: Build from Dockerfile
<!-- ```bash
docker build -t llava_vit:25.11 .
``` -->

> #### Option 2: Load pre-built Docker image
```bash
docker load -i /video_vit/docker_images/llava_vit_tag_25.11.15.tar && \
docker tag $(docker images -q | head -n 1) llava_vit:25.11.15
```


#### 2. Multi Node

> [!IMPORTANT]
> 多机必须使用预编译的镜像，且镜像必须一致

```
# Run container with -w to set working directory directly to the mounted volume
docker run -it --gpus all --ipc host --net host --privileged --cap-add IPC_LOCK \
    --ulimit memlock=-1 --ulimit stack=67108864 --rm \
    -v "$(pwd)":/workspace/LLaVA-ViT -v /train_tmp:/train_tmp \
    -v /vlm:/vlm -v /video_vit:/video_vit -v /rice_ocr:/rice_ocr \
    -v /data_0:/data_0 -v /data_1:/data_1 -v /data_2:/data_2 -v /data_3:/data_3 \
    -w /workspace/LLaVA-ViT/ \
    -e NCCL_TIMEOUT=1800 -e CUDA_DEVICE_MAX_CONNECTIONS=1 -e NCCL_SOCKET_IFNAME=eth0 -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA="mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_1" -e NCCL_NET_GDR_LEVEL=2 -e NCCL_IB_QPS_PER_CONNECTION=4 -e NCCL_IB_TC=160 -e NCCL_IB_TIMEOUT=22 -e NCCL_CROSS_NIC=1 -e NCCL_MIN_NCHANNELS=8 -e NCCL_MAX_NCHANNELS=16 \
    -e http_proxy=http://172.16.5.77:8889 -e https_proxy=http://172.16.5.77:8889 \
    llava_vit:25.11.15 bash -c "service ssh restart; bash; "

# Inside the container, install the package in editable mode
pip install -e .
```

## 🚀 Training

1. Sigle Node

<!-- ```bash
# Example command to start training
torchrun -m --nproc_per_node 8 training.train_univit \
  --list_batch_size 64 \
  --output ./output/baseline
```

2. Multi Node

```bash
```

## 🚀 Evaluation
```bash
DATASETS=ucf101 \
MODEL_FAMILY=llava_vit \
MODEL_NAME=pretrain_encoder_small_patch16_224_v10_12_rms_unmask_with_head \
CKPT_PATH=/video_vit/xiangan/checkpoint_llava_vit/baseline_continue_with_mlcd/00117188/backbone.pt \
EMBEDDING_SIZE=384 \
NUM_EPOCH=100 \
NUM_FRAMES=8 \
BATCH_SIZE=4 \
LR=5e-4 bash video_attentive_probe.sh
``` -->

## Eval

```
torchrun--nproc_per_node 8 --master_port 15555 attentive_probe.py --eval_freq 1 --default_lr_list 0.0003 --batch_size 16 --default_weight_decay 0 --dali_py_num_workers 8 --model_family llava_vit_sampling --dataset ssv2
```


