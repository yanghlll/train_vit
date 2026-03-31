"""
Quadtree CU Multi-Scale ViT Encoder
=====================================
Extends LlavaViTEncoder architecture with:
1. MultiScalePatchEmbed (APT-style): 16/32/64px patch embedding with dual-path fusion
2. ScaleEmbedding: per-token scale indicator
3. Continuous RoPE: accepts non-integer (t, h, w) positions for variable-size patches

Reuses LLaVA-ViT's TransformerCausal, VideoRotaryEmbeddingSplit466, and
Siglip2MultiheadAttentionPoolingHead.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model
from typing import Optional, Dict, Tuple, List

from model_factory.layers import (
    TransformerCausal,
    VideoRotaryEmbeddingSplit466,
    Siglip2MultiheadAttentionPoolingHead,
)


# ---------------------------------------------------------------------------
# PI-Resize: Pseudo-Inverse kernel resizing (FlexiViT)
# ---------------------------------------------------------------------------

_PI_RESIZE_CACHE: Dict[Tuple[int, int, int, int], torch.Tensor] = {}


def pi_resize_weight(original_weight: torch.Tensor, target_patch_size: int) -> torch.Tensor:
    C_out, C_in, src_h, src_w = original_weight.shape
    dst_h = dst_w = target_patch_size
    if src_h == dst_h and src_w == dst_w:
        return original_weight.clone()

    cache_key = (src_h, src_w, dst_h, dst_w)
    if cache_key not in _PI_RESIZE_CACHE:
        B_mat = torch.zeros(dst_h * dst_w, src_h * src_w)
        for dr in range(dst_h):
            for dc in range(dst_w):
                sr = (dr + 0.5) * src_h / dst_h - 0.5
                sc = (dc + 0.5) * src_w / dst_w - 0.5
                r0, c0 = int(math.floor(sr)), int(math.floor(sc))
                r1, c1 = r0 + 1, c0 + 1
                fdr, fdc = sr - r0, sc - c0
                for (r, c, w) in [(r0, c0, (1-fdr)*(1-fdc)), (r0, c1, (1-fdr)*fdc),
                                  (r1, c0, fdr*(1-fdc)), (r1, c1, fdr*fdc)]:
                    if 0 <= r < src_h and 0 <= c < src_w:
                        B_mat[dr * dst_w + dc, r * src_w + c] = w
        _PI_RESIZE_CACHE[cache_key] = torch.linalg.pinv(B_mat.T)

    Bt_pinv = _PI_RESIZE_CACHE[cache_key].to(device=original_weight.device, dtype=original_weight.dtype)
    w_flat = original_weight.reshape(C_out * C_in, src_h * src_w)
    w_resized = (Bt_pinv @ w_flat.T).T
    return w_resized.reshape(C_out, C_in, dst_h, dst_w)


# ---------------------------------------------------------------------------
# Multi-Scale Patch Embedding (APT-style with ZeroMLP)
# ---------------------------------------------------------------------------

class MultiScalePatchEmbed(nn.Module):
    """APT-style multi-scale patch embedding: 16x16, 32x32, 64x64."""

    def __init__(self, d_model: int, base_patch: int = 16, num_channels: int = 3):
        super().__init__()
        self.d_model = d_model
        self.base_patch = base_patch

        self.base_proj = nn.Conv2d(num_channels, d_model, kernel_size=base_patch,
                                   stride=base_patch, bias=False)

        self.aggregate_32 = nn.Conv1d(d_model, d_model, kernel_size=4, bias=True)
        self.aggregate_64 = nn.Conv1d(d_model, d_model, kernel_size=16, bias=True)

        self.zero_mlp_32 = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.zero_mlp_32.weight)
        nn.init.zeros_(self.zero_mlp_32.bias)
        self.zero_mlp_64 = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.zero_mlp_64.weight)
        nn.init.zeros_(self.zero_mlp_64.bias)

        self._pi_kernel_cache: Dict[int, torch.Tensor] = {}

    def _get_pi_kernel(self, target_size: int) -> torch.Tensor:
        if target_size not in self._pi_kernel_cache or \
           self._pi_kernel_cache[target_size].device != self.base_proj.weight.device:
            self._pi_kernel_cache[target_size] = pi_resize_weight(self.base_proj.weight, target_size)
        return self._pi_kernel_cache[target_size]

    def embed_16x16(self, patches: torch.Tensor) -> torch.Tensor:
        return self.base_proj(patches).squeeze(-1).squeeze(-1)

    def embed_32x32(self, patches: torch.Tensor) -> torch.Tensor:
        N = patches.shape[0]
        if N == 0:
            return patches.new_zeros(0, self.d_model)
        coarse = self.base_proj(
            F.interpolate(patches, size=(16, 16), mode='bilinear', align_corners=False)
        ).squeeze(-1).squeeze(-1)
        subs = torch.cat([patches[:, :, i*16:(i+1)*16, j*16:(j+1)*16]
                          for i in range(2) for j in range(2)], dim=0)
        sub_embeds = self.base_proj(subs).squeeze(-1).squeeze(-1).reshape(N, 4, self.d_model)
        fine = self.aggregate_32(sub_embeds.transpose(1, 2)).squeeze(-1)
        return coarse + self.zero_mlp_32(fine)

    def embed_64x64(self, patches: torch.Tensor) -> torch.Tensor:
        N = patches.shape[0]
        if N == 0:
            return patches.new_zeros(0, self.d_model)
        coarse = self.base_proj(
            F.interpolate(patches, size=(16, 16), mode='bilinear', align_corners=False)
        ).squeeze(-1).squeeze(-1)
        subs = torch.cat([patches[:, :, i*16:(i+1)*16, j*16:(j+1)*16]
                          for i in range(4) for j in range(4)], dim=0)
        sub_embeds = self.base_proj(subs).squeeze(-1).squeeze(-1).reshape(N, 16, self.d_model)
        fine = self.aggregate_64(sub_embeds.transpose(1, 2)).squeeze(-1)
        return coarse + self.zero_mlp_64(fine)

    def forward(self, patches_16=None, patches_32=None, patches_64=None):
        result = {}
        if patches_16 is not None and patches_16.shape[0] > 0:
            result[16] = self.embed_16x16(patches_16)
        if patches_32 is not None and patches_32.shape[0] > 0:
            result[32] = self.embed_32x32(patches_32)
        if patches_64 is not None and patches_64.shape[0] > 0:
            result[64] = self.embed_64x64(patches_64)
        return result


# ---------------------------------------------------------------------------
# Scale Embedding
# ---------------------------------------------------------------------------

class ScaleEmbedding(nn.Module):
    def __init__(self, d_model: int, num_scales: int = 4):
        super().__init__()
        self.embed = nn.Embedding(num_scales, d_model)

    def forward(self, scale_indices: torch.Tensor) -> torch.Tensor:
        return self.embed(scale_indices)


# ---------------------------------------------------------------------------
# Continuous 3D RoPE (extends VideoRotaryEmbeddingSplit466)
# ---------------------------------------------------------------------------

class ContinuousVideoRoPE(VideoRotaryEmbeddingSplit466):
    @torch.no_grad()
    def forward_from_positions(self, positions_thw: torch.Tensor) -> torch.Tensor:
        """Compute RoPE from continuous (t, h, w) positions.
        Args: positions_thw: (L, 3) or (B, L, 3) float tensor
        Returns: freqs: same leading dims + (half,)
        """
        device = positions_thw.device
        inv_t = self.inv_freq_t.to(device=device)
        inv_h = self.inv_freq_h.to(device=device)
        inv_w = self.inv_freq_w.to(device=device)

        t_pos = positions_thw[..., 0].float()
        h_pos = positions_thw[..., 1].float()
        w_pos = positions_thw[..., 2].float()

        ft = torch.einsum("...s,d->...sd", t_pos, inv_t)
        fh = torch.einsum("...s,d->...sd", h_pos, inv_h)
        fw = torch.einsum("...s,d->...sd", w_pos, inv_w)

        return torch.cat([ft, fh, fw], dim=-1)


# ---------------------------------------------------------------------------
# QuadtreeViTEncoder
# ---------------------------------------------------------------------------

class QuadtreeViTEncoder(nn.Module):
    """Multi-scale ViT encoder using LLaVA-ViT architecture.

    All samples have FIXED token count (target_num), so:
    - No padding, no attention mask
    - Direct batch forward through Transformer
    - Same as LlavaViTEncoder but with multi-scale patch embedding
    """

    def __init__(
        self,
        patch_size=16,
        hidden_size=384,
        head_dim=64,
        num_hidden_layers=12,
        intermediate_size=1536,
        act_layer=nn.GELU,
        use_gradient_checkpointing=False,
        attn_dropout=0.0,
        norm_cls=nn.LayerNorm,
        use_head=True,
    ):
        super().__init__()
        assert hidden_size % head_dim == 0
        num_attention_heads = hidden_size // head_dim

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.patch_size = patch_size

        self.patch_embed = MultiScalePatchEmbed(
            d_model=hidden_size, base_patch=patch_size, num_channels=3
        )
        self.scale_embed = ScaleEmbedding(d_model=hidden_size, num_scales=4)
        self.ln_pre = norm_cls(hidden_size)
        self.ln_post = norm_cls(hidden_size)

        self.transformer = TransformerCausal(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            act_layer=act_layer,
            gradient_checkpointing=use_gradient_checkpointing,
            attn_dropout=attn_dropout,
            norm_cls=norm_cls,
        )

        self.video_rope = ContinuousVideoRoPE(head_dim)

        self.use_head = use_head
        if use_head:
            self.head = Siglip2MultiheadAttentionPoolingHead(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
            )

    def load_pretrained_base(self, state_dict: dict):
        """Load pretrained LlavaViTEncoder weights, mapping conv1 → patch_embed.base_proj."""
        new_sd = {}
        for k, v in state_dict.items():
            k = k.replace("_orig_mod.", "").replace("module.", "")
            if k == "conv1.weight":
                new_sd["patch_embed.base_proj.weight"] = v
            elif k.startswith("conv1."):
                new_sd[k.replace("conv1.", "patch_embed.base_proj.")] = v
            elif k == "class_embedding":
                continue
            else:
                new_sd[k] = v
        missing, unexpected = self.load_state_dict(new_sd, strict=False)
        return missing, unexpected

    def forward(
        self,
        patches_by_scale: Dict[int, torch.Tensor],
        positions_thw: torch.Tensor,
        scale_indices: torch.Tensor,
    ) -> dict:
        """Single-clip forward.
        Args:
            patches_by_scale: {16: (N16,C,16,16), 32: (N32,C,32,32), 64: (N64,C,64,64)}
            positions_thw: (L, 3) continuous positions
            scale_indices: (L,) scale index per token
        """
        # 1. Multi-scale patch embedding
        embeddings_by_scale = self.patch_embed(
            patches_16=patches_by_scale.get(16),
            patches_32=patches_by_scale.get(32),
            patches_64=patches_by_scale.get(64),
        )
        token_parts = [embeddings_by_scale[s] for s in [16, 32, 64] if s in embeddings_by_scale]
        hidden_states = torch.cat(token_parts, dim=0)  # (L, D)

        # 2. Scale + pre-norm
        hidden_states = hidden_states + self.scale_embed(scale_indices)
        hidden_states = self.ln_pre(hidden_states)

        # 3. RoPE
        freqs = self.video_rope.forward_from_positions(positions_thw)  # (L, half)

        # 4. Transformer (L, 1, D)
        x_in = hidden_states.unsqueeze(0).permute(1, 0, 2)
        out = self.transformer(x_in, rotary_pos_emb=freqs.unsqueeze(0))
        out = out.permute(1, 0, 2).squeeze(0)  # (L, D)

        # 5. Post-norm + pooling
        out = self.ln_post(out)
        head_output = self.head(out.unsqueeze(0)).squeeze(0) if self.use_head else None

        return {"visible_embeddings": out, "head_output": head_output}

    def forward_packed(
        self,
        patches_by_scale: Dict[int, torch.Tensor],
        positions_thw: torch.Tensor,
        scale_indices: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> dict:
        """Batched forward for FIXED-length sequences (all clips have same token count).

        No padding, no attention mask — all clips are the same length (target_num).
        Reshapes packed (total_L, D) into (B, L, D) and batch-forwards through Transformer.
        """
        device = positions_thw.device
        D = self.hidden_size
        num_clips = len(cu_seqlens) - 1
        L = max_seqlen  # all clips have same length

        # 1. Multi-scale patch embedding
        embeddings_by_scale = self.patch_embed(
            patches_16=patches_by_scale.get(16),
            patches_32=patches_by_scale.get(32),
            patches_64=patches_by_scale.get(64),
        )
        token_parts = [embeddings_by_scale[s] for s in [16, 32, 64] if s in embeddings_by_scale]
        hidden_states = torch.cat(token_parts, dim=0)  # (total_L, D)

        # 2. Scale embedding + pre-norm
        hidden_states = hidden_states + self.scale_embed(scale_indices)
        hidden_states = self.ln_pre(hidden_states)

        # 3. RoPE
        freqs = self.video_rope.forward_from_positions(positions_thw)  # (total_L, half)

        # 4. Reshape packed → batch: (total_L, D) → (B, L, D)
        hidden_states = hidden_states.view(num_clips, L, D)
        freqs = freqs.view(num_clips, L, -1)

        # 5. Transformer: expects (L, B, D)
        x_in = hidden_states.permute(1, 0, 2)  # (L, B, D)
        out = self.transformer(x_in, rotary_pos_emb=freqs)  # no attention_mask needed
        out = out.permute(1, 0, 2)  # (B, L, D)

        # 6. Post-norm
        out = self.ln_post(out)

        # 7. Pooling per-clip
        head_output = None
        if self.use_head:
            pooled_list = []
            for i in range(num_clips):
                pooled = self.head(out[i:i+1])  # (1, D)
                pooled_list.append(pooled)
            head_output = torch.cat(pooled_list, dim=0)  # (B, D)

        # Flatten back for compatibility
        all_output = out.reshape(-1, D)  # (total_L, D)

        return {"visible_embeddings": all_output, "head_output": head_output}


# ---------------------------------------------------------------------------
# Model registration (timm)
# ---------------------------------------------------------------------------

@register_model
def quadtree_encoder_small_patch16_224(pretrained=False, **kwargs):
    return QuadtreeViTEncoder(
        patch_size=16, hidden_size=384, head_dim=64,
        num_hidden_layers=12, intermediate_size=1536,
        act_layer=nn.GELU, norm_cls=nn.LayerNorm, use_head=True,
    )

@register_model
def quadtree_encoder_base_patch16_224(pretrained=False, **kwargs):
    return QuadtreeViTEncoder(
        patch_size=16, hidden_size=768, head_dim=64,
        num_hidden_layers=12, intermediate_size=3072,
        act_layer=nn.GELU, norm_cls=nn.LayerNorm, use_head=True,
    )
