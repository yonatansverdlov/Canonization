import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from check_data import *

# ===========================================================================
#  RoPE Utilities
# ===========================================================================

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precomputes the cosine and sine frequencies for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cos = torch.cos(freqs)  # (seq_len, dim / 2)
    freqs_sin = torch.sin(freqs)  # (seq_len, dim / 2)
    return freqs_cos, freqs_sin


def apply_rotary_emb(q, k, freqs_cos, freqs_sin):
    """Applies Rotary Position Embedding to queries and keys."""
    # q, k shapes: (B, num_heads, N, head_dim)
    # freqs_cos, freqs_sin shapes: (N, head_dim // 2)

    # Reshape q and k to split the head dimension into pairs
    q_ = q.reshape(*q.shape[:-1], -1, 2)  # (B, num_heads, N, head_dim // 2, 2)
    k_ = k.reshape(*k.shape[:-1], -1, 2)

    q_0, q_1 = q_[..., 0], q_[..., 1]
    k_0, k_1 = k_[..., 0], k_[..., 1]

    # Expand freqs to match q and k shapes
    fc = freqs_cos.view(1, 1, q.shape[2], -1)
    fs = freqs_sin.view(1, 1, q.shape[2], -1)

    # Apply the rotation matrix
    q_out_0 = q_0 * fc - q_1 * fs
    q_out_1 = q_0 * fs + q_1 * fc
    k_out_0 = k_0 * fc - k_1 * fs
    k_out_1 = k_0 * fs + k_1 * fc

    # Re-stack and flatten back to original shape
    q_out = torch.stack([q_out_0, q_out_1], dim=-1).flatten(3)
    k_out = torch.stack([k_out_0, k_out_1], dim=-1).flatten(3)

    return q_out, k_out


# ===========================================================================
#  Model Baseline: Transformer with RoPE
# ===========================================================================

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class RoPEAttention(nn.Module):
    def __init__(self, dim, num_heads, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        assert dim % num_heads == 0, f"Attention dimension ({dim}) must be divisible by num_heads ({num_heads})."

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, freqs_cos, freqs_sin):
        B, N, C = x.shape

        # Extract Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # Shapes: (B, num_heads, N, head_dim)

        # Apply RoPE to Q and K
        q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin)

        # Standard Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Mix values and project
        x = (attn @ v).transpose(1, 2).contiguous().view(B, N, C)
        x = self.proj(x)
        return self.proj_drop(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2., drop=0.1, attn_drop=0.1, drop_path=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RoPEAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x, freqs_cos, freqs_sin):
        x = x + self.drop_path(self.attn(self.norm1(x), freqs_cos, freqs_sin))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PointTransformerClassifier(nn.Module):
    def __init__(self,
                 num_classes=40,
                 in_channels=3,
                 dim=216,
                 depth=4,
                 heads=6,
                 mlp_ratio=2.0,
                 drop_rate=0.1,
                 drop_path_rate=0.1,
                 max_seq_len=2048):  # Added max_seq_len for RoPE initialization
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.GELU(),
            nn.Linear(64, dim)
        )

        # Precompute RoPE frequencies
        head_dim = dim // heads
        freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, end=max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=drop_rate,
                drop_path=dpr[i]
            ) for i in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        # NOTE: Expects x to be pre-ordered and shape (B, N, 3)
        if x.shape[1] == 3 and x.shape[2] > 3:
            x = x.transpose(1, 2).contiguous()

        N = x.shape[1]

        # Slice RoPE frequencies up to current sequence length
        freqs_cos = self.freqs_cos[:N]
        freqs_sin = self.freqs_sin[:N]

        x = self.embedding(x)

        for block in self.blocks:
            x = block(x, freqs_cos, freqs_sin)

        x = self.norm(x)
        x = x.max(dim=1)[0]
        return self.head(x)


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from hilbertcurve.hilbertcurve import HilbertCurve
from space_filling_pytorch import encode


# ===========================================================================
#  Helper: Robust PCA Canonicalization (Skew-based)
# ===========================================================================
class Canonicalizer:
    """
    Contract: All canonicalizers must return (canonical_pc, perm, R), where `perm` is
    strictly the argsort index array used to form `canonical_pc` via `torch.gather`.
    """

    @staticmethod
    def center(pc, eps=1e-8):
        pc = pc - pc.mean(dim=1, keepdim=True)
        return pc

    @staticmethod
    def _enforce_so3(R):
        det = torch.linalg.det(R)
        flip = (det < 0).to(R.dtype).view(-1, 1, 1)
        Ffix = torch.eye(3, device=R.device, dtype=R.dtype).view(1, 3, 3).repeat(R.shape[0], 1, 1)
        Ffix[:, 2, 2] = 1.0 - 2.0 * flip.squeeze(-1).squeeze(-1)
        return torch.bmm(R, Ffix)

    @staticmethod
    def _fix_eig_signs(vecs):
        B, N, K = vecs.shape
        max_idx = torch.argmax(vecs.abs(), dim=1, keepdim=True)
        max_vals = torch.gather(vecs, 1, max_idx)
        signs = torch.sign(max_vals)
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        return vecs * signs

    @staticmethod
    def _apply_data_signs(canonical_pc, R, s):
        s = torch.where(s == 0, torch.ones_like(s), s)
        flips = (s < 0).sum(dim=-1)
        odd = (flips % 2 == 1).view(-1)
        s[odd, 2] *= -1
        canonical_pc2 = canonical_pc * s.unsqueeze(1)
        R2 = Canonicalizer._enforce_so3(R * s.unsqueeze(1))
        return canonical_pc2, R2

    @staticmethod
    def _order(canonical_pc):
        B, N, D = canonical_pc.shape
        device = canonical_pc.device

        perm = torch.arange(N, device=device).unsqueeze(0).expand(B, N)

        for dim_idx in (2, 1, 0):
            vals = torch.gather(canonical_pc[..., dim_idx], 1, perm)
            sort_idx = torch.argsort(vals, dim=1, stable=True)
            perm = torch.gather(perm, 1, sort_idx)

        ordered = torch.gather(canonical_pc, 1, perm.unsqueeze(-1).expand(-1, -1, D))
        return ordered, perm

    @staticmethod
    def pca_skew(pc, epsilon=1e-8):
        B, N, D = pc.shape

        # 1. Covariance matrix
        cov = torch.bmm(pc.transpose(1, 2), pc) / (N - 1)
        cov += torch.eye(D, device=pc.device).unsqueeze(0) * epsilon

        # 2. Eigendecomposition
        _, eigenvectors = torch.linalg.eigh(cov)
        eigenvectors = eigenvectors.flip(dims=[2])
        eigenvectors = Canonicalizer._fix_eig_signs(eigenvectors)
        eigenvectors = Canonicalizer._enforce_so3(eigenvectors)

        # 3. Initial Canonicalization
        canonical_pc = torch.bmm(pc, eigenvectors)

        # 4. Robust Skewness Sign Fix
        skew = (canonical_pc ** 3).mean(dim=1)
        s = torch.sign(skew)

        canonical_pc2, R_final = Canonicalizer._apply_data_signs(canonical_pc, eigenvectors, s)

        # 5. Final Lexicographical Ordering
        ordered, perm = Canonicalizer._order(canonical_pc2)

        return ordered, perm, R_final


# ===========================================================================
#  Model 1: The Global MLP Baseline (Modularized)
# ===========================================================================

class FourierFeatureMap(nn.Module):
    def __init__(self, in_features=3, num_bands=4, scale=10.0):
        super().__init__()
        self.register_buffer('B_matrix', torch.randn(in_features, num_bands * in_features) * scale)

    def forward(self, x):
        x_proj = (2. * math.pi * x) @ self.B_matrix
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class MLPResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, drop=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.skip(x) + self.drop(self.act(self.linear1(self.norm(x))))


# -------------------------
# Dynamic Ordering Module (100% GPU)
# -------------------------
class DynamicOrdering(nn.Module):
    def __init__(self, ordering_type='lex', hilbert_m=12):
        super().__init__()
        self.ordering_type = ordering_type

        # space_size is calculated as 2^m.
        # If hilbert_m=12, space_size is 4096.
        self.space_size = 1 << hilbert_m

    def batched_normalize(self, X):
        """Batched affine normalization to [0, 1]: (B, N, 3)"""
        m = X.amin(dim=1, keepdim=True)
        Z = X - m
        M = Z.amax(dim=(1, 2), keepdim=True)
        return Z / torch.clamp(M, min=1e-8)

    def batched_lex_perm(self, X):
        """Fully batched, pure-GPU Lexicographical sort."""
        B, N, _ = X.shape
        idx = torch.arange(N, device=X.device).unsqueeze(0).expand(B, N)

        for k in (2, 1, 0):
            vals = torch.gather(X[..., k], 1, idx)
            sort_idx = torch.argsort(vals, dim=1, stable=True)
            idx = torch.gather(idx, 1, sort_idx)

        return idx

    def batched_hilbert_perm(self, X):
        """
        Blazingly fast GPU Hilbert sorting using space-filling-pytorch.
        """
        # 1. The library requires coordinates strictly in [-1, 1].
        # Since X is in [0, 1] from batched_normalize, we scale it.
        X_encoded_input = (X * 2.0) - 1.0

        # 2. Get the Hilbert codes using the Triton kernel
        # Returns shape (B, N) with giant integers
        h_codes = encode(X_encoded_input, space_size=self.space_size, method='hilbert', convention='xyz')

        # 3. Sort the codes to get the permutation indices
        perm = torch.argsort(h_codes, dim=1, stable=True)

        return perm

    @torch.no_grad()
    def forward(self, x):
        # --- NEW: Robust Skew-based PCA Canonicalization ---
        if self.ordering_type == 'pca':
            # Center the data first
            x_centered = Canonicalizer.center(x)
            # pca_skew returns canonicalized ordered points, permutation, and rotation matrix
            x_ordered, perm, R = Canonicalizer.pca_skew(x_centered)
            return x_ordered

        # --- Existing Lexicographical / Hilbert Logic ---
        # 1. Normalize the batch to [0, 1]
        x_norm = self.batched_normalize(x)

        # 2. Find permutations
        if self.ordering_type == 'lex':
            idx = self.batched_lex_perm(x_norm)
        elif self.ordering_type == 'hilbert':
            idx = self.batched_hilbert_perm(x_norm)
        else:
            return x_norm  # 'ply' baseline remains unchanged

        # 3. Gather the points into their new strict 1D sequence
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, 3)
        x_ordered = torch.gather(x_norm, 1, idx_expanded)

        return x_ordered


# -------------------------
# The Classifier
# -------------------------
class GlobalMLPClassifier(nn.Module):
    def __init__(self,
                 num_classes=40,
                 num_points=1024,
                 in_channels=3,
                 num_bands=4,
                 fourier_scale=10.0,
                 dropout=0.5,
                 point_dropout=0.2,
                 mlp_dims=[256, 128, 64],  # Using the tighter bottleneck
                 ordering_type='lex',  # Can be 'pca', 'lex', 'hilbert', 'ply'
                 hilbert_m=12):
        super().__init__()

        self.num_points = num_points

        # --- Internal Point Ordering ---
        self.dynamic_order = DynamicOrdering(ordering_type=ordering_type, hilbert_m=hilbert_m)

        # Fourier Mapping
        self.fourier_map = FourierFeatureMap(in_features=in_channels, num_bands=num_bands, scale=fourier_scale)
        self.input_drop = nn.Dropout(0.2)  # Input dropout to fight overfitting

        # Use Dropout1d to drop entire points (it expects shape: B, Channels, Sequence)
        # Point-Level Dropout
        self.point_dropout = nn.Dropout1d(p=point_dropout)
        self.input_dim = num_points * (in_channels * num_bands * 2)

        blocks = []
        current_dim = self.input_dim
        for dim in mlp_dims:
            blocks.append(MLPResidualBlock(current_dim, dim, drop=dropout))
            current_dim = dim

        self.blocks = nn.Sequential(*blocks)
        self.final_norm = nn.LayerNorm(current_dim)
        self.head = nn.Linear(current_dim, num_classes)

    def forward(self, x):
        if x.shape[1] == 3 and x.shape[2] > 3:
            x = x.transpose(1, 2).contiguous()

        B, N, _ = x.shape
        assert N == self.num_points, f"GlobalMLP strictly requires N={self.num_points} points, got {N}."

        # --- 1. Order the Points Internally ---
        # The model takes raw data, applies canonicalization/normalization, and sorts it
        x = self.dynamic_order(x)

        # --- 2. Extract Features ---
        x = self.fourier_map(x)

        # Apply Point-Level Dropout
        # Dropout1d expects (Batch, Channels, Length), so we permute
        x = self.point_dropout(x)

        x = x.view(B, -1)
        x = self.input_drop(x)

        # --- 3. Pass through MLP ---
        x = self.blocks(x)
        x = self.final_norm(x)
        return self.head(x)