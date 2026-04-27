import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MLP
from torch_geometric.nn.aggr import DeepSetsAggregation


class Canonicalizer:
    """
    All canonicalizers return (pc, perm, R)
    """

    @staticmethod
    def center(pc):
        return pc - pc.mean(dim=1, keepdim=True)

    @staticmethod
    def _enforce_so3(R):
        det = torch.linalg.det(R)
        flip = (det < 0).to(R.dtype).view(-1, 1, 1)

        Ffix = torch.eye(3, device=R.device, dtype=R.dtype).unsqueeze(0).repeat(R.shape[0], 1, 1)
        Ffix[:, 2, 2] = 1.0 - 2.0 * flip.view(-1)

        return torch.bmm(R, Ffix)

    @staticmethod
    def _fix_eig_signs(vecs):
        max_idx = torch.argmax(vecs.abs(), dim=1, keepdim=True)
        max_vals = torch.gather(vecs, 1, max_idx)
        signs = torch.sign(max_vals)
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        return vecs * signs

    @staticmethod
    def _order(pc):
        """
        True lexicographic: x → y → z (primary x)
        """
        B, N, D = pc.shape
        device = pc.device

        perm = torch.arange(N, device=device).unsqueeze(0).expand(B, N)

        for dim_idx in (0, 1, 2):
            vals = torch.gather(pc[..., dim_idx], 1, perm)
            sort_idx = torch.argsort(vals, dim=1, stable=True)
            perm = torch.gather(perm, 1, sort_idx)

        ordered = torch.gather(pc, 1, perm.unsqueeze(-1).expand(-1, -1, D))
        return ordered, perm

    @staticmethod
    def pca(pc, epsilon=1e-8):
        pc = Canonicalizer.center(pc)
        B, N, _ = pc.shape

        cov = torch.bmm(pc.transpose(1, 2), pc) / max(N - 1, 1)
        cov = cov + torch.eye(3, device=pc.device, dtype=pc.dtype).unsqueeze(0) * epsilon

        _, vecs = torch.linalg.eigh(cov)
        vecs = vecs.flip(dims=[2])

        pc_pca = torch.bmm(pc, vecs)
        perm = torch.arange(N, device=pc.device).unsqueeze(0).expand(B, N)

        return pc_pca, perm, vecs

    @staticmethod
    def pca_skew(pc):
        pc_centered = Canonicalizer.center(pc)

        cov = torch.bmm(pc_centered.transpose(1, 2), pc_centered) / max(pc.shape[1] - 1, 1)
        _, vecs = torch.linalg.eigh(cov)

        vecs = vecs.flip(dims=[2])
        vecs = Canonicalizer._fix_eig_signs(vecs)
        vecs = Canonicalizer._enforce_so3(vecs)

        pc_pca = torch.bmm(pc_centered, vecs)

        skew = (pc_pca ** 3).mean(dim=1)
        s = torch.sign(skew)
        s = torch.where(s == 0, torch.ones_like(s), s)

        # Make parity even so the sign flip preserves proper rotation
        flips = (s < 0).sum(dim=-1)
        odd = (flips % 2 == 1).view(-1)
        s[odd, 2] *= -1

        pc_pca = pc_pca * s.unsqueeze(1)
        vecs = Canonicalizer._enforce_so3(vecs * s.unsqueeze(1))

        pc_ordered, perm = Canonicalizer._order(pc_pca)

        return pc_ordered, perm, vecs


def to_pyg_format(x):
    B, N, C = x.shape
    x_flat = x.reshape(B * N, C)
    batch = torch.arange(B, device=x.device).view(B, 1).expand(B, N).reshape(-1)
    return x_flat, batch


class DeepSetBase(nn.Module):
    def __init__(self, in_features=3, num_classes=40):
        super().__init__()
        self.local_nn = MLP([in_features, 64, 128, 256], batch_norm=True)
        self.global_nn = MLP([256, 128, num_classes], dropout=0.5, batch_norm=True)
        self.deepsets = DeepSetsAggregation(local_nn=self.local_nn, global_nn=self.global_nn)

    def forward(self, pc):
        x, batch = to_pyg_format(pc)
        return self.deepsets(x, index=batch)


# ========================= MODELS ========================= #

class Model1_PurePCA(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        self.net = DeepSetBase(num_classes=num_classes)

    def forward(self, pc):
        pc, _, _ = Canonicalizer.pca(pc)
        return F.log_softmax(self.net(pc), dim=-1)


class Model2_FrameAveraging(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        self.net = DeepSetBase(num_classes=num_classes)

        signs = list(itertools.product([1.0, -1.0], repeat=3))
        self.register_buffer("signs", torch.tensor(signs, dtype=torch.float32))

    def forward(self, pc):
        pc, _, _ = Canonicalizer.pca(pc)
        B, N, _ = pc.shape

        pc_all = (pc.unsqueeze(0) * self.signs.view(8, 1, 1, 3)).reshape(8 * B, N, 3)

        logits = self.net(pc_all)
        log_probs = F.log_softmax(logits, dim=-1).view(8, B, -1)

        return torch.logsumexp(log_probs, dim=0) - math.log(8)


class Model3_Skewness(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        self.net = DeepSetBase(num_classes=num_classes)

    def forward(self, pc):
        pc, _, _ = Canonicalizer.pca_skew(pc)
        return F.log_softmax(self.net(pc), dim=-1)


class Model4_RandomFrame(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        self.net = DeepSetBase(num_classes=num_classes)

    def forward(self, pc):
        pc, _, _ = Canonicalizer.pca(pc)
        B = pc.shape[0]

        if self.training:
            signs = (torch.randint(0, 2, (B, 1, 3), device=pc.device) * 2 - 1).to(pc.dtype)
            pc = pc * signs
        # eval: deterministic

        return F.log_softmax(self.net(pc), dim=-1)