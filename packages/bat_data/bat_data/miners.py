"""Online triplet/contrastive miners.

These run on a *batch* of L2-normalised embeddings + integer identity
labels and return the indices of useful (anchor, positive, negative)
triples for a margin-based loss.

Both miners follow the FaceNet conventions:

* :class:`HardNegativeMiner` — for every (anchor, positive) pair, picks
  the hardest negative (closest in distance) within the batch.
* :class:`SemiHardMiner` — for every (anchor, positive) pair, picks a
  negative that is **farther** than the positive but within ``margin``
  of it (Schroff et al., 2015). Falls back to the hardest available
  negative if no semi-hard candidate exists for that pair.

Distances are computed in squared-Euclidean space (operator-equivalent
to ``2 - 2 * cos`` once embeddings are L2-normalised, which is the
common training setup).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type-only import
    import torch

__all__ = ["MinedTriplets", "HardNegativeMiner", "SemiHardMiner"]


@dataclass
class MinedTriplets:
    """Indices of mined triples within the source batch.

    All four tensors are 1-D ``LongTensor`` of equal length.
    ``distances_ap``/``distances_an`` are the corresponding squared
    distances at mining time (handy for diagnostics).
    """

    anchor_idx: "torch.Tensor"
    positive_idx: "torch.Tensor"
    negative_idx: "torch.Tensor"
    distances_ap: "torch.Tensor"
    distances_an: "torch.Tensor"

    def __len__(self) -> int:
        return int(self.anchor_idx.numel())

    @property
    def num_triplets(self) -> int:
        return len(self)


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------


def _pairwise_squared_distances(embeddings: "torch.Tensor") -> "torch.Tensor":
    """Return the (N, N) matrix of squared Euclidean distances.

    Numerically clamped to ``>= 0`` because of float-eps artifacts.
    """
    import torch

    if embeddings.dim() != 2:
        raise ValueError(
            f"embeddings must be 2-D, got shape {tuple(embeddings.shape)}"
        )
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y
    sq = (embeddings * embeddings).sum(dim=1, keepdim=True)
    dot = embeddings @ embeddings.t()
    dist = sq + sq.t() - 2.0 * dot
    return torch.clamp(dist, min=0.0)


def _label_masks(
    labels: "torch.Tensor",
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Return ``(positive_mask, negative_mask)`` Boolean tensors.

    ``positive_mask[i, j] = labels[i] == labels[j] AND i != j``
    ``negative_mask[i, j] = labels[i] != labels[j]``
    """
    import torch

    if labels.dim() != 1:
        raise ValueError(
            f"labels must be 1-D, got shape {tuple(labels.shape)}"
        )
    eq = labels.unsqueeze(0) == labels.unsqueeze(1)
    eye = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    positive_mask = eq & ~eye
    negative_mask = ~eq
    return positive_mask, negative_mask


# ---------------------------------------------------------------------------
# Miners
# ---------------------------------------------------------------------------


class HardNegativeMiner:
    """Pick the hardest negative (smallest distance) per (a, p) pair."""

    def __init__(self, *, squared: bool = True) -> None:
        # ``squared=False`` returns Euclidean distances (sqrt) for users
        # who want a margin in raw distance units.
        self.squared = bool(squared)

    def mine(
        self,
        embeddings: "torch.Tensor",
        labels: "torch.Tensor",
    ) -> MinedTriplets:
        import torch

        if embeddings.size(0) != labels.size(0):
            raise ValueError(
                "embeddings and labels must agree on batch size"
            )

        dist = _pairwise_squared_distances(embeddings)
        pos_mask, neg_mask = _label_masks(labels)

        anchors: list[int] = []
        positives: list[int] = []
        negatives: list[int] = []
        d_ap_list: list[float] = []
        d_an_list: list[float] = []

        n = embeddings.size(0)
        for i in range(n):
            pos_indices = torch.nonzero(pos_mask[i], as_tuple=False).flatten()
            neg_indices = torch.nonzero(neg_mask[i], as_tuple=False).flatten()
            if pos_indices.numel() == 0 or neg_indices.numel() == 0:
                continue
            # hardest negative for this anchor
            neg_dists = dist[i, neg_indices]
            n_idx_local = int(torch.argmin(neg_dists).item())
            n_idx = int(neg_indices[n_idx_local].item())
            d_an = float(dist[i, n_idx].item())
            for p_idx_t in pos_indices:
                p_idx = int(p_idx_t.item())
                anchors.append(i)
                positives.append(p_idx)
                negatives.append(n_idx)
                d_ap_list.append(float(dist[i, p_idx].item()))
                d_an_list.append(d_an)

        device = embeddings.device
        anchor_t = torch.tensor(anchors, dtype=torch.long, device=device)
        pos_t = torch.tensor(positives, dtype=torch.long, device=device)
        neg_t = torch.tensor(negatives, dtype=torch.long, device=device)
        dap = torch.tensor(d_ap_list, dtype=torch.float32, device=device)
        dan = torch.tensor(d_an_list, dtype=torch.float32, device=device)
        if not self.squared and dap.numel():
            dap = dap.sqrt()
            dan = dan.sqrt()
        return MinedTriplets(
            anchor_idx=anchor_t,
            positive_idx=pos_t,
            negative_idx=neg_t,
            distances_ap=dap,
            distances_an=dan,
        )

    __call__ = mine


class SemiHardMiner:
    """Pick semi-hard negatives per FaceNet (Schroff et al., 2015).

    For each (a, p), choose ``n`` such that
    ``d(a, p) < d(a, n) < d(a, p) + margin`` if any exist; otherwise
    fall back to the hardest negative (so the loop produces something
    when the batch is too easy).
    """

    def __init__(
        self,
        *,
        margin: float = 0.2,
        squared: bool = True,
    ) -> None:
        if margin <= 0.0:
            raise ValueError(f"margin must be > 0; got {margin!r}")
        self.margin = float(margin)
        self.squared = bool(squared)

    def mine(
        self,
        embeddings: "torch.Tensor",
        labels: "torch.Tensor",
    ) -> MinedTriplets:
        import torch

        if embeddings.size(0) != labels.size(0):
            raise ValueError(
                "embeddings and labels must agree on batch size"
            )

        dist = _pairwise_squared_distances(embeddings)
        pos_mask, neg_mask = _label_masks(labels)

        anchors: list[int] = []
        positives: list[int] = []
        negatives: list[int] = []
        d_ap_list: list[float] = []
        d_an_list: list[float] = []

        n = embeddings.size(0)
        for i in range(n):
            pos_indices = torch.nonzero(pos_mask[i], as_tuple=False).flatten()
            neg_indices = torch.nonzero(neg_mask[i], as_tuple=False).flatten()
            if pos_indices.numel() == 0 or neg_indices.numel() == 0:
                continue
            neg_dists = dist[i, neg_indices]

            for p_idx_t in pos_indices:
                p_idx = int(p_idx_t.item())
                d_ap = float(dist[i, p_idx].item())
                # Semi-hard window: d_ap < d_an < d_ap + margin
                lo = d_ap
                hi = d_ap + self.margin
                semi_mask = (neg_dists > lo) & (neg_dists < hi)
                if bool(semi_mask.any().item()):
                    candidates = neg_indices[semi_mask]
                    cand_dists = neg_dists[semi_mask]
                    n_idx_local = int(torch.argmin(cand_dists).item())
                    n_idx = int(candidates[n_idx_local].item())
                else:
                    # Fall back to hardest negative.
                    n_idx_local = int(torch.argmin(neg_dists).item())
                    n_idx = int(neg_indices[n_idx_local].item())

                anchors.append(i)
                positives.append(p_idx)
                negatives.append(n_idx)
                d_ap_list.append(d_ap)
                d_an_list.append(float(dist[i, n_idx].item()))

        device = embeddings.device
        anchor_t = torch.tensor(anchors, dtype=torch.long, device=device)
        pos_t = torch.tensor(positives, dtype=torch.long, device=device)
        neg_t = torch.tensor(negatives, dtype=torch.long, device=device)
        dap = torch.tensor(d_ap_list, dtype=torch.float32, device=device)
        dan = torch.tensor(d_an_list, dtype=torch.float32, device=device)
        if not self.squared and dap.numel():
            dap = dap.sqrt()
            dan = dan.sqrt()
        return MinedTriplets(
            anchor_idx=anchor_t,
            positive_idx=pos_t,
            negative_idx=neg_t,
            distances_ap=dap,
            distances_an=dan,
        )

    __call__ = mine
