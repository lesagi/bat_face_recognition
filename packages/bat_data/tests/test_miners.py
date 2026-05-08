"""Hard / semi-hard miner correctness on hand-built fixtures."""

from __future__ import annotations

import pytest

pytest.importorskip("pydantic")
torch = pytest.importorskip("torch")

from bat_data.miners import HardNegativeMiner, SemiHardMiner  # noqa: E402


def _fixture_embeddings():
    """4 samples in 2 classes:
        idx 0 -> class 0, vec (1, 0)
        idx 1 -> class 0, vec (0, 1)
        idx 2 -> class 1, vec (0.6, 0)   # close to anchor 0 (hardest neg)
        idx 3 -> class 1, vec (-1, -1)   # far away

    With this layout:
      - For anchor 0 (positive 1), the hardest negative is 2.
      - For anchor 1 (positive 0), the hardest negative is 2.
    """
    emb = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.6, 0.0],
            [-1.0, -1.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    return emb, labels


def test_hard_negative_miner_returns_hardest() -> None:
    emb, labels = _fixture_embeddings()
    miner = HardNegativeMiner(squared=True)
    triplets = miner.mine(emb, labels)

    # Two anchors with positives: anchor 0 -> pos 1 (then anchor 1 -> pos 0,
    # plus anchor 2 -> pos 3, anchor 3 -> pos 2). 4 triplets total.
    assert len(triplets) == 4

    # For anchor 0, the picked negative must be index 2.
    picks = {
        (int(a.item()), int(p.item())): int(n.item())
        for a, p, n in zip(
            triplets.anchor_idx,
            triplets.positive_idx,
            triplets.negative_idx,
            strict=False,
        )
    }
    assert picks[(0, 1)] == 2
    assert picks[(1, 0)] == 2
    # For anchors 2 and 3 (class 1), negatives are class 0; the hardest
    # negative for anchor 2 is index 0 (distance 0.16) vs index 1 (1.36).
    assert picks[(2, 3)] == 0
    # For anchor 3 (vec -1,-1) the closer of (1,0)/(0,1) is index 1
    # (distance ((-1-0)^2 + (-1-1)^2)=5 vs ((-1-1)^2+(-1-0)^2)=5; tie ->
    # argmin returns the lowest-index).
    assert picks[(3, 2)] in (0, 1)


def test_hard_miner_skips_anchors_with_no_positives() -> None:
    # 3 distinct classes => no positives => no triplets.
    emb = torch.eye(3)
    labels = torch.tensor([0, 1, 2], dtype=torch.long)
    triplets = HardNegativeMiner().mine(emb, labels)
    assert len(triplets) == 0


def test_semi_hard_miner_picks_within_window() -> None:
    emb, labels = _fixture_embeddings()
    # margin=2.0 is loose enough to allow both negatives 2 and 3 for
    # anchor 0 (d_ap=2; window (2, 4)).
    miner = SemiHardMiner(margin=2.0, squared=True)
    triplets = miner.mine(emb, labels)
    assert len(triplets) > 0
    # For (anchor=0, positive=1): d_ap=2; semi-hard window (2, 4).
    # Distances to negatives: 2->0.16, 3->5. Neither in (2, 4) — falls
    # back to hardest negative => index 2.
    pick = None
    for a, p, n in zip(
        triplets.anchor_idx, triplets.positive_idx, triplets.negative_idx, strict=False
    ):
        if int(a.item()) == 0 and int(p.item()) == 1:
            pick = int(n.item())
    assert pick == 2  # hardest fallback


def test_semi_hard_miner_uses_window_when_present() -> None:
    # Construct a batch where a true semi-hard negative exists.
    #   class 0: a (0,0), p (0.1,0)   d_ap = 0.01
    #   class 1: easy neg (10, 10),  semi-hard neg (0.5, 0)   d_an=0.25
    emb = torch.tensor(
        [
            [0.0, 0.0],  # 0: anchor (class 0)
            [0.1, 0.0],  # 1: positive (class 0)
            [0.5, 0.0],  # 2: semi-hard negative (class 1)  d=0.25
            [10.0, 10.0],  # 3: easy negative (class 1)       d=200
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    miner = SemiHardMiner(margin=0.5, squared=True)
    trips = miner.mine(emb, labels)

    chosen = {
        (int(a), int(p)): int(n)
        for a, p, n in zip(trips.anchor_idx, trips.positive_idx, trips.negative_idx, strict=False)
    }
    # For (0, 1): d_ap=0.01, window (0.01, 0.51). 0.25 is in-window.
    assert chosen[(0, 1)] == 2
