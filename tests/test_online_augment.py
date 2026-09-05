# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Unit tests for the project's online augmentation module (slice / ratio / blur / compose).

Covers the index-space arithmetic in ``BaseDataset.__len__``, ``_n_per``, ``_origin_index`` and
``_compose_at`` -- the four spots where ``n_per = 5 + (1 if ratio) + (2 if blur)`` was duplicated.
Also covers the motion-blur kernel degeneration guard, which used to silently produce NaN when
the length / angle combination left the kernel empty.
"""

import math
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers: construct a minimal BaseDataset skeleton without going through
# __init__ (which would scan a real dataset directory).
# ---------------------------------------------------------------------------
class _Stub:
    """Minimal attribute stub for BaseDataset so we can call the methods we care about."""


def _make_dataset(labels=None, **flags):
    """Build a ``BaseDataset``-shaped object with only the attributes the helpers read."""
    from ultralytics.data.base import BaseDataset

    if labels is None:
        labels = [
            {
                "bboxes": np.array([[0.25, 0.25, 0.5, 0.5]], dtype=np.float32),
                "cls": np.array([[0]], dtype=np.float32),
                "segments": [],
                "keypoints": None,
                "normalized": True,
                "shape": (48, 64),
            }
            for _ in range(5)
        ]
    ds = BaseDataset.__new__(BaseDataset)
    ds.labels = labels
    ds.im_files = [f"img_{i}.jpg" for i in range(len(labels))]
    ds.augment = False
    # Default: slicing on (the four flags the online module gates on).
    # ``slice_transform`` must be non-None for _n_per/__len__ to expand the index space;
    # ``slice_keep_origin`` flips it from x4 (tiles only) to x5 (tiles + origin + extras).
    flags.setdefault("slice_all_tiles", True)
    flags.setdefault("slice_transform", "fake")
    flags.setdefault("slice_keep_origin", True)
    for k, v in flags.items():
        setattr(ds, k, v)
    return ds


# ---------------------------------------------------------------------------
# _n_per single source of truth
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "ratio,blur,want",
    [
        (False, False, 5),
        (True, False, 6),
        (False, True, 7),
        (True, True, 8),
    ],
)
def test_n_per_combos(ratio, blur, want):
    """``_n_per`` is the single source of truth for the per-image multiplier.

    If this changes, ``__len__``, ``_origin_index`` and ``_compose_at`` must agree -- otherwise
    samples silently fall off the end of the index space.
    """
    ds = _make_dataset(ratio_pad_keep=ratio, blur_keep=blur, slice_keep_origin=True, slice_all_tiles=True)
    assert ds._n_per() == want


# ---------------------------------------------------------------------------
# Index space closure: every index has exactly one (origin, sub) tuple
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("N", [1, 2, 3, 4, 5, 7, 8, 9])
def test_index_space_full_pool(N):
    """For N originals + keep_all + compose, len = 8*N + ceil(N/4) and every index maps to a real sample."""
    ds = _make_dataset(
        labels=[{"bboxes": np.empty((0, 4)), "cls": np.empty((0, 1)), "segments": [],
                "keypoints": None, "normalized": True, "shape": (1, 1)}] * N,
        slice_all_tiles=True, slice_transform="fake", slice_keep_origin=True,
        ratio_pad_keep=True, blur_keep=True, compose_keep=True,
    )
    assert len(ds) == 8 * N + (N + 3) // 4

    n_per = ds._n_per()
    n_origin = N
    for idx in range(len(ds)):
        if idx >= n_per * n_origin:
            # compose group: (idx - n_per*N) // 1 maps to a group of 4 originals
            group = idx - n_per * n_origin
            assert 0 <= group < (N + 3) // 4
        else:
            origin = idx // n_per
            assert 0 <= origin < N
            sub = idx % n_per
            assert 0 <= sub < n_per


def test_index_space_no_compose():
    """When compose_keep=False, len = 8*N (no tail)."""
    ds = _make_dataset(
        slice_all_tiles=True, slice_transform="fake", slice_keep_origin=True,
        ratio_pad_keep=True, blur_keep=True, compose_keep=False,
    )
    assert len(ds) == 8 * len(ds.labels)


def test_index_space_keep_origin_only():
    """When keep_origin is True but ratio/blur off, len = 5*N (+ tail if compose)."""
    ds = _make_dataset(
        slice_all_tiles=True, slice_transform="fake", slice_keep_origin=True,
        ratio_pad_keep=False, blur_keep=False, compose_keep=False,
    )
    assert len(ds) == 5 * len(ds.labels)


def test_index_space_emit_all_only():
    """When slice_all_tiles but slice_keep_origin False, len = 4*N (no extras)."""
    ds = _make_dataset(
        slice_all_tiles=True, slice_transform="fake", slice_keep_origin=False,
    )
    assert len(ds) == 4 * len(ds.labels)


# ---------------------------------------------------------------------------
# _origin_index
# ---------------------------------------------------------------------------
def test_origin_index_basic():
    """``_origin_index`` divides by ``n_per`` (5/6/7/8), so all sub-samples of image i share origin i."""
    ds = _make_dataset(slice_all_tiles=True, slice_transform="fake", slice_keep_origin=True,
                       ratio_pad_keep=True, blur_keep=True)
    n_per = ds._n_per()
    assert n_per == 8
    for i in range(5):
        for s in range(n_per):
            assert ds._origin_index(i * n_per + s) == i
    # Compose tail (>= 40 = 8*5) maps to image 5 (wrap), which is a valid image.
    assert ds._origin_index(40) == 5


# ---------------------------------------------------------------------------
# Motion blur kernel: degenerate input must not produce NaN
# ---------------------------------------------------------------------------
def test_motion_blur_kernel_degenerate_safe():
    """A degenerate motion-blur kernel (length / angle that leaves no rasterised line) used to
    divide by zero and produce NaN -- a failure that ``train.py``'s blanket
    ``filterwarnings('ignore')`` silently hid. The guard now degrades to an identity kernel.
    """
    from ultralytics.data.base import _motion_blur_kernel

    # length=0 + any angle -> the cv2.line call rasterises nothing -> sum() == 0
    k = _motion_blur_kernel(length=0.0, angle=0.0)
    assert k.shape[0] >= 3 and k.shape[1] >= 3
    assert math.isfinite(k.sum())
    assert np.isclose(k.sum(), 1.0)  # guard degrades to identity -> sums to 1
    # Center pixel should be 1.0 (identity), everything else 0
    c = k.shape[0] // 2
    assert k[c, c] == 1.0


def test_motion_blur_kernel_normal():
    """A normal kernel still rasterises a line and sums to 1 (regression: don't break the happy path)."""
    from ultralytics.data.base import _motion_blur_kernel

    k = _motion_blur_kernel(length=15.0, angle=30.0)
    assert math.isfinite(k.sum())
    assert np.isclose(k.sum(), 1.0, atol=1e-5)
    assert k.sum() > 0


# ---------------------------------------------------------------------------
# Safe-imwrite helper
# ---------------------------------------------------------------------------
def test_safe_imwrite_returns_bool(tmp_path: Path):
    """``_imwrite`` returns True on success; never raises for empty paths."""
    from ultralytics.data.base import _imwrite

    img = np.zeros((50, 50, 3), dtype=np.uint8)
    out = tmp_path / "中文 路径.jpg"
    ok = _imwrite(str(out), img)
    assert ok is True
    assert out.exists()


# ---------------------------------------------------------------------------
# cache='ram' warning hook (smoke: just importing the module does not crash)
# ---------------------------------------------------------------------------
def test_module_imports_with_all_online_aug_flags():
    """Sanity: importing + setting every online-aug flag does not raise."""
    from ultralytics.data.base import BaseDataset

    ds = BaseDataset.__new__(BaseDataset)
    ds.labels = [{"bboxes": np.empty((0, 4)), "cls": np.empty((0, 1))}]
    ds.im_files = ["x.jpg"]
    ds.augment = False
    # All online-aug switches on; n_per must be 8 without raising.
    ds.slice_all_tiles = True
    ds.slice_transform = "fake"
    ds.slice_keep_origin = True
    ds.ratio_pad_keep = True
    ds.blur_keep = True
    ds.compose_keep = True
    assert ds._n_per() == 8


# ---------------------------------------------------------------------------
# P2-3 per-branch save cap decoupling
# ---------------------------------------------------------------------------
def test_save_cap_fallback_to_legacy():
    """When only the legacy ``slice_save_max`` is set, every branch inherits it."""
    from ultralytics.data.base import _save_cap

    ds = _make_dataset(slice_all_tiles=True, slice_transform="fake", slice_keep_origin=True)
    ds.slice_save_max = 100
    assert _save_cap(ds, "blur") == 100
    assert _save_cap(ds, "ratio") == 100
    assert _save_cap(ds, "compose") == 100


def test_save_cap_per_branch_override():
    """Per-branch overrides win over the legacy cap; ``0`` still means unlimited."""
    from ultralytics.data.base import _save_cap

    ds = _make_dataset(slice_all_tiles=True, slice_transform="fake", slice_keep_origin=True)
    ds.slice_save_max = 100
    ds.slice_save_max_blur = 5
    ds.slice_save_max_compose = 0  # explicit unlimited
    assert _save_cap(ds, "blur") == 5
    assert _save_cap(ds, "ratio") == 100  # falls back
    assert _save_cap(ds, "compose") == 0  # explicit unlimited still wins


def test_save_cap_none_falls_back():
    """``None`` attribute (i.e. attribute never set) falls back to legacy cap."""
    from ultralytics.data.base import _save_cap

    ds = _make_dataset(slice_all_tiles=True, slice_transform="fake", slice_keep_origin=True)
    ds.slice_save_max = 7
    assert not hasattr(ds, "slice_save_max_tile")  # baseline
    assert _save_cap(ds, "tile") == 7
