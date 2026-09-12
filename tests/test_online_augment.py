# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Unit tests for the project's online augmentation module (slice / ratio / blur / compose /
weather / occlusion).

Covers the index-space arithmetic of the 7-segment mixed sample pool -- ``BaseDataset.__len__`` /
``_n_per`` / ``_segment_bases`` share one source of truth, so these tests guard against the
"len vs decodable index range drift" failure mode this project has hit before (S2 regression:
the tests used to assert the pre-``keep_origin``-decoupling layout ``n_per = 5 + ratio + 2*blur``
and a now-removed ``_origin_index`` method, so CI turned red and the layout was left unguarded).

Layout under test (see ``_segment_bases``; each optional branch is an independent segment gated
ONLY by its own switch, slicing lives entirely inside the base segment)::

    [0, base_len)                base:      _n_per samples per original (slicing pipeline)
    [base_len, +N)               origin:    1 un-sliced original per image (slice_keep_origin)
    [base_len+N, +2N)            ratio:     1 aspect-ratio-padded image per original
    [base_len+2N, +4N)           blur:      short + long motion-blurred images per original
    [base_len+4N, +ceil(N/4))    compose:   one 2x2 stitched image per group of 4 originals
    [base_len+4N+ceil(N/4), +N)  weather:   1 rain/haze/noise-degraded image per original
    [.. +N)                      occlusion: 1 rect/stripe-occluded image per original

Also covers the motion-blur kernel degeneration guard (used to silently produce NaN), the
unicode-safe imwrite helper and the per-branch save caps.
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


def _make_label(shape=(48, 64)):
    """One fake label dict shaped like ``BaseDataset.labels[i]``."""
    return {
        "bboxes": np.array([[0.25, 0.25, 0.5, 0.5]], dtype=np.float32),
        "cls": np.array([[0]], dtype=np.float32),
        "segments": [],
        "keypoints": None,
        "normalized": True,
        "shape": shape,
    }


def _make_dataset(labels=None, **flags):
    """Build a ``BaseDataset``-shaped object with only the attributes the helpers read.

    All seven branch switches default to ON (ratio/blur/compose/weather/occlusion + slicing with
    emit_all + keep_origin), so the full-pool layout is the default case and individual tests
    turn specific branches off.
    """
    from ultralytics.data.base import BaseDataset

    if labels is None:
        labels = [_make_label() for _ in range(5)]
    ds = BaseDataset.__new__(BaseDataset)
    ds.labels = labels
    ds.im_files = [f"img_{i}.jpg" for i in range(len(labels))]
    ds.augment = False
    defaults = dict(
        slice_all_tiles=True,
        slice_transform="fake",  # non-None: slicing pipeline active
        slice_keep_origin=True,
        ratio_pad_keep=True,
        blur_keep=True,
        compose_keep=True,
        weather_keep=True,
        occlusion_keep=True,
    )
    defaults.update(flags)
    for k, v in defaults.items():
        setattr(ds, k, v)
    return ds


def _bases(ds):
    """Return the 7 segment boundaries + total as a plain dict for easy assertions."""
    sb = ds._segment_bases()
    return dict(
        base=sb.base, origin=sb.origin, ratio=sb.ratio, blur=sb.blur,
        compose=sb.compose, weather=sb.weather, occlusion=sb.occlusion, total=sb.total,
    )


# ---------------------------------------------------------------------------
# _n_per single source of truth (base segment only: slicing output)
# ---------------------------------------------------------------------------
def test_n_per_emit_all():
    """emit_all: each original expands to 4 tiles."""
    ds = _make_dataset(slice_all_tiles=True, slice_transform="fake")
    assert ds._n_per() == 4


def test_n_per_no_emit_all():
    """slice_all_tiles=False: 1 sample per original (random tile)."""
    ds = _make_dataset(slice_all_tiles=False, slice_transform="fake")
    assert ds._n_per() == 1


def test_n_per_no_slicing():
    """Slicing pipeline off: 1 sample per original (plain originals, no expansion)."""
    ds = _make_dataset(slice_transform=None)
    assert ds._n_per() == 1


# ---------------------------------------------------------------------------
# 7-segment index space closure: __len__ == total, boundaries match the layout
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("N", [1, 2, 3, 4, 5, 7, 8, 9])
def test_segment_bases_full_pool(N):
    """All branches on: fields are SEGMENT STARTS; total = len = 4N + N + N + 2N + ceil(N/4) + N + N."""
    ds = _make_dataset(labels=[_make_label() for _ in range(N)])  # all switches on by default
    compose_len = (N + 3) // 4 if N >= 4 else 0  # _compose_on requires >= 4 originals
    base = 4 * N
    expected = dict(
        base=base,
        origin=base,                    # origin segment starts right after the base segment
        ratio=base + N,
        blur=base + 2 * N,
        compose=base + 4 * N,
        weather=base + 4 * N + compose_len,
        occlusion=base + 5 * N + compose_len,
        total=base + 6 * N + compose_len,
    )
    got = _bases(ds)
    assert got == expected, f"N={N}: {got} != {expected}"
    assert len(ds) == expected["total"]  # __len__ and total must never drift


def test_segment_bases_no_compose():
    """compose off: compose start collapses onto the blur END (start + 2N); len = 4N+N+N+2N+N+N = 10N."""
    ds = _make_dataset(compose_keep=False)
    n = len(ds.labels)
    sb = _bases(ds)
    assert sb["compose"] == sb["blur"] + 2 * n  # blur segment is 2N long
    assert len(ds) == 10 * n


def test_segment_bases_keep_origin_only():
    """Only slicing + keep_origin: len = 4N + N = 5N."""
    ds = _make_dataset(
        ratio_pad_keep=False, blur_keep=False, compose_keep=False,
        weather_keep=False, occlusion_keep=False,
    )
    assert len(ds) == 5 * len(ds.labels)


def test_segment_bases_emit_all_only():
    """emit_all without any extra branch: len = 4N."""
    ds = _make_dataset(
        slice_keep_origin=False, ratio_pad_keep=False, blur_keep=False, compose_keep=False,
        weather_keep=False, occlusion_keep=False,
    )
    assert len(ds) == 4 * len(ds.labels)


def test_segment_bases_no_slicing():
    """Slicing off (slice_transform=None): base = N, origin auto-suppressed (would duplicate)."""
    ds = _make_dataset(slice_transform=None, slice_keep_origin=False,
                       ratio_pad_keep=False, blur_keep=False, compose_keep=False,
                       weather_keep=False, occlusion_keep=False)
    sb = _bases(ds)
    assert ds._n_per() == 1
    assert sb["base"] == len(ds.labels)
    assert sb["origin"] == sb["base"]  # keep_origin needs the slicing pipeline -> off
    assert len(ds) == len(ds.labels)


def test_segment_bases_weather_occlusion_lengths():
    """weather/occlusion each add exactly N samples, laid out after origin when mid branches off."""
    ds = _make_dataset(ratio_pad_keep=False, blur_keep=False, compose_keep=False)
    n = len(ds.labels)
    sb = _bases(ds)
    assert len(ds) == 7 * n  # 4N + N(origin) + N(weather) + N(occlusion)
    assert sb["weather"] == sb["origin"] + n
    assert sb["occlusion"] == sb["weather"] + n


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
# Smoke: importing + setting every online-aug flag does not crash
# ---------------------------------------------------------------------------
def test_module_imports_with_all_online_aug_flags():
    """Sanity: importing + setting every online-aug flag does not raise."""
    from ultralytics.data.base import BaseDataset

    ds = BaseDataset.__new__(BaseDataset)
    ds.labels = [{"bboxes": np.empty((0, 4)), "cls": np.empty((0, 1))}]
    ds.im_files = ["x.jpg"]
    ds.augment = False
    # All online-aug switches on; n_per must be 4 (emit_all) without raising.
    ds.slice_all_tiles = True
    ds.slice_transform = "fake"
    ds.slice_keep_origin = True
    ds.ratio_pad_keep = True
    ds.blur_keep = True
    ds.compose_keep = True
    ds.weather_keep = True
    ds.occlusion_keep = True
    assert ds._n_per() == 4


# ---------------------------------------------------------------------------
# P2-3 per-branch save cap decoupling
# ---------------------------------------------------------------------------
def test_save_cap_fallback_to_legacy():
    """When only the legacy ``slice_save_max`` is set, every branch inherits it."""
    from ultralytics.data.base import _save_cap

    ds = _make_dataset()
    ds.slice_save_max = 100
    assert _save_cap(ds, "blur") == 100
    assert _save_cap(ds, "ratio") == 100
    assert _save_cap(ds, "compose") == 100


def test_save_cap_per_branch_override():
    """Per-branch overrides win over the legacy cap; ``0`` still means unlimited."""
    from ultralytics.data.base import _save_cap

    ds = _make_dataset()
    ds.slice_save_max = 100
    ds.slice_save_max_blur = 5
    ds.slice_save_max_compose = 0  # explicit unlimited
    assert _save_cap(ds, "blur") == 5
    assert _save_cap(ds, "ratio") == 100  # falls back
    assert _save_cap(ds, "compose") == 0  # explicit unlimited still wins


def test_save_cap_none_falls_back():
    """``None`` attribute (i.e. attribute never set) falls back to legacy cap."""
    from ultralytics.data.base import _save_cap

    ds = _make_dataset()
    ds.slice_save_max = 7
    assert not hasattr(ds, "slice_save_max_tile")  # baseline
    assert _save_cap(ds, "tile") == 7


# ---------------------------------------------------------------------------
# Second-review regression guards (P0-1 / M-1 / M-5 / L-2 / L-3 / L-4)
# ---------------------------------------------------------------------------
def _tiny_detect_dataset(tmp_path: Path):
    """Write a 1-image YOLO detection dataset on disk and return (img_dir, data dict)."""
    import cv2

    img_dir = tmp_path / "images" / "train"
    lbl_dir = tmp_path / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    cv2.imwrite(str(img_dir / "a.jpg"), np.zeros((64, 64, 3), dtype=np.uint8))
    (lbl_dir / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    data = {"train": str(img_dir), "val": str(img_dir), "names": {0: "a"}, "nc": 1, "channels": 3}
    return img_dir, data


@pytest.mark.parametrize("configured,expected", [(8, 8), (0, 0), (5, 5)])
def test_raw_cache_size_reaches_dataset(tmp_path, configured, expected):
    """M-1: ``slice_raw_cache_size`` must actually reach the dataset.

    It used to be read from ``self`` inside ``BaseDataset.__init__`` -- before ``v8_transforms`` copied
    hyp's keys onto the dataset -- so the knob (including its "0 = off" semantics and the "raise it to
    speed up decoding" hint in the cache='ram' warning) was permanently stuck at the default 2.
    """
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset

    img_dir, data = _tiny_detect_dataset(tmp_path)
    cfg = get_cfg(
        overrides={
            "data": str(tmp_path / "d.yaml"),
            "imgsz": 64,
            "task": "detect",
            "mode": "train",
            "slice_prob": 0.5,
            "slice_all_tiles": True,
            "slice_raw_cache_size": configured,
            "workers": 0,
            "cache": False,
            "close_aug_epoch": 0,
        }
    )
    ds = build_yolo_dataset(cfg, str(img_dir), 1, data, mode="train")
    assert ds._raw_cache_size == expected  # the value base.py actually enforces
    assert getattr(ds, "slice_raw_cache_size", None) == expected  # mirrored by v8_transforms


def test_run_val_forces_rebuild_when_loader_is_not_sliced():
    """P0-1: sliced validation must actually run during training.

    ``DetectionTrainer.get_validator`` hands the validator a PREBUILT whole-image ``test_loader``. The
    old "rebuild only when the mode changes" shortcut saw ``True == True`` and reused it, so
    ``val_slice_enable=True`` validated on whole images and ``SliceValDataset`` was never constructed.
    """
    import types

    from ultralytics.data.base import SliceValDataset
    from ultralytics.engine.trainer import BaseTrainer

    class _Validator:
        def __init__(self, loader, enable):
            self.dataloader = loader
            self.args = types.SimpleNamespace(val_slice_enable=enable)
            self.called_with = "unset"

        def __call__(self, trainer):
            # dataloader=None here means the trainer set it to None to force a rebuild.
            self.called_with = self.dataloader
            return {"ok": True}

    def _trainer_with(loader, enable):
        trainer = BaseTrainer.__new__(BaseTrainer)
        trainer.validator = _Validator(loader, enable)
        return trainer

    whole_loader = types.SimpleNamespace(dataset=object())  # NOT a SliceValDataset
    sliced_loader = types.SimpleNamespace(dataset=SliceValDataset.__new__(SliceValDataset))

    # (1) sliced requested while the prebuilt whole-image loader is in place -> force a rebuild
    trainer = _trainer_with(whole_loader, True)
    trainer._run_val(True)
    assert trainer.validator.called_with is None
    assert trainer.validator.dataloader is whole_loader  # state restored after the pass

    # (2) already-sliced loader -> reuse as-is (no pointless rebuild)
    trainer = _trainer_with(sliced_loader, True)
    trainer._run_val(True)
    assert trainer.validator.called_with is sliced_loader

    # (3) switched to the whole-image reference pass -> rebuild again
    trainer = _trainer_with(sliced_loader, True)
    trainer._run_val(False)
    assert trainer.validator.called_with is None

    # (4) vanilla training (slice off) -> keep reusing the prebuilt loader (upstream behaviour)
    trainer = _trainer_with(whole_loader, False)
    trainer._run_val(False)
    assert trainer.validator.called_with is whole_loader


def test_save_metrics_realigns_changed_metric_set(tmp_path):
    """M-5: rows are positional, so a changed metric set (e.g. dual-metric on resume) must not
    silently out-grow the header."""
    from ultralytics.engine.trainer import BaseTrainer

    trainer = BaseTrainer.__new__(BaseTrainer)  # bypass __init__
    trainer.csv = tmp_path / "results.csv"
    trainer.train_time_start = 0.0
    trainer.epoch = 0
    trainer.save_metrics({"metrics/a": 1.0})
    trainer.epoch = 1
    trainer.save_metrics({"metrics/a": 2.0, "whole_metrics/a": 9.0})  # column set changed

    rows = [line.split(",") for line in trainer.csv.read_text(encoding="utf-8").strip().splitlines()]
    assert rows[0] == ["epoch", "time", "metrics/a", "whole_metrics/a"]
    assert {len(r) for r in rows} == {len(rows[0])}  # header and every row stay aligned
    assert rows[1][2] == "1" and rows[1][3] == ""  # old row preserved, new column left blank
    assert rows[2][2] == "2" and rows[2][3] == "9"


def test_sliced_metrics_guard_requires_whole_image_gt():
    """L-2: sliced batches without ``_slice_base_labels`` must fail loudly."""
    import torch

    from ultralytics.models.yolo.detect.val import DetectionValidator

    v = DetectionValidator.__new__(DetectionValidator)
    v._slice_base_labels = None
    batch = {"img": torch.zeros(1, 3, 64, 64), "val_slice_meta": [{"orig_idx": 0, "n_tiles": 4}]}
    with pytest.raises(RuntimeError, match="_slice_base_labels"):
        v._update_metrics_sliced([], batch)


def test_sliced_metrics_guard_rejects_non_square_canvas():
    """L-3: remap assumes a square canvas; a non-square one would silently mis-place every box."""
    import torch

    from ultralytics.models.yolo.detect.val import DetectionValidator

    v = DetectionValidator.__new__(DetectionValidator)
    v._slice_base_labels = []
    batch = {"img": torch.zeros(1, 3, 64, 96), "val_slice_meta": [{"orig_idx": 0, "n_tiles": 4}]}
    with pytest.raises(RuntimeError, match="square validation canvas"):
        v._update_metrics_sliced([], batch)


def test_finalize_metrics_clears_unfinished_slice_acc():
    """L-4: originals that never saw all sub-tiles must not leak into the next pass."""
    import types

    from ultralytics.models.yolo.detect.val import DetectionValidator

    v = DetectionValidator.__new__(DetectionValidator)
    v._slice_acc = {0: {"preds": [], "done": 1, "n_tiles": 4, "im_file": "a.jpg"}}
    v.seen = 3
    v.args = types.SimpleNamespace(plots=False)
    v.metrics = types.SimpleNamespace(speed=None, confusion_matrix=None, save_dir=None)
    v.speed, v.confusion_matrix, v.save_dir = {}, None, "."
    v.finalize_metrics()
    assert v._slice_acc == {}
