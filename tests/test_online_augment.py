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
    segment_bases = ds._segment_bases()
    return dict(
        base=segment_bases.base, origin=segment_bases.origin, ratio=segment_bases.ratio, blur=segment_bases.blur,
        compose=segment_bases.compose, weather=segment_bases.weather, occlusion=segment_bases.occlusion, total=segment_bases.total,
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
    segment_bases = _bases(ds)
    assert segment_bases["compose"] == segment_bases["blur"] + 2 * n  # blur segment is 2N long
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
    segment_bases = _bases(ds)
    assert ds._n_per() == 1
    assert segment_bases["base"] == len(ds.labels)
    assert segment_bases["origin"] == segment_bases["base"]  # keep_origin needs the slicing pipeline -> off
    assert len(ds) == len(ds.labels)


def test_segment_bases_weather_occlusion_lengths():
    """weather/occlusion each add exactly N samples, laid out after origin when mid branches off."""
    ds = _make_dataset(ratio_pad_keep=False, blur_keep=False, compose_keep=False)
    n = len(ds.labels)
    segment_bases = _bases(ds)
    assert len(ds) == 7 * n  # 4N + N(origin) + N(weather) + N(occlusion)
    assert segment_bases["weather"] == segment_bases["origin"] + n
    assert segment_bases["occlusion"] == segment_bases["weather"] + n


# ---------------------------------------------------------------------------
# Motion blur kernel: degenerate input must not produce NaN
# ---------------------------------------------------------------------------
def test_motion_blur_kernel_degenerate_safe():
    """A degenerate motion-blur kernel (length / angle that leaves no rasterised line) used to
    divide by zero and produce NaN -- a failure that ``train.py``'s blanket
    ``filterwarnings('ignore')`` silently hid. The guard now degrades to an identity kernel.
    """
    from ultralytics.data.online_degrade import _motion_blur_kernel

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
    from ultralytics.data.online_degrade import _motion_blur_kernel

    k = _motion_blur_kernel(length=15.0, angle=30.0)
    assert math.isfinite(k.sum())
    assert np.isclose(k.sum(), 1.0, atol=1e-5)
    assert k.sum() > 0


# ---------------------------------------------------------------------------
# Safe-imwrite helper
# ---------------------------------------------------------------------------
def test_safe_imwrite_returns_bool(tmp_path: Path):
    """``_imwrite`` returns True on success; never raises for empty paths."""
    from ultralytics.data.online_io import _imwrite

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
# per-branch save cap decoupling
# ---------------------------------------------------------------------------
def test_save_cap_fallback_to_legacy():
    """When only the legacy ``slice_save_max`` is set, every branch inherits it."""
    from ultralytics.data.online_io import _save_cap

    ds = _make_dataset()
    ds.slice_save_max = 100
    assert _save_cap(ds, "blur") == 100
    assert _save_cap(ds, "ratio") == 100
    assert _save_cap(ds, "compose") == 100


def test_save_cap_per_branch_override():
    """Per-branch overrides win over the legacy cap; ``0`` still means unlimited."""
    from ultralytics.data.online_io import _save_cap

    ds = _make_dataset()
    ds.slice_save_max = 100
    ds.slice_save_max_blur = 5
    ds.slice_save_max_compose = 0  # explicit unlimited
    assert _save_cap(ds, "blur") == 5
    assert _save_cap(ds, "ratio") == 100  # falls back
    assert _save_cap(ds, "compose") == 0  # explicit unlimited still wins


def test_save_cap_none_falls_back():
    """``None`` attribute (i.e. attribute never set) falls back to legacy cap."""
    from ultralytics.data.online_io import _save_cap

    ds = _make_dataset()
    ds.slice_save_max = 7
    assert not hasattr(ds, "slice_save_max_tile")  # baseline
    assert _save_cap(ds, "tile") == 7


# ---------------------------------------------------------------------------
# Second-review regression guards
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
    """Sliced validation must actually run during training.

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


# ---------------------------------------------------------------------------
# Third-review regression guards (S-1 / S-2 / S-3 / M-3 / M-5 / M-10 / L-2)
# ---------------------------------------------------------------------------
def _make_loader_dataset(n=12, cap=4, extended=True):
    """Minimal BaseDataset skeleton able to run ``load_image`` (no directory scan).

    ``extended=True`` turns on ONE extended-pool branch (ratio_pad_keep) while leaving slicing off
    -- exactly the combination that used to let ``self.ims`` grow without bound.
    """
    from collections import deque

    from ultralytics.data.base import BaseDataset

    ds = BaseDataset.__new__(BaseDataset)
    ds.ni = n
    ds.labels = [_make_label() for _ in range(n)]
    ds.im_files = [f"img_{i}.jpg" for i in range(n)]
    ds.npy_files = [Path(f"img_{i}.npy") for i in range(n)]
    ds.channels = 3
    ds.cv2_flag = 1
    ds.imgsz = 64
    ds.prefix = ""
    ds.augment = True
    ds.cache = None
    ds.max_buffer_length = cap + 1
    ds.buffer = deque(maxlen=cap)
    ds._ims_cap = cap
    ds._ims_keys = {}
    ds.ims = [None] * n
    ds.im_hw0 = [None] * n
    ds.im_hw = [None] * n
    ds.slice_transform = None  # slicing OFF -> load_image owns the ims write
    ds.slice_all_tiles = False
    ds.slice_keep_origin = False
    ds.ratio_pad_keep = extended
    ds.blur_keep = False
    ds.compose_keep = False
    ds.weather_keep = False
    ds.occlusion_keep = False
    return ds


def test_ims_cache_bounded_when_extended_pool_is_on(monkeypatch):
    """S-1: ``self.ims`` must stay FIFO-bounded even though the mosaic buffer cannot evict it.

    Regression: the only eviction sat behind ``not self._extended_pool_on()``, so with slicing off
    plus any extended branch on, every visited image stayed resident -- one frame per image in the
    dataset (~29 GB/worker for 8520 images at imgsz=1280) and a monotone RSS climb with no warning.
    """
    from ultralytics.data import base as base_mod

    ds = _make_loader_dataset(n=12, cap=4, extended=True)
    monkeypatch.setattr(base_mod, "imread", lambda f, flags=1: np.zeros((32, 32, 3), np.uint8))

    assert ds._extended_pool_on() is True  # the leaking configuration
    for i in range(ds.ni):
        ds.load_image(i)

    resident = sum(im is not None for im in ds.ims)
    assert resident <= 4, f"self.ims grew to {resident} entries; cap is 4"
    assert len(ds._ims_keys) <= 4
    assert ds.ims[ds.ni - 1] is not None  # the freshest frame is still resident


def test_ims_cache_bounded_in_pure_ultralytics_mode(monkeypatch):
    """S-1 parity: the pure-ultralytics path must stay bounded too (legacy buffer behaviour).

    One owner (``_remember_ims``) now evicts ``self.ims`` in every mode; this guards that moving the
    eviction out of ``load_image``'s buffer branch did not lose the bound.
    """
    from ultralytics.data import base as base_mod

    ds = _make_loader_dataset(n=12, cap=4, extended=False)
    monkeypatch.setattr(base_mod, "imread", lambda f, flags=1: np.zeros((32, 32, 3), np.uint8))

    assert ds._extended_pool_on() is False
    for i in range(ds.ni):
        ds.load_image(i)

    assert sum(im is not None for im in ds.ims) <= 4
    assert len(ds.buffer) <= 4  # the mosaic buffer is still fed in this mode


def test_degrade_frame_caps_resolution_and_scales_parameters():
    """S-2: the degradation branches work at ``degrade_max_side``, not at the sensor resolution.

    ``_degrade_frame`` must report the applied scale so pixel-typed parameters (PSF length, defocus
    sigma, rain-line length) can shrink with it -- that is what keeps the post-resize result
    equivalent to the uncapped path.
    """
    ds = _make_dataset(labels=[_make_label()])
    ds.imgsz = 64
    big = np.zeros((300, 400, 3), np.uint8)
    ds._load_image_cached = lambda idx: big.copy()

    ds.degrade_max_side = 0  # auto = 2 x imgsz
    im, scale = ds._degrade_frame(0)
    assert im.shape[:2] == (96, 128) and scale == pytest.approx(0.32)

    ds.degrade_max_side = 200  # explicit cap
    im, scale = ds._degrade_frame(0)
    assert im.shape[:2] == (150, 200) and scale == pytest.approx(0.5)

    ds.degrade_max_side = -1  # disabled -> untouched, scale 1.0 (legacy behaviour)
    im, scale = ds._degrade_frame(0)
    assert im.shape[:2] == (300, 400) and scale == 1.0

    small = np.zeros((40, 60, 3), np.uint8)
    ds._load_image_cached = lambda idx: small
    ds.degrade_max_side = 0
    im, scale = ds._degrade_frame(0)  # already under the cap: no resize at all
    assert im is small and scale == 1.0


def test_weather_noise_uses_cv2_randn_on_a_single_channel_view(monkeypatch):
    """S-4: noise must be drawn by ``cv2.randn`` into ONE float32 buffer, on a C1 view.

    Two properties are load-bearing and both are silent when broken:
    * ``cv2.randn`` on a 3-channel matrix applies ``sigma/sqrt(3)`` (measured std 8.67 for sigma=15),
      so the buffer has to be handed over as ``(h, w * c)``. Verified numerically by
      ``test_weather_noise_sigma_matches_the_configured_std``; here we pin the call shape.
    * ``np.random.normal`` must no longer be involved at all -- it allocated a float64 temporary and
      was the slowest operator in the whole online pipeline (137.8 ms vs 27.0 ms at 1280x960).
    """
    from ultralytics.data import online_degrade

    calls = []
    real_randn = online_degrade.cv2.randn

    def spy_randn(dst, mean, stddev):
        calls.append((dst.shape, float(mean), float(stddev), dst.dtype))
        return real_randn(dst, mean, stddev)

    monkeypatch.setattr(online_degrade.cv2, "randn", spy_randn)

    calls_to_normal = []
    real_normal = online_degrade.np.random.normal

    def spy_normal(*args, **kwargs):
        calls_to_normal.append(args)
        return real_normal(*args, **kwargs)

    monkeypatch.setattr(online_degrade.np.random, "normal", spy_normal)

    img = np.full((64, 96, 3), 128, np.uint8)
    out = online_degrade._apply_weather(img, "noise", noise_std=7.5)

    assert not calls_to_normal, "the noise branch must not use np.random.normal any more"
    assert out.dtype == np.uint8 and out.shape == img.shape
    assert len(calls) == 1, "one draw per sample"
    shape, mean, stddev, dtype = calls[0]
    assert len(shape) == 2, f"cv2.randn must see a single-channel view, got {shape}"
    assert shape[0] * shape[1] == img.size, "every element filled exactly once"
    assert dtype == np.float32, "float32 keeps the transient at 4x the frame, not 8x"
    assert mean == 0.0 and stddev == 7.5, "the configured sigma is handed over unscaled"


def test_weather_noise_sigma_matches_the_configured_std():
    """S-4: the produced noise must have the configured std -- the cv2.randn channel trap.

    ``cv2.randn(dst, 0, sigma)`` silently yields ``sigma/sqrt(3)`` when ``dst`` is 3-channel, so a
    regression here looks like a working run with a 42% weaker augmentation. Mid-gray input keeps
    clipping out of the measurement.
    """
    from ultralytics.data import online_degrade

    rng = np.random.default_rng(0)
    img = rng.integers(60, 200, size=(256, 256, 3), dtype=np.uint8)
    for sigma in (5.0, 15.0):
        np.random.seed(3)
        out = online_degrade._apply_weather(img, "noise", noise_std=sigma)
        d = out.astype(np.int16) - img.astype(np.int16)
        assert abs(float(d.std()) - sigma) < 0.05 * sigma, f"noise std {d.std():.3f} != {sigma}"


def test_weather_noise_is_reproducible_and_advances_the_numpy_stream_once():
    """S-4: cv2's RNG is seeded from the numpy stream, so reproducibility is unchanged.

    The branch used to consume the numpy stream itself; it now consumes exactly one ``randint`` and
    hands it to ``cv2.setRNGSeed``. Same ``np.random.seed`` -> same pixels, and every consumer
    downstream of the call sees the stream positioned exactly as before.
    """
    from ultralytics.data import online_degrade

    img = np.random.default_rng(1).integers(0, 256, (64, 64, 3), dtype=np.uint8)

    np.random.seed(11)
    a = online_degrade._apply_weather(img, "noise", noise_std=12.0)
    np.random.seed(11)
    b = online_degrade._apply_weather(img, "noise", noise_std=12.0)
    np.random.seed(12)
    c = online_degrade._apply_weather(img, "noise", noise_std=12.0)
    assert np.array_equal(a, b), "same numpy seed must reproduce the pixels"
    assert not np.array_equal(a, c), "a different seed must give different noise"

    np.random.seed(99)
    online_degrade._apply_weather(img, "noise", noise_std=12.0)
    after_call = np.random.random(3)
    np.random.seed(99)
    np.random.randint(0, 2**31 - 1)  # the single draw the branch performs
    after_one_draw = np.random.random(3)
    assert np.allclose(after_call, after_one_draw), "the numpy stream advance must be unchanged"


def test_motion_blur_crop_is_bit_identical_and_shrinks_the_kernel():
    """S-4: trimming the zero border off the PSF must not change a single pixel.

    ``cv2.filter2D`` walks the whole ``size x size`` kernel even though a rasterized line only
    occupies a thin diagonal band, and its cost has a cliff around 11 px. Cropping the zeros away and
    passing the matching ``anchor`` is exact (a zero tap contributes nothing) and measured 1.27x over
    the configured length distribution, so this guards both the equality and the anchor contract.
    """
    from ultralytics.data import online_degrade as od

    import cv2

    img = np.random.default_rng(2).integers(0, 256, (120, 160, 3), dtype=np.uint8)
    for length, angle in ((5.0, 0.0), (12.0, 30.0), (12.0, 90.0), (20.0, 45.0), (35.0, 137.0)):
        kernel = od._motion_blur_kernel(length, angle)
        cropped, anchor = od._crop_kernel(kernel)
        assert 0 <= anchor[0] < cropped.shape[1] and 0 <= anchor[1] < cropped.shape[0], "OpenCV asserts this"
        assert cropped.shape[0] <= kernel.shape[0] and cropped.shape[1] <= kernel.shape[1]
        assert cropped.shape[0] * cropped.shape[1] < kernel.shape[0] * kernel.shape[1], "the padding is real"
        ref = cv2.filter2D(img, -1, kernel)
        alt = cv2.filter2D(img, -1, cropped, anchor=anchor)
        assert np.array_equal(ref, alt), f"crop changed the result for length={length} angle={angle}"
        # the public entry point must go through the cropped kernel too
        assert np.array_equal(od._apply_motion_blur(img, length=length, angle=angle), ref)


def test_weather_noise_draws_in_chunks_never_full_frame(monkeypatch):
    """S-3 (superseded by S-4): kept as a guard that no full-frame float64 temporary is allocated.

    Regression: ``np.random.normal(0, sigma, img.shape)`` allocated 274 MB (float64) for a
    4000x3000 frame and was immediately truncated to uint8. S-4 replaced the chunked draw with a
    single ``cv2.randn`` into a float32 buffer, which is stricter than chunking: the only full-frame
    temporary left is the float32 accumulator (4x the frame), not an 8x float64 one.
    """
    from ultralytics.data import online_degrade

    allocated = []
    real_empty = np.empty

    def spy_empty(shape, *a, **k):
        allocated.append((shape, k.get("dtype", a[0] if a else None)))
        return real_empty(shape, *a, **k)

    monkeypatch.setattr(online_degrade.np, "empty", spy_empty)
    img = np.zeros((2048, 640, 3), np.uint8)
    out = online_degrade._apply_weather(img, "noise", noise_std=10.0)

    assert out.dtype == np.uint8 and out.shape == img.shape
    assert allocated, "the noise buffer must be allocated explicitly"
    assert all(dt is np.float32 for _, dt in allocated), "a float64 buffer is 2x the memory for nothing"
    assert all(np.prod(s) < img.size or tuple(s) == img.shape for s, _ in allocated), "no full-frame extra"


def test_load_image_cached_lru_capacity_and_hit_order(monkeypatch):
    """M-3: explicit LRU -- capacity honoured, hits refresh order, eviction drops exactly one entry."""
    from collections import OrderedDict

    from ultralytics.data import base as base_mod
    from ultralytics.data.base import BaseDataset

    ds = BaseDataset.__new__(BaseDataset)
    ds.im_files = [f"img_{i}.jpg" for i in range(6)]
    ds.cv2_flag = 1
    ds._raw_cache_size = 4
    ds._raw_cache = OrderedDict()

    decoded = []

    def fake_imread(f, flags=1):
        decoded.append(f)
        return np.full((8, 8, 3), len(decoded), np.uint8)

    monkeypatch.setattr(base_mod, "imread", fake_imread)

    for i in range(4):
        ds._load_image_cached(i)
    assert len(decoded) == 4 and len(ds._raw_cache) == 4

    hit = ds._load_image_cached(0)
    assert len(decoded) == 4, "a cache hit must not re-decode"
    assert list(ds._raw_cache)[-1] == 0, "a hit must move the entry to the MRU end"
    hit[:] = 0  # callers take ownership; the cached array must survive that
    assert ds._raw_cache[0].any()

    ds._load_image_cached(4)
    assert len(ds._raw_cache) == 4, "the cache must not grow past its capacity"
    assert set(ds._raw_cache) == {0, 2, 3, 4}, "exactly the LRU entry (1) is evicted, not half the cache"


def test_slice_val_subset_is_deterministic_across_rebuilds():
    """M-5: ``val_slice_ratio < 1`` must score the SAME subset every round.

    Regression: an unseeded ``random.sample`` redrew the sliced subset on each validation round (the
    wrapper is rebuilt per round), so mAP moved between epochs partly because a different set of
    images was being scored -- and that jitter drives best.pt / early stopping.
    """
    from ultralytics.data.base import SliceValDataset

    labels = [{**_make_label(shape=(64, 64)), "im_file": f"img_{i}.jpg"} for i in range(20)]

    def build_mask():
        base = _Stub()
        base.labels = labels
        base.transforms = None
        base.collate_fn = None
        base.prefix = ""
        ds = SliceValDataset.__new__(SliceValDataset)
        ds.base = base
        ds.labels = labels
        ds.n = len(labels)
        ds.overlap_ratio = 0.2
        ds.all_tiles = True
        ds.ratio = 0.5
        ds._build()
        return ds._mask

    first, second = build_mask(), build_mask()
    assert first is not None and np.array_equal(first, second)
    assert int(first.sum()) == 10


def test_occlusion_segment_mismatch_warns_instead_of_silently_truncating(monkeypatch, caplog):
    """M-10: a segment/box length mismatch must be reported, not silently paired up by ``zip``."""
    from ultralytics.data import base as base_mod

    ds = _make_dataset(labels=[_make_label()])
    ds.augment = False
    ds.imgsz = 64
    ds.degrade_max_side = -1  # keep the frame at its native (tiny) size
    ds._occlusion_mask = None
    ds.occlusion_types = "rect"
    ds.occlusion_blocks = 1
    ds.occlusion_size_ratio = 0.9
    ds.occlusion_color = "black"
    ds.occlusion_max_cover = 0.0  # every box counts as fully covered -> the keep mask drops all
    ds.occlusion_save_dir = ""
    ds.im_files = ["a.jpg"]
    ds._load_image_cached = lambda idx: np.zeros((48, 64, 3), np.uint8)
    monkeypatch.setattr(base_mod, "_apply_occlusion", lambda img, t, **k: (img, [(0, 0, 10, 10)]))

    lb = _make_label(shape=(48, 64))
    lb["bboxes"] = np.array([[0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.4, 0.4]], dtype=np.float32)
    lb["cls"] = np.array([[0], [0]], dtype=np.float32)
    lb["segments"] = [np.zeros((3, 2), dtype=np.float32)]  # deliberately 1 segment for 2 boxes
    ds.labels = [lb]

    with caplog.at_level("WARNING"):
        ds._build_occlusion_sample(0, 0)

    assert "1 segments vs 2 boxes" in caplog.text


def test_online_defaults_match_default_cfg():
    """L-2: the in-code fallbacks and ``default.yaml`` must not drift.

    The 40+ ``getattr(self, <key>, _online_default(<key>))`` fallbacks are only correct if the table
    agrees with the authoritative config, and nothing else guards that.
    """
    from ultralytics.cfg import DEFAULT_CFG_DICT
    from ultralytics.data.base import _ONLINE_DEFAULTS

    mismatched = {
        k: (v, DEFAULT_CFG_DICT.get(k, "<missing from DEFAULT_CFG>"))
        for k, v in _ONLINE_DEFAULTS.items()
        if DEFAULT_CFG_DICT.get(k, object()) != v
    }
    assert not mismatched, f"_ONLINE_DEFAULTS drifted from default.yaml: {mismatched}"


# ---------------------------------------------------------------------------
# S-2 grouped sampling: keep one original's sub-samples inside the raw-image LRU
# ---------------------------------------------------------------------------
def _sampler_stub(**flags):
    """``BaseDataset`` skeleton carrying every attribute ``grouped_sample_units`` reads."""
    ds = _make_dataset(**{"augment": True, "_raw_cache_size": 4, **flags})
    ds.prefix = "test: "
    return ds


@pytest.mark.parametrize("N", [1, 2, 3, 4, 5, 6, 7, 8, 9, 16])
def test_grouped_units_partition_the_pool(N):
    """Grouping is a pure REORDERING: every pool index survives exactly once, none is invented.

    A layout that drops or duplicates indices would silently change which samples an epoch sees, so
    ``grouped_sample_units`` validates it too -- this test pins the contract it validates against.
    """
    ds = _sampler_stub(labels=[_make_label() for _ in range(N)])
    units = ds.grouped_sample_units()
    assert units is not None
    flat = [index for unit in units for block in unit for index in block]
    assert sorted(flat) == list(range(len(ds)))
    assert all(0 < len(unit) <= 4 for unit in units)
    assert all(len(block) > 0 for unit in units for block in unit)


@pytest.mark.parametrize("N", [4, 5, 8, 9, 16])
def test_grouped_units_place_compose_sample_first(N):
    """The compose sample must be the first index of the unit owning its group.

    It is the single sample that reads four originals at once, so it is what primes the LRU for the
    whole unit; leaving it mid-unit would make its four decodes evict what the round-robin just built.
    """
    ds = _sampler_stub(labels=[_make_label() for _ in range(N)])
    segment_bases = ds._segment_bases()
    units = ds.grouped_sample_units()
    for group in range(segment_bases.weather - segment_bases.compose):
        assert units[group][0][0] == segment_bases.compose + group


def test_grouped_units_disabled_when_grouping_cannot_pay_off():
    """No repeated decode to absorb => return None so the loader keeps its plain global shuffle."""
    no_reuse = _sampler_stub(
        slice_transform=None,
        slice_keep_origin=False,
        ratio_pad_keep=False,
        blur_keep=False,
        compose_keep=False,
        weather_keep=False,
        occlusion_keep=False,
    )
    assert no_reuse.grouped_sample_units() is None  # one index per image: nothing to group
    assert _sampler_stub(_raw_cache_size=0).grouped_sample_units() is None  # LRU off (and it is the only consumer)
    assert _sampler_stub(augment=False).grouped_sample_units() is None  # validation/inference dataset


def test_grouped_sampler_decodes_each_original_once():
    """Behavioural S-2 check: grouped order decodes each original ~once, global shuffle ~once per sample.

    Simulates the worker-side LRU (capacity 4, keyed by original image) over both orders and counts
    the reads that would need a real JPEG decode -- the cost S-2 is about (203 ms vs 21 ms a read).
    """
    import collections

    import torch

    from ultralytics.data.base import GroupedImageSampler

    n = 8
    ds = _sampler_stub(labels=[_make_label() for _ in range(n)])
    segment_bases = ds._segment_bases()
    units = ds.grouped_sample_units()
    total = segment_bases.total

    # Ground truth, independent of the sampler: which ORIGINAL image(s) each pool index reads.
    owner = {}
    for unit_index, unit in enumerate(units):
        for block_index, block in enumerate(unit):
            for index in block:
                owner[index] = 4 * unit_index + block_index

    def decodes(order, capacity=4):
        resident = collections.OrderedDict()
        misses = 0
        for index in order:
            if index >= segment_bases.compose:  # compose reads its whole group of four
                group = index - segment_bases.compose
                images = [(group * 4 + j) % n for j in range(4)]
            else:
                images = [owner[index]]
            if not all(image in resident for image in images):
                misses += 1
            for image in images:
                resident[image] = None
                resident.move_to_end(image)
            while len(resident) > capacity:
                resident.popitem(last=False)
        return misses

    sampler = GroupedImageSampler(units, seed=0)
    order = list(sampler)
    assert sorted(order) == list(range(total))  # covers the pool exactly once
    assert len(sampler) == total == len(ds)

    grouped = decodes(order)
    shuffled = decodes(torch.randperm(total).tolist())
    assert grouped <= 2 * n, f"grouped sampling should need ~1 decode per original, needed {grouped}"
    assert shuffled > 3 * n, f"a global shuffle must thrash the LRU, only needed {shuffled}"
    assert grouped * 3 < shuffled


def _tiny_tiled_dataset(tmp_path: Path, n: int, size=(64, 64)):
    """Write an ``n``-image YOLO detection dataset on disk and return ``(img_dir, data dict)``."""
    import cv2

    img_dir = tmp_path / "images" / "train"
    lbl_dir = tmp_path / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        cv2.imwrite(str(img_dir / f"{i}.jpg"), np.full((size[0], size[1], 3), 20 * i, dtype=np.uint8))
        (lbl_dir / f"{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    data = {"train": str(img_dir), "val": str(img_dir), "names": {0: "a"}, "nc": 1, "channels": 3}
    return img_dir, data


@pytest.mark.parametrize("enabled,grouped", [(True, True), (False, False)])
def test_build_dataloader_wires_grouped_sampler(tmp_path, enabled, grouped):
    """``slice_grouped_sampler`` must actually reach the loader, in both directions.

    Same failure mode as ``slice_raw_cache_size``: a config key that is registered and documented but
    never read looks perfectly healthy while doing nothing at all.
    """
    from ultralytics.cfg import get_cfg
    from ultralytics.data.base import GroupedImageSampler
    from ultralytics.data.build import build_dataloader, build_yolo_dataset

    img_dir, data = _tiny_tiled_dataset(tmp_path, 8)
    cfg = get_cfg(
        overrides={
            "data": str(tmp_path / "d.yaml"),
            "imgsz": 64,
            "task": "detect",
            "mode": "train",
            "slice_prob": 1.0,
            "slice_all_tiles": True,
            "slice_raw_cache_size": 4,
            "slice_grouped_sampler": enabled,
            "mosaic": 0.0,
            "workers": 0,
            "cache": False,
            "close_aug_epoch": 0,
        }
    )
    ds = build_yolo_dataset(cfg, str(img_dir), 8, data, mode="train")
    assert ds.slice_grouped_sampler is enabled

    loader = build_dataloader(ds, batch=4, workers=0, shuffle=True)
    sampler = getattr(loader, "sampler", None) or loader._index_sampler
    assert isinstance(sampler, GroupedImageSampler) is grouped
    if grouped:
        assert len(sampler) == len(ds)
        assert sorted(sampler) == list(range(len(ds)))


# ---------------------------------------------------------------------------
# S-3: compose is capped BEFORE the canvas is allocated (was: stitch at full
# sensor resolution, then downscale the whole canvas -- 5.6x time / 8.7x peak)
# ---------------------------------------------------------------------------
def _cap_long_side():
    from ultralytics.data.base import _cap_long_side as fn

    return fn


def test_cap_long_side_is_a_noop_under_the_cap():
    """Under the cap (or with the cap disabled) nothing is copied and scale is 1.0.

    The identity matters: callers reuse the returned object, and compose relies on
    ``scale == 1.0`` meaning "the input array itself, unmodified".
    """
    cap = _cap_long_side()
    im = np.zeros((40, 60, 3), dtype=np.uint8)
    for limit in (100, 0, -5):
        out, scale = cap(im, limit)
        assert out is im, f"limit={limit} must not copy"
        assert scale == 1.0, f"limit={limit} must report no scaling"


def test_cap_long_side_downscales_and_reports_the_factor():
    """A real downscale returns a NEW array whose long side is exactly the cap."""
    cap = _cap_long_side()
    im = np.zeros((300, 400, 3), dtype=np.uint8)
    out, scale = cap(im, 100)
    assert out is not im
    assert out.shape == (75, 100, 3), "long side must land exactly on the cap"
    assert scale == pytest.approx(0.25)


def _compose_stub(tmp_path: Path, n=4, size=(256, 192), max_side=64, imgsz=64, levels=None):
    """A ``BaseDataset`` skeleton whose compose group is ``n`` constant-colour JPEGs.

    Constant colours make the quadrant layout readable straight off the composed pixels (any
    interpolation of a constant region is that same constant), so the tests can assert on content
    rather than on a golden file.
    """
    import cv2
    from collections import OrderedDict, deque

    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    levels = levels if levels is not None else [20 * (i + 1) for i in range(n)]
    files = []
    for i in range(n):
        f = img_dir / f"{i}.jpg"
        cv2.imwrite(str(f), np.full((size[0], size[1], 3), levels[i], dtype=np.uint8))
        files.append(str(f))

    ds = _make_dataset(labels=[_make_label() for _ in range(n)], compose_keep=True)
    ds.im_files = files
    ds.imgsz = imgsz
    ds.compose_max_side = max_side
    ds.cv2_flag = cv2.IMREAD_COLOR
    ds.cache = None
    ds.augment = False
    ds.buffer = deque(maxlen=8)
    ds.prefix = ""
    ds._raw_cache = OrderedDict()
    ds._raw_cache_size = 4
    ds._raw_hits = 0
    ds._raw_misses = 0
    ds._compose_mask = None
    ds._seg_cache = None
    ds.compose_save = False
    return ds


def test_compose_canvas_is_allocated_at_the_capped_size(tmp_path, monkeypatch):
    """The pre-S-3 path allocated the canvas at FULL resolution and downscaled it afterwards.

    Both implementations render the same final geometry (labels are normalized, so the resize is
    label-neutral) -- which is exactly why the fix cannot be pinned down by looking at the output.
    Watching what actually gets allocated can.
    """
    import ultralytics.data.base as base_mod

    ds = _compose_stub(tmp_path, n=4, size=(256, 192), max_side=64, imgsz=64)
    allocs = []
    real_empty = np.empty

    def spy(shape, *args, **kwargs):
        allocs.append(tuple(shape) if isinstance(shape, (tuple, list)) else (shape,))
        return real_empty(shape, *args, **kwargs)

    monkeypatch.setattr(base_mod.np, "empty", spy)
    label = ds._build_compose_sample(ds._segment_bases().compose)
    monkeypatch.undo()

    # 4 sources of 256x192 capped to half=32 -> quadrants 32x24 -> canvas 64x48
    assert label["ori_shape"] == (64, 48)
    assert label["img"].shape == (64, 48, 3)
    canvas_allocs = [s for s in allocs if len(s) == 3]
    assert canvas_allocs, "no canvas allocation was observed"
    worst = max(max(s[:2]) for s in canvas_allocs)
    assert worst <= 64, f"allocated a {worst}-px canvas; the uncapped path would allocate 512"


def test_compose_caps_with_linear_interpolation(tmp_path, monkeypatch):
    """The compose cap must use ``INTER_LINEAR``, NOT the degradation branches' ``INTER_AREA``.

    Both shrink the image, but for a non-integer ratio OpenCV's INTER_AREA box path costs ~24x more
    (measured 40 ms vs 1.7 ms per 4000x3000 source at 6.25x decimation) -- enough to make compose
    slower than the full-resolution stitch it replaced. The kernel choice is invisible in the
    output, so it is asserted where it happens.
    """
    import cv2

    import ultralytics.data.base as base_mod

    ds = _compose_stub(tmp_path, n=4, size=(256, 192), max_side=64, imgsz=64)
    seen = []
    real_resize = base_mod.cv2.resize

    def spy(src, dsize, *args, **kwargs):
        seen.append(kwargs.get("interpolation"))
        return real_resize(src, dsize, *args, **kwargs)

    monkeypatch.setattr(base_mod.cv2, "resize", spy)
    ds._build_compose_sample(ds._segment_bases().compose)
    monkeypatch.undo()

    assert seen, "compose must resize (the sources are above half of compose_max_side)"
    assert set(seen) == {cv2.INTER_LINEAR}, f"unexpected kernels: {seen}"


def test_compose_quadrants_keep_the_group_order(tmp_path):
    """Capping must not scramble the 2x2 placement: TL, TR, BL, BR = group images 0..3.

    ``imgsz == compose_max_side`` makes ``_finalize_label`` a no-op (r == 1), so the returned image
    IS the canvas and the quadrants can be read off directly.
    """
    ds = _compose_stub(tmp_path, n=4, size=(256, 192), max_side=64, imgsz=64, levels=[20, 60, 100, 140])
    label = ds._build_compose_sample(ds._segment_bases().compose)
    img = label["img"]
    assert img.shape[:2] == (64, 48)
    quad_h, quad_w = 32, 24
    got = [
        float(img[r * quad_h:(r + 1) * quad_h, c * quad_w:(c + 1) * quad_w].mean())
        for r in (0, 1)
        for c in (0, 1)
    ]
    assert got == pytest.approx([20, 60, 100, 140], abs=2.0)


def test_compose_max_side_negative_disables_the_cap(tmp_path):
    """``< 0`` mirrors ``degrade_max_side``: keep the legacy full-resolution stitch untouched."""
    ds = _compose_stub(tmp_path, n=4, size=(256, 192), max_side=-1, imgsz=64)
    label = ds._build_compose_sample(ds._segment_bases().compose)
    assert label["ori_shape"] == (512, 384), "4 x 256x192 stitched at full resolution"


def test_compose_does_not_corrupt_the_worker_raw_cache(tmp_path):
    """compose reads its sources with ``copy=False``; the shared LRU buffer must stay pristine.

    If compose ever wrote THROUGH the shared buffer (or handed it to something that did), the second
    compose of the same group would differ from the first -- a silent augmentation bug with no crash
    and no obvious symptom.
    """
    from ultralytics.data.base import imread

    ds = _compose_stub(tmp_path, n=4, size=(256, 192), max_side=64, imgsz=64, levels=[20, 60, 100, 140])
    index = ds._segment_bases().compose
    assert np.array_equal(ds._build_compose_sample(index)["img"], ds._build_compose_sample(index)["img"])

    for i, f in enumerate(ds.im_files):
        cached = ds._raw_cache.get(i)
        assert cached is not None, f"image {i} should be resident (LRU capacity is 4 for 4 images)"
        assert np.array_equal(cached, imread(f, flags=ds.cv2_flag)), f"cached image {i} was mutated"


def test_compose_branch_runs_through_the_real_dataset(tmp_path):
    """End-to-end with a real ``BaseDataset``: auto cap (``0`` -> 2*imgsz) and all four labels kept."""
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset

    img_dir, data = _tiny_tiled_dataset(tmp_path, 8, size=(128, 96))
    cfg = get_cfg(
        overrides={
            "data": str(tmp_path / "d.yaml"),
            "imgsz": 64,
            "task": "detect",
            "mode": "train",
            "slice_prob": 0.0,
            "slice_all_tiles": False,
            "slice_keep_origin": False,
            "ratio_pad_keep": False,
            "blur_keep": False,
            "weather_keep": False,
            "occlusion_keep": False,
            "compose_keep": True,
            "compose_ratio": 1.0,
            "mosaic": 0.0,
            "workers": 0,
            "cache": False,
            "close_aug_epoch": 0,
        }
    )
    ds = build_yolo_dataset(cfg, str(img_dir), 8, data, mode="train")
    assert ds.compose_max_side == 0, "config key must reach the dataset (three-place rule)"

    compose_index = ds._segment_bases().compose
    label = ds._build_compose_sample(compose_index)
    # compose_max_side=0 -> auto 2*imgsz=128 -> half=64 -> sources 128x96 become 64x48 -> canvas 128x96
    assert label["ori_shape"] == (128, 96)
    assert label["img"].shape[:2] == (64, 48), "then resized to imgsz by the shared tail"
    # the real YOLODataset tail converts the normalized boxes into an Instances object
    instances = label["instances"]
    assert len(instances) == 4, "one box per quadrant"
    assert np.isfinite(instances.bboxes).all()
    assert float(instances.bboxes.max()) <= 1.0 + 1e-6, "normalized coords stay in [0, 1]"


def test_blur_and_weather_branches_run_through_the_real_dataset(tmp_path):
    """End-to-end: the two S-4 branches must survive a real ``__getitem__``.

    S-4 replaced the noise RNG (``cv2.randn`` on a single-channel view, seeded from the numpy
    stream) and trimmed the blur PSF before handing it to ``filter2D``. Both only execute inside
    DataLoader workers, so a mistake there surfaces as a crashed epoch rather than a failed
    assertion -- so drive the real segments instead of the bare helpers.

    The source images are *constant*, which makes "the branch actually ran" directly assertable:
    a degradation that silently no-ops would leave a zero-variance image.
    """
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset

    img_dir, data = _tiny_tiled_dataset(tmp_path, 8, size=(96, 96))
    cfg = get_cfg(
        overrides={
            "data": str(tmp_path / "d.yaml"),
            "imgsz": 64,
            "task": "detect",
            "mode": "train",
            "slice_prob": 0.0,
            "slice_all_tiles": False,
            "slice_keep_origin": False,
            "ratio_pad_keep": False,
            "compose_keep": False,
            "occlusion_keep": False,
            "blur_keep": True,
            "blur_ratio": 1.0,
            "weather_keep": True,
            "weather_ratio": 1.0,
            "weather_types": "noise",
            "weather_noise_std": 15.0,
            "mosaic": 0.0,
            "workers": 0,
            "cache": False,
            "close_aug_epoch": 0,
        }
    )
    ds = build_yolo_dataset(cfg, str(img_dir), 8, data, mode="train")
    bases = ds._segment_bases()
    assert ds.weather_noise_std == 15.0, "config key must reach the dataset (three-place rule)"

    # blur segment holds 2 samples per original (short + long); weather holds 1
    for name, start, count in (("blur", bases.blur, 16), ("weather", bases.weather, 8)):
        for offset in range(count):
            label = ds[start + offset]
            img = label["img"]
            # the real YOLODataset tail hands back a CHW torch tensor
            assert str(img.dtype).endswith("uint8"), f"{name}[{offset}] dtype {img.dtype}"
            assert img.ndim == 3 and img.shape[0] == 3, f"{name}[{offset}] shape {tuple(img.shape)}"
            assert max(img.shape[1:]) == 64, f"{name}[{offset}] was not resized to imgsz"
            # the transform tail unpacks Instances into cls/bboxes; the branch must not drop them
            assert len(label["cls"]) >= 1, f"{name}[{offset}] lost its labels"

    # the noise branch must leave visible noise on a constant source (std 0 if it silently no-oped)
    for offset in range(8):
        img = ds[bases.weather + offset]["img"]
        assert float(np.asarray(img, dtype=np.float32).std()) > 1.0, "noise branch did not apply"


def test_weather_occlusion_whitelist_single_source_of_truth():
    """M-2: the type whitelist must come straight from ``online_degrade``, not be laundered via base.

    ``base.py`` used to import ``_WEATHER_TYPES`` / ``_OCCLUSION_TYPES`` for the sole purpose of
    letting ``augment.py`` re-import them from there -- it never used them itself (Ruff F401 x2).
    A single ``ruff --fix``, or an IDE "optimise imports", would therefore have deleted those two
    lines and silently removed the construction-time validation, leaving ``weather_types="rian"`` to
    fall through to the runtime random fallback instead of raising. Importing directly makes the
    dependency real (the name is used), so no linter has a reason to touch it.
    """
    from ultralytics.data import augment, online_degrade

    assert augment._WEATHER_TYPES is online_degrade._WEATHER_TYPES
    assert augment._OCCLUSION_TYPES is online_degrade._OCCLUSION_TYPES


def test_base_carries_no_unused_online_degrade_import():
    """M-2 guard: every name ``base.py`` pulls from ``online_degrade`` must actually be used there.

    A name imported only to be re-exported is invisible to behavioural tests and is exactly what an
    auto-fix deletes, so assert it statically. This deliberately duplicates Ruff F401: the local loop
    has no lint step (upstream only runs Ruff on pull requests).
    """
    import ast

    import ultralytics.data.base as base_mod

    tree = ast.parse(Path(base_mod.__file__).read_text(encoding="utf-8"))
    imported = {
        a.asname or a.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "ultralytics.data.online_degrade"
        for a in node.names
    }
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}

    assert imported, "base.py is expected to import from online_degrade"
    assert imported <= used, f"base.py imports names it never uses (re-export trap): {sorted(imported - used)}"


@pytest.mark.parametrize(
    ("key", "bad", "valid"),
    (("weather_types", "rian", "noise"), ("occlusion_types", "rectangle", "rect")),
)
def test_typo_in_type_whitelist_fails_at_construction(tmp_path, key, bad, valid):
    """M-2: a typo must raise while the dataset is built, not silently pick a runtime fallback.

    This is the behaviour the re-export was enabling; if the import ever goes missing again the
    validation disappears, and this test is what notices.
    """
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset

    img_dir, data = _tiny_tiled_dataset(tmp_path, 2, size=(64, 64))
    cfg = get_cfg(
        overrides={
            "data": str(tmp_path / "d.yaml"),
            "imgsz": 64,
            "task": "detect",
            "mode": "train",
            "cache": False,
            key: bad,
        }
    )
    with pytest.raises(ValueError, match=key) as exc:
        build_yolo_dataset(cfg, str(img_dir), 2, data, mode="train")
    assert valid in str(exc.value), f"the error must name the valid types, got: {exc.value}"

