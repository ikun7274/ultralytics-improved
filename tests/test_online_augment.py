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

    All seven branch switches default to ON (ratio/blur/compose/weather/occlusion + slicing with emit_all +
    keep_origin), so the full-pool layout is the default case and individual tests turn specific branches off.
    """
    from ultralytics.data.base import BaseDataset

    if labels is None:
        labels = [_make_label() for _ in range(5)]
    ds = BaseDataset.__new__(BaseDataset)
    ds.labels = labels
    ds.im_files = [f"img_{i}.jpg" for i in range(len(labels))]
    ds.augment = False
    defaults = {
        "slice_all_tiles": True,
        "slice_transform": "fake",  # non-None: slicing pipeline active
        "slice_keep_origin": True,
        "ratio_pad_keep": True,
        "blur_keep": True,
        "compose_keep": True,
        "weather_keep": True,
        "occlusion_keep": True,
    }
    defaults.update(flags)
    for k, v in defaults.items():
        setattr(ds, k, v)
    return ds


def _bases(ds):
    """Return the 7 segment boundaries + total as a plain dict for easy assertions."""
    segment_bases = ds._segment_bases()
    return {
        "base": segment_bases.base,
        "origin": segment_bases.origin,
        "ratio": segment_bases.ratio,
        "blur": segment_bases.blur,
        "compose": segment_bases.compose,
        "weather": segment_bases.weather,
        "occlusion": segment_bases.occlusion,
        "total": segment_bases.total,
    }


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
    expected = {
        "base": base,
        "origin": base,  # origin segment starts right after the base segment
        "ratio": base + N,
        "blur": base + 2 * N,
        "compose": base + 4 * N,
        "weather": base + 4 * N + compose_len,
        "occlusion": base + 5 * N + compose_len,
        "total": base + 6 * N + compose_len,
    }
    got = _bases(ds)
    assert got == expected, f"N={N}: {got} != {expected}"
    assert len(ds) == expected["total"]  # __len__ and total must never drift


def test_segment_bases_no_compose():
    """Compose off: compose start collapses onto the blur END (start + 2N); len = 4N+N+N+2N+N+N = 10N."""
    ds = _make_dataset(compose_keep=False)
    n = len(ds.labels)
    segment_bases = _bases(ds)
    assert segment_bases["compose"] == segment_bases["blur"] + 2 * n  # blur segment is 2N long
    assert len(ds) == 10 * n


def test_segment_bases_keep_origin_only():
    """Only slicing + keep_origin: len = 4N + N = 5N."""
    ds = _make_dataset(
        ratio_pad_keep=False,
        blur_keep=False,
        compose_keep=False,
        weather_keep=False,
        occlusion_keep=False,
    )
    assert len(ds) == 5 * len(ds.labels)


def test_segment_bases_emit_all_only():
    """emit_all without any extra branch: len = 4N."""
    ds = _make_dataset(
        slice_keep_origin=False,
        ratio_pad_keep=False,
        blur_keep=False,
        compose_keep=False,
        weather_keep=False,
        occlusion_keep=False,
    )
    assert len(ds) == 4 * len(ds.labels)


def test_segment_bases_no_slicing():
    """Slicing off (slice_transform=None): base = N, origin auto-suppressed (would duplicate)."""
    ds = _make_dataset(
        slice_transform=None,
        slice_keep_origin=False,
        ratio_pad_keep=False,
        blur_keep=False,
        compose_keep=False,
        weather_keep=False,
        occlusion_keep=False,
    )
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
# Segment-bases cache: every layout switch is attached AFTER __init__
# ---------------------------------------------------------------------------
# Every attribute that can move a boundary. `slice_transform` is the odd one out (None vs an
# object rather than a bool), so it is set to None / "fake" instead of False / True.
_LAYOUT_SWITCHES = (
    "slice_all_tiles",
    "slice_keep_origin",
    "ratio_pad_keep",
    "blur_keep",
    "compose_keep",
    "weather_keep",
    "occlusion_keep",
)


def _off_value(switch):
    """The switch's OFF value (see ``_LAYOUT_SWITCHES``)."""
    return None if switch == "slice_transform" else False


@pytest.mark.parametrize("switch", [*_LAYOUT_SWITCHES, "slice_transform"])
def test_segment_cache_invalidates_when_a_switch_changes(switch):
    """A pool layout cached before a switch flip must follow that flip.

    The switches are NOT settled by ``__init__``: ``v8_transforms`` attaches them afterwards (and ``__init__`` itself
    only reads ``len(self)`` after ``build_transforms``). A "first call wins" cache therefore made that ordering a
    *silent* correctness requirement -- read ``len()`` too early (a log line, a sanity check, a test that configures a
    dataset after constructing it) and the boundaries stayed frozen on the old switches, so the pool length disagreed
    with the decodable index range and samples were dropped / indexed out of range with no error at all.

    Asserted on both sides: the version stamp must move, AND the rendered layout must equal a dataset that was built
    with the flipped switch from the start.
    """
    ds = _make_dataset()
    ds._segment_bases()  # populate the cache (and the stamp)
    stamp = ds._seg_key
    setattr(ds, switch, _off_value(switch))
    fresh = _make_dataset(**{switch: _off_value(switch)})
    # _segment_bases() is what refreshes the stamp, so it must be called before the stamp is read.
    assert _bases(ds) == _bases(fresh)
    assert ds._seg_key != stamp, f"flipping {switch} must move the version stamp"
    assert len(ds) == len(fresh)


def test_segment_bases_are_cached_not_recomputed():
    """The invalidation must not give back the caching win it is protecting.

    Without this pin the "fix" could be to drop the cache entirely -- correct, but it would hand back the L8 gain and
    re-allocate a ``SegmentBases`` (+ a list + a loop) on every ``__getitem__``. A rebuild would return an EQUAL but
    different object, so identity is the only assertion that can tell the two apart.
    """
    ds = _make_dataset()
    first = ds._segment_bases()
    assert ds._segment_bases() is first


def test_early_len_does_not_poison_the_final_layout():
    """The reported failure, reproduced end to end: read ``len()`` BEFORE the switches exist.

    Mirrors ``__init__`` -> ``build_transforms``: mid-``__init__`` the object has no ``slice_transform`` and every
    ``*_keep`` is False, so an early ``len(self)`` used to freeze the plain-original layout into the cache and leave the
    fully configured dataset advertising N samples instead of the expanded pool.
    """
    ds = _make_dataset(**{k: _off_value(k) for k in _LAYOUT_SWITCHES} | {"slice_transform": None})
    early = len(ds)  # the poison
    # v8_transforms now attaches the switches, as it does at the end of __init__.
    all_on = {
        "slice_all_tiles": True,
        "slice_transform": "fake",
        "slice_keep_origin": True,
        "ratio_pad_keep": True,
        "blur_keep": True,
        "compose_keep": True,
        "weather_keep": True,
        "occlusion_keep": True,
    }
    for k, v in all_on.items():
        setattr(ds, k, v)
    assert early == len(ds.labels), "the early read must see the unconfigured (plain original) layout"
    assert len(ds) == len(_make_dataset()), "the late read must see the full expanded pool"


def test_segment_cache_invalidates_when_labels_change():
    """``len(self.labels)`` is a layout input too: replacing ``labels`` must resize the pool."""
    ds = _make_dataset()
    ds._segment_bases()
    ds.labels = [_make_label() for _ in range(9)]
    fresh = _make_dataset(labels=[_make_label() for _ in range(9)])
    assert _bases(ds) == _bases(fresh)
    assert len(ds) == len(fresh)


# ---------------------------------------------------------------------------
# Motion blur kernel: degenerate input must not produce NaN
# ---------------------------------------------------------------------------
def test_motion_blur_kernel_degenerate_safe():
    """A degenerate motion-blur kernel (length / angle that leaves no rasterized line) used to divide by zero and
    produce NaN -- a failure that the blanket ``filterwarnings('ignore')`` in ``detect.py`` / ``export.py`` (kept
    as-is by user decision) would have silently hidden. The guard now degrades to an identity kernel.
    """
    from ultralytics.data.online_degrade import _motion_blur_kernel

    # length=0 + any angle -> the cv2.line call rasterizes nothing -> sum() == 0
    k = _motion_blur_kernel(length=0.0, angle=0.0)
    assert k.shape[0] >= 3 and k.shape[1] >= 3
    assert math.isfinite(k.sum())
    assert np.isclose(k.sum(), 1.0)  # guard degrades to identity -> sums to 1
    # Center pixel should be 1.0 (identity), everything else 0
    c = k.shape[0] // 2
    assert k[c, c] == 1.0


def test_motion_blur_kernel_normal():
    """A normal kernel still rasterizes a line and sums to 1 (regression: don't break the happy path)."""
    from ultralytics.data.online_degrade import _motion_blur_kernel

    k = _motion_blur_kernel(length=15.0, angle=30.0)
    assert math.isfinite(k.sum())
    assert np.isclose(k.sum(), 1.0, atol=1e-5)
    assert k.sum() > 0


# ---------------------------------------------------------------------------
# blur_axis_aligned: on an axis the PSF is exactly a uniform box, so cv2.blur is a bit-exact
# (and much faster) stand-in for the dense filter2D convolution. Guarded at four levels: the
# invariant, the bit-for-bit equality, the mechanism (which OpenCV call is made), and the sampler.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("length", [3.0, 5.0, 6.0, 9.5, 21.0, 35.0])
def test_axis_psf_is_a_uniform_box_of_psf_size(length):
    """Invariant the fast path rests on: on-axis the PSF is a uniform box exactly ``_psf_size`` wide.

    If the rasterization or the size formula ever drifts, the box filter silently stops being an exact replacement --
    this is the test that would catch it first.
    """
    from ultralytics.data.online_degrade import _motion_blur_kernel, _psf_size

    n = _psf_size(length)
    assert n % 2 == 1, "odd size keeps the PSF center on a pixel"
    for angle, vertical in ((0.0, False), (90.0, True)):
        k = _motion_blur_kernel(length, angle)
        taps = k[np.nonzero(k)]
        assert k.shape == (n, n)
        assert np.count_nonzero(k) == n, "the segment must occupy exactly one row/column"
        assert np.unique(np.round(taps, 6)).size == 1, "on-axis taps must be uniform"
        assert np.isclose(taps[0], 1.0 / n, atol=1e-6)
        rows, cols = np.nonzero(k)
        # a horizontal smear spans COLUMNS, a vertical one spans ROWS
        assert np.ptp(rows if vertical else cols) + 1 == n


@pytest.mark.parametrize("angle", [0.0, 90.0])
@pytest.mark.parametrize("defocus", [0.0, 1.0])
@pytest.mark.parametrize("length", [5.0, 7.25, 12.0, 20.0, 21.0, 27.5, 35.0])
def test_axis_aligned_blur_is_bit_identical_to_the_psf(length, angle, defocus):
    """``axis_aligned=True`` must be a pure replacement, not an approximation.

    The switch is allowed to change WHICH angles are sampled; it is not allowed to change any pixel for a given angle.
    Anything but exact equality here means the optimization traded quality for speed and must not be enabled by default.
    """
    from ultralytics.data.online_degrade import _apply_motion_blur

    img = np.random.default_rng(11).integers(0, 256, (96, 128, 3), dtype=np.uint8)
    fast = _apply_motion_blur(img, length=length, angle=angle, defocus_sigma=defocus, axis_aligned=True)
    ref = _apply_motion_blur(img, length=length, angle=angle, defocus_sigma=defocus, axis_aligned=False)
    assert np.array_equal(fast, ref)


def test_axis_aligned_blur_uses_the_box_filter(monkeypatch):
    """Guard the mechanism, not just the numbers: the fast path must make the O(1) cv2.blur call.

    A later refactor could keep every output identical while quietly falling back to a dense filter2D, handing back the
    whole speed-up; only watching the calls catches that.
    """
    import cv2

    from ultralytics.data import online_degrade as od

    calls = []
    real_blur = cv2.blur

    def spy_blur(im, ksize, *args, **kwargs):
        calls.append(tuple(ksize))
        return real_blur(im, ksize, *args, **kwargs)

    monkeypatch.setattr(od.cv2, "blur", spy_blur)
    monkeypatch.setattr(od.cv2, "filter2D", lambda *_a, **_k: pytest.fail("axis-aligned path must not convolve"))

    img = np.zeros((32, 48, 3), np.uint8)
    od._apply_motion_blur(img, length=21.0, angle=0.0, axis_aligned=True)
    od._apply_motion_blur(img, length=21.0, angle=90.0, axis_aligned=True)
    assert calls == [(21, 1), (1, 21)], "horizontal -> (n, 1) kernel, vertical -> (1, n)"


def _stub_blur_dataset(monkeypatch, **flags):
    """A dataset whose blur sampling can be observed without touching any pixels."""
    from ultralytics.data import base as base_mod

    ds = _make_dataset(**flags)
    ds._blur_mask = None
    monkeypatch.setattr(ds, "_degrade_frame", lambda _i: (np.zeros((32, 32, 3), np.uint8), 1.0))
    monkeypatch.setattr(ds, "_finalize_label", lambda label, _img: label)
    seen = []

    def spy(im, length=15.0, angle=30.0, defocus_sigma=0.0, axis_aligned=False):
        seen.append({"length": length, "angle": angle, "sigma": defocus_sigma, "axis": axis_aligned})
        return im

    monkeypatch.setattr(base_mod, "_apply_motion_blur", spy)
    return ds, seen


def test_build_blur_sample_restricts_angles_when_axis_aligned(monkeypatch):
    """With the switch on the sampler must only ever emit 0/90 -- and must request the fast path."""
    ds, seen = _stub_blur_dataset(monkeypatch, blur_axis_aligned=True)
    for _ in range(8):
        ds._build_blur_sample(0, 0, long=False)  # short tier
        ds._build_blur_sample(0, 0, long=True)  # long tier (both tiers share this angle choice)
    assert len(seen) == 16
    assert all(s["axis"] is True for s in seen)
    assert {s["angle"] for s in seen} == {0.0, 90.0}, "both axes must be reachable"
    # Only the direction is replaced -- the tier wiring (length range / defocus) must be untouched.
    assert all(5.0 <= s["length"] <= 12.0 and s["sigma"] == 0.0 for s in seen[0::2])
    assert all(20.0 <= s["length"] <= 35.0 and s["sigma"] == 1.0 for s in seen[1::2])


def test_build_blur_sample_keeps_continuous_angles_when_disabled(monkeypatch):
    """``blur_axis_aligned: False`` must restore the original U[0, 180) smear direction."""
    ds, seen = _stub_blur_dataset(monkeypatch, blur_axis_aligned=False)
    for _ in range(16):
        ds._build_blur_sample(0, 0)
    assert all(s["axis"] is False for s in seen)
    assert any(s["angle"] not in (0.0, 90.0) for s in seen), "angles must be free again"


def test_blur_axis_aligned_keeps_the_rng_stream_position(monkeypatch):
    """Flipping the switch must consume the same number of ``random`` draws.

    Both branches draw exactly one length and one direction, so the stream is left in the same place and an A/B of the
    switch compares the same augmentation sequence rather than two different samplings.
    """
    import random

    def next_value_after_one_sample(flag):
        ds, _ = _stub_blur_dataset(monkeypatch, blur_axis_aligned=flag)
        random.seed(1234)
        ds._build_blur_sample(0, 0)
        return random.random()

    assert next_value_after_one_sample(True) == next_value_after_one_sample(False)


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
    """``slice_raw_cache_size`` must actually reach the dataset.

    It used to be read from ``self`` inside ``BaseDataset.__init__`` -- before ``v8_transforms`` copied hyp's keys onto
    the dataset -- so the knob (including its "0 = off" semantics and the "raise it to speed up decoding" hint in the
    cache='ram' warning) was permanently stuck at the default 2.
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

    ``DetectionTrainer.get_validator`` hands the validator a PREBUILT whole-image ``test_loader``. The old "rebuild only
    when the mode changes" shortcut saw ``True == True`` and reused it, so ``val_slice_enable=True`` validated on whole
    images and ``SliceValDataset`` was never constructed.
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

    # (4) vanilla training (slice off) -> keep reusing the prebuilt loader (upstream behavior)
    trainer = _trainer_with(whole_loader, False)
    trainer._run_val(False)
    assert trainer.validator.called_with is whole_loader


def test_save_metrics_realigns_changed_metric_set(tmp_path):
    """Rows are positional, so a changed metric set (e.g. dual-metric on resume) must not silently out-grow the header.
    """
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
    """Sliced batches without ``_slice_base_labels`` must fail loudly."""
    import torch

    from ultralytics.models.yolo.detect.val import DetectionValidator

    v = DetectionValidator.__new__(DetectionValidator)
    v._slice_base_labels = None
    batch = {"img": torch.zeros(1, 3, 64, 64), "val_slice_meta": [{"orig_idx": 0, "n_tiles": 4}]}
    with pytest.raises(RuntimeError, match="_slice_base_labels"):
        v._update_metrics_sliced([], batch)


def test_sliced_metrics_guard_rejects_non_square_canvas():
    """Remap assumes a square canvas; a non-square one would silently mis-place every box."""
    import torch

    from ultralytics.models.yolo.detect.val import DetectionValidator

    v = DetectionValidator.__new__(DetectionValidator)
    v._slice_base_labels = []
    batch = {"img": torch.zeros(1, 3, 64, 96), "val_slice_meta": [{"orig_idx": 0, "n_tiles": 4}]}
    with pytest.raises(RuntimeError, match="square validation canvas"):
        v._update_metrics_sliced([], batch)


def test_finalize_metrics_clears_unfinished_slice_acc():
    """Originals that never saw all sub-tiles must not leak into the next pass."""
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
# Regression guards from the 2026-09 code reviews: cache bounds, determinism,
# sampling locality, config hygiene, warning transparency.
# ---------------------------------------------------------------------------
def _make_loader_dataset(n=12, cap=4, extended=True):
    """Minimal BaseDataset skeleton able to run ``load_image`` (no directory scan).

    ``extended=True`` turns on ONE extended-pool branch (ratio_pad_keep) while leaving slicing off -- exactly the
    combination that used to let ``self.ims`` grow without bound.
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
    """``self.ims`` must stay FIFO-bounded even though the mosaic buffer cannot evict it.

    Regression: the only eviction sat behind ``not self._extended_pool_on()``, so with slicing off plus any extended
    branch on, every visited image stayed resident -- one frame per image in the dataset (~29 GB/worker for 8520 images
    at imgsz=1280) and a monotone RSS climb with no warning.
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
    """Parity: the pure-ultralytics path must stay bounded too (legacy buffer behavior).

    One owner (``_remember_ims``) now evicts ``self.ims`` in every mode; this guards that moving the eviction out of
    ``load_image``'s buffer branch did not lose the bound.
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
    """The degradation branches work at ``degrade_max_side``, not at the sensor resolution.

    ``_degrade_frame`` must report the applied scale so pixel-typed parameters (PSF length, defocus sigma, rain-line
    length) can shrink with it -- that is what keeps the post-resize result equivalent to the uncapped path.
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

    ds.degrade_max_side = -1  # disabled -> untouched, scale 1.0 (legacy behavior)
    im, scale = ds._degrade_frame(0)
    assert im.shape[:2] == (300, 400) and scale == 1.0

    small = np.zeros((40, 60, 3), np.uint8)
    ds._load_image_cached = lambda idx: small
    ds.degrade_max_side = 0
    im, scale = ds._degrade_frame(0)  # already under the cap: no resize at all
    assert im is small and scale == 1.0


def test_weather_noise_uses_cv2_randn_on_a_single_channel_view(monkeypatch):
    """Noise must be drawn by ``cv2.randn`` into ONE float32 buffer, on a C1 view.

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
    """The produced noise must have the configured std -- the cv2.randn channel trap.

    ``cv2.randn(dst, 0, sigma)`` silently yields ``sigma/sqrt(3)`` when ``dst`` is 3-channel, so a regression here looks
    like a working run with a 42% weaker augmentation. Mid-gray input keeps clipping out of the measurement.
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
    """cv2's RNG is seeded from the numpy stream, so reproducibility is unchanged.

    The branch used to consume the numpy stream itself; it now consumes exactly one ``randint`` and hands it to
    ``cv2.setRNGSeed``. Same ``np.random.seed`` -> same pixels, and every consumer downstream of the call sees the
    stream positioned exactly as before.
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
    """Trimming the zero border off the PSF must not change a single pixel.

    ``cv2.filter2D`` walks the whole ``size x size`` kernel even though a rasterized line only occupies a thin diagonal
    band, and its cost has a cliff around 11 px. Cropping the zeros away and passing the matching ``anchor`` is exact (a
    zero tap contributes nothing) and measured 1.27x over the configured length distribution, so this guards both the
    equality and the anchor contract.
    """
    import cv2

    from ultralytics.data import online_degrade as od

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
    """Kept as a guard that no full-frame float64 temporary is allocated.

    Regression: ``np.random.normal(0, sigma, img.shape)`` allocated 274 MB (float64) for a 4000x3000 frame and was
    immediately truncated to uint8. The draw is now done as a single ``cv2.randn`` into a float32 buffer, which is
    stricter than chunking: the only full-frame temporary left is the float32 accumulator (4x the frame), not an 8x
    float64 one.
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
    """Explicit LRU -- capacity honored, hits refresh order, eviction drops exactly one entry."""
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
    """``val_slice_ratio < 1`` must score the SAME subset every round.

    Regression: an unseeded ``random.sample`` redrew the sliced subset on each validation round (the wrapper is rebuilt
    per round), so mAP moved between epochs partly because a different set of images was being scored -- and that jitter
    drives best.pt / early stopping.
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
    """A segment/box length mismatch must be reported, not silently paired up by ``zip``."""
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
    """The in-code fallbacks and ``default.yaml`` must not drift.

    The 40+ ``getattr(self, <key>, _online_default(<key>))`` fallbacks are only correct if the table agrees with the
    authoritative config, and nothing else guards that.
    """
    from ultralytics.cfg import DEFAULT_CFG_DICT
    from ultralytics.data.base import _ONLINE_DEFAULTS

    mismatched = {
        k: (v, DEFAULT_CFG_DICT.get(k, "<missing from DEFAULT_CFG>"))
        for k, v in _ONLINE_DEFAULTS.items()
        if DEFAULT_CFG_DICT.get(k, object()) != v
    }
    assert not mismatched, f"_ONLINE_DEFAULTS drifted from default.yaml: {mismatched}"


def test_online_defaults_cover_every_online_cfg_key():
    """Reverse drift guard (pairs with the test above): every DEFAULT_CFG key in the online-augment namespace MUST have
    a fallback in ``_ONLINE_DEFAULTS``, so no training-side ``getattr(hyp, <key>, <literal>)`` can silently drift
    when a new online key is added but the table is not. Excluded: ``mask_ratio`` (upstream segment-Mosaic key, not
    an online branch) and the ``val_slice_*`` family (validator-side slicing has its own defaults in the
    val pipeline).
    """
    from ultralytics.cfg import DEFAULT_CFG_DICT
    from ultralytics.data.base import _ONLINE_DEFAULTS

    excluded = {"mask_ratio", "val_slice_ratio", "val_slice_overlap_ratio"}
    missing = sorted(
        k
        for k in DEFAULT_CFG_DICT
        if k not in excluded
        and (k.startswith("slice_") or k.endswith(("_keep", "_ratio")))
        and k not in _ONLINE_DEFAULTS
    )
    assert not missing, f"online cfg keys without _ONLINE_DEFAULTS fallback: {missing}"


# ---------------------------------------------------------------------------
# Grouped sampling: keep one original's sub-samples inside the raw-image LRU
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

    It is the single sample that reads four originals at once, so it is what primes the LRU for the whole unit; leaving
    it mid-unit would make its four decodes evict what the round-robin just built.
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
    """Behavioral check: grouped order decodes each original ~once, global shuffle ~once per sample.

    Simulates the worker-side LRU (capacity 4, keyed by original image) over both orders and counts the reads that would
    need a real JPEG decode -- the cost being guarded (203 ms vs 21 ms a read).
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

    Same failure mode as ``slice_raw_cache_size``: a config key that is registered and documented but never read looks
    perfectly healthy while doing nothing at all.
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
# compose is capped BEFORE the canvas is allocated (was: stitch at full
# sensor resolution, then downscale the whole canvas -- 5.6x time / 8.7x peak)
# ---------------------------------------------------------------------------
def _cap_long_side():
    from ultralytics.data.base import _cap_long_side as fn

    return fn


def test_cap_long_side_is_a_noop_under_the_cap():
    """Under the cap (or with the cap disabled) nothing is copied and scale is 1.0.

    The identity matters: callers reuse the returned object, and compose relies on ``scale == 1.0`` meaning "the input
    array itself, unmodified".
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

    Constant colors make the quadrant layout readable straight off the composed pixels (any interpolation of a constant
    region is that same constant), so the tests can assert on content rather than on a golden file.
    """
    from collections import OrderedDict, deque

    import cv2

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
    """The original implementation allocated the canvas at FULL resolution and downscaled it afterwards.

    Both implementations render the same final geometry (labels are normalized, so the resize is label-neutral) -- which
    is exactly why the fix cannot be pinned down by looking at the output. Watching what actually gets allocated can.
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

    Both shrink the image, but for a non-integer ratio OpenCV's INTER_AREA box path costs ~24x more (measured 40 ms vs
    1.7 ms per 4000x3000 source at 6.25x decimation) -- enough to make compose slower than the full-resolution stitch it
    replaced. The kernel choice is invisible in the output, so it is asserted where it happens.
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

    ``imgsz == compose_max_side`` makes ``_finalize_label`` a no-op (r == 1), so the returned image IS the canvas and
    the quadrants can be read off directly.
    """
    ds = _compose_stub(tmp_path, n=4, size=(256, 192), max_side=64, imgsz=64, levels=[20, 60, 100, 140])
    label = ds._build_compose_sample(ds._segment_bases().compose)
    img = label["img"]
    assert img.shape[:2] == (64, 48)
    quad_h, quad_w = 32, 24
    got = [
        float(img[r * quad_h : (r + 1) * quad_h, c * quad_w : (c + 1) * quad_w].mean()) for r in (0, 1) for c in (0, 1)
    ]
    assert got == pytest.approx([20, 60, 100, 140], abs=2.0)


def test_compose_max_side_negative_disables_the_cap(tmp_path):
    """``< 0`` mirrors ``degrade_max_side``: keep the legacy full-resolution stitch untouched."""
    ds = _compose_stub(tmp_path, n=4, size=(256, 192), max_side=-1, imgsz=64)
    label = ds._build_compose_sample(ds._segment_bases().compose)
    assert label["ori_shape"] == (512, 384), "4 x 256x192 stitched at full resolution"


def test_compose_does_not_corrupt_the_worker_raw_cache(tmp_path):
    """Compose reads its sources with ``copy=False``; the shared LRU buffer must stay pristine.

    If compose ever wrote THROUGH the shared buffer (or handed it to something that did), the second compose of the same
    group would differ from the first -- a silent augmentation bug with no crash and no obvious symptom.
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
    """End-to-end: the two degrade branches (noise + blur) must survive a real ``__getitem__``.

    The noise draw now goes through ``cv2.randn`` (on a single-channel view, seeded from the numpy stream), and the blur
    PSF is trimmed before being handed to ``filter2D``. Both only execute inside DataLoader workers, so a mistake there
    surfaces as a crashed epoch rather than a failed assertion -- so drive the real segments instead of the bare
    helpers.

    The source images are *constant*, which makes "the branch actually ran" directly assertable: a degradation that
    silently no-ops would leave a zero-variance image.
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
    """The type whitelist must come straight from ``online_degrade``, not be laundered via base.

    ``base.py`` used to import ``_WEATHER_TYPES`` / ``_OCCLUSION_TYPES`` for the sole purpose of letting ``augment.py``
    re-import them from there -- it never used them itself (Ruff F401 x2). A single ``ruff --fix``, or an IDE "optimize
    imports", would therefore have deleted those two lines and silently removed the construction-time validation,
    leaving ``weather_types="rian"`` to fall through to the runtime random fallback instead of raising. Importing
    directly makes the dependency real (the name is used), so no linter has a reason to touch it.
    """
    from ultralytics.data import augment, online_degrade

    assert augment._WEATHER_TYPES is online_degrade._WEATHER_TYPES
    assert augment._OCCLUSION_TYPES is online_degrade._OCCLUSION_TYPES


def test_base_carries_no_unused_online_degrade_import():
    """Guard: every name ``base.py`` pulls from ``online_degrade`` must actually be used there.

    A name imported only to be re-exported is invisible to behavioral tests and is exactly what an auto-fix deletes, so
    assert it statically. This deliberately duplicates Ruff F401: the local loop has no lint step (upstream only runs
    Ruff on pull requests).
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
    """A typo must raise while the dataset is built, not silently pick a runtime fallback.

    This is the behavior the re-export was enabling; if the import ever goes missing again the validation disappears,
    and this test is what notices.
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


# ---------------------------------------------------------------------------
# ONE 2x2 grid per (epoch, image), shared by all 4 tiles
# ---------------------------------------------------------------------------
def _biased_slice(**flags):
    """A real ``OnlineSlice`` with the target-aware seam ON -- the grid math itself is under test."""
    from ultralytics.data.augment import OnlineSlice

    defaults = {"p": 1.0, "overlap_ratio": 0.2, "center_bias": True, "bias_jitter": 0.05}
    defaults.update(flags)
    return OnlineSlice(**defaults)


def _spread_boxes(w=4000, h=3000, n=6, seed=0):
    """Pixel ``xyxy`` boxes spread over the frame, so the seam search has a real signal."""
    rng = np.random.default_rng(seed)
    cx, cy = rng.uniform(0.1, 0.9, n), rng.uniform(0.1, 0.9, n)
    bw, bh = rng.uniform(0.05, 0.15, n), rng.uniform(0.05, 0.15, n)
    return np.stack([cx * w - bw * w / 2, cy * h - bh * h / 2, cx * w + bw * w / 2, cy * h + bh * h / 2], axis=1)


def _tile_rects(tiles):
    """The 4 tile rects as plain int tuples (hashable, so a set can prove they are identical)."""
    return [(int(x0), int(y0), int(x1), int(y1)) for x0, y0, x1, y1 in tiles]


def test_all_four_tiles_share_one_grid():
    """``slice_all_tiles`` emits 4 samples per image; they must be the 4 tiles of ONE grid.

    Regression: ``slice_at`` recomputed the geometry -- and re-drew ``bias_jitter`` -- on every call, so k=0..3 landed
    on 4 different grids. Measured at the default jitter: 100% of images affected, seams up to ~3% of the image extent
    apart, and the two seam decisions contradicting each other.
    """
    t = _biased_slice()
    t.set_epoch(3)
    w, h, xyxy = 4000, 3000, _spread_boxes()

    grids = {tuple(_tile_rects(t._grid(w, h, xyxy, 17)[0])) for _ in range(4)}
    assert len(grids) == 1, "the 4 tiles of one image must be the 4 tiles of one grid"


def test_missing_key_keeps_the_per_call_seam():
    """``key=None`` stays backward compatible: a fresh seam per call (the historical bug shape: a fresh seam every call)."""
    import random

    t = _biased_slice()
    t.set_epoch(3)
    w, h, xyxy = 4000, 3000, _spread_boxes()

    random.seed(0)
    assert len({t._grid(w, h, xyxy, None)[1:] for _ in range(4)}) == 4


def test_grid_is_stable_within_an_epoch_and_changes_across_epochs():
    """The grid must be a pure function of (epoch, image) -- independent of the global RNG state.

    ``bias_jitter`` exists so the seams are not frozen across epochs; pinning them to the global stream instead made
    them depend on how many random draws happened to precede them.
    """
    import random

    t = _biased_slice()
    t.set_epoch(3)
    w, h, xyxy = 4000, 3000, _spread_boxes()

    random.seed(123)
    first = t._grid(w, h, xyxy, 17)[1:]
    random.seed(999)
    assert t._grid(w, h, xyxy, 17)[1:] == first, "the grid must not depend on the global RNG state"

    t.set_epoch(4)
    assert t._grid(w, h, xyxy, 17)[1:] != first, "per-epoch variation is the whole point of bias_jitter"
    t.set_epoch(3)
    assert t._grid(w, h, xyxy, 99)[1:] != first, "different images must still get different seams"


def test_biased_grid_union_covers_the_whole_image():
    """``slice_geometry`` promises every pixel is in >= 1 tile; that must survive the 4 emitted tiles."""
    t = _biased_slice()
    t.set_epoch(0)
    w, h, xyxy = 4000, 3000, _spread_boxes()

    for key in range(50):
        cov = np.zeros((h, w), np.uint8)
        for x0, y0, x1, y1 in _tile_rects(t._grid(w, h, xyxy, key)[0]):
            cov[y0:y1, x0:x1] = 1
        assert cov.all(), f"hole in the union of the 4 tiles for key={key}"


def test_slice_at_forwards_the_image_key_not_the_tile_key(monkeypatch):
    """``key`` must be the IMAGE index -- ``src`` is ``(img_index, k)`` and would defeat the whole point."""
    from ultralytics.data.augment import OnlineSlice

    t = _biased_slice()
    t.set_epoch(2)
    seen = []
    real_grid = OnlineSlice._grid

    def spy(self, w, h, xyxy=None, key=None):
        out = real_grid(self, w, h, xyxy, key)
        seen.append((key, round(out[1], 9), round(out[2], 9)))
        return out

    monkeypatch.setattr(OnlineSlice, "_grid", spy)
    img = np.zeros((3000, 4000, 3), np.uint8)
    label = {
        "bboxes": _spread_boxes(4000, 3000, n=6),
        "cls": np.zeros((6, 1), dtype=np.float32),
        "segments": [],
        "keypoints": None,
        "normalized": False,
        "bbox_format": "xyxy",
    }
    for k in range(4):
        t.slice_at(img, label, k, src=(17, k), key=17)

    assert [k for k, _, _ in seen] == [17] * 4, "src=(img_index, k) must NOT be used as the grid key"
    assert len({(bx, by) for _, bx, by in seen}) == 1, "4 tiles -> 1 grid"


def test_rebuild_epoch_masks_pushes_the_epoch_to_the_slice_transform():
    """The epoch must reach ``OnlineSlice`` from ``_rebuild_epoch_masks``, not just the main process.

    Workers rebuild through ``_sync_epoch_masks``; publishing the epoch anywhere else would leave them pinned at epoch
    0, i.e. one grid in the main process and a different one in every worker.
    """

    class _FakeSlice:
        def __init__(self):
            self.resets = 0
            self.epochs = []

        def reset_counters(self):
            self.resets += 1

        def set_epoch(self, epoch):
            self.epochs.append(int(epoch))

    ds = _make_dataset()
    ds._mask_seed = 12345
    fake = _FakeSlice()
    ds.slice_transform = fake

    ds._rebuild_epoch_masks(7, 100)
    ds._rebuild_epoch_masks(8, 100)
    assert fake.epochs == [7, 8], "every rebuild must publish its own epoch"
    assert fake.resets == 2


def test_real_dataset_feeds_the_image_index_and_epoch_to_the_grid(tmp_path, monkeypatch):
    """End-to-end guard for the base.py wiring: image index in, one grid out, current epoch set.

    The unit tests above prove the grid math; this one proves the dataset actually hands it the ingredients (image index
    + current epoch). Without it, ``key`` would stay ``None`` and the whole fix would silently do nothing while every
    unit test kept passing.
    """
    from ultralytics.cfg import get_cfg
    from ultralytics.data.augment import OnlineSlice
    from ultralytics.data.build import build_yolo_dataset

    img_dir, data = _tiny_tiled_dataset(tmp_path, 3, size=(96, 96))
    cfg = get_cfg(
        overrides={
            "data": str(tmp_path / "d.yaml"),
            "imgsz": 64,
            "task": "detect",
            "mode": "train",
            "slice_prob": 1.0,
            "slice_all_tiles": True,
            "slice_ratio": 1.0,
            "slice_center_bias": True,
            "slice_bias_jitter": 0.05,
            "slice_keep_origin": False,
            "ratio_pad_keep": False,
            "blur_keep": False,
            "compose_keep": False,
            "weather_keep": False,
            "occlusion_keep": False,
            "mosaic": 0.0,
            "workers": 0,
            "cache": False,
            "close_aug_epoch": 0,
        }
    )
    ds = build_yolo_dataset(cfg, str(img_dir), 3, data, mode="train")
    ds.set_epoch(5, epochs=50)

    seen = []
    real_grid = OnlineSlice._grid

    def spy(self, w, h, xyxy=None, key=None):
        out = real_grid(self, w, h, xyxy, key)
        seen.append((key, round(out[1], 9), round(out[2], 9), int(self._slice_epoch)))
        return out

    monkeypatch.setattr(OnlineSlice, "_grid", spy)
    assert len(ds) == 3 * 4  # emit_all only: 4 tiles per original, every other branch off
    for i in range(len(ds)):
        ds[i]

    assert len(seen) == 12, "every base-segment sample must go through the biased grid"
    assert {epoch for *_, epoch in seen} == {5}, "the grid must see the CURRENT epoch"
    per_image = {}
    for key, bx, by, _ in seen:
        per_image.setdefault(key, set()).add((bx, by))
    assert set(per_image) == {0, 1, 2}, f"the key must be the image index, got {sorted(per_image)}"
    assert all(len(v) == 1 for v in per_image.values()), "4 tiles per image must share exactly one grid"


# ---------------------------------------------------------------------------
# Project keys are read through _hyp_get, so default.yaml is always the fallback
# ---------------------------------------------------------------------------
_PROJECT_KEY_PREFIXES = ("slice_", "compose_", "ratio_pad_", "blur_", "weather_", "occlusion_", "mosaic_save_")
_PROJECT_EXTRA_KEYS = {"degrade_max_side", "close_aug_epoch"}


def _hyp_ns(strip_project_keys=False, **overrides):
    """A DEFAULT_CFG-derived hyp; ``strip_project_keys`` simulates an args.yaml from before the feature."""
    from ultralytics.cfg import DEFAULT_CFG_DICT
    from ultralytics.utils import IterableSimpleNamespace

    cfg = dict(DEFAULT_CFG_DICT)
    if strip_project_keys:
        cfg = {k: v for k, v in cfg.items() if not k.startswith(_PROJECT_KEY_PREFIXES) and k not in _PROJECT_EXTRA_KEYS}
    cfg.update(overrides)
    return IterableSimpleNamespace(**cfg)


def _stub_augment_dataset():
    """Only the attributes ``v8_transforms`` / ``Mosaic.__init__`` actually read off the dataset."""
    import types

    return types.SimpleNamespace(data={}, rect=False, use_obb=False, use_keypoints=False, cache=False)


def _mirror_onto_dataset(hyp):
    """Run ``v8_transforms`` and return only the attributes it published (the OnlineSlice object excluded)."""
    from ultralytics.data.augment import v8_transforms

    ds = _stub_augment_dataset()
    pre_existing = set(vars(ds))
    v8_transforms(ds, 64, hyp)
    return {k: v for k, v in vars(ds).items() if k != "slice_transform" and k not in pre_existing}


def test_hyp_without_any_project_key_still_builds_the_augmentation_pipeline():
    """Core: a hyp that predates every project key must not abort the augmentation build.

    ``getattr(hyp, "<key>")`` with no default *is* ``hyp.<key>`` -- Ruff's B009 says as much -- so all 60 of those reads
    raised ``AttributeError`` from deep inside the augmentation build for any hyp that was not freshly derived from the
    current ``DEFAULT_CFG``: a third-party namespace, or the ``train_args`` restored from an older ``args.yaml`` /
    checkpoint that predates the key. They now fall back to ``default.yaml``, so an old config must produce the very
    same dataset attributes.
    """
    assert _mirror_onto_dataset(_hyp_ns(strip_project_keys=True)) == _mirror_onto_dataset(_hyp_ns())


def test_mirrored_defaults_come_from_default_yaml():
    """With no overrides, every mirrored value must equal its ``default.yaml`` entry.

    Pairs with ``test_online_defaults_match_default_cfg`` (which guards ``base._ONLINE_DEFAULTS``): this one guards the
    ``v8_transforms`` side, so neither copy of the defaults can drift.
    """
    from ultralytics.cfg import DEFAULT_CFG_DICT

    mirror = _mirror_onto_dataset(_hyp_ns())
    checked = {k: v for k, v in mirror.items() if k in DEFAULT_CFG_DICT}
    assert len(checked) >= 40, f"the mirror shrank unexpectedly: {sorted(checked)}"
    wrong = {k: (v, DEFAULT_CFG_DICT[k]) for k, v in checked.items() if v != DEFAULT_CFG_DICT[k]}
    assert not wrong, f"v8_transforms published values that are not default.yaml's: {wrong}"


def test_hyp_get_resolution_order():
    """Attribute first, then an explicit default, then ``default.yaml``."""
    from ultralytics.cfg import DEFAULT_CFG_DICT
    from ultralytics.data.augment import _hyp_get
    from ultralytics.utils import IterableSimpleNamespace

    hyp = IterableSimpleNamespace(slice_prob=0.7)
    assert _hyp_get(hyp, "slice_prob") == 0.7, "the value on hyp must win"
    assert DEFAULT_CFG_DICT["slice_prob"] != 0.7, "guard the assertion above against a coincidental default"
    assert _hyp_get(hyp, "blur_short_len_min") == DEFAULT_CFG_DICT["blur_short_len_min"], "missing -> default.yaml"
    assert _hyp_get(hyp, "augmentations", []) == [], "an explicit default must beat the default.yaml lookup"
    assert _hyp_get(hyp, "slice_save_max_tile") is None, "an explicit None is a value, not a missing key"


def test_hyp_get_rejects_a_key_that_is_registered_nowhere():
    """A key in neither ``hyp`` nor ``default.yaml`` must fail loudly and name itself.

    Silently taking a built-in literal is what made the old spelling dangerous: a new config key read before it was
    registered went unnoticed until a user's hyp happened to lack it. Raising here makes the "register it in
    default.yaml too" convention self-enforcing instead of a comment.
    """
    from ultralytics.data.augment import _hyp_get
    from ultralytics.utils import IterableSimpleNamespace

    with pytest.raises(ValueError, match="definitely_not_a_hyperparameter"):
        _hyp_get(IterableSimpleNamespace(), "definitely_not_a_hyperparameter")


def _ast_function(module, name):
    """Return the AST node of the top-level function ``name`` in ``module``'s source file."""
    import ast

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def test_v8_transforms_has_no_bare_getattr_on_hyp():
    """Guard: ``getattr(hyp, "<key>")`` must not come back -- it silently loses the fallback.

    This deliberately duplicates Ruff B009 (the local loop has no lint step), and the mechanism is the point: the call
    still *works* whenever hyp happens to carry the key, so no behavioral test would notice the fallback disappearing
    until a user hit it.
    """
    import ast

    from ultralytics.data import augment

    offenders = [
        f"line {node.lineno}: getattr(hyp, {node.args[1].value!r})"
        for node in ast.walk(_ast_function(augment, "v8_transforms"))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "hyp"
        and isinstance(node.args[1], ast.Constant)
    ]
    assert not offenders, "read project keys through _hyp_get so default.yaml stays the fallback: " + "; ".join(
        offenders
    )


def test_every_project_key_read_by_v8_transforms_is_registered():
    """Guard: each key ``v8_transforms`` reads must be in ``default.yaml`` -- or carry a literal.

    A 2-arg ``_hyp_get`` call asserts "default.yaml has it", so an unregistered key there would raise at build time;
    checking it statically names the offender before a user does. A 3-arg call means the key is deliberately outside
    ``default.yaml`` (today only ``augmentations``), which is the one case where a literal default is allowed -- writing
    one for a registered key would be a second copy of a default that already lives in the config.
    """
    import ast

    from ultralytics.cfg import DEFAULT_CFG_DICT
    from ultralytics.data import augment

    unregistered, duplicated = [], []
    for node in ast.walk(_ast_function(augment, "v8_transforms")):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_hyp_get"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
        ):
            continue
        key, has_default = node.args[1].value, len(node.args) >= 3
        if not has_default and key not in DEFAULT_CFG_DICT:
            unregistered.append(f"line {node.lineno}: {key}")
        if has_default and key in DEFAULT_CFG_DICT:
            duplicated.append(f"line {node.lineno}: {key}")
    assert not unregistered, "these keys are read without a default but are not in default.yaml: " + "; ".join(
        unregistered
    )
    assert not duplicated, "these keys get a literal default although default.yaml already defines one: " + "; ".join(
        duplicated
    )


def test_old_args_yaml_can_still_construct_a_real_dataset(tmp_path):
    """End-to-end: the real user path -- restoring an older ``args.yaml`` -- must not crash.

    The unit test above proves the reads fall back; this one proves ``YOLODataset.__init__`` gets all the way through
    the augmentation build with such a hyp instead of dying in the middle of it.
    """
    from ultralytics.data.dataset import YOLODataset

    img_dir, data = _tiny_detect_dataset(tmp_path)
    ds = YOLODataset(
        img_path=str(img_dir),
        imgsz=64,
        cache=False,
        data=data,
        augment=True,
        hyp=_hyp_ns(strip_project_keys=True),
    )
    assert ds.transforms is not None
    assert getattr(ds, "blur_axis_aligned", None) is True, "the mirror must still run on a legacy hyp"


# ---------------------------------------------------------------------------
# The `self.ims` whole-image cache is bounded by a BYTE BUDGET and exposed as config
# ---------------------------------------------------------------------------
# Upstream's cap is `min(ni, batch*8, 1000) - 1`, which only tracks `batch`. One frame is
# `imgsz*imgsz*channels`, so the two axes multiply: at batch=64/imgsz=1280 that is 511 frames
# ~= 2.3 GiB per worker and at imgsz=1920 ~= 5.3 GiB, on top of the raw-image LRU. The shipped
# 1 GiB budget caps those at 218 and 97 frames (~1 GiB) while leaving `imgsz <= 640`/`batch <= 64`
# byte-for-byte unchanged.
def _ims_hyp(**overrides):
    """A DEFAULT_CFG-derived hyp with the ims-cache keys overridden."""
    return _hyp_ns(**overrides)


@pytest.mark.parametrize(
    "imgsz,channels,budget_mb,expected",
    [
        (640, 3, 256, 218),  # 256 MiB // (640*640*3 B)
        (1280, 3, 256, 54),
        (1920, 3, 256, 24),
        (640, 1, 256, 655),  # grayscale frames are 3x smaller -> 3x more frames
        (2048, 3, 1, 1),  # 1 MiB is less than a single 2048 frame -> the floor of 1, never zero
    ],
)
def test_ims_cap_for_budget_converts_bytes_to_frames(imgsz, channels, budget_mb, expected):
    """The budget is a byte budget, so the frame count must scale with 1/(imgsz^2 * channels)."""
    from ultralytics.data.base import _ims_cap_for_budget, _ims_frame_bytes

    assert _ims_cap_for_budget(budget_mb, _ims_frame_bytes(imgsz, channels)) == expected


def test_ims_cap_for_budget_reports_no_budget_as_zero():
    """`0` is the "no budget configured" sentinel -- and NOT a usable cap (see the trap test)."""
    from ultralytics.data.base import _ims_cap_for_budget

    for budget in (0, -1, -256):
        assert _ims_cap_for_budget(budget, 1228800) == 0


@pytest.mark.parametrize("frames_cfg", [1, 7, 16])
def test_ims_cache_frames_explicit_is_honoured_exactly(frames_cfg):
    """`>0` is an exact frame count, independent of ni/batch/imgsz/budget."""
    from ultralytics.data.base import _resolve_ims_cap

    hyp = _ims_hyp(ims_cache_frames=frames_cfg, ims_cache_mb=1)
    for ni, batch, imgsz in [(8520, 64, 1920), (300, 8, 640), (1, 1, 640)]:
        assert _resolve_ims_cap(hyp, ni, batch, imgsz, 3, True) == frames_cfg


def test_ims_cache_frames_explicit_may_exceed_upstream():
    """`>0` is the documented way to spend memory to buy back JPEG re-decodes."""
    from ultralytics.data.base import _legacy_ims_cap, _resolve_ims_cap

    assert _legacy_ims_cap(8520, 8) == 63
    assert _resolve_ims_cap(_ims_hyp(ims_cache_frames=999), 8520, 8, 640, 3, True) == 999


def test_ims_cache_frames_negative_reproduces_upstream_exactly():
    """`<0` is the A/B escape hatch: it must reproduce the upstream formula verbatim."""
    from ultralytics.data.base import _legacy_ims_cap, _resolve_ims_cap

    hyp = _ims_hyp(ims_cache_frames=-1)
    for ni, batch in [(8520, 8), (8520, 16), (8520, 64), (2000, 200), (300, 1)]:
        legacy = max(1, min(ni, batch * 8, 1000) - 1)
        assert _legacy_ims_cap(ni, batch) == legacy, "the helper must equal upstream's formula"
        assert _resolve_ims_cap(hyp, ni, batch, 1920, 3, True) == legacy


@pytest.mark.parametrize("imgsz", [320, 640, 1280, 1920])
@pytest.mark.parametrize("batch", [1, 8, 16, 64, 200])
def test_ims_auto_never_exceeds_upstream_so_the_default_cannot_regress_memory(imgsz, batch):
    """The default (`0` = auto) is `min(upstream, budget)`: it may only ever LOWER the footprint.

    That is what keeps the blast radius to the pathological configurations -- typical runs resolve to exactly the legacy
    upstream value (pinned by the end-to-end test below).
    """
    from ultralytics.data.base import _legacy_ims_cap, _resolve_ims_cap

    for ni in (300, 8520):
        assert _resolve_ims_cap(_ims_hyp(), ni, batch, imgsz, 3, True) <= _legacy_ims_cap(ni, batch)


@pytest.mark.parametrize("frames", [0, -1])
@pytest.mark.parametrize("budget", [0, -1, 1e-9, 1, 256, 1e9])
def test_ims_cap_is_never_zero_for_an_augmenting_dataset(frames, budget):
    """`_ims_cap <= 0` means "never evict" in `_remember_ims`, so zero is an unbounded LEAK.

    `_ims_cap_for_budget` legitimately returns 0 for "no budget"; that is exactly why the caller may not pass it
    straight through -- there is no safe zero while the dataset is augmenting.
    """
    from ultralytics.data.base import _resolve_ims_cap

    hyp = _ims_hyp(ims_cache_frames=frames, ims_cache_mb=budget)
    for ni, batch, imgsz in [(8520, 64, 1920), (8520, 64, 640), (8, 8, 64)]:
        assert _resolve_ims_cap(hyp, ni, batch, imgsz, 3, True) >= 1


def test_ims_cap_is_zero_only_when_the_dataset_does_not_augment():
    """`0` is correct there: the cache WRITE is behind `self.augment`, so nothing would fill it."""
    from ultralytics.data.base import _resolve_ims_cap

    for hyp in (_ims_hyp(), _ims_hyp(ims_cache_frames=32), _ims_hyp(ims_cache_frames=-1)):
        assert _resolve_ims_cap(hyp, 100, 16, 640, 3, False) == 0


def test_ims_cache_keys_are_registered_int_config():
    """§1 three-place rule: an unregistered key is silently not validated/coerced on the CLI.

    Registration is asserted by BEHAVIOR (a float must be rejected), not just by membership: that is the part users
    actually feel, and it is what breaks if someone drops the key from ``CFG_INT_KEYS``.
    """
    from ultralytics.cfg import CFG_INT_KEYS, DEFAULT_CFG_DICT, get_cfg

    for key in ("ims_cache_frames", "ims_cache_mb"):
        assert key in DEFAULT_CFG_DICT, f"{key} must live in default.yaml"
        assert key in CFG_INT_KEYS, f"{key} must be in CFG_INT_KEYS or '--{key}=8' is not validated"

    cfg = get_cfg(overrides={"ims_cache_frames": 8, "ims_cache_mb": 64})
    assert type(cfg.ims_cache_frames) is int and cfg.ims_cache_frames == 8
    assert type(cfg.ims_cache_mb) is int and cfg.ims_cache_mb == 64

    with pytest.raises(TypeError, match="ims_cache_mb"):
        get_cfg(overrides={"ims_cache_mb": 1.5})


@pytest.mark.parametrize("imgsz,batch", [(320, 8), (320, 64), (640, 8), (640, 16), (640, 64)])
def test_ims_auto_leaves_common_configs_at_the_upstream_value(imgsz, batch):
    """The shipped 1 GiB budget must be INVISIBLE for configs whose upstream cap already fits it.

    This is what makes the new default safe to land: it is not "a new policy for everyone", it is a ceiling that only
    the pathological axes (``imgsz`` >= 1280, ``batch`` >= 128) ever reach. Checked rather than argued, because a budget
    tweak that quietly shrinks the 640-column range would cost re-decodes in the most common configuration and no test
    would notice.
    """
    from ultralytics.data.base import _legacy_ims_cap, _resolve_ims_cap

    for ni in (2000, 8520):
        assert _resolve_ims_cap(_ims_hyp(), ni, batch, imgsz, 3, True) == _legacy_ims_cap(ni, batch)


@pytest.mark.parametrize("imgsz", [1280, 1920, 2560])
def test_ims_auto_caps_the_high_resolution_configs_within_budget(imgsz):
    """The pathological axis: batch=64 at 1280/1920 used to be 2.3/5.3 GiB per worker."""
    from ultralytics.data.base import _ims_frame_bytes, _legacy_ims_cap, _resolve_ims_cap

    legacy = _legacy_ims_cap(8520, 64)
    cap = _resolve_ims_cap(_ims_hyp(), 8520, 64, imgsz, 3, True)
    assert cap < legacy, "the budget must bind here -- that is the whole point"
    assert cap * _ims_frame_bytes(imgsz, 3) / (1 << 20) <= 1024, "and it must fit the documented budget"


@pytest.mark.parametrize(
    "imgsz,overrides,expected",
    [
        (64, {"ims_cache_frames": 6}, 6),  # explicit: read from hyp (the upstream cap would be 7)
        (64, {"ims_cache_frames": -1}, 7),  # upstream formula
        (640, {"ims_cache_mb": 8}, 6),  # auto, budget binds: min(7, 8 MiB / 1.17 MiB) = 6
        (640, {"ims_cache_mb": 256}, 7),  # auto, budget loose: exactly the legacy upstream value
    ],
)
def test_ims_cache_budget_reaches_the_dataset(tmp_path, imgsz, overrides, expected):
    """End-to-end: the knob must actually reach ``BaseDataset._ims_cap``.

    Same failure mode as ``slice_raw_cache_size`` (see its test): the value is read inside ``BaseDataset.__init__``, so
    reading it off ``self`` instead of ``hyp`` would leave the config registered, documented -- and completely inert.
    The two "auto" cases pin both directions (budget binding and not binding), which is what proves the `min(...)` is
    wired.
    """
    from ultralytics.cfg import get_cfg
    from ultralytics.data.base import _legacy_ims_cap
    from ultralytics.data.build import build_yolo_dataset

    img_dir, data = _tiny_tiled_dataset(tmp_path, 8)
    cfg = get_cfg(
        overrides={
            "data": str(tmp_path / "d.yaml"),
            "imgsz": imgsz,
            "task": "detect",
            "mode": "train",
            "workers": 0,
            "cache": False,
            "close_aug_epoch": 0,
            **overrides,
        }
    )
    ds = build_yolo_dataset(cfg, str(img_dir), 8, data, mode="train")
    assert ds.channels == 3  # precondition for the frame-size arithmetic
    assert _legacy_ims_cap(8, 8) == 7  # ni=8 -> upstream's cap is 7, so 6 and 4 are distinguishable
    assert ds._ims_cap == expected


def test_ims_cache_frames_bounds_the_cache_end_to_end(tmp_path, monkeypatch):
    """The resolved cap must be the one ``_remember_ims`` actually enforces (config -> eviction)."""
    from ultralytics.cfg import get_cfg
    from ultralytics.data import base as base_mod
    from ultralytics.data.build import build_yolo_dataset

    img_dir, data = _tiny_tiled_dataset(tmp_path, 8)
    cfg = get_cfg(
        overrides={
            "data": str(tmp_path / "d.yaml"),
            "imgsz": 64,
            "task": "detect",
            "mode": "train",
            "workers": 0,
            "cache": False,
            "close_aug_epoch": 0,
            "ims_cache_frames": 3,
        }
    )
    ds = build_yolo_dataset(cfg, str(img_dir), 8, data, mode="train")
    assert ds._ims_cap == 3
    assert getattr(ds, "slice_transform", None) is None, "the test needs load_image to own the ims write"

    monkeypatch.setattr(base_mod, "imread", lambda _f, **__: np.zeros((64, 64, 3), np.uint8))
    for i in list(range(ds.ni)) * 3:
        ds.load_image(i)
    assert len(ds._ims_keys) == 3, "the configured cap, not the 7 the mosaic buffer would allow"
    assert sum(im is not None for im in ds.ims) == 3


def test_ims_cache_budget_is_reported(tmp_path, monkeypatch):
    """The estimate must be logged: that line is how the documented memory guidance is verified."""
    from ultralytics.cfg import get_cfg
    from ultralytics.data import base as base_mod
    from ultralytics.data.build import build_yolo_dataset

    class _Recorder:
        def __init__(self):
            self.messages = []

        def info(self, msg, *_, **__):
            self.messages.append(str(msg))

        def warning(self, msg, *_, **__):
            self.messages.append(str(msg))

    rec = _Recorder()
    monkeypatch.setattr(base_mod, "LOGGER", rec)
    img_dir, data = _tiny_tiled_dataset(tmp_path, 8)
    cfg = get_cfg(
        overrides={
            "data": str(tmp_path / "d.yaml"),
            "imgsz": 640,  # makes the budget bind at this tiny ni: min(7, 8 MiB / 1.17 MiB) = 6
            "task": "detect",
            "mode": "train",
            "workers": 0,
            "cache": False,
            "close_aug_epoch": 0,
            "ims_cache_mb": 8,
        }
    )
    build_yolo_dataset(cfg, str(img_dir), 8, data, mode="train")

    hit = [m for m in rec.messages if "self.ims whole-image cache" in m]
    assert hit, "the resolved budget must be reported (nothing else makes the knob observable)"
    assert "6 frames" in hit[0], hit[0]
    assert "MiB/worker" in hit[0], hit[0]  # the per-worker total is what makes the line actionable
    assert "auto=min(upstream, budget)" in hit[0], hit[0]


def test_remember_ims_treats_a_zero_cap_as_no_eviction_at_all(monkeypatch):
    """Documents the trap that makes "0 frames" unsafe, i.e. why `_resolve_ims_cap` floors at 1."""
    from ultralytics.data import base as base_mod

    ds = _make_loader_dataset(n=4, cap=1, extended=True)
    ds._ims_cap = 0  # the value a naive "budget with nothing left in it" would hand over
    monkeypatch.setattr(base_mod, "imread", lambda _f, **__: np.zeros((32, 32, 3), np.uint8))
    for i in range(ds.ni):
        ds.load_image(i)
    assert sum(im is not None for im in ds.ims) == ds.ni, "a zero cap evicts nothing -- it is a leak"


def test_ims_eviction_does_not_change_the_returned_frame(monkeypatch):
    """Why this knob needs no A/B: a miss only costs a re-decode, never different pixels.

    Each source decodes to a value derived from its own file name, so a stale or mis-keyed entry would come back as a
    different number. ``_make_loader_dataset`` caps `self.ims` at 2, so the reads below genuinely evict each other.
    """
    from ultralytics.data import base as base_mod

    ds = _make_loader_dataset(n=6, cap=2, extended=True)

    def fake_imread(f, **__):  # `flags=` arrives as a keyword from load_image
        return np.full((32, 32, 3), int(Path(str(f)).stem.split("_")[-1]), np.uint8)

    monkeypatch.setattr(base_mod, "imread", fake_imread)
    first = [ds.load_image(i)[0].copy() for i in range(ds.ni)]
    assert len({int(im.flat[0]) for im in first}) == ds.ni, "the fixture must make frames distinguishable"

    for i in (0, 3, ds.ni - 1):  # every one of these was evicted by the reads above
        im, _, hw = ds.load_image(i)
        assert (im == i).all(), f"image {i} came back with the wrong content"
        assert hw == (64, 64)
