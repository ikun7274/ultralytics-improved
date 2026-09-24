# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""In-memory image-degradation primitives for the online-augmentation branches.

Split out of ``base.py`` so the numeric kernels (aspect-ratio padding, motion blur, weather
degradation, occlusion drawing, exact rectangle-union area) live in one focused module instead of
being interleaved with the dataset's index/mask/buffer bookkeeping. These are the in-memory ports of
the offline ``pytools/`` tools; all of them are pure functions over a NumPy image.

Every kernel is written to keep the transient allocation low: the online branches run inside
DataLoader workers, where an extra full-size float64 temporary per sample is the difference between
a smooth epoch and a stalled run.
"""

from __future__ import annotations

import math
import random

import cv2
import numpy as np

# --- Online aspect-ratio pad helpers (in-memory port of pytools/change_image_resolution_slice_dataset_auto_improved.py) ---
_RATIO_REL_TOL = 0.02
_RATIO_43 = 4.0 / 3.0
_RATIO_169 = 16.0 / 9.0
_RATIO_PAD_COLORS = {"black": (0, 0, 0), "gray": (114, 114, 114), "white": (255, 255, 255)}

# Legal weather / occlusion types (kept in sync with the dispatch in _apply_weather / _apply_occlusion
# and validated up-front against the config by `v8_transforms`).
_WEATHER_TYPES: frozenset[str] = frozenset({"rain", "haze", "noise"})
_OCCLUSION_TYPES: frozenset[str] = frozenset({"rect", "stripe"})


def _channels(img: np.ndarray) -> int:
    """Return the channel count of an image, treating 2-D arrays as single-channel."""
    return int(img.shape[2]) if img.ndim == 3 else 1


def _ratio_pad_params(w: int, h: int, target_ratio: str, auto: bool) -> tuple[int, int, int, int] | None:
    """Compute border-padding to reach a target aspect ratio (in-memory port of the offline tool).

    - ratio < target (portrait-ish): keep height, widen with left/right symmetric borders
    - ratio > target (landscape-ish): keep width, heighten with top/bottom symmetric borders
    - already close to the target: return None (no pad needed -> copy / use original)
    - auto: 4:3 <-> 16:9 bidirectional; any other ratio goes to its NEAREST of 4:3 or 16:9
    Returns (new_w, new_h, pad_left, pad_top) or None.
    """
    ratio = w / h
    if auto:
        if math.isclose(ratio, _RATIO_43, rel_tol=_RATIO_REL_TOL):
            target = _RATIO_169  # 4:3 -> 16:9
        elif math.isclose(ratio, _RATIO_169, rel_tol=_RATIO_REL_TOL):
            target = _RATIO_43  # 16:9 -> 4:3
        else:
            target = _RATIO_43 if abs(ratio - _RATIO_43) <= abs(ratio - _RATIO_169) else _RATIO_169
    else:
        target = _RATIO_43 if target_ratio == "4:3" else _RATIO_169
        if math.isclose(ratio, target, rel_tol=_RATIO_REL_TOL):
            return None
    if ratio < target:  # widen
        new_w = max(1, round(h * target))
        new_h = h
        pad_left = (new_w - w) // 2
        pad_top = 0
    else:  # heighten
        new_w = w
        new_h = max(1, round(w / target))
        pad_left = 0
        pad_top = (new_h - h) // 2
    return new_w, new_h, pad_left, pad_top


def _psf_size(length: float) -> int:
    """Odd PSF side length that holds a motion-blur segment of ``length`` pixels.

    Shared by the dense PSF path and the axis-aligned box path so the two cannot drift apart: when the segment lies
    exactly along an axis the rasterized line covers ``_psf_size(length)`` pixels at uniform weight, which is what makes
    the box filter in ``_apply_motion_blur`` an exact stand-in rather than an approximation.
    """
    # ceil (not int()) so the half-length from the center never gets truncated by the border: int() would
    # silently shorten the effective blur for fractional lengths (e.g. 9.5 -> size 9, only 8px of blur).
    # `| 1` forces an odd size so the center pixel is well defined. Matches the offline tool's `ceil|1`.
    return max(3, math.ceil(length) | 1)


def _motion_blur_kernel(length: float, angle: float) -> np.ndarray:
    """Build a line-segment PSF motion-blur kernel (in-memory port of the offline motion_blur tool).

    0 deg = horizontal-right; kernel is squared with an odd size (>=3); the segment is drawn anti-aliased and normalized
    to sum = 1 (keeps brightness unchanged after convolution).
    """
    rad = np.deg2rad(angle)
    size = _psf_size(length)
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    dx = np.cos(rad)
    dy = np.sin(rad)
    length_scaled = length / 2.0
    x1 = center - dx * length_scaled
    y1 = center - dy * length_scaled
    x2 = center + dx * length_scaled
    y2 = center + dy * length_scaled
    cv2.line(kernel, (round(x1), round(y1)), (round(x2), round(y2)), 1.0, thickness=1, lineType=cv2.LINE_AA)
    # Guard against a zero/degenerate kernel (e.g. blur_*_len_*: 0 -> a single point may not be rasterized).
    # Dividing by 0 would produce NaN pixels -> NaN loss; where an entry point still installs a blanket
    # `filterwarnings('ignore')` (detect.py / export.py -- see 代码审查报告-全面复审.md M-5) the
    # RuntimeWarning is hidden, so the failure would surface only as a silently ruined run. train.py / val.py
    # already narrow their filters, but this guard is unconditional on purpose: it must not depend on which
    # entry point called it.
    ksum = float(kernel.sum())
    if not ksum > 0:
        kernel[:] = 0.0
        kernel[size // 2, size // 2] = 1.0  # degenerate to an identity (no-op) kernel
    else:
        kernel /= ksum
    return kernel


def _crop_kernel(kernel: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Trim the all-zero border off a convolution kernel, returning ``(kernel, anchor)``.

    ``cv2.filter2D`` is handed a ``size x size`` PSF in which the rasterized line occupies one thin diagonal band, so
    most of the kernel is zeros it still walks over. Its cost was measured to depend on the kernel's DIMENSIONS (a cliff
    around 11 px: a 9x9 kernel runs in ~6 ms, a 13x13 one in ~45 ms at 1280x960) as well as on the non-zero tap count,
    so handing over the padded square is pure waste. Dropping the zero border is *bit-identical* -- a zero tap
    contributes nothing, and the explicit ``anchor`` keeps the kernel aligned on the original center -- and measured
    1.85x on the short tier / 1.19x on the long tier (1.35x over the configured length distribution, 600 draws). It also
    means a larger ``blur_short_len_max`` no longer falls off the cliff.

    The bbox is unioned with the center pixel so ``anchor`` always stays inside the kernel (OpenCV asserts
    ``anchor.inside(Rect(0, 0, ksize.width, ksize.height))``); that extra zero row/column costs nothing.
    """
    size = kernel.shape[0]
    c = size // 2
    nz = np.nonzero(kernel)
    y0, y1 = min(int(nz[0].min()), c), max(int(nz[0].max()), c)
    x0, x1 = min(int(nz[1].min()), c), max(int(nz[1].max()), c)
    cropped = np.ascontiguousarray(kernel[y0 : y1 + 1, x0 : x1 + 1])
    return cropped, (c - x0, c - y0)


def _apply_motion_blur(
    img: np.ndarray, length: float = 15.0, angle: float = 30.0, defocus_sigma: float = 0.0, axis_aligned: bool = False
) -> np.ndarray:
    """Apply motion blur (and optional defocus) to a BGR/grayscale image (in-memory port of the offline tool).

    ``axis_aligned=True`` restricts the smear to the image axes, where ``angle`` is expected to be 0 (horizontal) or 90
    (vertical). This is NOT a cheaper approximation of the general path -- when the segment lies exactly along an axis
    the anti-aliased line rasterizes to ``_psf_size(length)`` pixels of uniform weight 1/n, i.e. a plain box filter, so
    the output is bit-identical while ``cv2.blur`` runs it on an O(1) running-sum path whose cost is independent of the
    kernel length. Verified over 242 (length, axis) pairs -- every 0.25 px from 5 to 35, both axes, 3-channel 1280x960
    -- ``cv2.blur`` and the PSF's ``filter2D`` agree exactly (max|diff| = 0), at the production working resolution it
    runs ~1.3x faster on the short tier (7.0 -> 5.2 ms) and ~1.9x on the long tier (20.5 -> 10.5 ms, defocus included),
    1.82x over both tiers (28.1 -> 15.4 ms/image). The bare convolution gain is larger still (2.94x on the long tier)
    but the fixed-cost defocus pass that follows it dilutes it.

    Prefer it when the motion's image-plane projection really is axis-aligned (fixed camera mounting, e.g. along-track
    aerial/vehicle imagery): there the constraint is the more faithful model, and it also costs
    less. Note that it does narrow the smear direction from U[0, 180) to {0, 90}, which is a genuine change
    to the augmentation distribution -- hence the ``blur_axis_aligned`` config switch rather than a silent replacement.
    """
    if axis_aligned:
        # The kernel length comes from the same helper the dense PSF uses, so the two paths cannot drift;
        # `_psf_size` is odd by construction, so cv2's default centered anchor matches the PSF's center.
        n = _psf_size(length)
        blurred = cv2.blur(img, (1, n) if float(angle) % 180.0 >= 45.0 else (n, 1))
    else:
        kernel, anchor = _crop_kernel(_motion_blur_kernel(length, angle))
        blurred = cv2.filter2D(img, -1, kernel, anchor=anchor)
    if defocus_sigma > 0:
        ksize = int(6 * defocus_sigma) | 1  # odd kernel size
        blurred = cv2.GaussianBlur(blurred, (ksize, ksize), defocus_sigma)
    return blurred


def _apply_weather(
    img: np.ndarray,
    weather_type: str,
    rain_density: float = 0.15,
    rain_length: float = 15.0,
    haze_beta: float = 0.4,
    noise_std: float = 15.0,
) -> np.ndarray:
    """Apply one weather degradation (rain / haze / Gaussian noise) to a BGR image, in memory.

    Labels are UNCHANGED (degradation never moves targets). Intensities are sampled randomly per call so the model does
    not overfit to a single degradation level:
    - rain:  ~density*max(h,w) semi-transparent streaks at a fixed 20-degree slant; alpha 0.2-0.6.
    - haze:  atmospheric-scattering model I = J*t + A*(1-t), t = 1-beta, A = gray atmosphere (200).
    - noise: additive Gaussian noise (RGB independent), sensor/low-light simulation.
    `weather_type` is one of "rain" / "haze" / "noise" (caller picks randomly from `weather_types`).

    Every branch runs in a single pass over the image and keeps the transient allocation close to 1x the frame: the
    branches execute per sample inside DataLoader workers, so an extra full-size float64 temporary (274 MB for a
    4000x3000 frame) is directly a memory ceiling on `workers`.
    """
    if weather_type == "rain":
        h, w = img.shape[:2]
        n = max(1, int(rain_density * max(h, w)))
        ang = math.radians(20.0)
        dx, dy = math.sin(ang), math.cos(ang)
        # Vectorized streak end-points, drawn in two thickness batches with cv2.polylines. The old
        # per-streak cv2.line cost ~600 Python->C calls per sample at density 0.15 on a 4000px frame
        # (1-2 orders of magnitude slower than the vectorized haze/noise); the distribution of
        # count / position / length / thickness is unchanged.
        x0s = np.random.uniform(0.0, float(w), size=n)
        y0s = np.random.uniform(0.0, float(h), size=n)
        lens = rain_length * np.random.uniform(0.5, 1.0, size=n)
        x1s = np.clip(x0s - dx * lens, 0.0, float(w))
        y1s = np.clip(y0s - dy * lens, 0.0, float(h))
        pts = np.stack([np.stack([x0s, y0s], axis=1), np.stack([x1s, y1s], axis=1)], axis=1).astype(np.int32)
        thick2 = np.random.rand(n) < 0.5  # ~half thickness 2, half thickness 1 (same as random.choice((1, 2)))
        overlay = img.copy()
        for t in (1, 2):
            batch = pts[thick2] if t == 2 else pts[~thick2]
            if len(batch):
                cv2.polylines(
                    overlay, list(batch), isClosed=False, color=(205, 205, 225), thickness=t, lineType=cv2.LINE_AA
                )
        alpha = np.random.uniform(0.2, 0.6)  # single RNG family (np.random) with the rain lines
        return cv2.addWeighted(img, 1.0 - alpha, overlay, alpha, 0)
    if weather_type == "haze":
        # One cv2.convertScaleAbs pass: saturate_cast<uchar>(|img*alpha + beta|), i.e. exactly the old
        # clip(img*t + 200*(1-t), 0, 255). The previous expression chain
        # (`img.astype(f32) * t + np.full_like(img, 200, f32) * (1-t)`) materialized 3-4 full-size
        # float32 temporaries -- measured 549 MB peak on a 4000x3000 frame.
        t = max(0.0, min(1.0, 1.0 - haze_beta))
        return cv2.convertScaleAbs(img, alpha=t, beta=200.0 * (1.0 - t))
    # noise: cv2.randn draws the whole frame into ONE float32 buffer -- measured 137.8 -> 27.0 ms
    # (5.1x) on a 1280x960 BGR frame, with no row chunking and no full-frame float64 temporary (the
    # 824 MB spike this branch used to have). Peak is ~5x the frame (float32 accumulator + uint8
    # result, clipped in place).
    #
    # Two non-obvious details, both load-bearing:
    #  * cv2.randn must be handed a SINGLE-channel view. On a 3-channel matrix OpenCV scales the
    #    requested sigma and the noise comes out at sigma/sqrt(3) -- measured std 8.67 for the
    #    configured 15.0, i.e. a 42% weaker augmentation with no error anywhere. `out.reshape(h, -1)`
    #    is that view: it aliases the same memory, so the image keeps its shape.
    #    (np.random can't replace this: the legacy stream emits float64 only -- neither
    #    `standard_normal` nor `normal` accepts a `dtype`, in numpy 2.0 -- and a separate Generator
    #    would be slower (2.2x) *and* still a different stream.)
    #  * cv2 has its own RNG, which `np.random.seed` does not reach. Rather than seeding cv2 out of
    #    band (which would mean teaching init_seeds/seed_worker about it), ONE integer is pulled from
    #    the existing numpy stream and handed to ``setRNGSeed``. The stream therefore advances exactly
    #    as it did before (one draw per noise sample), every downstream decision stays in lockstep,
    #    and the branch is reproducible whenever numpy is.
    sigma = max(0.0, float(noise_std))
    cv2.setRNGSeed(int(np.random.randint(0, 2**31 - 1)))
    out = np.empty(img.shape, dtype=np.float32)
    if sigma > 0:
        cv2.randn(out.reshape(out.shape[0], -1), 0.0, sigma)
    else:
        out.fill(0.0)  # cv2.randn rejects a zero stddev; sigma=0 means "no noise"
    out += img
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)


def _union_area(rects: list[tuple[int, int, int, int]]) -> float:
    """Exact area of the union of axis-aligned integer rectangles (x-sweep + y-interval merge).

    Replaces the full-image bool mask (h*w bytes per occluded sample -- 4000x3000 ~ 12 MB) with an exact small
    computation: occluder counts are 1~3 per sample, so the sweep is O(k^2 log k). Integer pixel semantics match the old
    mask exactly: a rect [x0, x1) x [y0, y1) covers (x1-x0) * (y1-y0) pixels.
    """
    if not rects:
        return 0.0
    xs = sorted({x0 for x0, _, _, _ in rects} | {x1 for _, _, x1, _ in rects})
    total = 0.0
    for a, b in zip(xs, xs[1:]):
        if b <= a:
            continue
        ys = sorted((y0, y1) for x0, y0, x1, y1 in rects if x0 <= a and b <= x1)
        if not ys:
            continue
        cur0, cur1 = ys[0]
        span = 0.0
        for y0, y1 in ys:
            if y0 <= cur1:
                cur1 = max(cur1, y1)
            else:
                span += cur1 - cur0
                cur0, cur1 = y0, y1
        span += cur1 - cur0  # last merged interval
        total += span * (b - a)
    return total


def _apply_occlusion(
    img: np.ndarray,
    occlusion_type: str,
    blocks: int = 1,
    size_ratio: float = 0.1,
    color: str = "auto",
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Draw semantic occlusion blocks (rect / stripe) on a BGR image, in memory.

    Simulates real drone-view occluders -- tree crowns, shadows, power lines, cloud edges. Labels are NOT moved by the
    drawing itself (the caller decides max_cover-based removal); returns the occluded image plus the pixel ``(x0, y0,
    x1, y1)`` boxes of every block so the caller can compute per-target covered ratios.

    - ``rect``: random rectangle, side in [0.5, 1.0] * base (base = sqrt(size_ratio * area)).
    - ``stripe``: thin band (width ~2% of the short side, length 0.5-1.0 of the long side),
    near-horizontal or near-vertical (models power lines / branches / cloud edges).
    - ``color='auto'``: sample the image's dark-quartile mean so the block blends into the scene
    instead of being a stark black blob; ``black`` / ``gray`` are fixed alternatives.
    """
    h, w = img.shape[:2]
    out = img.copy()
    base = max(2.0, math.sqrt(max(1.0, size_ratio) * h * w))
    # auto color: mean of the darkest ~25% pixels (per-channel), a plausible tree/shadow tone
    if color == "auto":
        # The old implementation reshaped the whole image (12 MP) and ran a full np.quantile sort --
        # 1-2 s of pure waste per sample on large frames. A large prime stride (37, coprime with
        # typical row widths) samples the whole frame at ~320k pixels, which is plenty to estimate a
        # per-channel 25th percentile.
        flat = img.reshape(-1, _channels(img))[::37]
        q = np.quantile(flat, 0.25, axis=0)
        oc_color = tuple(round(float(v)) for v in q)
    elif color == "black":
        oc_color = (0, 0, 0)
    else:  # gray
        oc_color = (128, 128, 128)
    boxes = []
    for _ in range(max(1, int(blocks))):
        if occlusion_type == "stripe":
            thick = max(2, int(0.02 * min(h, w)))
            length = int(max(h, w) * random.uniform(0.5, 1.0))
            vertical = random.random() < 0.5
            if vertical:
                cx = random.uniform(0.0, float(w))
                cy = random.uniform(0.0, float(h))
                x0, x1 = max(0, int(cx - thick // 2)), min(w, int(cx + thick // 2) + 1)
                y0, y1 = max(0, int(cy - length // 2)), min(h, int(cy + length // 2) + 1)
                cv2.rectangle(out, (x0, y0), (x1, y1), oc_color, -1)
                boxes.append((x0, y0, x1, y1))
            else:
                cx = random.uniform(0.0, float(w))
                cy = random.uniform(0.0, float(h))
                x0, x1 = max(0, int(cx - length // 2)), min(w, int(cx + length // 2) + 1)
                y0, y1 = max(0, int(cy - thick // 2)), min(h, int(cy + thick // 2) + 1)
                cv2.rectangle(out, (x0, y0), (x1, y1), oc_color, -1)
                boxes.append((x0, y0, x1, y1))
        else:  # rect
            side = base * random.uniform(0.5, 1.0)
            bw, bh = int(side * random.uniform(0.7, 1.3)), int(side * random.uniform(0.7, 1.3))
            x0 = random.uniform(0.0, max(1.0, float(w - bw)))
            y0 = random.uniform(0.0, max(1.0, float(h - bh)))
            x0i, y0i = int(x0), int(y0)
            x1i, y1i = min(w, x0i + bw), min(h, y0i + bh)
            cv2.rectangle(out, (x0i, y0i), (x1i, y1i), oc_color, -1)
            boxes.append((x0i, y0i, x1i, y1i))
    return np.ascontiguousarray(out), boxes
