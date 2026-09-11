# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import glob
import math
import os
import random
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from torch.utils.data import Dataset

from ultralytics.data.utils import FORMATS_HELP_MSG, HELP_URL, IMG_FORMATS, check_file_speeds, get_split_fraction
from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, NUM_THREADS, TQDM
from ultralytics.utils.patches import imread, imwrite


# --- Online aspect-ratio pad helpers (in-memory port of pytools/change_image_resolution_slice_dataset_auto_improved.py) ---
_RATIO_REL_TOL = 0.02
_RATIO_43 = 4.0 / 3.0
_RATIO_169 = 16.0 / 9.0
_RATIO_PAD_COLORS = {"black": (0, 0, 0), "gray": (114, 114, 114), "white": (255, 255, 255)}


def _ratio_pad_params(w: int, h: int, target_ratio: str, auto: bool):
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
        new_w = max(1, int(round(h * target)))
        new_h = h
        pad_left = (new_w - w) // 2
        pad_top = 0
    else:  # heighten
        new_w = w
        new_h = max(1, int(round(w / target)))
        pad_left = 0
        pad_top = (new_h - h) // 2
    return new_w, new_h, pad_left, pad_top


def _motion_blur_kernel(length: float, angle: float) -> np.ndarray:
    """Build a line-segment PSF motion-blur kernel (in-memory port of the offline motion_blur tool).

    0 deg = horizontal-right; kernel is squared with an odd size (>=3); the segment is drawn anti-aliased
    and normalized to sum = 1 (keeps brightness unchanged after convolution).
    """
    rad = np.deg2rad(angle)
    # ceil (not int()) so the half-length from the center never gets truncated by the border: int() would
    # silently shorten the effective blur for fractional lengths (e.g. 9.5 -> size 9, only 8px of blur).
    # `| 1` forces an odd size so the center pixel is well defined. Matches the offline tool's `ceil|1`.
    size = max(3, math.ceil(length) | 1)
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    dx = np.cos(rad)
    dy = np.sin(rad)
    length_scaled = length / 2.0
    x1 = center - dx * length_scaled
    y1 = center - dy * length_scaled
    x2 = center + dx * length_scaled
    y2 = center + dy * length_scaled
    cv2.line(kernel, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))),
             1.0, thickness=1, lineType=cv2.LINE_AA)
    # Guard against a zero/degenerate kernel (e.g. blur_*_len_*: 0 -> a single point may not be rasterized).
    # Dividing by 0 would produce NaN pixels -> NaN loss, and train.py's blanket `filterwarnings('ignore')`
    # hides the RuntimeWarning, so the failure would surface only as a silently ruined run.
    ksum = float(kernel.sum())
    if not ksum > 0:
        kernel[:] = 0.0
        kernel[size // 2, size // 2] = 1.0  # degenerate to an identity (no-op) kernel
    else:
        kernel /= ksum
    return kernel


def _apply_motion_blur(img: np.ndarray, length: float = 15.0, angle: float = 30.0,
                       defocus_sigma: float = 0.0) -> np.ndarray:
    """Apply motion blur (and optional defocus) to a BGR/grayscale image (in-memory port of the offline tool)."""
    kernel = _motion_blur_kernel(length, angle)
    blurred = cv2.filter2D(img, -1, kernel)
    if defocus_sigma > 0:
        ksize = int(6 * defocus_sigma) | 1  # odd kernel size
        blurred = cv2.GaussianBlur(blurred, (ksize, ksize), defocus_sigma)
    return blurred


# Directories already created in this process (see _ensure_dir).
_MKDIR_DONE: set[str] = set()


def _ensure_dir(path) -> None:
    """Create ``path`` (with parents) once per process; later calls are a no-op.

    The save branches used to call ``mkdir(parents=True, exist_ok=True)`` before *every* write, i.e. once
    per sample per worker -- a pointless syscall storm on spinning disks, and a lock convoy on network
    storage when 8 workers race on the same directory.
    """
    key = str(path)
    if key in _MKDIR_DONE:
        return
    Path(path).mkdir(parents=True, exist_ok=True)
    _MKDIR_DONE.add(key)


def _imwrite(path, img) -> bool:
    """Write an image using Ultralytics' unicode-safe ``imwrite`` and warn when it fails.

    OpenCV's own ``cv2.imwrite`` silently fails on non-ASCII paths -- verified in this very project, where
    it *returned True* while the file never appeared on disk. Never ignore the return value.
    """
    ok = bool(imwrite(str(path), img))
    if not ok:
        LOGGER.warning(f"{img.shape} save failed: imwrite returned False for '{path}' (check path/permissions).")
    return ok


def _save_cap(dataset, branch: str) -> int:
    """Return the per-branch save cap (P2-3 decoupling).

    Each branch (tile/ratio/blur/compose) used to share the legacy ``slice_save_max`` budget, so enabling
    compose_save would silently cap tile/ratio/blur to the same total and vice versa. The new params
    ``slice_save_max_{tile,ratio,blur,compose}`` -- if set -- override the global cap per branch. A None
    or missing attribute falls back to ``slice_save_max``. ``0`` means unlimited in either case.

    Args:
        dataset: BaseDataset instance (read attribute from ``self``).
        branch (str): One of "tile", "ratio", "blur", "compose" -- picks the override attribute name.
    """
    attr = f"slice_save_max_{branch}"
    val = getattr(dataset, attr, None)
    if val is None:
        val = getattr(dataset, "slice_save_max", 0) or 0
    return int(val)



class BaseDataset(Dataset):
    """Base dataset class for loading and processing image data.

    This class provides core functionality for loading images, caching, and preparing data for training and inference in
    object detection tasks.

    Attributes:
        img_path (str | list[str]): Path to the folder containing images.
        imgsz (int): Target image size for resizing.
        augment (bool): Whether to apply data augmentation.
        single_cls (bool): Whether to treat all objects as a single class.
        prefix (str): Prefix to print in log messages.
        fraction (float | int): Dataset ratio or image count to use.
        channels (int): Number of channels in the images (1 for grayscale, 3 for color). Color images loaded with OpenCV
            are in BGR channel order.
        cv2_flag (int): OpenCV flag for reading images.
        im_files (list[str]): List of image file paths.
        labels (list[dict]): List of label data dictionaries.
        ni (int): Number of images in the dataset.
        rect (bool): Whether to use rectangular training.
        batch_size (int): Size of batches.
        stride (int): Stride used in the model.
        pad (float): Padding value.
        buffer (list): Buffer for mosaic images.
        max_buffer_length (int): Maximum buffer size.
        ims (list): List of loaded images.
        im_hw0 (list): List of original image dimensions (h, w).
        im_hw (list): List of resized image dimensions (h, w).
        npy_files (list[Path]): List of numpy file paths.
        cache (str | None): Cache setting ('ram', 'disk', or None for no caching).
        transforms (callable): Image transformation function.
        batch_shapes (np.ndarray): Batch shapes for rectangular training.
        batch (np.ndarray): Batch index of each image.

    Methods:
        get_img_files: Read image files from the specified path.
        update_labels: Update labels to include only specified classes.
        load_image: Load an image from the dataset.
        cache_images: Cache images to memory or disk.
        cache_images_to_disk: Save an image as an *.npy file for faster loading.
        check_cache_disk: Check image caching requirements vs available disk space.
        check_cache_ram: Check image caching requirements vs available memory.
        set_rectangle: Sort images by aspect ratio and set batch shapes for rectangular training.
        get_image_and_label: Get and return label information from the dataset.
        update_labels_info: Custom label format method to be implemented by subclasses.
        build_transforms: Build transformation pipeline to be implemented by subclasses.
        get_labels: Get labels method to be implemented by subclasses.
    """

    class _ImageCache:
        """Store images in one contiguous array to preserve copy-on-write sharing between workers."""

        def __init__(self, images: list[np.ndarray]):
            """Pack images and their layouts into contiguous NumPy arrays."""
            self.shapes = np.array([im.shape for im in images])
            self.dtypes = np.array([im.dtype.str for im in images])
            self.offsets = np.concatenate(([0], np.cumsum([im.nbytes for im in images])))
            self.buffer = np.empty(self.offsets[-1], dtype=np.uint8)
            for i, im in enumerate(images):
                self.buffer[self.offsets[i] : self.offsets[i + 1]] = im.reshape(-1).view(np.uint8)
                images[i] = None

        def __getitem__(self, i: int) -> np.ndarray:
            """Return an image view by index."""
            i = range(len(self.shapes))[i]
            return self.buffer[self.offsets[i] : self.offsets[i + 1]].view(self.dtypes[i]).reshape(self.shapes[i])

    def __init__(
        self,
        img_path: str | list[str],
        imgsz: int = 640,
        cache: bool | str = False,
        augment: bool = True,
        hyp: dict[str, Any] = DEFAULT_CFG,
        prefix: str = "",
        rect: bool = False,
        batch_size: int = 16,
        stride: int = 32,
        pad: float = 0.5,
        single_cls: bool = False,
        classes: list[int] | None = None,
        fraction: float = 1.0,
        channels: int = 3,
    ):
        """Initialize BaseDataset with given configuration and options.

        Args:
            img_path (str | list[str]): Path to the folder containing images or list of image paths.
            imgsz (int): Image size for resizing.
            cache (bool | str): Cache images to RAM or disk during training.
            augment (bool): If True, data augmentation is applied.
            hyp (dict[str, Any]): Hyperparameters to apply data augmentation.
            prefix (str): Prefix to print in log messages.
            rect (bool): If True, rectangular training is used.
            batch_size (int): Size of batches.
            stride (int): Stride used in the model.
            pad (float): Padding value.
            single_cls (bool): If True, single class training is used.
            classes (list[int], optional): List of included classes.
            fraction (float | int): Dataset ratio or image count to use.
            channels (int): Number of channels in the images (1 for grayscale, 3 for color). Color images loaded with
                OpenCV are in BGR channel order.
        """
        super().__init__()
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.fraction = get_split_fraction(fraction, "train")
        self.channels = channels
        self.cv2_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
        self.im_files = self.get_img_files(self.img_path)
        self.labels = self.get_labels()
        self.update_labels(include_class=classes)  # single_cls and include_class
        self.ni = len(self.labels)  # number of images
        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()

        # Buffer thread for mosaic images
        self.buffer = []  # buffer size = batch size
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0

        # P0-4: per-worker LRU cache of ORIGINAL-resolution images. With n_per=8 one source image is
        # otherwise decoded 8 times per epoch (4 tiles + 1 origin + 1 ratio + 2 blur), plus 4 more for
        # compose -- measured at 9.0x on the reference dataset. Sub-samples of the same image live at
        # consecutive indices, so a tiny cache absorbs nearly all of that once the sampler keeps them
        # adjacent (see GroupedRandomSampler in data/build.py). Keyed by original image index.
        self._raw_cache = {}
        self._raw_cache_size = int(getattr(self, "slice_raw_cache_size", 2) or 0)
        # Per-epoch slice mask (slice_ratio exact ratio, original-level). None = pure slicing or
        # slicing off. Rebuilt by set_epoch(epoch) at every epoch start; see get_image_and_label.
        self._slice_mask = None
        # ratio_pad_ratio / blur_ratio / compose_ratio: per-epoch exact-ratio masks (rebuilt in
        # set_epoch; None = all samples augmented / branch off). See set_epoch for semantics.
        self._ratio_mask = None
        self._blur_mask = None
        self._compose_mask = None

        # Cache images (options are cache = True, False, None, "ram", "disk")
        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        # .npy 磁盘缓存已移除(实测无收益): npy_files 仅保留原生 cache='disk' 的默认位置(与 jpg 同目录)
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
        self.cache = cache.lower() if isinstance(cache, str) else "ram" if cache is True else None
        # P0-3: the online branches (slice / blur / ratio / compose) need the ORIGINAL-resolution image, but
        # cache='ram' stores the image ALREADY resized to imgsz, so it can never serve them. It would just
        # eat tens of GB and -- via Mosaic.buffer_enabled = (cache != "ram") -- silently switch mosaic from
        # buffer sampling to uniform sampling. Degrade loudly instead of pretending to help.
        if self.cache == "ram" and float(getattr(hyp, "slice_prob", 0.0) or 0.0) > 0.0:
            LOGGER.warning(
                f"{self.prefix}cache='ram' cannot accelerate online slicing: the online branches read images at "
                f"their original resolution, while RAM cache stores imgsz-resized copies. Degrading to "
                f"cache=False to avoid wasting memory. To speed up decoding raise slice_raw_cache_size "
                f"(per-worker memory LRU)."
            )
            self.cache = None
        if self.cache == "ram" and self.check_cache_ram():
            if hyp.deterministic:
                LOGGER.warning(
                    "cache='ram' may produce non-deterministic training results. "
                    "Consider cache='disk' as a deterministic alternative if your disk space allows."
                )
            self.cache_images()
        elif self.cache == "disk" and self.check_cache_disk():
            self.cache_images()

        # Transforms
        self.transforms = self.build_transforms(hyp=hyp)

        # Report the actual number of training samples after online augmentation expansion (train only),
        # so the user sees e.g. 140 samples from 28 images (4 slices + 1 origin + 1 ratio + 2 blur +
        # N/4 compose) before training starts. Every branch is reported independently: slicing may be
        # off while compose/ratio/blur are on, or vice versa.
        if self.augment:
            _n_total = len(self)
            _n_origin = self.ni
            _parts = []
            if getattr(self, "slice_transform", None) is not None:
                _parts.append("4 slices" if bool(getattr(self, "slice_all_tiles", False)) else "random 1 slice")
            if self._keep_origin_on():
                _parts.append("1 origin")
            if bool(getattr(self, "ratio_pad_keep", False)):
                _parts.append("1 ratio")
            if bool(getattr(self, "blur_keep", False)):
                _parts.append("2 blur")
            if self._compose_on():
                _parts.append("N/4 compose")
            if _parts:
                LOGGER.info(
                    f"{self.prefix}Online augment: {_n_total} training samples from {_n_origin} images "
                    f"({' + '.join(_parts)} per image)"
                )

    def get_img_files(self, img_path: str | list[str]) -> list[str]:
        """Read image files from the specified path.

        Args:
            img_path (str | list[str]): Path or list of paths to image directories or files.

        Returns:
            (list[str]): List of image file paths.

        Raises:
            FileNotFoundError: If no images are found or the path doesn't exist.
        """
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(Path(glob.escape(p)) / "**" / "*.*"), recursive=True)
                    # F = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p, encoding="utf-8") as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent, 1) if x.startswith("./") else x for x in t]  # local to global
                        # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global (pathlib)
                else:
                    raise FileNotFoundError(f"{self.prefix}{p} does not exist")
            im_files = sorted(x.replace("/", os.sep) for x in f if x.rpartition(".")[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert im_files, f"{self.prefix}No images found in {img_path}. {FORMATS_HELP_MSG}"
        except Exception as e:
            raise FileNotFoundError(f"{self.prefix}Error loading data from {img_path}\n{HELP_URL}") from e
        count = self.fraction if isinstance(self.fraction, int) else round(len(im_files) * self.fraction)
        im_files = im_files[:count] if count < len(im_files) else im_files
        check_file_speeds(im_files, prefix=self.prefix)  # check image read speeds
        return im_files

    def update_labels(self, include_class: list[int] | None) -> None:
        """Update labels to include only specified classes.

        Args:
            include_class (list[int], optional): List of classes to include. If None, all classes are included.
        """
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]["cls"]
                bboxes = self.labels[i]["bboxes"]
                segments = self.labels[i]["segments"]
                keypoints = self.labels[i].get("keypoints")
                j = (cls == include_class_array).any(1)
                self.labels[i]["cls"] = cls[j]
                self.labels[i]["bboxes"] = bboxes[j]
                if segments:
                    self.labels[i]["segments"] = [segments[si] for si, idx in enumerate(j) if idx]
                if keypoints is not None:
                    self.labels[i]["keypoints"] = keypoints[j]
            if self.single_cls:
                self.labels[i]["cls"][:] = 0

    def load_image(
        self, i: int, rect_mode: bool = True, resize_short: bool = False
    ) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        """Load an image from dataset index 'i'.

        Args:
            i (int): Index of the image to load.
            rect_mode (bool): Whether to use rectangular resizing (long side to imgsz).
            resize_short (bool): Whether to resize the shorter side to imgsz while maintaining aspect ratio. Overrides
                rect_mode when True.

        Returns:
            im (np.ndarray): Loaded image as a NumPy array.
            hw_original (tuple[int, int]): Original image dimensions in (height, width) format.
            hw_resized (tuple[int, int]): Resized image dimensions in (height, width) format.

        Raises:
            FileNotFoundError: If the image file is not found.
        """
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                try:
                    im = np.load(fn)
                    npy_channels = im.shape[-1] if im.ndim >= 3 else 1
                    if npy_channels != self.channels:
                        LOGGER.warning(
                            f"{self.prefix}Removing stale *.npy image file {fn} with {npy_channels} channels, expected {self.channels}"
                        )
                        Path(fn).unlink(missing_ok=True)
                        im = imread(f, flags=self.cv2_flag)
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}Removing corrupt *.npy image file {fn} due to: {e}")
                    Path(fn).unlink(missing_ok=True)
                    im = imread(f, flags=self.cv2_flag)  # BGR
            else:  # read image
                im = imread(f, flags=self.cv2_flag)  # BGR
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")

            h0, w0 = im.shape[:2]  # orig hw
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                if resize_short:  # resize short side to imgsz while maintaining aspect ratio
                    r = self.imgsz / min(h0, w0)  # ratio
                    if r != 1:  # if sizes are not equal
                        w, h = (math.ceil(w0 * r), self.imgsz) if h0 < w0 else (self.imgsz, math.ceil(h0 * r))
                        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
                else:
                    r = self.imgsz / max(h0, w0)  # ratio
                    if r != 1:  # if sizes are not equal
                        w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
            if im.ndim == 2:
                im = im[..., None]

            # Add to buffer if training with augmentations
            if self.augment and self.cache != "ram":
                if getattr(self, "slice_transform", None) is None:
                    # Without slicing, load_image's index is the dataset index, so buffer + ims cache are managed here.
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
                    if not self._extended_pool_on():
                        # Pure-ultralytics mode (no slicing, no project extension): the buffer can only
                        # contain original-image indices, so self-managed append/pop/clear is safe.
                        self.buffer.append(i)
                        if 1 < len(self.buffer) >= self.max_buffer_length:  # prevent empty buffer
                            j = self.buffer.pop(0)
                            if self.cache != "ram":
                                self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None
                    # Extended pool on: the buffer is bookkept centrally by get_image_and_label with
                    # EXPANDED indices; load_image must not pop/clear ims with them (j >= n would
                    # overflow ims) -- e.g. slicing off + compose/blur/ratio/keep_origin on. The ims
                    # cache above stays safe (i is an original index) and avoids re-reading disk on
                    # the full-image path.
                # With slicing: do NOT cache in self.ims here. get_image_and_label centrally manages the buffer
                # with expanded indices; caching origin-indexed images here would leak memory (never released).

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def cache_images(self) -> None:
        """Cache images to memory or disk for faster training."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        fcn, storage = (self.cache_images_to_disk, "Disk") if self.cache == "disk" else (self.load_image, "RAM")
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if self.cache == "disk":
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims[i].nbytes
                pbar.desc = f"{self.prefix}Caching images ({b / gb:.1f}GB {storage})"
            pbar.close()
        if self.cache == "ram":
            self.ims = self._ImageCache(self.ims)

    def cache_images_to_disk(self, i: int) -> None:
        """Save an image as an *.npy file for faster loading."""
        f = self.npy_files[i]
        if not f.exists():
            try:
                np.save(f.as_posix(), imread(self.im_files[i], flags=self.cv2_flag), allow_pickle=False)
            except Exception as e:
                f.unlink(missing_ok=True)
                LOGGER.warning(f"{self.prefix}WARNING ⚠️ Failed to cache image {f}: {e}")

    def check_cache_disk(self, safety_margin: float = 0.5) -> bool:
        """Check if there's enough disk space for caching images.

        Args:
            safety_margin (float): Safety margin factor for disk space calculation.

        Returns:
            (bool): True if there's enough disk space, False otherwise.
        """
        import shutil

        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im_file = random.choice(self.im_files)
            im = imread(im_file)
            if im is None:
                continue
            b += im.nbytes
            if not os.access(Path(im_file).parent, os.W_OK):
                self.cache = None
                LOGGER.warning(f"{self.prefix}Skipping caching images to disk, directory not writable")
                return False
        disk_required = b * self.ni / n * (1 + safety_margin)  # bytes required to cache dataset to disk
        _check_path = Path(self.im_files[0]).parent
        total, _used, free = shutil.disk_usage(_check_path)
        if disk_required > free:
            self.cache = None
            LOGGER.warning(
                f"{self.prefix}{disk_required / gb:.1f}GB disk space required, "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{free / gb:.1f}/{total / gb:.1f}GB free, not caching images to disk"
            )
            return False
        return True

    def check_cache_ram(self, safety_margin: float = 1.0) -> bool:
        """Check if there's enough RAM for caching images.

        Args:
            safety_margin (float): Safety margin factor for RAM calculation.

        Returns:
            (bool): True if there's enough RAM, False otherwise.
        """
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            b += self.load_image(random.randrange(self.ni))[0].nbytes
        mem_required = b * self.ni / n * (1 + safety_margin)  # GB required to cache dataset into RAM
        mem = __import__("psutil").virtual_memory()
        if mem_required > mem.available:
            self.cache = None
            LOGGER.warning(
                f"{self.prefix}{mem_required / gb:.1f}GB RAM required to cache images "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, not caching images"
            )
            return False
        return True

    def set_rectangle(self) -> None:
        """Sort images by aspect ratio and set batch shapes for rectangular training."""
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches

        s = np.array([x.pop("shape") for x in self.labels])  # hw
        ar = s[:, 0] / s[:, 1]  # aspect ratio
        irect = ar.argsort()
        self.im_files = [self.im_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi  # batch index of image

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return transformed label information for given index."""
        return self.transforms(self.get_image_and_label(index))

    def set_epoch(self, epoch: int = 0) -> None:
        """Rebuild per-epoch masks for exact-ratio ORIGINAL-level augmentation selection.

        Exactly ``round(x * N)`` ORIGINAL images are randomly chosen this epoch for each
        image-level branch (slice_ratio / ratio_pad_ratio / blur_ratio; blur 短+长 同命运),
        and ``round(x * ceil(N/4))`` 4-image GROUPS for compose_ratio. Un-selected positions keep
        their segment slot but fall back to the ORIGINAL full image (len stays constant; same
        principle as the mode-A background fallback). Resampled every epoch by the trainer
        (``trainer.py`` calls ``dataset.set_epoch(epoch)`` at each epoch start).

        Mask = None (all augmented / branch off) when the branch is off or its ratio >= 1.
        """
        # P3 fix: reset OnlineSlice's positive/background counters every epoch so the neg_ratio
        # background quota restarts per epoch instead of accumulating monotonically across epochs
        # (later epochs would otherwise keep ever fewer background tiles). Like the slice mask,
        # the reset propagates to DataLoader workers via fork inheritance (same architectural
        # assumption as the mask rebuild).
        st = getattr(self, "slice_transform", None)
        if st is not None and hasattr(st, "reset_counters"):
            st.reset_counters()
        n = len(self.labels)
        aug_on = bool(self.augment)
        # --- slice (original-level) ---
        x = float(getattr(self, "slice_ratio", 1.0))
        if aug_on and getattr(self, "slice_transform", None) is not None and 0.0 <= x < 1.0:
            mask = np.zeros(n, dtype=bool)
            if x > 0:
                mask[random.sample(range(n), int(round(x * n)))] = True
            self._slice_mask = mask
        else:
            self._slice_mask = None
        # --- ratio_pad (original-level) ---
        x = float(getattr(self, "ratio_pad_ratio", 1.0))
        if aug_on and bool(getattr(self, "ratio_pad_keep", False)) and 0.0 <= x < 1.0:
            mask = np.zeros(n, dtype=bool)
            if x > 0:
                mask[random.sample(range(n), int(round(x * n)))] = True
            self._ratio_mask = mask
        else:
            self._ratio_mask = None
        # --- blur (original-level, short+long same fate) ---
        x = float(getattr(self, "blur_ratio", 1.0))
        if aug_on and bool(getattr(self, "blur_keep", False)) and 0.0 <= x < 1.0:
            mask = np.zeros(n, dtype=bool)
            if x > 0:
                mask[random.sample(range(n), int(round(x * n)))] = True
            self._blur_mask = mask
        else:
            self._blur_mask = None
        # --- compose (group-level) ---
        x = float(getattr(self, "compose_ratio", 1.0))
        if aug_on and self._compose_on() and 0.0 <= x < 1.0:
            n_groups = (n + 3) // 4
            mask = np.zeros(n_groups, dtype=bool)
            if x > 0:
                mask[random.sample(range(n_groups), int(round(x * n_groups)))] = True
            self._compose_mask = mask
        else:
            self._compose_mask = None

    def _n_per(self) -> int:
        """Single source of truth: how many samples each ORIGINAL image expands to in the BASE segment.

        The base segment contains ONLY the slicing pipeline output (slice_prob / slice_all_tiles);
        the un-sliced origin copy is now an INDEPENDENT segment (see _segment_bases), laid out after
        the base segment just like ratio / blur / compose:
            slicing off       -> 1  (no expansion, plain originals)
            slice_all_tiles   -> 4  (the 4 sliced tiles)

        EVERY site that needs this number (_segment_bases / get_image_and_label / __len__)
        must call this instead of re-deriving it inline. The formula used to be copy-pasted in 4
        places; adding a new online branch and missing one of them desynchronises __len__ from the
        decodable index range and drops samples with no error at all.
        """
        if not (bool(getattr(self, "slice_all_tiles", False)) and getattr(self, "slice_transform", None) is not None):
            return 1
        return 4

    def _keep_origin_on(self) -> bool:
        """True when the un-sliced ORIGIN segment allocates samples (slice_keep_origin independent switch).

        ``slice_keep_origin`` is now an independent switch -- it is no longer forced off when
        ``slice_prob == 0`` in ``augment.py``, and it no longer changes ``_n_per``. It only makes
        sense together with slicing though: without slicing the originals are already the whole
        sample pool, so keeping them again would duplicate every image. Auto-suppressed here when
        the slicing pipeline is off.
        """
        return (
            bool(getattr(self, "slice_keep_origin", False))
            and bool(getattr(self, "slice_all_tiles", False))
            and getattr(self, "slice_transform", None) is not None
        )

    def _segment_bases(self) -> tuple[int, int, int, int, int]:
        """Return the five segment boundaries of the mixed sample pool.

        Layout (each optional branch is an independent, contiguous segment gated ONLY by its own
        switch; slicing lives entirely inside the base segment):
            [0, base_len)                 base:     _n_per samples per original (slicing pipeline)
            [base_len, +N)                origin:   1 un-sliced original per image (keep_origin)
            [base_len+N, +2N)             ratio:    1 aspect-ratio-padded image per original
            [base_len+2N, +4N)            blur:     short + long motion-blurred images per original
            [base_len+4N, +ceil(N/4))     compose:  one 2x2 stitched image per group of 4 originals

        Returns (base_len, origin_base, ratio_base, blur_base, compose_base).
        """
        n = len(self.labels)
        base_len = n * self._n_per()
        o_base = base_len  # origin segment starts right after the base slicing segment
        r_base = o_base + (n if self._keep_origin_on() else 0)
        b_base = r_base + (n if bool(getattr(self, "ratio_pad_keep", False)) else 0)
        c_base = b_base + (2 * n if bool(getattr(self, "blur_keep", False)) else 0)
        return base_len, o_base, r_base, b_base, c_base

    def _origin_at(self, index: int, img_index: int) -> dict[str, Any]:
        """Build one un-sliced ORIGINAL-resolution sample (slice_keep_origin independent segment).

        The origin segment is laid out right after the base slicing segment (see _segment_bases) and
        behaves exactly like a plain full image in the mixed pool: load_image + training resize +
        ratio_pad, and it enters the Mosaic mix pool (dataset.buffer) like every other sample.
        Nothing is written to disk.

        ``index`` is the EXPANDED mixed-pool index (for correct Mosaic buffer bookkeeping);
        ``img_index`` is the ORIGINAL image index this sample derives from.
        """
        label = deepcopy(self.labels[img_index])
        label.pop("shape", None)  # shape is for rect, remove it
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(img_index)
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        # Keep the original on the same Mosaic mix pool as every other sample (cache != 'ram')
        if self.augment and self.cache != "ram":
            self.buffer.append(index)
            if 1 < len(self.buffer) >= self.max_buffer_length:
                self.buffer.pop(0)
        return self.update_labels_info(label)

    def _compose_on(self) -> bool:
        """True when the compose branch allocates samples (switch on AND >= 4 originals).

        With fewer than 4 originals a group would have to reuse the same image in 2+ quadrants,
        duplicating its targets and skewing the label distribution, so compose is switched off
        entirely in that case.
        """
        return bool(getattr(self, "compose_keep", False)) and len(self.labels) >= 4

    def _extended_pool_on(self) -> bool:
        """True when any project extension allocates samples beyond plain originals.

        ``load_image``'s self-managed buffer (pure-ultralytics path) is only safe when the buffer
        can only contain original-image indices. Once any extended segment (keep_origin / ratio /
        blur / compose) writes EXPANDED indices into the same buffer, ``load_image`` must not
        pop/clear ``ims`` with them; buffer bookkeeping is then entirely owned by
        ``get_image_and_label`` (expanded indices).
        """
        return (
            self._keep_origin_on()
            or bool(getattr(self, "ratio_pad_keep", False))
            or bool(getattr(self, "blur_keep", False))
            or self._compose_on()
        )

    def _load_image_cached(self, img_index: int) -> np.ndarray:
        """Load original-resolution image, with a tiny per-worker memory LRU (no .npy disk cache).

        Unified read path used by all online-augmentation branches (slice / blur / ratio /
        compose). Reads the original JPEG directly via ``imread``; a small in-memory LRU
        (``slice_raw_cache_size``) absorbs repeated decoding of the same image across its
        sub-samples. The custom .npy disk cache was removed -- it measured no benefit.

        Returns the image as a contiguous uint8 array with at least 3 dims (H, W, C); grayscale
        is expanded to (H, W, 1).

        Caching: a hit returns a COPY, not the cached array. Callers legitimately take ownership of the
        result (e.g. ``_ratio_at`` keeps ``big = im`` when no padding is needed, and downstream affine /
        mosaic transforms write in place), so handing out the cached buffer would corrupt it for the next
        sub-sample of the same image. A memcpy is still ~10-30x cheaper than decoding a large JPEG.
        """
        # Per-worker LRU lookup (see __init__). Disabled when slice_raw_cache_size <= 0.
        size = int(getattr(self, "_raw_cache_size", 2) or 0)
        cache = getattr(self, "_raw_cache", None)
        if cache is None:
            cache = self._raw_cache = {}
        if size > 0:
            hit = cache.get(img_index)
            if hit is not None:
                # P4 fix: refresh the insertion order on hit (true LRU, not FIFO). The dict entry
                # previously stayed in its original slot on a hit, so hot entries were evicted by
                # the "drop oldest half" sweep purely by first-load time.
                cache[img_index] = cache.pop(img_index)
                return hit.copy()

        f = self.im_files[img_index]
        im = imread(f, flags=self.cv2_flag)
        if im is None:
            raise FileNotFoundError(f"Image Not Found {f}")
        if im.ndim == 2:
            im = im[..., None]

        if size > 0:
            if len(cache) >= size:
                # dicts preserve insertion order: drop the oldest half to make room.
                for k in list(cache)[: max(1, len(cache) // 2)]:
                    cache.pop(k, None)
            cache[img_index] = im
            return im.copy()
        return im

    def _blur_at(self, index: int, img_index: int, long: bool = False) -> dict[str, Any]:
        """Build one in-memory motion-blurred image from a single original image (online port of the offline
        motion_blur tool).

        Two tiers per image: ``short`` (light, length in [blur_short_len_min, blur_short_len_max], no
        defocus) and ``long`` (heavy, length in [blur_long_len_min, blur_long_len_max], optional defocus
        sigma up to blur_long_defocus_sigma). The blur angle is sampled uniformly in [0, 180) per call
        (augmentation randomness, consistent with fliplr etc.). Labels are UNCHANGED (blur does not move
        targets). The blurred image is resized to the training size like the other branches and enters the
        Mosaic mix pool (dataset.buffer). Nothing is written to disk (save via blur_save_dir).

        ``index`` is the EXPANDED mixed-pool index (for correct Mosaic buffer bookkeeping);
        ``img_index`` is the ORIGINAL image index this blurred sample derives from.
        """
        f = self.im_files[img_index]
        im = self._load_image_cached(img_index)
        h, w = im.shape[:2]
        if long:
            lo = float(getattr(self, "blur_long_len_min", 20))
            hi = float(getattr(self, "blur_long_len_max", 35))
            sigma = float(getattr(self, "blur_long_defocus_sigma", 1.0))
            tier = "long"
        else:
            lo = float(getattr(self, "blur_short_len_min", 5))
            hi = float(getattr(self, "blur_short_len_max", 12))
            sigma = 0.0
            tier = "short"
        # blur_ratio: 未选中原图跳过模糊, 整图直通 (位保留, len 恒定); 短+长同命运
        if self._blur_mask is not None and not self._blur_mask[img_index]:
            blur = im
        else:
            length = random.uniform(lo, hi)
            angle = random.uniform(0.0, 180.0)
            blur = _apply_motion_blur(im, length=length, angle=angle, defocus_sigma=sigma)

        label = deepcopy(self.labels[img_index])
        label.pop("shape", None)
        label["im_file"] = f
        label["img"] = np.ascontiguousarray(blur)

        # Keep the blurred image on the same Mosaic mix pool as every other sample (cache != 'ram')
        if self.augment and self.cache != "ram":
            self.buffer.append(index)
            if 1 < len(self.buffer) >= self.max_buffer_length:
                self.buffer.pop(0)

        # Optional save for visual inspection (blur_save_dir set). Annotated per
        # slice_save_annotated, capped by slice_save_max_blur (P2-3: per-branch override; falls back to
        # slice_save_max when the per-branch cap is not set), deduplicated per (image, tier) across epochs.
        sdir = str(getattr(self, "blur_save_dir", "") or "")
        st = getattr(self, "slice_transform", None)
        if sdir and st is not None:
            cdir = Path(sdir)
            if not hasattr(self, "_blur_saved"):
                self._blur_saved = 0
                self._blur_saved_keys = set()
            key = ("blur", tier, index)
            save_cap = _save_cap(self, "blur")
            if key not in self._blur_saved_keys and (save_cap == 0 or self._blur_saved < save_cap):
                _ensure_dir(cdir)
                img = blur
                boxes = np.asarray(label.get("bboxes", np.empty((0, 4))), dtype=np.float64)
                if st.save_annotated and len(boxes):
                    img = blur.copy()
                    H2, W2 = blur.shape[:2]
                    cls = np.asarray(label.get("cls", np.empty((0, 1))))
                    for b, c in zip(boxes, np.asarray(cls).reshape(-1)):
                        cx, cy, bw, bh = (float(v) for v in b)
                        x0 = int(round((cx - bw / 2) * W2))
                        y0 = int(round((cy - bh / 2) * H2))
                        x1 = int(round((cx + bw / 2) * W2))
                        y1 = int(round((cy + bh / 2) * H2))
                        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
                        cv2.putText(img, f"cls{int(c)}", (x0, max(0, y0 - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                # p{pid}: counters are per-worker, so two workers can emit the same sequence number.
                # img{img_index} already disambiguates in practice, but Mosaic proved the pattern is fragile.
                _imwrite(cdir / f"blur_{tier}_p{os.getpid()}_img{img_index}_"
                                f"{self._blur_saved:05d}_n{len(boxes)}.jpg", img)
                self._blur_saved += 1
                self._blur_saved_keys.add(key)

        # Resize to the training size (same as the sliced / original / ratio / composed branches)
        h1, w1 = blur.shape[:2]
        r = self.imgsz / max(h1, w1)
        if r != 1:
            blur = cv2.resize(blur, (math.ceil(w1 * r), math.ceil(h1 * r)), interpolation=cv2.INTER_LINEAR)
        if blur.ndim == 2:
            blur = blur[..., None]
        label["img"] = np.ascontiguousarray(blur)
        label["ori_shape"] = (h1, w1)
        label["resized_shape"] = blur.shape[:2]
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )
        return self.update_labels_info(label)

    def _ratio_at(self, index: int, img_index: int) -> dict[str, Any]:
        """Build one in-memory aspect-ratio-padded image from a single original image (online port of the
        offline change_image_resolution tool).

        The original image is read at its ORIGINAL resolution, padded with borders (left/right or top/bottom,
        symmetric) to its target aspect ratio -- ``auto``: 4:3 <-> 16:9 bidirectional, any other ratio goes to
        its nearest of 4:3 or 16:9; ``4:3``/``16:9``: unify every image to that ratio -- and every annotation
        (bboxes/segments/keypoints) is remapped by the pad offset: ``xc' = (xc*W + pad_left)/new_w``,
        ``yc' = (yc*H + pad_top)/new_h``, ``w' = w*W/new_w``, ``h' = h*H/new_h``, then boundary-clamped and
        invalid boxes dropped (identical to the offline tool). The padded image is resized to the training size
        like the other branches and enters the Mosaic mix pool (dataset.buffer). Nothing is written to disk.

        ``index`` is the EXPANDED mixed-pool index (for correct Mosaic buffer bookkeeping);
        ``img_index`` is the ORIGINAL image index this padded sample derives from.
        """
        f = self.im_files[img_index]
        im = self._load_image_cached(img_index)
        h, w = im.shape[:2]
        # ratio_pad_ratio: 未选中原图跳过加框, 整图直通 (位保留, len 恒定)
        skip_pad = self._ratio_mask is not None and not self._ratio_mask[img_index]
        target = str(getattr(self, "ratio_pad_target", "auto") or "auto")
        color_key = str(getattr(self, "ratio_pad_color", "black") or "black")
        # P1-7: validate up front. Previously an unknown color raised a bare KeyError halfway through
        # training, and an unknown target silently fell back to 16:9 (any non-"4:3" string did).
        if color_key not in _RATIO_PAD_COLORS:
            raise ValueError(f"ratio_pad_color must be one of {sorted(_RATIO_PAD_COLORS)}, got '{color_key}'.")
        if target not in ("auto", "4:3", "16:9"):
            raise ValueError(f"ratio_pad_target must be one of 'auto', '4:3', '16:9', got '{target}'.")
        pad = None if skip_pad else _ratio_pad_params(w, h, target, auto=(target == "auto"))
        if pad is None:
            # Already at the target ratio: no padding needed (use the original as-is)
            big = im
            new_w, new_h, pad_left, pad_top = w, h, 0, 0
        else:
            new_w, new_h, pad_left, pad_top = pad
            C = im.shape[2]
            big = np.full((new_h, new_w, C), _RATIO_PAD_COLORS[color_key], dtype=im.dtype)
            big[pad_top:pad_top + h, pad_left:pad_left + w] = im
            del im

        lb = self.labels[img_index]
        boxes = np.asarray(lb.get("bboxes", np.empty((0, 4))), dtype=np.float64).copy()
        # P1-3: explicit .copy(). np.asarray() returns the SAME array when the dtype already matches, so this
        # used to hand out a live view of self.labels[i]["cls"] -- and augment._update_label_text writes
        # label["cls"] in place, which would permanently corrupt the cached label for all later epochs.
        # _blur_at and the main path already deepcopy; this branch and _compose_at did not.
        cls = np.asarray(lb.get("cls", np.empty((0, 1))), dtype=np.float32).copy()
        keep = None
        if len(boxes) and (pad_left or pad_top):
            boxes[:, 0] = (boxes[:, 0] * w + pad_left) / new_w
            boxes[:, 1] = (boxes[:, 1] * h + pad_top) / new_h
            boxes[:, 2] = (boxes[:, 2] * w) / new_w
            boxes[:, 3] = (boxes[:, 3] * h) / new_h
            boxes[:, 0] = np.clip(boxes[:, 0], 0.0, 1.0)
            boxes[:, 1] = np.clip(boxes[:, 1], 0.0, 1.0)
            boxes[:, 2] = np.clip(boxes[:, 2], 0.0, 1.0 - boxes[:, 0])
            boxes[:, 3] = np.clip(boxes[:, 3], 0.0, 1.0 - boxes[:, 1])
            keep = (boxes[:, 2] > 0) & (boxes[:, 3] > 0)
            boxes = boxes[keep]
            cls = cls[keep]
        # segments: normalized polys remapped by the pad offset (no clipping needed: padding only enlarges)
        segs = []
        for si, s in enumerate(lb.get("segments", []) or []):
            # P1-4: apply the same `keep` mask as the boxes. Padding never drops anything today, so this is
            # dormant -- but the moment any box is filtered, len(bboxes) != len(segments) would silently
            # mislabel every seg/pose task with no error anywhere.
            if keep is not None and not (si < len(keep) and bool(keep[si])):
                continue
            s = np.asarray(s, dtype=np.float64).copy()
            if pad_left or pad_top:
                s[..., 0] = (s[..., 0] * w + pad_left) / new_w
                s[..., 1] = (s[..., 1] * h + pad_top) / new_h
            segs.append(s.astype(np.float32))
        label = {
            "im_file": f,
            "img": np.ascontiguousarray(big),
            "bboxes": boxes.astype(np.float32),
            "bbox_format": "xywh",
            "normalized": True,
            "cls": cls,
            "segments": segs,
        }
        kpts = lb.get("keypoints", None)
        if kpts is not None:
            k = np.asarray(kpts, dtype=np.float64).copy()
            if pad_left or pad_top:
                k[..., 0] = (k[..., 0] * w + pad_left) / new_w
                k[..., 1] = (k[..., 1] * h + pad_top) / new_h
            if keep is not None and k.shape[0] == len(keep):
                k = k[keep]  # P1-4: keep keypoints aligned with the filtered boxes
            label["keypoints"] = k.astype(np.float32)

        # Keep the ratio-padded image on the same Mosaic mix pool as every other sample (cache != 'ram')
        if self.augment and self.cache != "ram":
            self.buffer.append(index)
            if 1 < len(self.buffer) >= self.max_buffer_length:
                self.buffer.pop(0)

        # Optional save of the ratio-padded image for visual inspection (ratio_pad_save_dir set).
        # Annotated per slice_save_annotated, capped by slice_save_max_ratio (P2-3: per-branch override,
        # falls back to slice_save_max), deduplicated per (image) across epochs/mix visits.
        # Disabled by default (empty dir).
        sdir = str(getattr(self, "ratio_pad_save_dir", "") or "")
        st = getattr(self, "slice_transform", None)
        if sdir and st is not None:
            cdir = Path(sdir)
            if not hasattr(self, "_ratio_saved"):
                self._ratio_saved = 0
                self._ratio_saved_keys = set()
            key = ("ratio", index)
            save_cap = _save_cap(self, "ratio")
            if key not in self._ratio_saved_keys and (save_cap == 0 or self._ratio_saved < save_cap):
                _ensure_dir(cdir)
                img = big
                if st.save_annotated and len(boxes):
                    img = big.copy()
                    H2, W2 = big.shape[:2]
                    for b, c in zip(boxes, np.asarray(cls).reshape(-1)):
                        cx, cy, bw, bh = (float(v) for v in b)
                        x0 = int(round((cx - bw / 2) * W2))
                        y0 = int(round((cy - bh / 2) * H2))
                        x1 = int(round((cx + bw / 2) * W2))
                        y1 = int(round((cy + bh / 2) * H2))
                        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
                        cv2.putText(img, f"cls{int(c)}", (x0, max(0, y0 - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                _imwrite(cdir / f"ratio_p{os.getpid()}_img{img_index}_"
                                f"{self._ratio_saved:05d}_n{len(boxes)}.jpg", img)
                self._ratio_saved += 1
                self._ratio_saved_keys.add(key)

        # Resize to the training size (same as the sliced / original / composed branches)
        h1, w1 = big.shape[:2]
        r = self.imgsz / max(h1, w1)
        if r != 1:
            big = cv2.resize(big, (math.ceil(w1 * r), math.ceil(h1 * r)), interpolation=cv2.INTER_LINEAR)
        if big.ndim == 2:
            big = big[..., None]
        label["img"] = np.ascontiguousarray(big)
        label["ori_shape"] = (h1, w1)
        label["resized_shape"] = big.shape[:2]
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )
        return self.update_labels_info(label)

    def _compose_at(self, index: int) -> dict[str, Any]:
        """Build a 2x2 composed image from 4 original images (online port of the offline compose tool).

        Each group of 4 original images (``[group*4, group*4+3]``, wrapped for the tail group) is stitched
        into one larger image (2 columns x 2 rows) in memory, and every annotation (bboxes/segments/keypoints)
        is remapped from each sub-image's normalized coords to the composed-image coordinate system:
        ``xc' = (xc + col) / 2, yc' = (yc + row) / 2, w' = w / 2, h' = h / 2`` (col/row = 0/1). The composed
        image is then resized to the training size like the other branches. It enters the Mosaic mix pool
        (dataset.buffer) so it participates in mosaic stitching. Nothing is written to disk.
        """
        n_origin = len(self.labels)
        # The compose segment starts right after the base + ratio + blur segments (_segment_bases);
        # group = index - compose_base selects the group of 4 originals.
        _base_len, _o_base, _r_base, _b_base, c_base = self._segment_bases()
        group = index - c_base
        base = group * 4
        # P1-2: with fewer than 4 originals the modulo wrap would put the SAME image in 2+ quadrants,
        # duplicating its targets and skewing the label distribution. _compose_on() never allocates
        # compose indices in that case, so this is a defensive guard for direct calls only.
        if n_origin < 4:
            raise IndexError(
                f"Compose samples are disabled with fewer than 4 images (got {n_origin}); index {index} is "
                f"outside the valid range."
            )
        idxs = [(base + j) % n_origin for j in range(4)]  # wrap the tail group to always have 4 images
        # compose_ratio: 未选中组退回组内第 1 张原图整图 (位保留, len 恒定)
        if self._compose_mask is not None and not self._compose_mask[group]:
            return self._origin_at(index, base)
        imgs = []
        for i in idxs:
            im = self._load_image_cached(i)
            imgs.append(im)
        # Unify sub-image size to the max in this group (stretching preserves normalized coords linearly)
        W = max(im.shape[1] for im in imgs)
        H = max(im.shape[0] for im in imgs)
        # Allocate the composed image ONCE and fill each 2x2 sub-block in place (memory-friendly: avoids the
        # intermediate row1/row2 hstack buffers, ~2x peak reduction on a large composed image).
        C = imgs[0].shape[2]
        big = np.empty((2 * H, 2 * W, C), dtype=imgs[0].dtype)
        for j, im in enumerate(imgs):
            r, c = divmod(j, 2)
            if im.shape[1] != W or im.shape[0] != H:
                im = cv2.resize(im, (W, H), interpolation=cv2.INTER_LINEAR)
            big[r * H:(r + 1) * H, c * W:(c + 1) * W] = im
            imgs[j] = None  # P2-9: free each source the moment it is copied, instead of holding all 4
        del imgs  # release the 4 source images as early as possible

        # 拼后降采样到最长边 = compose_max_side (0=自动=2×imgsz), 降低内存与后续处理开销。
        # bbox/segment/keypoint 均为归一化坐标, 只缩放像素不影响标签; 保存/再resize 均用降采样后的图。
        max_side = int(getattr(self, "compose_max_side", 0) or 0)
        if max_side <= 0:
            max_side = 2 * int(getattr(self, "imgsz", 1280))
        _H, _W = big.shape[:2]
        if max(_H, _W) > max_side:
            _r = max_side / max(_H, _W)
            big = cv2.resize(big, (math.ceil(_W * _r), math.ceil(_H * _r)), interpolation=cv2.INTER_LINEAR)

        boxes_all, cls_all, segs_all, kpts_all, has_kpts = [], [], [], [], False
        for j, i in enumerate(idxs):
            row, col = divmod(j, 2)
            lb = self.labels[i]
            boxes = np.asarray(lb.get("bboxes", np.empty((0, 4))), dtype=np.float64).copy()
            # P1-3: explicit .copy() (np.asarray is a no-op view when the dtype already matches), otherwise
            # the composed label shares cls storage with self.labels and in-place text updates corrupt it.
            cls = np.asarray(lb.get("cls", np.empty((0, 1))), dtype=np.float32).copy()
            if len(boxes):
                boxes[:, 0] = (boxes[:, 0] + col) / 2.0
                boxes[:, 1] = (boxes[:, 1] + row) / 2.0
                boxes[:, 2] = boxes[:, 2] / 2.0
                boxes[:, 3] = boxes[:, 3] / 2.0
                boxes_all.append(boxes)
                cls_all.append(cls)
            # segments: list of normalized polys -> composed-image normalized
            for s in lb.get("segments", []) or []:
                s = np.asarray(s, dtype=np.float64).copy()
                s[..., 0] = (s[..., 0] + col) / 2.0
                s[..., 1] = (s[..., 1] + row) / 2.0
                segs_all.append(s.astype(np.float32))
            # keypoints: (N, K, 3) normalized x,y + visibility
            kpts = lb.get("keypoints", None)
            if kpts is not None:
                has_kpts = True
                k = np.asarray(kpts, dtype=np.float64).copy()
                k[..., 0] = (k[..., 0] + col) / 2.0
                k[..., 1] = (k[..., 1] + row) / 2.0
                kpts_all.append(k)

        bboxes = np.concatenate(boxes_all, 0).astype(np.float32) if boxes_all else np.empty((0, 4), np.float32)
        cls = np.concatenate(cls_all, 0).astype(np.float32) if cls_all else np.empty((0, 1), np.float32)
        label = {
            "im_file": self.im_files[idxs[0]],
            "img": np.ascontiguousarray(big),
            "bboxes": bboxes,
            "bbox_format": "xywh",
            "normalized": True,
            "cls": cls,
            "segments": segs_all,
        }
        if has_kpts:
            label["keypoints"] = np.concatenate(kpts_all, 0).astype(np.float32)

        # Optional save of the composed 2x2 image for visual inspection (compose_save=True).
        # Save dir: compose_save_dir (custom) if set, else slice_save_dir/compose/. Annotated per
        # slice_save_annotated, capped by slice_save_max_compose (P2-3: per-branch override, falls back to
        # slice_save_max), deduplicated per (group) across epochs/mix visits.
        st = getattr(self, "slice_transform", None)
        if st is not None and getattr(self, "compose_save", False):
            comp_dir = str(getattr(self, "compose_save_dir", "") or "")
            cdir = Path(comp_dir) if comp_dir else (st.save_dir / "compose" if st.save_dir is not None else None)
            if cdir is None and not getattr(self, "_compose_warned", False):
                # P2-4: args.yaml ships compose_save=True with an empty compose_save_dir and no
                # slice_save_dir -> nothing was ever written and nothing said why.
                self._compose_warned = True
                LOGGER.warning(
                    f"{self.prefix}compose_save=True but no output directory is configured: set "
                    f"compose_save_dir, or slice_save_dir (images then go to <slice_save_dir>/compose). "
                    f"Skipping compose save."
                )
            if cdir is not None:
                if not hasattr(self, "_compose_saved"):
                    self._compose_saved = 0
                    self._compose_saved_keys = set()
                key = ("compose", group)
                save_cap = _save_cap(self, "compose")
                if key not in self._compose_saved_keys and (save_cap == 0 or self._compose_saved < save_cap):
                    _ensure_dir(cdir)
                    img = big
                    if st.save_annotated and len(bboxes):
                        img = big.copy()
                        H2, W2 = big.shape[:2]
                        for b, c in zip(bboxes, np.asarray(cls).reshape(-1)):
                            cx, cy, bw, bh = (float(v) for v in b)
                            x0 = int(round((cx - bw / 2) * W2))
                            y0 = int(round((cy - bh / 2) * H2))
                            x1 = int(round((cx + bw / 2) * W2))
                            y1 = int(round((cy + bh / 2) * H2))
                            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
                            cv2.putText(img, f"cls{int(c)}", (x0, max(0, y0 - 4)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    if _imwrite(
                        str(cdir / f"compose_p{os.getpid()}_g{group}_{self._compose_saved:05d}_n{len(bboxes)}.jpg"),
                        img,
                    ):
                        self._compose_saved += 1
                        self._compose_saved_keys.add(key)

        # Keep the composed image on the same Mosaic mix pool as every other sample (cache != 'ram')
        if self.augment and self.cache != "ram":
            self.buffer.append(index)
            if 1 < len(self.buffer) >= self.max_buffer_length:
                self.buffer.pop(0)

        # Resize to the training size (same as the sliced / original branches)
        h1, w1 = big.shape[:2]
        r = self.imgsz / max(h1, w1)
        if r != 1:
            big = cv2.resize(big, (math.ceil(w1 * r), math.ceil(h1 * r)), interpolation=cv2.INTER_LINEAR)
        if big.ndim == 2:
            big = big[..., None]
        label["img"] = np.ascontiguousarray(big)
        label["ori_shape"] = (h1, w1)
        label["resized_shape"] = big.shape[:2]
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )
        return self.update_labels_info(label)

    def get_image_and_label(self, index: int, count_slice: bool = True) -> dict[str, Any]:
        """Get and return label information from the dataset.

        Args:
            index (int): Index of the image to retrieve.
            count_slice (bool): Whether OnlineSlice updates its positive/background counters and saves tiles.
                Auxiliary "mix" samples requested by Mosaic/CutMix/MixUp pass ``False`` so they slice normally
                but do not inflate the ``neg_ratio`` quota or duplicate saved slices.
        """
        # Mixed-pool layout (see _segment_bases): a base segment holding ONLY the slicing pipeline's
        # samples (4 tiles per image), followed by four INDEPENDENT segments gated only by their own
        # switches -- origin (1 un-sliced original per image, keep_origin), ratio (1 per image),
        # blur (2 per image), compose (1 per 4 images). None of them requires slicing or keep_origin.
        emit_all = getattr(self, "slice_all_tiles", False)
        origin_on = self._keep_origin_on()
        ratio_on = bool(getattr(self, "ratio_pad_keep", False))
        blur_on = bool(getattr(self, "blur_keep", False))
        compose_on = self._compose_on()
        base_len, o_base, r_base, b_base, c_base = self._segment_bases()

        if compose_on and index >= c_base:
            return self._compose_at(index)  # composed 2x2 sample from 4 original images
        if blur_on and index >= b_base:
            j = index - b_base  # 0..2N-1: even -> short tier, odd -> long tier
            return self._blur_at(index, j // 2, long=(j % 2 == 1))
        if ratio_on and index >= r_base:
            return self._ratio_at(index, index - r_base)  # origin index = offset inside the ratio segment
        if origin_on and index >= o_base:
            return self._origin_at(index, index - o_base)  # un-sliced original (keep_origin segment)

        # ---- base segment (slicing pipeline) ----
        n_per = self._n_per()
        if n_per > 1:  # slicing is active and expands each original image
            img_index = index // n_per
            k = index % n_per  # tile index for emit_all mode
        else:  # no expansion: sample index == original image index
            img_index = index
            k = None
        label = deepcopy(self.labels[img_index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop("shape", None)  # shape is for rect, remove it
        # Online slicing runs on the ORIGINAL-resolution image (before any training resize) so small
        # objects are genuinely enlarged when the sliced sub-image is resized to the training size.
        # slice_ratio: each epoch set_epoch() rebuilds a mask that picks exactly round(slice_ratio*N)
        # ORIGINAL images to slice (original-level decision: with emit_all all 4 tiles share one fate);
        # the rest are fed as un-sliced full images. Mask None = pure slicing (default) or slicing off.
        # The un-sliced originals live in their own independent segment (_origin_at) and never slice.
        slice_t = getattr(self, "slice_transform", None)
        # Lazy init: if the trainer never called set_epoch (direct sampling / validation path),
        # build the mask once on first access so 0 < slice_ratio < 1 still works.
        if self._slice_mask is None and 0.0 <= float(getattr(self, "slice_ratio", 1.0)) < 1.0:
            self.set_epoch(getattr(self, "epoch", 0))
        slice_ok = self._slice_mask is None or bool(self._slice_mask[img_index])
        # The mosaic buffer is maintained centrally here using DATASET (expanded) indices whenever
        # load_image does NOT self-manage it: slicing on, or any project extension on (load_image's
        # self-managed path is only active in pure-ultralytics mode; mixing original indices from
        # load_image with expanded indices here would corrupt the buffer).
        if (
            self.augment
            and self.cache != "ram"
            and (slice_t is not None or self._extended_pool_on())
        ):
            self.buffer.append(index)
            if 1 < len(self.buffer) >= self.max_buffer_length:  # prevent unbounded buffer
                self.buffer.pop(0)
        if slice_t is not None and self.augment and slice_ok:
            # _load_image_cached: 直接读原图 jpg + worker 内存 LRU (.npy 磁盘缓存已移除)
            im = self._load_image_cached(img_index)
            im, label = (
                slice_t.slice_at(im, label, k, src=(img_index, k), count=count_slice) if emit_all else slice_t(
                    im, label, src=img_index, count=count_slice
                )
            )
            h1, w1 = im.shape[:2]
            # Resize the sliced sub-image to the training size, preserving aspect ratio (same as load_image).
            r = self.imgsz / max(h1, w1)
            if r != 1:
                im = cv2.resize(im, (math.ceil(w1 * r), math.ceil(h1 * r)), interpolation=cv2.INTER_LINEAR)
            if im.ndim == 2:
                im = im[..., None]
            label["img"] = np.ascontiguousarray(im)
            label["ori_shape"] = (h1, w1)  # sliced sub-image is the new "original" for downstream transforms
            label["resized_shape"] = im.shape[:2]
            label["ratio_pad"] = (
                label["resized_shape"][0] / label["ori_shape"][0],
                label["resized_shape"][1] / label["ori_shape"][1],
            )
            return self.update_labels_info(label)
        # load_image indexes the ORIGINAL image files, so always use img_index (== index in the non-sliced
        # case); in emit_all mode index is the expanded (4N) sample index and would overflow.
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(img_index)
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)

    def __len__(self) -> int:
        """Return the number of samples in the mixed pool.

        Base segment (``_n_per`` samples per original) plus four independent segments: +N origin
        (keep_origin), +N ratio, +2N blur, +ceil(N/4) compose -- each present only when its own
        switch is on. Boundaries are centralized in ``_segment_bases`` so ``get_image_and_label``
        / ``_compose_at`` cannot drift apart; a mismatch would silently drop samples with no error.
        """
        n = len(self.labels)
        total = n * self._n_per()
        if self._keep_origin_on():
            total += n
        if bool(getattr(self, "ratio_pad_keep", False)):
            total += n
        if bool(getattr(self, "blur_keep", False)):
            total += 2 * n
        if self._compose_on():
            total += (n + 3) // 4
        return total

    def update_labels_info(self, label: dict[str, Any]) -> dict[str, Any]:
        """Customize your label format here."""
        return label

    def build_transforms(self, hyp: dict[str, Any] | None = None):
        """Users can customize augmentations here.

        Examples:
            >>> if self.augment:
            ...     # Training transforms
            ...     return Compose([])
            >>> else:
            ...    # Val transforms
            ...    return Compose([])
        """
        raise NotImplementedError

    def get_labels(self) -> list[dict[str, Any]]:
        """Users can customize their own format here.

        Examples:
            Ensure output is a dictionary with the following keys:
            >>> dict(
            ...     im_file=im_file,
            ...     shape=shape,  # format: (height, width)
            ...     cls=cls,
            ...     bboxes=bboxes,  # xywh
            ...     segments=segments,  # xy
            ...     keypoints=keypoints,  # xy
            ...     normalized=True,  # or False
            ...     bbox_format="xyxy",  # or xywh, ltwh
            ... )
        """
        raise NotImplementedError


class SliceValDataset(Dataset):
    """Validation-side online slicing: expand each val image into 2x2 (+overlap) sub-tiles (SAHI eval).

    Wraps the native validation ``YOLODataset`` and reuses its labels / transforms / collate_fn, so the
    validator pipeline stays untouched except for prediction remapping + NMS fusion (see
    ``DetectionValidator``). Every sub-tile is inferred -- no target filtering (the training-side
    filters are about what the model SEES; the val side evaluates slicing INFERENCE, so all boxes of
    every tile are kept and overlapping duplicates are merged later by NMS).

    ``val_slice_ratio < 1`` picks ``round(x * N)`` originals to slice this round, the rest pass through
    as full images (1 "tile" = the whole image). ``len`` is fixed at construction; the dataset is
    rebuilt by the validator on every validation round.

    Each sample carries ``val_slice_meta`` with the metadata needed to remap predictions back to the
    original image coordinates: ``orig_idx``, ``k``, ``offset`` (tile origin), ``tile_shape``,
    ``orig_shape``, ``n_tiles``, ``sliced``.
    """

    def __init__(
        self,
        base: Dataset,
        overlap_ratio: float = 0.2,
        all_tiles: bool = True,
        ratio: float = 1.0,
    ) -> None:
        self.base = base
        self.labels = base.labels
        self.n = len(self.labels)
        self.overlap_ratio = overlap_ratio
        self.all_tiles = all_tiles
        self.ratio = float(ratio)
        self.transforms = base.transforms
        # Forward the base dataset's collate_fn so build_dataloader assembles batches identically.
        self.collate_fn = base.collate_fn
        self._build()

    def _build(self) -> None:
        """Build the per-original tile layout: mask (val_slice_ratio) + per-orig tile boxes."""
        import bisect

        from ultralytics.data.augment import slice_geometry

        n = self.n
        mask = None
        if 0.0 <= self.ratio < 1.0:
            mask = np.zeros(n, dtype=bool)
            if self.ratio > 0:
                mask[random.sample(range(n), int(round(self.ratio * n)))] = True
        self._mask = mask
        counts: list[int] = []
        metas: list[list[tuple[int, int, int, int]]] = []
        self._shapes: list[tuple[int, int]] = []
        for i, lb in enumerate(self.labels):
            h, w = lb.get("shape", (0, 0))[:2]
            self._shapes.append((int(h), int(w)))
            slice_it = bool(self.all_tiles) and (mask is None or bool(mask[i])) and h >= 2 and w >= 2
            if slice_it:
                tiles = slice_geometry(w, h, self.overlap_ratio)
                counts.append(len(tiles))
                metas.append(tiles)
            else:
                counts.append(1)
                metas.append([(0, 0, w, h)])
        self._counts = counts
        self._metas = metas
        self._cum = [0]
        for c in counts:
            self._cum.append(self._cum[-1] + c)

    def __len__(self) -> int:
        """Total number of validation samples (sum of per-original tile counts)."""
        return self._cum[-1]

    def _decode(self, index: int) -> tuple[int, int]:
        """Map an expanded sample index to ``(original_index, tile_index)``."""
        import bisect

        i = bisect.bisect_right(self._cum, index) - 1
        return i, index - self._cum[i]

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Build one validation sample: (sub-image or full image) + remapped label + remap metadata."""
        oi, k = self._decode(index)
        x0, y0, x1, y1 = self._metas[oi][k]
        h, w = self._shapes[oi]
        sliced = (x0, y0, x1, y1) != (0, 0, w, h)
        label = deepcopy(self.labels[oi])
        label.pop("shape", None)  # shape is for rect, remove it
        # Load the ORIGINAL-resolution image (worker LRU cache when available).
        if hasattr(self.base, "_load_image_cached"):
            im = self.base._load_image_cached(oi)
        else:
            im = cv2.imread(label["im_file"])
        if im is None:
            im = cv2.imread(label["im_file"])
        tw, th = x1 - x0, y1 - y0
        sub = np.ascontiguousarray(im[y0:y1, x0:x1])
        if sliced:
            # Remap ALL boxes to the sub-image (clip at tile borders, no filtering): normalized
            # xywh (orig) -> pixel xyxy -> clip to tile -> normalized xywh (sub).
            boxes = np.asarray(label["bboxes"], dtype=np.float64)
            if len(boxes):
                cx, cy, bw, bh = boxes[:, 0] * w, boxes[:, 1] * h, boxes[:, 2] * w, boxes[:, 3] * h
                xa, ya = cx - bw / 2, cy - bh / 2
                xb, yb = cx + bw / 2, cy + bh / 2
                lx0 = np.clip(xa - x0, 0.0, tw)
                ly0 = np.clip(ya - y0, 0.0, th)
                lx1 = np.clip(xb - x0, 0.0, tw)
                ly1 = np.clip(yb - y0, 0.0, th)
                nw2, nh2 = lx1 - lx0, ly1 - ly0
                keep = (nw2 > 1) & (nh2 > 1)
                if keep.any():
                    lx0, ly0, lx1, ly1 = lx0[keep], ly0[keep], lx1[keep], ly1[keep]
                    label["bboxes"] = np.stack(
                        [(lx0 + lx1) / 2 / tw, (ly0 + ly1) / 2 / th, (lx1 - lx0) / tw, (ly1 - ly0) / th], axis=1
                    ).astype(np.float32)
                    label["cls"] = np.asarray(label["cls"])[keep]
                else:
                    label["bboxes"] = np.empty((0, 4), dtype=np.float32)
                    label["cls"] = np.empty((0, 1), dtype=np.float32)
            else:
                label["bboxes"] = np.empty((0, 4), dtype=np.float32)
            label["bbox_format"] = "xywh"
            label["normalized"] = True
            label["segments"] = []
            if label.get("keypoints") is not None:
                label["keypoints"] = np.empty((0, 0, 3), dtype=np.float32)
        label["img"] = sub
        # Convert to the "instances" structure expected by the val transforms chain (LetterBox+Format).
        label = self.base.update_labels_info(label) if hasattr(self.base, "update_labels_info") else label
        label["ori_shape"] = (th, tw)  # the sub-image is the new "original" for downstream transforms
        label["resized_shape"] = sub.shape[:2]
        label["ratio_pad"] = (1.0, 1.0)
        label["val_slice_meta"] = {
            "orig_idx": oi,
            "k": k,
            "offset": (x0, y0),
            "tile_shape": (th, tw),
            "orig_shape": (h, w),
            "n_tiles": len(self._metas[oi]),
            "sliced": bool(sliced),
        }
        return self.transforms(label)
