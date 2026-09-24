# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Save-path helpers shared by the online-augmentation branches.

Split out of ``base.py`` (which had grown to ~2000 lines mixing seven unrelated responsibilities) so
that both ``base.py`` and ``augment.py`` can reuse the same directory / write handling without an
import cycle. Pure utilities: no dataset state, no ultralytics data imports.
"""

from __future__ import annotations

from pathlib import Path

from ultralytics.utils import LOGGER
from ultralytics.utils.patches import imwrite

# Directories already created in this process (see _ensure_dir).
_MKDIR_DONE: set[str] = set()


def _ensure_dir(path) -> None:
    """Create ``path`` (with parents) once per process; later calls are a no-op.

    The save branches used to call ``mkdir(parents=True, exist_ok=True)`` before *every* write, i.e. once per sample per
    worker -- a pointless syscall storm on spinning disks, and a lock convoy on network storage when 8 workers race on
    the same directory.
    """
    key = str(path)
    if key in _MKDIR_DONE:
        return
    Path(path).mkdir(parents=True, exist_ok=True)
    _MKDIR_DONE.add(key)


def _forget_dir(path) -> None:
    """Drop ``path`` from the per-process "already created" set so the next write retries ``mkdir``.

    Without this the cache never expires: if the directory is removed mid-run (cleanup script, flaky network share)
    every later write kept failing against a path nobody re-created.
    """
    _MKDIR_DONE.discard(str(path))


def _imwrite(path, img) -> bool:
    """Write an image using Ultralytics' unicode-safe ``imwrite`` and warn when it fails.

    OpenCV's own ``cv2.imwrite`` silently fails on non-ASCII paths -- verified in this very project, where it *returned
    True* while the file never appeared on disk. Never ignore the return value.
    """
    ok = bool(imwrite(str(path), img))
    if not ok:
        LOGGER.warning(f"{img.shape} save failed: imwrite returned False for '{path}' (check path/permissions).")
        # Allow the next attempt to re-create the directory: the failure is often a vanishing/roaming path.
        _forget_dir(Path(path).parent)
    return ok


def _save_cap(dataset, branch: str) -> int:
    """Return the per-branch save cap.

    Each branch (tile/ratio/blur/compose/weather/occlusion) would otherwise share the single legacy ``slice_save_max``
    budget, so enabling one branch's saving would silently cap the others to the same total. ``slice_save_max_<branch>``
    -- when set -- overrides the global cap; a missing or ``None`` attribute falls back to ``slice_save_max``. ``0``
    means unlimited in either case.

    Args:
        dataset: dataset instance to read the cap attributes from.
        branch (str): branch name, i.e. the suffix of ``slice_save_max_<branch>``.
    """
    val = getattr(dataset, f"slice_save_max_{branch}", None)
    if val is None:
        val = getattr(dataset, "slice_save_max", 0) or 0
    return int(val)
