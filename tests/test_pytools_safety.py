"""pytools 删除护栏的回归测试（对应复审报告 S-1）。.

核心断言：站在「数据集根目录」里不带参数运行这两个脚本，**不能**删掉这个目录。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PYTOOLS = REPO / "pytools"
COMPOSE = PYTOOLS / "compose_slice_dataset_auto_improved.py"
RESIZE = PYTOOLS / "change_image_resolution_slice_dataset_auto_improved.py"

sys.path.insert(0, str(PYTOOLS))
import _safe_io  # noqa: E402

SENTINEL = "keep_me.txt"


def _make_fake_dataset(root: Path, n: int = 2) -> Path:
    """造一个最小的 YOLO 数据集：images/ + labels/ + 一个哨兵文件。."""
    from PIL import Image

    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", (64, 48), (20 * i % 256, 60, 90)).save(root / "images" / f"img_{i}.jpg")
        (root / "labels" / f"img_{i}.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    (root / SENTINEL).write_text("sentinel", encoding="utf-8")
    return root


def _run(script: Path, cwd: Path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )


def _assert_dataset_intact(ds: Path) -> None:
    assert (ds / "images" / "img_0.jpg").exists(), f"{ds}/images 被破坏"
    assert (ds / SENTINEL).exists(), f"{ds} 被整体删除"


# ---------------------------------------------------------------- 护栏单元测试


def test_safe_rmtree_refuses_cwd(tmp_path, monkeypatch):
    """空串 / '.' 都退化成 cwd，必须被拒绝且不产生任何删除。."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / SENTINEL).write_text("x", encoding="utf-8")
    for candidate in ("", ".", "./"):
        with pytest.raises(_safe_io.UnsafePathError):
            _safe_io.safe_rmtree(candidate, description="输出目录")
    assert (tmp_path / SENTINEL).exists()


def test_safe_rmtree_refuses_cwd_ancestor_and_repo_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(_safe_io.UnsafePathError):
        _safe_io.safe_rmtree(tmp_path.parent, description="上级目录")
    with pytest.raises(_safe_io.UnsafePathError):
        _safe_io.safe_rmtree(REPO, description="仓库根目录")


def test_safe_rmtree_still_deletes_normal_subdir(tmp_path):
    """正常子目录必须仍可删除，否则脚本就没法用了（护栏不能过头）。."""
    target = tmp_path / "out"
    (target / "images").mkdir(parents=True)
    (target / "images" / "a.jpg").write_bytes(b"x")
    assert _safe_io.safe_rmtree(target, description="输出目录") is True
    assert not target.exists()
    assert _safe_io.safe_rmtree(target, description="输出目录") is False


def test_require_nonempty_rejects_blank():
    with pytest.raises(SystemExit):
        _safe_io.require_nonempty({"--input_dir": ""})
    with pytest.raises(SystemExit):
        _safe_io.require_nonempty({"--output_dir": None})
    _safe_io.require_nonempty({"--input_dir": "D:/datasets/base_0_0"})


def test_validate_io_dirs_rejects_same_and_cwd(tmp_path, monkeypatch):
    ds = _make_fake_dataset(tmp_path / "base_0_0")
    with pytest.raises(SystemExit):
        _safe_io.validate_io_dirs("", str(tmp_path / "out"))
    with pytest.raises(ValueError):
        _safe_io.validate_io_dirs(str(ds), str(ds))
    monkeypatch.chdir(ds)
    with pytest.raises(_safe_io.UnsafePathError):
        _safe_io.validate_io_dirs(str(ds), ".")


# ------------------------------------------------------- S-1 复现场景（子进程）


@pytest.mark.parametrize("script", [COMPOSE, RESIZE])
def test_no_arg_run_in_dataset_root_keeps_cwd(tmp_path, script):
    """S-1 原始触发路径：cd 到数据集根目录、不带参数运行 → 目录必须完好。."""
    ds = _make_fake_dataset(tmp_path / "base_0_0")
    proc = _run(script, ds)
    assert proc.returncode != 0, f"{script.name} 竟然成功退出了"
    assert "--input_dir" in (proc.stderr + proc.stdout) or "required" in (proc.stderr + proc.stdout)
    _assert_dataset_intact(ds)


@pytest.mark.parametrize("script", [COMPOSE, RESIZE])
def test_output_dir_pointing_at_cwd_is_refused(tmp_path, script):
    """显式 --output_dir . 也必须被挡下（防止 "存在即删除" 删掉 cwd）。."""
    ds = _make_fake_dataset(tmp_path / "base_0_0")
    proc = _run(script, ds, "--input_dir", str(ds), "--output_dir", ".")
    assert proc.returncode != 0
    assert "UnsafePathError" in (proc.stderr + proc.stdout)
    _assert_dataset_intact(ds)


# ------------------------------------------------------------------ 正常流程


def test_compose_end_to_end_and_overwrite_gate(tmp_path):
    src = _make_fake_dataset(tmp_path / "base_0_0", n=4)
    out = tmp_path / "base_0_2"

    first = _run(COMPOSE, tmp_path, "--input_dir", str(src), "--output_dir", str(out))
    assert first.returncode == 0, first.stderr
    produced = sorted(p.name for p in (out / "images").glob("*.jpg"))
    assert produced, "没有生成任何合成图"

    # 输出已存在且未加 --overwrite：必须报错而不是静默删除
    second = _run(COMPOSE, tmp_path, "--input_dir", str(src), "--output_dir", str(out))
    assert second.returncode != 0
    assert "FileExistsError" in second.stderr
    assert sorted(p.name for p in (out / "images").glob("*.jpg")) == produced

    # 显式 --overwrite：允许覆盖
    third = _run(COMPOSE, tmp_path, "--input_dir", str(src), "--output_dir", str(out), "--overwrite")
    assert third.returncode == 0, third.stderr
    assert sorted(p.name for p in (out / "images").glob("*.jpg")) == produced


def test_resize_end_to_end(tmp_path):
    src = _make_fake_dataset(tmp_path / "base_0_0", n=2)
    out = tmp_path / "base_0_1"
    proc = _run(RESIZE, tmp_path, "--input_dir", str(src), "--output_dir", str(out))
    assert proc.returncode == 0, proc.stderr
    assert sorted(p.name for p in (out / "images").glob("*.jpg"))
