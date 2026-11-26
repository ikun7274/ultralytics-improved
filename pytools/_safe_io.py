"""pytools 共享的路径安全工具：删目录护栏 + 目录参数校验。.

背景
----
本目录多个数据集脚本都带「输出目录已存在就整目录删除重建」（``--overwrite``）语义，
最初各自实现了一份 ``safe_rmtree``，护栏只有「磁盘根 / 用户 home」两条。于是下面这条
路径可以静默删掉整个数据集：

    cd D:/datasets/base_0_0                                  # 数据集根，含 images/ labels/
    python pytools/compose_slice_dataset_auto_improved.py     # 忘了传路径

原因是 ``--input_dir/--output_dir`` 默认 ``""``，而 ``Path("") == Path(".")`` 且
``Path("").resolve()`` 就是 cwd —— 脚本前面所有 ``input_path.exists()`` / ``images/``
检查都会通过，接着 ``shutil.rmtree(Path("."))`` 直接把当前工作目录删掉（不进回收站）。

本模块把护栏收敛到一处：

1. :func:`safe_rmtree` / :func:`assert_deletable` 拒绝：磁盘根、用户 home、cwd、
   cwd 的任一祖先、本仓库根目录；
2. :func:`validate_io_dirs` 要求输入/输出目录**必须显式给出且非空**，输出目录不得指向
   cwd、不得等于输入目录，否则在删除动作发生之前就报错退出。

新增 pytools 脚本请直接复用这两个函数，不要再各自拷一份弱化版本。
"""

from __future__ import annotations

import shutil
from pathlib import Path

__all__ = [
    "UnsafePathError",
    "assert_deletable",
    "require_nonempty",
    "safe_rmtree",
    "validate_io_dirs",
]

# 本仓库根目录（pytools/ 的上一级）：禁止通过本模块删除它
_REPO_ROOT = Path(__file__).resolve().parent.parent


class UnsafePathError(RuntimeError):
    """待删除路径命中安全护栏。."""


def _iter_forbidden() -> dict[Path, str]:
    """返回 {禁止路径: 原因}。cwd 可能在运行期变化，故每次调用重新计算。."""
    forbidden: dict[Path, str] = {}

    def add(path, reason: str) -> None:
        try:
            resolved = Path(path).resolve()
        except OSError:  # 极端情况：盘符不存在等，跳过该条
            return
        forbidden.setdefault(resolved, reason)

    cwd = Path.cwd().resolve()
    add(Path(cwd.anchor), "磁盘根目录")
    add(Path.home(), "用户 home 目录")
    add(cwd, "当前工作目录")
    add(_REPO_ROOT, "本仓库根目录")
    for parent in cwd.parents:  # 删掉 cwd 的祖先，等效于删掉 cwd
        add(parent, "当前工作目录的上级目录")
    return forbidden


def assert_deletable(path, description="目录") -> Path:
    """若 ``path`` 命中护栏则抛 :class:`UnsafePathError`，否则返回其绝对路径。.

    注意：本函数只做安全检查，不关心路径是否存在（存在性由调用方决定）， 因此也能用于「删除前尚未创建」的输出目录校验。
    """
    try:
        resolved = Path(path).expanduser().resolve()
    except OSError:
        resolved = Path(path).expanduser().absolute()
    for bad, reason in _iter_forbidden().items():
        if resolved == bad:
            raise UnsafePathError(
                f"拒绝操作{description}：{resolved}\n"
                f"  原因：该路径是{reason}，继续执行会破坏工作区或数据集。\n"
                f"  请显式指定一个位于数据集目录下的子目录。"
            )
    return resolved


def require_nonempty(values: dict) -> None:
    """给定「参数名 -> 值」必须全部非空（非 ``None`` 且非空白串），否则 :class:`SystemExit`。.

    专治 ``Path("")``：它看着像「没传」，实际等价于 ``Path(".")``，即当前工作目录。
    """
    blank = [name for name, value in values.items() if value is None or not str(value).strip()]
    if blank:
        raise SystemExit(
            "错误：以下参数必须显式提供且不能为空（空值会退化为当前工作目录，可能导致数据集被删除）：\n  - "
            + "\n  - ".join(blank)
        )


def safe_rmtree(path, description="目录") -> bool:
    """带护栏地删除目录；目录不存在时直接返回 ``False``。."""
    resolved = assert_deletable(path, description)
    if not resolved.exists():
        return False
    try:
        top_level = sum(1 for _ in resolved.iterdir())
        detail = f"顶层 {top_level} 项"
    except OSError:
        detail = "无法统计条目数"
    print(f"将删除{description}：{resolved}（{detail}）")
    shutil.rmtree(resolved)
    return True


def validate_io_dirs(
    input_dir,
    output_dir,
    input_description="输入目录（--input_dir）",
    output_description="输出目录（--output_dir）",
) -> tuple[Path, Path]:
    """校验一对输入/输出目录，返回 ``(input_path, output_path)`` 两个绝对路径。.

    规则：

    1. 两者都必须显式给出且非空（见 :func:`require_nonempty`）；
    2. 输出目录不得是 cwd / 磁盘根 / home / 仓库根 / cwd 的祖先（见 :func:`assert_deletable`）；
    3. 输入目录必须存在；
    4. 输出目录不得与输入目录相同 —— 否则「先删输出」等于删掉原始数据；
    5. 输出目录位于输入目录内部时打印警告（语义可疑，但不立即删数据）。
    """
    require_nonempty({input_description: input_dir, output_description: output_dir})
    src = Path(str(input_dir)).expanduser().resolve()
    dst = assert_deletable(output_dir, description=output_description)
    if not src.exists():
        raise FileNotFoundError(f"{input_description}不存在：{src}")
    if src == dst:
        raise ValueError(f"{input_description}与{output_description}相同：{src}，继续执行会删除原始数据！")
    if src in dst.parents:
        print(f"警告：{output_description} {dst} 位于 {input_description} {src} 内部，请确认路径配置。")
    return src, dst
