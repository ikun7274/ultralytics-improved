"""
扫描指定目录下的图片(不递归子目录), 打乱顺序后按 7:2:1 划分, 写入 train.txt、val.txt、test.txt。

三个列表文件保存在「图片目录的上一级」目录中(与图片文件夹同级)。

每行仅写入图片文件名(含扩展名), 不包含任何路径。

打乱顺序使用随机种子(默认见 DEFAULT_SHUFFLE_SEED, 可通过 --seed 覆盖), 相同种子下划分结果可复现。

比例按整数分块保证条数之和等于图片总数: train = n*7//10, val = n*2//10, test 为剩余。
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

DEFAULT_SHUFFLE_SEED = 42

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}


def list_images(folder: Path) -> list[Path]:
    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def split_counts(n: int, r_train: int, r_val: int, r_test: int) -> tuple[int, int, int]:
    s = r_train + r_val + r_test
    nt = n * r_train // s
    nv = n * r_val // s
    nte = n - nt - nv
    return nt, nv, nte


def write_list(paths: list[Path], out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(''.join(p.name + '\n' for p in paths), encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser(
        description='打乱图片顺序, 按 7:2:1 写入 train.txt / val.txt / test.txt(仅文件名, 保存在图片目录上一级)'
    )
    parser.add_argument(
        '--img_dir',
        type=str,
        default=r"C:\Users\ASUS\Desktop\Datasets\水面漂浮物\base_4_0\images",
        help='存放图片的目录(列表文件将写入该目录的上一级)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=DEFAULT_SHUFFLE_SEED,
        help=f'随机种子(默认 {DEFAULT_SHUFFLE_SEED}, 相同种子下划分顺序一致)',
    )
    args = parser.parse_args()

    img_dir = Path(args.img_dir).resolve()
    if not img_dir.is_dir():
        print(f'错误: 目录不存在或不是文件夹: {img_dir}')
        return 1

    parent = img_dir.parent.resolve()
    paths = list_images(img_dir)
    if not paths:
        print(f'未找到图片: {img_dir}(扩展名 {sorted(IMAGE_EXTENSIONS)})')
        return 0

    rng = random.Random(args.seed)
    rng.shuffle(paths)

    nt, nv, _ = split_counts(len(paths), 7, 2, 1)
    all_paths = paths[:]
    train_paths = paths[:nt]
    val_paths = paths[nt : nt + nv]
    test_paths = paths[nt + nv :]

    for name, subset in (
        ('all.txt', all_paths),
        ('train.txt', train_paths),
        ('val.txt', val_paths),
        ('test.txt', test_paths),
    ):
        write_list(subset, parent / name)

    print(f'上级目录: {parent}')
    print(f'图片目录: {img_dir}(共 {len(paths)} 张, seed={args.seed})')
    print(f'train: {len(train_paths)}, val: {len(val_paths)}, test: {len(test_paths)}, all: {len(all_paths)}')
    print(f'已写入: {parent / "train.txt"}, {parent / "val.txt"}, {parent / "test.txt"}, {parent / "all.txt"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
