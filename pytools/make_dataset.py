"""
合并多个子版本目录下的 train/val/test 列表, 并生成主目录下的 data.yaml。

假定主目录为 ROOT, 子版本为 SUB(ROOT 下的子文件夹), 每个 SUB 内含：
  train.txt, val.txt, test.txt, all.txt(每行为图片文件名, 可含路径, 脚本会取 basename)
  以及 images/ 存放对应图片。

合并规则：对每个子目录中列表里的每一行, 生成路径
  ROOT / SUB / images / 文件名
  并写入 ROOT/train.txt、ROOT/val.txt、ROOT/test.txt、ROOT/all.txt(全路径, POSIX 风格斜杠)。

未指定 --root 时, 通过 pwd 命令获取当前工作目录作为主目录；指定 --root 则使用给定路径。

类别从 ROOT/classes.txt 读取(每行一个类名, 与 auto_format_convert.load_classes 一致), 
参照 datasets/data.yaml 的结构生成 ROOT/data.yaml(键为 train / val / test, nc, names)。

用法:
    python make_dataset.py --subdirs set_a set_b set_c
    python make_dataset.py --root D:/mydata --subdirs set_a set_b set_c

任一子目录缺少 train.txt / val.txt / test.txt / all.txt 之一即报错退出(不生成主目录下的合并文件)。
某个列表 txt 若存在但无任何有效图片行(空文件或仅空白行), 则跳过该子目录在该划分下的贡献, 并打印提示。
若某划分合并后仍无任何条目, 打印警告但仍照常写入 train/val/test/all 列表与 yaml。

在所有校验通过之后、合并写入之前：若主目录下已存在 train.txt、val.txt、test.txt、all.txt
或 --yaml_name 指定文件(默认 data.yaml)中的任意一个, 会先删除这些已存在文件再重新生成。

"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

SPLIT_FILES = ('train.txt', 'val.txt', 'test.txt', 'all.txt')


def path_line_for_txt(p: Path) -> str:
    """列表 txt 中行格式：绝对路径, 正斜杠。"""
    return str(p.resolve()).replace('\\', '/')


def load_classes(classes_path: Path) -> list[str]:
    with open(classes_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def read_name_lines(list_file: Path) -> list[str]:
    out: list[str] = []
    for line in list_file.read_text(encoding='utf-8').splitlines():
        s = line.strip()
        if not s:
            continue
        out.append(Path(s).name)
    return out


def get_root_from_pwd() -> Path:
    """通过 pwd 命令获取当前工作目录；失败时回退为 Python 的 cwd。"""
    try:
        proc = subprocess.run(
            'pwd',
            capture_output=True,
            text=True,
            check=True,
            shell=True,
        )
        for line in proc.stdout.splitlines():
            p = line.strip()
            if p:
                return Path(p).resolve()
    except (subprocess.CalledProcessError, OSError):
        pass
    return Path.cwd().resolve()


def resolve_root(root_arg: str | None) -> Path:
    if root_arg is not None:
        return Path(root_arg).resolve()
    root = get_root_from_pwd()
    print(f'未指定 --root, 使用 pwd 获取主目录: {root}')
    return root


def check_subdir_split_lists(root: Path, subdirs: list[str]) -> int:
    """子目录内必须同时具备 train/val/test/all 列表文件；否则返回 1。"""
    for sd in subdirs:
        sub_root = (root / sd).resolve()
        for split in SPLIT_FILES:
            p = sub_root / split
            if not p.is_file():
                print(f'错误: 缺少 {p}')
                return 1
    return 0


def remove_main_outputs_if_exist(root: Path, yaml_path: Path) -> None:
    """删除主目录下本次会重写的合并产物(仅删除实际存在的文件)。"""
    targets: list[Path] = [root / name for name in SPLIT_FILES] + [yaml_path.resolve()]
    # yaml_path 可能与 root/SPLIT 中某名重复, 去重避免 unlink 两次
    seen: set[Path] = set()
    to_remove: list[Path] = []
    for p in targets:
        pr = p.resolve()
        if pr in seen:
            continue
        seen.add(pr)
        if pr.is_file():
            to_remove.append(pr)
    if not to_remove:
        return
    print('  主目录下已有 train/val/test/all 或 yaml, 先删除再生成:')
    for p in to_remove:
        p.unlink()
        print(f'    已删除: {p.name}')


def merge_paths_for_split(
    root: Path,
    subdirs: list[str],
    split_name: str,
) -> list[str]:
    """split_name: train.txt | val.txt | test.txt | all.txt(子目录列表文件须已校验存在)"""
    merged: list[str] = []
    for sd in subdirs:
        sub_root = (root / sd).resolve()
        list_path = sub_root / split_name
        names = read_name_lines(list_path)
        if not names:
            print(f'  跳过(列表为空或无有效行): {list_path}')
            continue

        images_dir = sub_root / 'images'
        for name in names:
            full = (images_dir / name).resolve()
            merged.append(path_line_for_txt(full))
    return merged


def write_lines(path: Path, lines: list[str]) -> None:
    path.write_text(''.join(s + '\n' for s in lines), encoding='utf-8')


def yaml_escape_single_quoted(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def build_data_yaml(
    root: Path,
    classes: list[str],
) -> str:
    train_p = path_line_for_txt(root / 'train.txt')
    val_p = path_line_for_txt(root / 'val.txt')
    test_p = path_line_for_txt(root / 'test.txt')
    lines: list[str] = [
        f'train: {train_p}',
        f'val: {val_p}',
        f'test: {test_p}',
        '',
        '# number of classes',
        f'nc: {len(classes)}',
        '',
        '# class names',
        'names: [',
    ]
    for i, name in enumerate(classes):
        sep = ',' if i < len(classes) - 1 else ''
        lines.append(f'   {yaml_escape_single_quoted(name)}{sep}')
    lines.append(']')
    lines.append('')
    return '\n'.join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='合并子目录 train/val/test/all 列表并生成主目录 data.yaml(类别来自 classes.txt)'
    )
    parser.add_argument(
        '--root',
        type=str,
        default=r'C:\Users\ASUS\Desktop\Datasets\水面漂浮物',
        help='数据集主目录；未指定时通过 pwd 获取当前工作目录',
    )
    parser.add_argument(
        '--subdirs',
        nargs='+',
        default=["base_0_0", "base_1_0", "base_2_0", "base_3_0", "base_3_1", "base_4_0", "base_4_1"],
        help='要扫描的子版本目录名(主目录下的直接子文件夹, 多个用空格分隔)',
    )
    parser.add_argument(
        '--yaml_name',
        type=str,
        default='data.yaml',
        help='生成的 YAML 文件名(默认 data.yaml, 保存在主目录下)',
    )
    args = parser.parse_args()

    root = resolve_root(args.root)
    if not root.is_dir():
        print(f'错误: 主目录不存在: {root}')
        return 1

    classes_path = root / 'classes.txt'
    if not classes_path.is_file():
        print(f'错误: 缺少类别文件: {classes_path}')
        return 1
    classes = load_classes(classes_path)
    if not classes:
        print(f'错误: {classes_path} 无有效类别行')
        return 1

    for sd in args.subdirs:
        sub = root / sd
        if not sub.is_dir():
            print(f'错误: 子目录不存在: {sub}')
            return 1

    if check_subdir_split_lists(root, args.subdirs):
        return 1

    yaml_path = root / args.yaml_name
    remove_main_outputs_if_exist(root, yaml_path)

    print(f'主目录: {root}')
    print(f'子目录: {args.subdirs}')

    train_m = merge_paths_for_split(root, args.subdirs, 'train.txt')
    val_m = merge_paths_for_split(root, args.subdirs, 'val.txt')
    test_m = merge_paths_for_split(root, args.subdirs, 'test.txt')
    all_m = merge_paths_for_split(root, args.subdirs, 'all.txt')

    merged_counts = {
        'train': train_m,
        'val': val_m,
        'test': test_m,
        'all': all_m,
    }
    empty_splits = [name for name, lines in merged_counts.items() if not lines]
    if empty_splits:
        print(
            f'警告: 合并后 {" / ".join(empty_splits)} 无条目(可能各子目录对应列表均为空), 仍将写入空列表文件'
        )

    write_lines(root / 'train.txt', train_m)
    write_lines(root / 'val.txt', val_m)
    write_lines(root / 'test.txt', test_m)
    write_lines(root / 'all.txt', all_m)

    yaml_path.write_text(build_data_yaml(root, classes), encoding='utf-8')

    print(
        f'合并 train/val/test/all 行数: {len(train_m)} / {len(val_m)} / {len(test_m)} / {len(all_m)}'
    )
    print(f'类别数 nc={len(classes)}')
    print(
        f'已写入: {root / "train.txt"}, {root / "val.txt"}, '
        f'{root / "test.txt"}, {root / "all.txt"}'
    )
    print(f'已写入: {yaml_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
