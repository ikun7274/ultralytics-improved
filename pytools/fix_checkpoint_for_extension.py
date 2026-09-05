"""
修补 YOLO checkpoint 元数据以延长训练（针对早停/中断导致的提前终止）。

背景：Ultralytics 在训练启动时把 epochs / patience 写入 checkpoint 元数据（train_args 或 args），
resume 时优先读 checkpoint 内存档配置，所以单纯改 epochs 参数无效，必须直接修补 last.pt。

用法（命令行）：
    python fix_checkpoint_for_extension.py --ckpt path/to/last.pt --epochs 400 --patience 20
    python fix_checkpoint_for_extension.py --ckpt last.pt --epochs 400 --dry-run     # 只预览不保存
    python fix_checkpoint_for_extension.py --ckpt last.pt --epochs 400 --save-as last_fixed.pt
"""

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path

import torch


def detect_finished_epoch(ckpt: dict) -> int:
    """从 checkpoint 中检测实际已完成的 epoch 数（兼容 train_args / args 两种 key）。

    - checkpoint 的 'epoch' 字段是 0-indexed，表示"下一轮"的索引：
      中断(手动停止/报错) → epoch >= 0 → 已完成 = epoch + 1
      正常跑完原 epochs  → epoch 缺失或为 -1 → 已完成 = 原 epochs
    """
    original_epochs = None
    for key in ("train_args", "args"):
        args = ckpt.get(key)
        if args is None:
            continue
        original_epochs = args.get("epochs") if isinstance(args, dict) else getattr(args, "epochs", None)
        break

    current_epoch = ckpt.get("epoch")
    if current_epoch is not None and current_epoch >= 0:
        return int(current_epoch) + 1  # 中断
    if original_epochs is not None:
        return int(original_epochs)    # 正常结束
    raise ValueError("无法从 Checkpoint 中检测已完成的 Epoch 数。")


def backup_with_timestamp(src: Path, backup_dir: Path) -> Path:
    """按时间戳备份 checkpoint（多次备份不会互相覆盖），返回备份文件路径。"""
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = backup_dir / f"{src.stem}_{stamp}{src.suffix}"
    shutil.copy2(src, dst)
    return dst


def fix_checkpoint_for_extension(
    ckpt_path,
    new_total_epochs,
    new_patience=20,
    backup=True,
    backup_dir=None,
    save_as=None,
    dry_run=False,
):
    """修补 YOLO checkpoint 元数据以延长训练。

    Args:
        ckpt_path (str | Path): last.pt 的路径。
        new_total_epochs (int): 新的总训练轮数（必须 > 已完成轮数，否则无意义）。
        new_patience (int): 新的早停耐心值。默认 20。
        backup (bool): 保存前是否先做时间戳备份。默认 True。
        backup_dir (str | Path | None): 备份目录；默认与 ckpt 同目录。
        save_as (str | Path | None): 修补结果保存为新文件；为 None 时覆盖原文件。
        dry_run (bool): True 时只打印将执行的修改，不备份、不保存。

    Returns:
        dict: 修补结果摘要 {finished_epoch, new_epochs, new_patience, ckpt_path,
                             backup_path, saved_path}。
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint 不存在: {ckpt_path}")

    # 1) 加载（weights_only=False 才能读取元数据）
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # 2) 检测实际已完成轮数
    finished_epoch = detect_finished_epoch(ckpt)

    # 3) 合法性校验：新的总轮数必须严格大于已完成轮数
    if not isinstance(new_total_epochs, int) or new_total_epochs <= finished_epoch:
        raise ValueError(
            f"new_total_epochs={new_total_epochs} 必须为整数且大于已完成轮数 {finished_epoch}，"
            "否则恢复训练无法继续/会倒退。"
        )

    # 4) 关键：'epoch' 存的是"下一轮"的 0-indexed 索引；
    #    已完成 150 轮则下一轮是 151，索引为 150
    ckpt["epoch"] = finished_epoch - 1

    # 5) 同步更新两种可能的参数 key
    for key in ("train_args", "args"):
        args = ckpt.get(key)
        if args is None:
            continue
        if isinstance(args, dict):
            args["epochs"] = new_total_epochs
            args["patience"] = new_patience
        else:
            setattr(args, "epochs", new_total_epochs)
            setattr(args, "patience", new_patience)

    target = Path(save_as) if save_as else ckpt_path

    print("=" * 56)
    print(f"检测到上次训练结束于 Epoch: {finished_epoch}")
    print(f"将设置: 总 Epochs -> {new_total_epochs}, Patience -> {new_patience}")
    print(f"恢复训练将从 Epoch {finished_epoch + 1} 开始，直至 {new_total_epochs}")
    if save_as:
        print(f"目标输出: {target}")
    else:
        print(f"目标输出: {ckpt_path}（覆盖原文件）")

    if dry_run:
        print("[dry-run] 未备份、未保存，以上仅为将要执行的修改。")
        print("=" * 56)
        return {
            "finished_epoch": finished_epoch,
            "new_epochs": new_total_epochs,
            "new_patience": new_patience,
            "ckpt_path": str(ckpt_path),
            "backup_path": None,
            "saved_path": None,
        }

    # 6) 保存前时间戳备份
    backup_path = None
    if backup:
        bdir = Path(backup_dir) if backup_dir else ckpt_path.parent
        backup_path = backup_with_timestamp(ckpt_path, bdir)
        print(f"已备份原文件 -> {backup_path}")

    # P1-10: 先写到临时文件 + 校验, 再原子替换; 避免断电/磁盘满导致 ckpt 截断损坏
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    torch.save(ckpt, tmp_path)

    # 7) 回读验证 (在临时文件上), 校验后再原子替换
    ckpt_check = torch.load(tmp_path, map_location="cpu", weights_only=False)
    check_epochs = None
    for key in ("train_args", "args"):
        args = ckpt_check.get(key)
        if args is None:
            continue
        check_epochs = args.get("epochs") if isinstance(args, dict) else getattr(args, "epochs", None)
        break
    assert ckpt_check.get("epoch") == finished_epoch - 1, "epoch 索引修补失败"
    assert check_epochs == new_total_epochs, "epochs 元数据修补失败"
    print(f"回读验证通过: epoch 索引={ckpt_check.get('epoch')}, epochs={check_epochs}")

    # 8) 原子替换 (Windows ReplaceFile fallback 走 os.replace 单调用)
    os.replace(tmp_path, target)

    print("=" * 56)

    return {
        "finished_epoch": finished_epoch,
        "new_epochs": new_total_epochs,
        "new_patience": new_patience,
        "ckpt_path": str(ckpt_path),
        "backup_path": str(backup_path) if backup_path else None,
        "saved_path": str(target),
    }


def main():
    parser = argparse.ArgumentParser(
        description="修补 YOLO checkpoint 元数据以延长训练（针对早停/中断提前终止）。"
    )
    parser.add_argument("--ckpt", help="last.pt 的路径（必传; 例: <项目根>/runs/<exp>/weights/last.pt）", default="")
    parser.add_argument("--epochs", type=int, help="新的总训练轮数（必须大于已完成轮数）",default=400)
    parser.add_argument("--patience", type=int, default=20, help="新的早停耐心值（默认 20）")
    parser.add_argument("--no-backup", action="store_true", help="跳过时间戳备份（默认会备份）")
    parser.add_argument("--backup-dir", default=None, help="备份目录（默认与 ckpt 同目录）")
    parser.add_argument("--save-as", default=None, help="保存为新文件，不覆盖原文件")
    parser.add_argument("--dry-run", action="store_true", help="只预览修改，不备份、不保存")
    args = parser.parse_args()

    fix_checkpoint_for_extension(
        ckpt_path=args.ckpt,
        new_total_epochs=args.epochs,
        new_patience=args.patience,
        backup=not args.no_backup,
        backup_dir=args.backup_dir,
        save_as=args.save_as,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
