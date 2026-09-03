import os
import argparse
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import math


def safe_rmtree(path, description="目录"):
    p = Path(path).resolve()
    if p == p.root or p == Path.home().resolve():
        raise RuntimeError(f"拒绝删除 {description}：{p} 疑似系统关键路径")
    if p.exists():
        print(f"将删除 {description}：{p}")
        shutil.rmtree(p)


def parse_yolo_label(txt_path):
    anns = []
    if not Path(txt_path).exists():
        return anns
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])
                anns.append((cls_id, xc, yc, w, h))
            except ValueError:
                continue
    return anns


def write_yolo_label(txt_path, anns):
    with open(txt_path, 'w') as f:
        for cls_id, xc, yc, w, h in anns:
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


TARGET_RATIOS = {"4:3": 4.0 / 3.0, "16:9": 16.0 / 9.0}
PAD_COLORS = {"black": (0, 0, 0), "gray": (114, 114, 114), "white": (255, 255, 255)}
REL_TOL = 0.02


def compute_padding(w, h, target_ratio, auto):
    """计算 padding 参数。返回 (new_w, new_h, pad_left, pad_top) 或 None(无需转换)。

    - ratio < target (偏瘦高): 高度不变, 左右对称加宽
    - ratio > target (偏宽扁): 宽度不变, 上下对称加高
    - 已接近目标比例: 返回 None (直接复制)
    - auto 模式: 4:3 <-> 16:9 双向转换; 其他比例转到离它最近的 4:3 或 16:9
    """
    ratio = w / h
    r43, r169 = 4.0 / 3.0, 16.0 / 9.0
    if auto:
        if math.isclose(ratio, r43, rel_tol=REL_TOL):
            target = r169  # 4:3 -> 16:9
        elif math.isclose(ratio, r169, rel_tol=REL_TOL):
            target = r43  # 16:9 -> 4:3
        else:
            # 其他比例: 转到离它最近的 4:3 或 16:9
            target = r43 if abs(ratio - r43) <= abs(ratio - r169) else r169
    else:
        target = TARGET_RATIOS[target_ratio]
        if math.isclose(ratio, target, rel_tol=REL_TOL):
            return None  # 已接近目标比例, 无需转换

    if ratio < target:  # 左右加宽
        new_w = max(1, int(round(h * target)))
        new_h = h
        pad_left = (new_w - w) // 2
        pad_top = 0
    else:  # 上下加高
        new_w = w
        new_h = max(1, int(round(w / target)))
        pad_left = 0
        pad_top = (new_h - h) // 2
    return new_w, new_h, pad_left, pad_top


def remap_labels(anns, w, h, new_w, new_h, pad_left, pad_top):
    """加黑边后重映射归一化坐标, 边界 clamp, 丢弃无效框。"""
    new_anns = []
    for cls_id, xc, yc, w_norm, h_norm in anns:
        # 原图绝对坐标
        abs_x = xc * w
        abs_y = yc * h
        abs_w = w_norm * w
        abs_h = h_norm * h
        # 新图上偏移
        new_abs_x = abs_x + pad_left
        new_abs_y = abs_y + pad_top
        # 新归一化坐标
        new_xc = new_abs_x / new_w
        new_yc = new_abs_y / new_h
        new_w_norm = abs_w / new_w
        new_h_norm = abs_h / new_h
        # 裁剪边界
        new_xc = max(0.0, min(1.0, new_xc))
        new_yc = max(0.0, min(1.0, new_yc))
        new_w_norm = max(0.0, min(1.0 - new_xc, new_w_norm))
        new_h_norm = max(0.0, min(1.0 - new_yc, new_h_norm))
        if new_w_norm > 0 and new_h_norm > 0:
            new_anns.append((cls_id, new_xc, new_yc, new_w_norm, new_h_norm))
    return new_anns


def process_image(img_path, label_path, out_img_dir, out_label_dir,
                  target_ratio="auto", pad_color="black", max_side=0, stats=None):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size

    pad = compute_padding(w, h, target_ratio, auto=(target_ratio == "auto"))
    if pad is None:
        # 无需比例转换: 直接复制, 但仍受 max_side 限制 (等比缩放, 归一化标签不变)
        if max_side > 0 and max(w, h) > max_side:
            scale = max_side / max(w, h)
            nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            img.resize((nw, nh), Image.LANCZOS).save(out_img_dir / img_path.name)
        else:
            shutil.copy(img_path, out_img_dir / img_path.name)
        if label_path and Path(label_path).exists():
            shutil.copy(label_path, out_label_dir / label_path.name)
        if stats is not None:
            stats["copied"] += 1
        return

    new_w, new_h, pad_left, pad_top = pad
    new_img = Image.new('RGB', (new_w, new_h), color=PAD_COLORS[pad_color])
    new_img.paste(img, (pad_left, pad_top))

    # 可选: 限制输出最大边长 (等比缩放, 归一化标签不受影响)
    if max_side > 0 and max(new_w, new_h) > max_side:
        scale = max_side / max(new_w, new_h)
        new_w = max(1, int(round(new_w * scale)))
        new_h = max(1, int(round(new_h * scale)))
        new_img = new_img.resize((new_w, new_h), Image.LANCZOS)

    new_img.save(out_img_dir / img_path.name)

    # 标签重映射 (若有)
    if label_path and Path(label_path).exists():
        anns = parse_yolo_label(label_path)
        new_anns = remap_labels(anns, w, h, new_w, new_h, pad_left, pad_top)
        write_yolo_label(out_label_dir / label_path.name, new_anns)
    if stats is not None:
        stats["converted"] += 1


def main():
    parser = argparse.ArgumentParser(
        description="统一图片宽高比(加黑边)并同步转换YOLO标签: auto双向(4:3<->16:9)或指定目标比例统一所有图"
    )
    parser.add_argument("--input_dir", type=str, default=r"C:\Users\ASUS\Desktop\ultralytics-main\test_images",
                        help="输入目录, 应包含 images/ 和 labels/ 子目录")
    parser.add_argument("--output_dir", type=str, default=r"C:\Users\ASUS\Desktop\ultralytics-main\test_images\train_out",
                        help="输出目录, 将自动创建 images/ 和 labels/")
    parser.add_argument("--target_ratio", type=str, default="auto",
                        choices=["auto", "4:3", "16:9"],
                        help="目标比例: auto=仅4:3<->16:9双向(其他复制); 4:3/16:9=所有图统一到该比例")
    parser.add_argument("--pad_color", type=str, default="gray", choices=list(PAD_COLORS.keys()),
                        help="黑边颜色: black/gray/white")
    parser.add_argument("--max_side", type=int, default=0,
                        help="输出图片最大边长(px), 0=不缩放")
    parser.add_argument("--overwrite", action="store_true", default=True,
                        help="覆盖已有输出目录")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_path}")

    img_dir = input_path / "images"
    label_dir = input_path / "labels"
    if not img_dir.exists():
        raise FileNotFoundError(f"未找到 images 子目录: {img_dir}")

    output_path = Path(args.output_dir)
    if output_path.exists():
        if args.overwrite:
            safe_rmtree(output_path, description="输出目录")
        else:
            raise FileExistsError(f"输出目录已存在, 请使用 --overwrite 覆盖")

    out_img_dir = output_path / "images"
    out_label_dir = output_path / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    img_files = [f for f in img_dir.iterdir() if f.suffix.lower() in img_exts]
    if not img_files:
        print("未找到任何图片")
        return

    stats = {"converted": 0, "copied": 0, "failed": 0}
    print(f"找到 {len(img_files)} 张图片, target_ratio={args.target_ratio}, pad_color={args.pad_color}, "
          f"max_side={args.max_side if args.max_side > 0 else '不限'}, 开始处理...")
    for img_path in tqdm(img_files, desc="处理进度"):
        label_path = label_dir / (img_path.stem + '.txt') if label_dir.exists() else None
        try:
            process_image(img_path, label_path, out_img_dir, out_label_dir,
                          target_ratio=args.target_ratio, pad_color=args.pad_color,
                          max_side=args.max_side, stats=stats)
        except Exception as e:
            stats["failed"] += 1
            print(f"处理 {img_path.name} 失败: {e}")

    print(f"\n全部处理完成! 结果保存在: {output_path}")
    print(f"  - images/ : 转换后的图片")
    print(f"  - labels/ : 对应的标签")
    print(f"  统计: 共 {len(img_files)} 张 | 转换 {stats['converted']} | 复制 {stats['copied']} | 失败 {stats['failed']}")


if __name__ == "__main__":
    main()
