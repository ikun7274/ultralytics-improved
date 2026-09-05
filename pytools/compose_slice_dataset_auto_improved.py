import os
import argparse
import shutil
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import math

# P1-10: Pillow >= 10 移除了 Image.LANCZOS, 改用 Image.Resampling.LANCZOS (旧版回退)
_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS", getattr(Image, "LANCZOS", None))
if _LANCZOS is None:
    raise RuntimeError("Pillow 不再提供 LANCZOS resampling filter; 请升级 Pillow 或回退低层调用.")


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


def stitch_group(img_paths, label_paths, output_img_path, output_txt_path):
    """
    拼接一组图片（数量任意，但为了网格整齐，通常为 group_size），按顺序从左到右、从上到下排列。
    自动计算行列。
    """
    n = len(img_paths)
    if n == 0:
        return
    # 自动计算行列：尽量接近方形
    rows = int(math.sqrt(n))
    cols = (n + rows - 1) // rows
    while rows * cols < n:
        rows += 1
    # P1-10: 用 with 上下文读图, 防止 Windows 下文件句柄泄漏 (Image.open 不会自己 close)
    imgs: list[Image.Image] = []
    try:
        for p in img_paths:
            with Image.open(p) as im:
                im.load()
                imgs.append(im.copy())  # 拷贝一份, 让 .open 释放后仍可用
    except Exception:
        for im in imgs:
            try:
                im.close()
            except Exception:
                pass
        raise
    widths = [img.width for img in imgs]
    heights = [img.height for img in imgs]
    # 统一尺寸（取最大值）
    W_sub = max(widths)
    H_sub = max(heights)
    if len(set(widths)) != 1 or len(set(heights)) != 1:
        print(f"  子图尺寸不一致，统一缩放至 {W_sub}x{H_sub}")
        imgs = [img.resize((W_sub, H_sub), _LANCZOS) for img in imgs]

    W_big = W_sub * cols
    H_big = H_sub * rows
    # P1-10: 灰度/RGBA 图也能合成 — 用 'RGBA' / 'LA' 模式, 但统一可视化仍按 RGB
    base_mode = imgs[0].mode
    if base_mode in ("L", "1"):  # 灰度
        big_img = Image.new("L", (W_big, H_big))
    elif base_mode in ("LA", "RGBA", "P"):  # 含 alpha
        big_img = Image.new("RGBA", (W_big, H_big))
    else:
        big_img = Image.new("RGB", (W_big, H_big))

    for idx, img in enumerate(imgs):
        row = idx // cols
        col = idx % cols
        x0 = col * W_sub
        y0 = row * H_sub
        # 把子图统一转成与大画布相同的 mode 再贴; 转换模式时默认按白色做 alpha 复合
        if img.mode != big_img.mode:
            if big_img.mode in ("RGB", "L") and img.mode in ("RGBA", "LA"):
                bg = Image.new(big_img.mode, img.size, "white" if big_img.mode == "RGB" else 255)
                bg.paste(img, mask=img.getchannel("A") if "A" in img.mode else None)
                img_to_paste = bg
            elif big_img.mode == "RGBA" and img.mode in ("RGB", "L"):
                img_to_paste = img.convert("RGBA")
            else:
                img_to_paste = img.convert(big_img.mode)
        else:
            img_to_paste = img
        big_img.paste(img_to_paste, (x0, y0))
    if big_img.mode not in ("RGB", "L"):
        big_img = big_img.convert("RGB")
    big_img.save(output_img_path)

    # 处理标签
    all_anns = []
    for idx, (txt_path, img) in enumerate(zip(label_paths, imgs)):
        if not Path(txt_path).exists():
            continue
        anns = parse_yolo_label(txt_path)
        if not anns:
            continue
        row = idx // cols
        col = idx % cols
        dx = col * W_sub
        dy = row * H_sub
        for cls_id, xc, yc, w_norm, h_norm in anns:
            x_pix = xc * W_sub + dx
            y_pix = yc * H_sub + dy
            w_pix = w_norm * W_sub
            h_pix = h_norm * H_sub
            xc_new = x_pix / W_big
            yc_new = y_pix / H_big
            w_new = w_pix / W_big
            h_new = h_pix / H_big
            xc_new = max(0.0, min(1.0, xc_new))
            yc_new = max(0.0, min(1.0, yc_new))
            w_new = max(0.0, min(1.0 - xc_new, w_new))
            h_new = max(0.0, min(1.0 - yc_new, h_new))
            if w_new > 0 and h_new > 0:
                all_anns.append((cls_id, xc_new, yc_new, w_new, h_new))
    write_yolo_label(output_txt_path, all_anns)


def get_image_groups(img_dir, labels_dir, group_size=4, random_fill=False):
    """
    读取 img_dir 下所有图片文件，按文件名排序，每 group_size 个一组。
    如果最后一组不足 group_size 且 random_fill=True，则从所有图片中随机选择（可重复）补足。
    返回列表，每个元素为 (group_id, [img_paths], [label_paths])
    """
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    img_files = [f for f in Path(img_dir).iterdir() if f.suffix.lower() in img_exts]
    img_files.sort(key=lambda x: x.name)
    total = len(img_files)
    if total == 0:
        return []
    groups = []
    # 确定要分多少组
    num_groups = (total + group_size - 1) // group_size
    for gid in range(num_groups):
        start = gid * group_size
        end = min(start + group_size, total)
        group_imgs = img_files[start:end]
        # 如果不足 group_size 且允许补全
        if len(group_imgs) < group_size and random_fill:
            # 从所有图片中随机选择（可重复）补足
            needed = group_size - len(group_imgs)
            # 如果总图片数大于0，从 img_files 中随机抽取（可重复）
            fill_imgs = random.choices(img_files, k=needed)
            group_imgs.extend(fill_imgs)
        # 对应的标签文件
        group_labels = []
        for img_p in group_imgs:
            lbl_p = Path(labels_dir) / (img_p.stem + '.txt') if labels_dir else None
            group_labels.append(lbl_p)
        groups.append((gid, group_imgs, group_labels))
    return groups


def main():
    parser = argparse.ArgumentParser(
        description="按顺序将 images 目录中的图片每 group_size 张合成一组（网格），最后一组不足时可选随机补全"
    )
    parser.add_argument("--input_dir", type=str, default="",
                        help="输入目录，应包含 images/ 和 labels/ 子目录（切片后数据; 必传; 例: D:/datasets/base_0_0）")
    parser.add_argument("--output_dir", type=str, default="",
                        help="输出目录，将自动创建 images/ 和 labels/ 存放合成结果（必传; 例: D:/datasets/base_0_2）")
    parser.add_argument("--img_suffix", type=str, default=".jpg",
                        help="输出图片扩展名")
    parser.add_argument("--overwrite", action="store_true", default=True,
                        help="覆盖已有输出目录")
    parser.add_argument("--group_size", type=int, default=4,
                        help="每组图片数，默认为4（2x2网格）")
    parser.add_argument("--start_index", type=int, default=0,
                        help="输出图片名称的起始编号")
    parser.add_argument("--prefix", type=str, default="base_0_2",
                        help="输出图片和标签的前缀名称，例如 prefix='myimg' 则输出 myimg_0000.jpg")
    parser.add_argument("--random_fill", action="store_true", default=True,
                        help="当最后一组图片不足 group_size 时，从所有图片中随机选择补全")
    parser.add_argument("--seed", type=int, default=0,
                        help="随机种子，用于可复现的补全")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    input_path = Path(args.input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_path}")

    img_dir = input_path / "images"
    label_dir = input_path / "labels"
    if not img_dir.exists():
        raise FileNotFoundError(f"未找到 images 子目录: {img_dir}")
    if not label_dir.exists():
        print("警告：未找到 labels 子目录，将跳过标签处理")

    output_path = Path(args.output_dir)
    if output_path.exists():
        if args.overwrite:
            safe_rmtree(output_path, description="输出目录")
        else:
            raise FileExistsError(f"输出目录已存在，请使用 --overwrite 覆盖")

    out_img_dir = output_path / "images"
    out_label_dir = output_path / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    groups = get_image_groups(img_dir, label_dir, args.group_size, args.random_fill)
    if not groups:
        print("未找到任何图片")
        return

    print(f"找到 {len(groups)} 个组，开始合成...")
    for idx, (gid, img_paths, label_paths) in enumerate(tqdm(groups, desc="合成进度")):
        # 输出文件名：prefix_序号
        output_img = out_img_dir / f"{args.prefix}_{args.start_index + idx:04d}{args.img_suffix}"
        output_txt = out_label_dir / f"{args.prefix}_{args.start_index + idx:04d}.txt"
        try:
            stitch_group(img_paths, label_paths, output_img, output_txt)
        except Exception as e:
            print(f"合成组 {gid} 失败：{e}")

    print(f"\n全部合成完成！结果保存在：{output_path}")
    print(f"共生成 {len(groups)} 张合成图片")


if __name__ == "__main__":
    main()