import cv2
import numpy as np
import argparse
import os
from pathlib import Path


def _safe_label_path(input_path: Path, stem: str) -> Path | None:
    """YOLO 标签查找 (P1-10):
    1. 优先同目录 (与 image 同级)
    2. 回退到同级 labels/ 子目录 (Ultralytics 标准 YOLO 布局: images/<...>/foo.jpg, labels/<...>/foo.txt)
    """
    same_dir = input_path / f"{stem}.txt"
    if same_dir.exists():
        return same_dir
    labels_dir = input_path.parent / "labels" / f"{stem}.txt" if input_path.parent else None
    if labels_dir is not None and labels_dir.exists():
        return labels_dir
    # 兼容: 同级 images/labels (用户传了 images 子目录时)
    images_labels = input_path / "labels" / f"{stem}.txt"
    if images_labels.exists():
        return images_labels
    return None


def _safe_imwrite(path: str | Path, img: np.ndarray) -> bool:
    """Unicode-safe imwrite (P1-10): cv2.imwrite 中文路径会静默返回 False (项目内已验证),
    用 cv2.imencode + .tofile() 绕开. 失败返回 False 但不抛异常."""
    try:
        ok, buf = cv2.imencode(Path(path).suffix or ".jpg", img)
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except Exception:
        return False


def motion_blur_kernel(length, angle):
    """生成线段型运动模糊PSF，并保证线段完整落在卷积核内。"""
    rad = np.deg2rad(angle)
    size = max(3, int(np.ceil(length)) | 1)
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2

    dx = np.cos(rad)
    dy = np.sin(rad)

    length_scaled = length / 2.0
    x1 = center - dx * length_scaled
    y1 = center - dy * length_scaled
    x2 = center + dx * length_scaled
    y2 = center + dy * length_scaled

    cv2.line(
        kernel,
        (int(round(x1)), int(round(y1))),
        (int(round(x2)), int(round(y2))),
        1.0,
        thickness=1,
        lineType=cv2.LINE_AA
    )
    kernel /= kernel.sum()
    return kernel


def apply_motion_blur(img, length=15, angle=30, defocus_sigma=0):
    """对图像应用运动模糊，并可叠加高斯失焦模糊。"""
    kernel = motion_blur_kernel(length, angle)
    blurred = cv2.filter2D(
        img,
        -1,
        kernel,
        borderType=cv2.BORDER_REPLICATE
    )
    if defocus_sigma > 0:
        ksize = int(6 * defocus_sigma) | 1
        blurred = cv2.GaussianBlur(
            blurred,
            (ksize, ksize),
            defocus_sigma
        )
    return blurred


def get_image_files(input_dir):
    """获取输入目录中的所有图片文件，不递归子目录。"""
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"输入目录不存在：{input_dir}")
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return [
        file for file in input_path.iterdir()
        if file.is_file() and file.suffix.lower() in exts
    ]


def sample_images(image_files, sample_ratio, sample_count, rng):
    """
    按比例或固定数量随机选取图片。
    sample_count 优先于 sample_ratio。
    """
    total = len(image_files)
    if total == 0:
        return []
    if sample_count is not None:
        count = max(0, int(sample_count))
        count = min(count, total)
    else:
        if sample_ratio <= 0:
            return []
        count = max(1, int(round(total * float(sample_ratio))))
        count = min(count, total)
    if count >= total:
        return image_files.copy()
    return list(rng.choice(image_files, size=count, replace=False))


def process_folder(
    input_dir,
    output_dir,
    short_len_range=(8, 12),
    long_len_range=(20, 35),
    angle_range=(0, 180),
    short_sigma_max=0.0,
    long_sigma_max=1.0,
    sample_ratio=1.0,
    sample_count=None,
    seed=None,
    copy_yolo_labels=True,
    copy_origin=True
):
    rng = np.random.default_rng(seed)
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = get_image_files(input_dir)
    selected_files = sample_images(image_files, sample_ratio, sample_count, rng)
    selected_files.sort()

    total_blur_img = 0
    total_blur_label = 0
    total_origin_img = 0
    total_origin_label = 0

    print(f"扫描图片总数：{len(image_files)}")
    print(f"实际选取数量：{len(selected_files)}")

    for file in selected_files:
        img = cv2.imread(str(file))
        if img is None:
            print(f"无法读取，已跳过：{file.name}")
            continue

        stem = file.stem
        suffix = file.suffix

        # ========== 复制选中的原图到输出目录 ==========
        if copy_origin:
            out_origin = output_path / f"{stem}{suffix}"
            out_origin.write_bytes(file.read_bytes())
            total_origin_img += 1
            # 同步复制原图对应的txt标签 (P1-10: 同时支持同目录与 labels/ 标准目录)
            if copy_yolo_labels:
                src_txt = _safe_label_path(input_path, stem)
                if src_txt is not None:
                    out_origin_txt = output_path / f"{stem}.txt"
                    out_origin_txt.write_bytes(src_txt.read_bytes())
                    total_origin_label += 1

        # 短模糊：轻度运动抖动。
        short_length = rng.uniform(*short_len_range)
        short_angle = rng.uniform(*angle_range)
        short_sigma = rng.uniform(0.0, short_sigma_max)
        short_image = apply_motion_blur(img, length=short_length, angle=short_angle, defocus_sigma=short_sigma)
        short_image_path = output_path / f"{stem}_blurred_short{suffix}"
        # P1-10: cv2.imwrite 在中文路径上静默失败, 改用 unicode-safe imwrite
        _safe_imwrite(short_image_path, short_image)

        # 长模糊：重度拖影或失焦。
        long_length = rng.uniform(*long_len_range)
        long_angle = rng.uniform(*angle_range)
        long_sigma = rng.uniform(0.0, long_sigma_max)
        long_image = apply_motion_blur(img, length=long_length, angle=long_angle, defocus_sigma=long_sigma)
        long_image_path = output_path / f"{stem}_blurred_long{suffix}"
        _safe_imwrite(long_image_path, long_image)

        total_blur_img += 2

        print(
            f"{file.name}  "
            f"短模糊：length={short_length:.1f}, angle={short_angle:.1f}, sigma={short_sigma:.2f}  |  "
            f"长模糊：length={long_length:.1f}, angle={long_angle:.1f}, sigma={long_sigma:.2f}"
        )

        # 复制模糊图对应的YOLO标签 (P1-10: 同目录/标准 labels/ 二选一)
        if copy_yolo_labels:
            label_file = _safe_label_path(input_path, stem)
            if label_file is not None:
                short_label_path = output_path / f"{stem}_blurred_short.txt"
                long_label_path = output_path / f"{stem}_blurred_long.txt"
                short_label_path.write_bytes(label_file.read_bytes())
                long_label_path.write_bytes(label_file.read_bytes())
                total_blur_label += 2

    print("\n===== 处理完成 =====")
    print(f"输出目录：{output_path}")
    print(f"采样原图数量：{len(selected_files)}")
    print(f"复制原图文件：{total_origin_img}")
    print(f"生成模糊图片：{total_blur_img}")
    if copy_yolo_labels:
        print(f"复制原图标签：{total_origin_label}")
        print(f"复制模糊图标签：{total_blur_label}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="按比例/数量采样，生成运动+失焦模糊；支持复制原图与YOLO标签"
    )

    parser.add_argument("--input_dir", default="", help="输入图片目录 (必传; 例: D:/datasets/base_3_1_background/images)")
    parser.add_argument("--output_dir", default="", help="输出目录, 自动创建 (必传; 例: D:/datasets/base_3_1_background/images/motion_blur)")

    parser.add_argument("--sample_ratio", type=float, default=0.5, help="随机选取图片比例，0~1")
    parser.add_argument("--sample_count", type=int, default=None, help="固定选取张数，优先级高于sample_ratio")

    parser.add_argument("--short_len_min", type=float, default=8)
    parser.add_argument("--short_len_max", type=float, default=12)
    parser.add_argument("--short_sigma_max", type=float, default=0.5)

    parser.add_argument("--long_len_min", type=float, default=20)
    parser.add_argument("--long_len_max", type=float, default=35)
    parser.add_argument("--long_sigma_max", type=float, default=1.0)

    parser.add_argument("--angle_min", type=float, default=0)
    parser.add_argument("--angle_max", type=float, default=180)
    parser.add_argument("--seed", type=int, default=None, help="随机种子，复现实验")

    parser.add_argument("--no_copy_label", action="store_true", help="关闭YOLO txt标签复制")
    parser.add_argument("--no_copy_origin", action="store_true", help="不复制选中的原图到输出文件夹")

    args = parser.parse_args()

    process_folder(
        args.input_dir,
        args.output_dir,
        short_len_range=(args.short_len_min, args.short_len_max),
        long_len_range=(args.long_len_min, args.long_len_max),
        angle_range=(args.angle_min, args.angle_max),
        short_sigma_max=args.short_sigma_max,
        long_sigma_max=args.long_sigma_max,
        sample_ratio=args.sample_ratio,
        sample_count=args.sample_count,
        seed=args.seed,
        copy_yolo_labels=not args.no_copy_label,
        copy_origin=not args.no_copy_origin
    )
