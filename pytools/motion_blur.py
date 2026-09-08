import cv2
import numpy as np
import os
import argparse
from pathlib import Path


def _safe_imwrite(path: str | Path, img: np.ndarray) -> bool:
    """P1-10: unicode-safe imwrite 替代 cv2.imwrite (中文路径上后者静默返回 False)."""
    try:
        ok, buf = cv2.imencode(Path(path).suffix or ".jpg", img)
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except Exception:
        return False


def motion_blur_kernel(length, angle):
    """生成运动模糊核（线段形PSF）"""
    # 角度转弧度，并调整使0°为水平向右
    rad = np.deg2rad(angle)
    # 核大小取奇数，至少为3
    size = max(3, int(length) if length % 2 == 1 else int(length) + 1)
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    # 计算线段两端点坐标
    dx = np.cos(rad)
    dy = np.sin(rad)
    # 线段长度归一化到length
    length_scaled = length / 2.0
    x1 = center - dx * length_scaled
    y1 = center - dy * length_scaled
    x2 = center + dx * length_scaled
    y2 = center + dy * length_scaled
    # 使用抗锯齿线段绘制（OpenCV的line函数）
    cv2.line(kernel, (int(round(x1)), int(round(y1))),
             (int(round(x2)), int(round(y2))), 1.0, thickness=1, lineType=cv2.LINE_AA)
    # 归一化
    kernel /= kernel.sum()
    return kernel


def apply_motion_blur(img, length=15, angle=30, defocus_sigma=0):
    """
    对图像应用运动模糊和可选失焦模糊
    :param img: 输入图像 (BGR或灰度)
    :param length: 运动模糊长度（像素）
    :param angle: 运动方向（度）
    :param defocus_sigma: 高斯模糊标准差，0表示不添加失焦
    """
    kernel = motion_blur_kernel(length, angle)
    blurred = cv2.filter2D(img, -1, kernel)
    if defocus_sigma > 0:
        # 失焦模糊，使用高斯近似
        ksize = int(6 * defocus_sigma) | 1  # 确保奇数
        blurred = cv2.GaussianBlur(blurred, (ksize, ksize), defocus_sigma)
    return blurred


def process_folder(input_dir, output_dir,
                   short_len_range=(8, 12), long_len_range=(20, 35),
                   angle_range=(0, 180), short_sigma_max=0.0, long_sigma_max=1.0,
                   seed=None):
    """
    批量处理文件夹中的图片，每张原图生成 2 张模糊副本：
      1) 短模糊(轻度): length 小, 无/微失焦 —— 模拟轻微运动抖动
      2) 长模糊(重度): length 大, 可选失焦  —— 模拟强拖影/对焦不准
    两档角度各自独立随机铺满 angle_range，保证模糊方向多样性。
    """
    if seed is not None:
        np.random.seed(seed)
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 支持的图片扩展名
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    files = [f for f in input_path.iterdir() if f.suffix.lower() in exts]

    total = 0
    read_failed = 0
    write_failed = 0
    for file in files:
        img = cv2.imread(str(file))
        if img is None:
            print(f"无法读取 {file}，跳过")
            read_failed += 1
            continue

        stem, ext = file.stem, file.suffix

        # ---- 副本1: 短模糊(轻度) ----
        s_len = np.random.uniform(*short_len_range)
        s_ang = np.random.uniform(*angle_range)
        s_sig = np.random.uniform(0.0, short_sigma_max)
        res_short = apply_motion_blur(img, length=s_len, angle=s_ang, defocus_sigma=s_sig)
        out_s = output_path / f"{stem}_blurred_short{ext}"
        # M3: 检查写盘返回值, 失败计入失败数而非静默吞掉
        if not _safe_imwrite(out_s, res_short):
            write_failed += 1
            print(f"  ⚠ 写入失败: {out_s}")

        # ---- 副本2: 长模糊(重度) ----
        l_len = np.random.uniform(*long_len_range)
        l_ang = np.random.uniform(*angle_range)
        l_sig = np.random.uniform(0.0, long_sigma_max)
        res_long = apply_motion_blur(img, length=l_len, angle=l_ang, defocus_sigma=l_sig)
        out_l = output_path / f"{stem}_blurred_long{ext}"
        if not _safe_imwrite(out_l, res_long):
            write_failed += 1
            print(f"  ⚠ 写入失败: {out_l}")

        total += 2
        print(f"{file.name} -> "
              f"{out_s.name} (len={s_len:.1f}, ang={s_ang:.1f}, σ={s_sig:.2f}) | "
              f"{out_l.name} (len={l_len:.1f}, ang={l_ang:.1f}, σ={l_sig:.2f})")

    print(f"\n完成: {len(files)} 张原图 -> {total} 张模糊副本 (每图 2 张) 保存于 {output_path}")
    # M3: 失败汇总, 让用户明确知道数据集是否完整
    if read_failed:
        print(f"⚠ 警告: {read_failed} 张图片读取失败被跳过")
    if write_failed:
        print(f"⚠ 警告: {write_failed} 张模糊副本写入失败 (输出数据集可能不完整)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模拟无人机运动失焦，每张原图生成 2 张模糊副本(短模糊+长模糊)")
    parser.add_argument("--input_dir", default="", help="输入图片目录 (必传; 例: D:/datasets/base_0_0/images)")
    parser.add_argument("--output_dir", default="", help="输出目录, 自动创建 (必传; 例: D:/datasets/base_0_0/motion_blur)")
    # 短模糊档 (轻度): 小 length, 无失焦
    parser.add_argument("--short_len_min", type=float, default=8, help="短模糊长度最小值")
    parser.add_argument("--short_len_max", type=float, default=12, help="短模糊长度最大值")
    parser.add_argument("--short_sigma_max", type=float, default=0.5, help="短模糊失焦sigma上限 (默认0=不加失焦)")
    # 长模糊档 (重度): 大 length, 可选失焦
    parser.add_argument("--long_len_min", type=float, default=20, help="长模糊长度最小值")
    parser.add_argument("--long_len_max", type=float, default=35, help="长模糊长度最大值")
    parser.add_argument("--long_sigma_max", type=float, default=1.0, help="长模糊失焦sigma上限")
    # 共用
    parser.add_argument("--angle_min", type=float, default=0, help="运动方向角度最小值")
    parser.add_argument("--angle_max", type=float, default=180, help="运动方向角度最大值")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，便于复现")
    args = parser.parse_args()

    process_folder(
        args.input_dir,
        args.output_dir,
        short_len_range=(args.short_len_min, args.short_len_max),
        long_len_range=(args.long_len_min, args.long_len_max),
        angle_range=(args.angle_min, args.angle_max),
        short_sigma_max=args.short_sigma_max,
        long_sigma_max=args.long_sigma_max,
        seed=args.seed
    )
