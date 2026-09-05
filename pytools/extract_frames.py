import cv2
import os
import glob

# ==================== 用户配置区 ====================
# P2-6: 移除硬编码的本机路径, 改为占位空串; 必传 INPUT_DIR 和 OUTPUT_DIR
INPUT_DIR = ""                                                       # 视频所在目录（必传; 例: D:/videos/xxx）
OUTPUT_DIR = ""                                                      # 输出图片的保存目录（自动创建; 例: D:/frames/xxx）
PREFIX = "base_1_0"                     # 文件名前缀（可改为 B, C, SCENE 等）
START_NUM = 1                    # 起始数字
DIGITS = 5                       # 数字位数（如 5 代表 00001）
EXTRACT_INTERVAL = 10            # 每隔多少帧提取1张（30fps视频下约每秒1张）
JPEG_QUALITY = 100                # JPG画质 (0-100)
# ===================================================

def main():
    # 必传检查
    if not INPUT_DIR or not OUTPUT_DIR:
        raise ValueError(
            "INPUT_DIR / OUTPUT_DIR 必须配置。在脚本顶部的 用户配置区 设置实际路径后重试。"
        )

    # 1. 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. 获取所有视频文件（支持常见格式, 大小写合并去重）
    # P1-10: Windows 下 *.mp4 / *.MP4 同时存在会拿到相同文件的重复条目, 导致每个视频被处理两次.
    video_extensions = ["mp4", "avi", "mov", "mkv", "flv"]
    seen_lower = set()
    video_files = []
    for ext in video_extensions:
        for p in glob.iglob(os.path.join(INPUT_DIR, f"*.{ext}")):
            key = os.path.basename(p).lower()
            if key in seen_lower:
                continue
            seen_lower.add(key)
            video_files.append(p)

    if not video_files:
        print(f"错误：在 '{INPUT_DIR}' 目录下未找到任何视频文件！")
        return

    print(f"共发现 {len(video_files)} 个视频文件，开始处理...\n")

    counter = START_NUM  # 全局计数器（所有视频的图片按顺序连续编号）

    # 3. 遍历每个视频
    for video_path in video_files:
        video_name = os.path.basename(video_path)
        print(f"▶ 正在处理: {video_name}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  ⚠ 无法打开视频，已跳过")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        extracted_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 按间隔提取帧
            if frame_count % EXTRACT_INTERVAL == 0:
                # 生成文件名：前缀 + 数字（自动补零），如 A00001.jpg
                filename = f"{PREFIX}{str(counter).zfill(DIGITS)}.jpg"
                filepath = os.path.join(OUTPUT_DIR, filename)

                # P1-10: cv2.imwrite 中文路径会静默返回 False, 必须检查并告警
                ok = cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if not ok:
                    print(f"  ⚠ 保存失败 (路径含中文 / 权限不足?): {filepath}")
                else:
                    extracted_count += 1

                # 打印进度（每10张打印一次，避免刷屏）
                if extracted_count % 10 == 0:
                    print(f"  进度: {frame_count}/{total_frames} 帧, 已提取 {extracted_count} 张")

                counter += 1  # 全局编号+1

            frame_count += 1
        cap.release()
        print(f"  ✅ 完成，从该视频提取了 {extracted_count} 张图片\n")

    print(f"🎉 全部处理完成！共提取 {counter - START_NUM} 张图片，保存在 '{OUTPUT_DIR}' 目录下")

if __name__ == "__main__":
    main()
