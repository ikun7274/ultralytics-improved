import os
import json
import glob
from PIL import Image

def yolo_to_xanylabeling(
    yolo_txt_dir: str,
    image_dir: str,
    output_dir: str,
    class_names: list,
    version: str = "4.5.3"
):
    """
    批量将 YOLO 检测框 txt 转换为 X-AnyLabeling JSON 格式。

    Args:
        yolo_txt_dir: YOLO txt 标注文件夹路径
        image_dir:    对应图片文件夹路径（用于读取图片宽高）
        output_dir:   输出 JSON 文件夹路径
        class_names:  类别名列表，索引对应 YOLO txt 中的 class_id
        version:      X-AnyLabeling 版本号，默认 4.5.3
    """
    os.makedirs(output_dir, exist_ok=True)

    # 支持的图片后缀
    img_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

    # 遍历所有 txt 文件
    txt_files = glob.glob(os.path.join(yolo_txt_dir, "*.txt"))
    total = len(txt_files)
    success = 0

    for txt_path in txt_files:
        txt_name = os.path.splitext(os.path.basename(txt_path))[0]

        # 查找对应图片（同名，任意后缀）
        img_path = None
        for ext in img_exts:
            candidate = os.path.join(image_dir, txt_name + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            print(f"[跳过] 未找到对应图片: {txt_name}")
            continue

        # 读取图片宽高
        with Image.open(img_path) as img:
            img_w, img_h = img.size
        img_filename = os.path.basename(img_path)

        # 解析 YOLO txt
        shapes = []
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width    = float(parts[3])
                height   = float(parts[4])

                # 归一化 → 像素坐标
                x1 = (x_center - width / 2) * img_w
                y1 = (y_center - height / 2) * img_h
                x2 = (x_center + width / 2) * img_w
                y2 = (y_center + height / 2) * img_h

                # 边界裁剪（防止越界）
                x1 = max(0.0, min(x1, img_w))
                y1 = max(0.0, min(y1, img_h))
                x2 = max(0.0, min(x2, img_w))
                y2 = max(0.0, min(y2, img_h))

                label = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"

                shape = {
                    "label": label,
                    "score": None,
                    "points": [
                        [x1, y1],
                        [x2, y1],
                        [x2, y2],
                        [x1, y2]
                    ],
                    "group_id": None,
                    "description": "",
                    "difficult": False,
                    "shape_type": "rectangle",
                    "flags": {},
                    "attributes": {},
                    "kie_linking": []
                }
                shapes.append(shape)

        # 组装 JSON
        result = {
            "version": version,
            "flags": {},
            "shapes": shapes,
            "imagePath": img_filename,
            "imageData": None,
            "imageHeight": img_h,
            "imageWidth": img_w,
            "description": ""
        }

        # 写出 JSON（与图片同名）
        json_name = txt_name + ".json"
        json_path = os.path.join(output_dir, json_name)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        success += 1
        print(f"[完成] {json_name}  ({len(shapes)} 个标注)")

    print(f"\n转换完成：成功 {success}/{total}")


# ============================================================
#  使用示例 —— 修改下面的路径和类别名即可
# ============================================================
if __name__ == "__main__":
    # 1. YOLO txt 文件夹
    YOLO_TXT_DIR = r"C:\Users\ROG\Desktop\human\base_0_3\labels"

    # 2. 对应图片文件夹（用于读取宽高）
    IMAGE_DIR    = r"C:\Users\ROG\Desktop\human\base_0_3\images"

    # 3. 输出 JSON 文件夹
    OUTPUT_DIR   = r"C:\Users\ROG\Desktop\human\base_0_3\images"

    # 4. 类别名列表（索引 = YOLO txt 中的 class_id）
    #    例如 classes.txt 内容每行一个类别，可直接读取：
    #    with open("classes.txt", "r") as f:
    #        CLASS_NAMES = [l.strip() for l in f if l.strip()]
    CLASS_NAMES =   ['human']

    yolo_to_xanylabeling(
        yolo_txt_dir=YOLO_TXT_DIR,
        image_dir=IMAGE_DIR,
        output_dir=OUTPUT_DIR,
        class_names=CLASS_NAMES
    )
