import os
import logging
import argparse
import json
import shutil
import random
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import cv2
from ultralytics.data.converter import convert_coco
import concurrent.futures

# 设置日志级别
logging.getLogger("sahi").setLevel(logging.WARNING)

# 调色板（用于可视化）
COLOR_PALETTE = [
    (0, 255, 0),    # 绿
    (255, 0, 0),    # 蓝
    (0, 0, 255),    # 红
    (255, 255, 0),  # 青
    (255, 0, 255),  # 品红
    (0, 255, 255),  # 黄
    (128, 0, 128),  # 紫
    (255, 165, 0),  # 橙
    (0, 128, 128),  # 深青
    (128, 128, 0),  # 橄榄
]


def safe_rmtree(path, description="目录"):
    """安全删除目录，防止误删关键路径"""
    p = Path(path).resolve()
    if p == p.root or p == Path.home().resolve():
        raise RuntimeError(f"拒绝删除 {description}：{p} 疑似系统关键路径")
    if p.exists():
        item_count = len(list(p.iterdir()))
        print(f"将删除 {description}：{p}（包含 {item_count} 个项目）")
        shutil.rmtree(p)


def clear_dir(path):
    """清空或创建目录（安全删除）"""
    p = Path(path)
    if p.exists():
        safe_rmtree(p, description="待清空目录")
    p.mkdir(parents=True, exist_ok=True)


def parse_yolo_label(label_path, img_w, img_h):
    """解析 YOLO txt，返回绝对像素坐标和跳过数量"""
    annotations = []
    skipped = 0
    label_path = Path(label_path)
    if not label_path.exists():
        return annotations, skipped
    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            bbox_w = float(parts[3])
            bbox_h = float(parts[4])
        except ValueError:
            continue
        x_min = (x_center - bbox_w / 2) * img_w
        y_min = (y_center - bbox_h / 2) * img_h
        w = bbox_w * img_w
        h = bbox_h * img_h
        if w <= 0 or h <= 0:
            skipped += 1
            continue
        # 边界裁剪
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        w = min(w, img_w - x_min)
        h = min(h, img_h - y_min)
        if w <= 0 or h <= 0:
            skipped += 1
            continue
        annotations.append({
            "class_id": class_id,
            "x_min": x_min,
            "y_min": y_min,
            "w": w,
            "h": h,
        })
    return annotations, skipped


def validate_args(args):
    """参数校验"""
    if not (0 <= args.overlap_ratio < 1):
        raise ValueError(f"overlap_ratio 应在 [0, 1) 范围内，当前值：{args.overlap_ratio}")
    if not (0 <= args.min_area_ratio <= 1):
        raise ValueError(f"min_area_ratio 应在 [0, 1] 范围内，当前值：{args.min_area_ratio}")
    if not (0 <= args.min_retain_ratio <= 1):
        raise ValueError(f"min_retain_ratio 应在 [0, 1] 范围内，当前值：{args.min_retain_ratio}")
    if args.workers < 1:
        raise ValueError(f"workers 必须 >= 1，当前值：{args.workers}")

    orig_resolved = Path(args.orig_root).resolve()
    for output_dir in [args.coco_tmp, args.slice_coco_dir, args.final_yolo_dir]:
        out_resolved = Path(output_dir).resolve()
        if orig_resolved == out_resolved:
            raise ValueError(f"输入目录与输出目录相同：{output_dir}，会导致原始数据被删除！")
        if str(out_resolved).startswith(str(orig_resolved) + os.sep):
            print(f"警告：输出目录 {out_resolved} 位于输入目录 {orig_resolved} 内部，请确认路径配置。")


def find_image_label_dirs(yolo_root):
    """自动检测数据集结构，优先查找 images/train, labels/train"""
    img_dir = Path(yolo_root) / "images" / "train"
    lab_dir = Path(yolo_root) / "labels" / "train"
    if img_dir.exists() and lab_dir.exists():
        return img_dir, lab_dir
    else:
        return Path(yolo_root) / "images", Path(yolo_root) / "labels"


def load_class_names_from_file(file_path):
    """从文件读取类别名称（每行一个）"""
    path = Path(file_path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    return names if names else None


def process_one_image(img_path, label_dir, out_img_folder, class_names):
    """单张图片处理：复制图片、解析标签，返回COCO格式标注"""
    try:
        with Image.open(img_path) as pil_img:
            width, height = pil_img.size
    except Exception as e:
        return None, 0, 0, [], 0, f"警告：无法读取图片 {img_path}，跳过。错误：{e}"

    try:
        shutil.copy(img_path, out_img_folder / img_path.name)
    except Exception as e:
        return None, 0, 0, [], 0, f"警告：复制图片 {img_path} 失败，跳过。错误：{e}"

    label_file = label_dir / (img_path.stem + ".txt")
    parsed_anns, skipped = parse_yolo_label(label_file, width, height)

    annotations = []
    for ann in parsed_anns:
        annotations.append({
            "category_id": ann["class_id"] + 1,
            "bbox": [ann["x_min"], ann["y_min"], ann["w"], ann["h"]],
            "area": ann["w"] * ann["h"],
            "iscrowd": 0
        })
    return img_path.name, width, height, annotations, skipped, None


def yolo2coco(yolo_root, save_coco_dir, class_names, workers=1):
    """YOLO -> COCO 转换，支持多进程"""
    clear_dir(save_coco_dir)
    img_dir, label_dir = find_image_label_dirs(yolo_root)
    out_img_folder = Path(save_coco_dir) / "images"
    out_img_folder.mkdir(parents=True, exist_ok=True)

    img_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        img_paths.extend(img_dir.glob(ext))
    if not img_paths:
        raise FileNotFoundError(f"在 {img_dir} 中没有找到任何图片文件，请检查路径。")

    coco_dict = {"images": [], "annotations": [], "categories": []}
    for idx, name in enumerate(class_names, start=1):
        coco_dict["categories"].append({"id": idx, "name": name, "supercategory": "none"})

    total_annotations = 0
    total_skipped = 0

    if workers <= 1:
        for img_path in tqdm(img_paths, desc="转换 YOLO → COCO"):
            result = process_one_image(img_path, label_dir, out_img_folder, class_names)
            file_name, width, height, anns, skipped, error_msg = result
            if error_msg:
                tqdm.write(error_msg)
                continue
            if file_name is None:
                continue
            image_id = len(coco_dict["images"]) + 1
            coco_dict["images"].append({"id": image_id, "file_name": file_name, "width": width, "height": height})
            for ann in anns:
                coco_dict["annotations"].append({
                    "id": len(coco_dict["annotations"]) + 1,
                    "image_id": image_id,
                    "category_id": ann["category_id"],
                    "bbox": ann["bbox"],
                    "area": ann["area"],
                    "iscrowd": ann["iscrowd"],
                    "keypoints": [],
                    "num_keypoints": 0
                })
            total_annotations += len(anns)
            total_skipped += skipped
    else:
        print(f"使用 {workers} 个进程并行处理...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_one_image, img_path, label_dir, out_img_folder, class_names): img_path
                       for img_path in img_paths}
            with tqdm(total=len(futures), desc="并行转换 YOLO → COCO") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    pbar.update(1)
                    img_path = futures[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        tqdm.write(f"处理图片 {img_path} 时出错：{e}")
                        continue
                    file_name, width, height, anns, skipped, error_msg = result
                    if error_msg:
                        tqdm.write(error_msg)
                        continue
                    if file_name is None:
                        continue
                    image_id = len(coco_dict["images"]) + 1
                    coco_dict["images"].append({"id": image_id, "file_name": file_name, "width": width, "height": height})
                    for ann in anns:
                        coco_dict["annotations"].append({
                            "id": len(coco_dict["annotations"]) + 1,
                            "image_id": image_id,
                            "category_id": ann["category_id"],
                            "bbox": ann["bbox"],
                            "area": ann["area"],
                            "iscrowd": ann["iscrowd"],
                            "keypoints": [],
                            "num_keypoints": 0
                        })
                    total_annotations += len(anns)
                    total_skipped += skipped

    json_path = Path(save_coco_dir) / "annotations.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(coco_dict, f, indent=2)
    print(f"COCO JSON 已生成：{json_path}")
    print(f"图片数：{len(coco_dict['images'])}，有效标注数：{total_annotations}，跳过的无效标注数：{total_skipped}")
    return str(json_path), str(out_img_folder)


# ================= 固定2x2网格切片（带重叠 + 双重面积过滤） =================
def fixed_grid_slice_coco(coco_json, img_dir, out_slice_dir,
                          min_area_ratio, overlap_ratio=0.0,
                          min_retain_ratio=0.333):
    """
    2x2网格切片，支持重叠比例。
    过滤条件：碎片面积 < min_area_ratio * 子图面积 且 碎片面积 < min_retain_ratio * 原始面积 时丢弃。
    默认 min_area_ratio=0 关闭子图比例过滤，仅使用 min_retain_ratio 过滤边缘碎片。
    """
    from PIL import Image
    import json
    from pathlib import Path

    out_dir = Path(out_slice_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_img_dir = out_dir / "images"
    out_img_dir.mkdir(exist_ok=True)

    with open(coco_json, 'r') as f:
        coco = json.load(f)

    print(f"原始COCO标注总数: {len(coco['annotations'])}")
    if len(coco['annotations']) == 0:
        print("警告：原始COCO无标注，切片后也将无标注。请检查YOLO→COCO转换步骤。")

    anns_by_img = {}
    for ann in coco['annotations']:
        anns_by_img.setdefault(ann['image_id'], []).append(ann)

    new_images = []
    new_annotations = []
    ann_id_counter = 0
    new_img_id = 1
    total_original_anns = 0
    total_kept_anns = 0

    for img_info in coco['images']:
        img_path = Path(img_dir) / img_info['file_name']
        if not img_path.exists():
            print(f"警告：图片 {img_path} 不存在，跳过")
            continue

        with Image.open(img_path) as pil_img:
            pil_img.load()
            w, h = pil_img.size

            # 计算子图尺寸
            sw = int((1 + overlap_ratio) * w / 2)
            sh = int((1 + overlap_ratio) * h / 2)
            if sw <= 0 or sh <= 0:
                print(f"警告：图片 {img_path} 尺寸过小，无法切片，跳过")
                continue
            if sw > w:
                sw = w
            if sh > h:
                sh = h

            # 四个子图起始位置
            x_starts = [0, w - sw]
            y_starts = [0, h - sh]
            x_starts = [max(0, x) for x in x_starts]
            y_starts = [max(0, y) for y in y_starts]

            ann_list = anns_by_img.get(img_info['id'], [])
            total_original_anns += len(ann_list)

            for row in range(2):
                for col in range(2):
                    x_start = x_starts[col]
                    y_start = y_starts[row]
                    x_end = x_start + sw
                    y_end = y_start + sh

                    # 裁剪子图
                    sub_img = pil_img.crop((x_start, y_start, x_end, y_end))
                    stem = img_path.stem
                    ext = img_path.suffix
                    sub_filename = f"{stem}_{row}_{col}{ext}"
                    sub_img_path = out_img_dir / sub_filename
                    sub_img.save(sub_img_path)

                    new_img_info = {
                        'id': new_img_id,
                        'file_name': sub_filename,
                        'width': sw,
                        'height': sh
                    }
                    new_images.append(new_img_info)

                    sub_area = sw * sh
                    for ann in ann_list:
                        x_min, y_min, bbox_w, bbox_h = ann['bbox']
                        x_max = x_min + bbox_w
                        y_max = y_min + bbox_h
                        ori_area = bbox_w * bbox_h  # 原始标注面积

                        # 计算交集
                        inter_x_min = max(x_min, x_start)
                        inter_y_min = max(y_min, y_start)
                        inter_x_max = min(x_max, x_end)
                        inter_y_max = min(y_max, y_end)

                        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
                            continue

                        new_x_min = inter_x_min - x_start
                        new_y_min = inter_y_min - y_start
                        new_w = inter_x_max - inter_x_min
                        new_h = inter_y_max - inter_y_min
                        box_area = new_w * new_h

                        # 双重过滤：只有同时满足两个条件才丢弃
                        if box_area < min_area_ratio * sub_area and box_area < min_retain_ratio * ori_area:
                            continue

                        new_ann = {
                            'id': ann_id_counter + 1,
                            'image_id': new_img_id,
                            'category_id': ann['category_id'],
                            'bbox': [new_x_min, new_y_min, new_w, new_h],
                            'area': box_area,
                            'iscrowd': ann.get('iscrowd', 0)
                        }
                        new_annotations.append(new_ann)
                        ann_id_counter += 1
                        total_kept_anns += 1

                    new_img_id += 1

    print(f"原始标注总数: {total_original_anns}, 保留标注数: {total_kept_anns}")

    new_coco = {
        'images': new_images,
        'annotations': new_annotations,
        'categories': coco['categories']
    }

    new_json_path = out_dir / 'sliced_annotations.json'
    with open(new_json_path, 'w') as f:
        json.dump(new_coco, f, indent=2)

    print(f"2x2网格切片完成，重叠比例：{overlap_ratio:.2f}")
    print(f"生成 {len(new_images)} 张切片图片，{len(new_annotations)} 个标注。")
    return str(new_json_path), str(out_img_dir)


def filter_background_slices(slice_json_path, slice_img_folder, neg_ratio):
    """控制背景切片保留比例"""
    if neg_ratio < 0:
        print("neg_ratio 为负，保留所有背景切片。")
        return
    json_path = Path(slice_json_path)
    if not json_path.exists():
        print(f"警告：JSON 文件 {json_path} 不存在，跳过背景筛选。")
        return
    with open(json_path, "r") as f:
        coco = json.load(f)
    pos_ids = set(ann["image_id"] for ann in coco["annotations"])
    all_ids = set(img["id"] for img in coco["images"])
    neg_ids = all_ids - pos_ids
    if not neg_ids:
        print("没有背景切片，无需筛选。")
        return
    keep_num = int(len(pos_ids) * neg_ratio)
    if keep_num >= len(neg_ids):
        print(f"背景切片数量 {len(neg_ids)} 不超过保留上限 {keep_num}，全部保留。")
        return
    keep_ids = set(random.sample(list(neg_ids), keep_num))
    remove_ids = neg_ids - keep_ids
    for img_info in coco["images"]:
        if img_info["id"] in remove_ids:
            img_path = Path(slice_img_folder) / img_info["file_name"]
            if img_path.exists():
                img_path.unlink()
    coco["images"] = [img for img in coco["images"] if img["id"] not in remove_ids]
    with open(json_path, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"已保留 {keep_num} 个背景切片，删除了 {len(remove_ids)} 个背景切片。")


def coco2yolo(coco_json_path, save_yolo_root, slice_img_dir):
    """切片COCO转回YOLO格式，并整理文件，补齐空标签"""
    save_root = Path(save_yolo_root)
    if save_root.exists():
        safe_rmtree(save_root, description="最终 YOLO 输出目录")

    labels_dir = Path(coco_json_path).parent
    print(f"调用 convert_coco，输入目录：{labels_dir}，输出目录：{save_root}")
    convert_coco(str(labels_dir), str(save_root), use_segments=False, use_keypoints=False)

    src_img_dir = Path(slice_img_dir)
    dst_img_dir = save_root / "images"
    dst_img_dir.mkdir(parents=True, exist_ok=True)

    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    img_files = [f for f in src_img_dir.iterdir() if f.suffix.lower() in img_extensions]
    if not img_files:
        print(f"警告：切片图片目录 {src_img_dir} 中没有找到任何图片文件。")
    for img_file in tqdm(img_files, desc="复制切片图片"):
        shutil.copy(img_file, dst_img_dir / img_file.name)
    print(f"已复制 {len(img_files)} 张切片图片到 {dst_img_dir}")

    labels_root = save_root / "labels"
    if labels_root.exists():
        txt_files = list(labels_root.glob("*/*.txt")) + list(labels_root.glob("*.txt"))
        if txt_files:
            for txt_file in tqdm(txt_files, desc="移动标签文件到根目录"):
                if txt_file.parent != labels_root:
                    shutil.move(str(txt_file), str(labels_root / txt_file.name))
            for subdir in labels_root.iterdir():
                if subdir.is_dir():
                    try:
                        subdir.rmdir()
                    except OSError:
                        pass
            print(f"已整理 {len(txt_files)} 个标签文件到 {labels_root}")
        else:
            print("警告：未生成任何标签文件，请检查 COCO JSON 是否包含标注。")
    else:
        print(f"警告：标签目录 {labels_root} 不存在。")

    # 补齐空标签
    print("检查并补齐缺失的标签文件（创建空 txt）...")
    missing_count = 0
    for img_file in tqdm(img_files, desc="补齐空标签"):
        txt_name = img_file.stem + ".txt"
        txt_path = labels_root / txt_name
        if not txt_path.exists():
            txt_path.touch()
            missing_count += 1
    if missing_count > 0:
        print(f"已创建 {missing_count} 个空标签文件（对应无标注的切片）。")
    else:
        print("所有切片图片均有对应的标签文件。")
    return save_root


def generate_final_statistics(yolo_root, output_json="statistics.json", slice_params=None):
    """
    扫描最终 YOLO 格式数据集，生成统计报告，并记录切片参数。
    """
    root = Path(yolo_root)
    img_dir = root / "images"
    labels_dir = root / "labels"
    if not img_dir.exists():
        print(f"警告：图片目录 {img_dir} 不存在，无法生成统计。")
        return

    img_files = [f for f in img_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    total_images = len(img_files)
    label_files = list(labels_dir.glob("*.txt")) if labels_dir.exists() else []
    total_labels = len(label_files)
    empty_labels = 0
    non_empty_labels = 0
    class_counts = {}
    total_instances = 0

    for lbl_file in tqdm(label_files, desc="解析标签统计"):
        if lbl_file.stat().st_size == 0:
            empty_labels += 1
            continue
        non_empty_labels += 1
        with open(lbl_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
                    total_instances += 1

    classes_file = root.parent / "classes.txt" if root.parent else None
    class_names = {}
    if classes_file and classes_file.exists():
        with open(classes_file, "r") as f:
            names = [line.strip() for line in f if line.strip()]
            for idx, name in enumerate(names):
                class_names[idx] = name
    else:
        for cls_id in class_counts.keys():
            class_names[cls_id] = f"class_{cls_id}"

    stats = {
        "total_images": total_images,
        "total_label_files": total_labels,
        "empty_label_files": empty_labels,
        "non_empty_label_files": non_empty_labels,
        "total_instances": total_instances,
        "class_distribution": {
            class_names.get(cls_id, f"class_{cls_id}"): count
            for cls_id, count in sorted(class_counts.items())
        },
        "slice_parameters": slice_params if slice_params else {},
        "description": "SAHI切片，该数据集是通过2x2网格切片（带重叠）从原始图片生成。"
    }

    out_path = root / output_json
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\n========== 最终数据集统计 ==========")
    print(f"图片总数：{total_images}")
    print(f"标签文件总数：{total_labels}")
    print(f"  其中空标签（无目标）：{empty_labels}")
    print(f"  非空标签（有目标）：{non_empty_labels}")
    print(f"目标实例总数：{total_instances}")
    if class_counts:
        print("各类别实例数：")
        for cls_id, count in sorted(class_counts.items()):
            name = class_names.get(cls_id, f"class_{cls_id}")
            print(f"  {name}: {count}")
    else:
        print("所有标签均为空，无目标实例。")
    if slice_params:
        print("\n切片参数：")
        for key, val in slice_params.items():
            print(f"  {key}: {val}")
    print(f"统计报告已保存至：{out_path}")
    print("====================================\n")
    return stats


def convert_yolo_to_xanylabeling(
    yolo_txt_dir: str,
    image_dir: str,
    output_dir: str,
    class_names: list,
    version: str = "4.5.3"
):
    """YOLO txt -> X-AnyLabeling JSON"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    img_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
    txt_files = list(Path(yolo_txt_dir).glob("*.txt"))
    total = len(txt_files)
    success = 0
    for txt_path in tqdm(txt_files, desc="转换为 X-AnyLabeling JSON"):
        txt_name = txt_path.stem
        img_path = None
        for ext in img_exts:
            candidate = Path(image_dir) / (txt_name + ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            tqdm.write(f"[跳过] 未找到对应图片: {txt_name}")
            continue
        with Image.open(img_path) as img:
            img_w, img_h = img.size
        img_filename = img_path.name

        parsed_anns, _ = parse_yolo_label(txt_path, img_w, img_h)
        shapes = []
        for ann in parsed_anns:
            x1 = ann["x_min"]
            y1 = ann["y_min"]
            x2 = ann["x_min"] + ann["w"]
            y2 = ann["y_min"] + ann["h"]
            label = class_names[ann["class_id"]] if ann["class_id"] < len(class_names) else f"class_{ann['class_id']}"
            shape = {
                "label": label,
                "score": None,
                "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                "group_id": None,
                "description": "",
                "difficult": False,
                "shape_type": "rectangle",
                "flags": {},
                "attributes": {},
                "kie_linking": []
            }
            shapes.append(shape)
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
        json_path = output_path / (txt_name + ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        success += 1
    print(f"X-AnyLabeling JSON 转换完成：成功 {success}/{total}")


def visualize_random_samples(
    yolo_root: str,
    vis_dir: str,
    class_names: list,
    num_samples: int = 10,
    colors: list = None
):
    """随机可视化带标注的图片"""
    root = Path(yolo_root)
    img_dir = root / "images"
    label_dir = root / "labels"
    if not img_dir.exists() or not label_dir.exists():
        print("警告：图片或标签目录不存在，跳过可视化。")
        return
    img_files = [f for f in img_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    if not img_files:
        print("警告：没有找到任何图片文件，跳过可视化。")
        return
    valid_img_files = []
    for img_file in img_files:
        label_file = label_dir / (img_file.stem + ".txt")
        if label_file.exists() and label_file.stat().st_size > 0:
            valid_img_files.append(img_file)
    if not valid_img_files:
        print("警告：没有找到任何带有标注的图片，跳过可视化。")
        return
    print(f"找到 {len(valid_img_files)} 张带有标注的图片。")
    selected = random.sample(valid_img_files, min(num_samples, len(valid_img_files)))
    vis_path = Path(vis_dir)
    vis_path.mkdir(parents=True, exist_ok=True)
    if colors is None:
        colors = COLOR_PALETTE
    print(f"开始可视化 {len(selected)} 张图片（均包含标注）...")
    for img_file in tqdm(selected, desc="可视化随机样本"):
        img_path = img_file
        label_path = label_dir / (img_file.stem + ".txt")
        img = cv2.imread(str(img_path))
        if img is None:
            tqdm.write(f"警告：无法读取图片 {img_path}，跳过。")
            continue
        h, w = img.shape[:2]
        parsed_anns, _ = parse_yolo_label(label_path, w, h)
        for ann in parsed_anns:
            cls_id = ann["class_id"]
            x1 = max(0, int(ann["x_min"]))
            y1 = max(0, int(ann["y_min"]))
            x2 = min(w, int(ann["x_min"] + ann["w"]))
            y2 = min(h, int(ann["y_min"] + ann["h"]))
            color = colors[cls_id % len(colors)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label_text = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        out_path = vis_path / img_file.name
        cv2.imwrite(str(out_path), img)
    print(f"可视化完成，结果保存在：{vis_path}")


def main():
    parser = argparse.ArgumentParser(description="YOLO 数据集固定2x2网格切片工具（支持重叠、双重面积过滤）")
    # 目录
    parser.add_argument("--orig_root", type=str,
                        default="",
                        help="原始 YOLO 数据集根目录 (必传; 例: D:/datasets/base_0_0)")
    parser.add_argument("--coco_tmp", type=str,
                        default="",
                        help="临时 COCO 存放目录 (必传; 例: D:/datasets/coco_temp)")
    parser.add_argument("--slice_coco_dir", type=str,
                        default="",
                        help="切片输出目录 (必传; 例: D:/datasets/slice_coco_output)")
    parser.add_argument("--final_yolo_dir", type=str,
                        default="",
                        help="最终 YOLO 切片数据集输出目录 (必传; 例: D:/datasets/base_0_1)")
    # 格式导出
    parser.add_argument("--export_json", action="store_true", default=True,
                        help="是否导出 X-AnyLabeling 格式的 JSON")
    parser.add_argument("--json_dir", type=str, help="JSON 输出目录，默认在 final_yolo_dir 下创建 json 子目录")

    # 切片参数
    parser.add_argument("--overlap_ratio", type=float, default=0.2,
                        help="重叠比例（0~1），防止目标被切碎")
    parser.add_argument("--min_area_ratio", type=float, default=0.005,
                        help="基于子图面积的过滤阈值（0~1），默认0关闭；可以过滤原图中本来就存在的极小目标(可以防止误标)")
    parser.add_argument("--min_retain_ratio", type=float, default=0.4,
                        help="基于原始标注面积的过滤阈值（0~1），可以过滤边缘被切片的目标(如目标小于原来的1/3)")
    
    parser.add_argument("--neg_ratio", type=float, default=0.2,
                        help="背景保留比例：<0 保留全部；>=0 保留数量 = 正样本数 × neg_ratio")
    parser.add_argument("--classes", type=str, default=None,
                        help="类别名称，逗号分隔，若未指定且自动读取失败则报错")
    parser.add_argument("--workers", type=int, default=1, help="并行进程数，默认1")
    parser.add_argument("--keep_temp", action="store_true", help="保留临时目录")
    parser.add_argument("--vis_num", type=int, default=10, help="可视化样本数，0表示不执行")
    parser.add_argument("--vis_dir", type=str, default=None, help="可视化保存目录")

    args = parser.parse_args()
    validate_args(args)

    # 读取类别
    classes_file = Path(args.orig_root).parent / "classes.txt"
    class_names = load_class_names_from_file(classes_file)
    if class_names:
        print(f"从 {classes_file} 自动读取类别：{class_names}")
    else:
        if args.classes is not None:
            class_names = [c.strip() for c in args.classes.split(",") if c.strip()]
            if class_names:
                print(f"使用命令行指定的类别：{class_names}")
            else:
                raise ValueError("--classes 参数为空，请提供有效类别名称")
        else:
            raise FileNotFoundError(f"未找到类别文件 {classes_file}，且未通过 --classes 指定类别，无法继续。")

    print("===== 1. YOLO原始数据集转COCO =====")
    coco_json_path, coco_img_path = yolo2coco(
        args.orig_root, args.coco_tmp, class_names, workers=args.workers
    )

    print("\n===== 2. 固定2x2网格切片（带重叠 + 双重面积过滤） =====")
    slice_json, slice_img_folder = fixed_grid_slice_coco(
        coco_json_path, coco_img_path, args.slice_coco_dir,
        args.min_area_ratio, args.overlap_ratio, args.min_retain_ratio
    )

    print("\n===== 2.5 背景切片比例控制 =====")
    filter_background_slices(slice_json, slice_img_folder, args.neg_ratio)

    print("\n===== 3. 切片COCO标注转回YOLO TXT =====")
    yolo_all_root = coco2yolo(slice_json, args.final_yolo_dir, slice_img_folder)

    # 切片参数用于统计报告
    slice_params = {
        "overlap_ratio": args.overlap_ratio,
        "min_area_ratio": args.min_area_ratio,
        "min_retain_ratio": args.min_retain_ratio,
        "slice_mode": "2x2_grid_with_overlap"
    }

    print("\n===== 4. 生成统计报告 =====")
    generate_final_statistics(args.final_yolo_dir, "statistics.json", slice_params)

    if args.vis_num > 0:
        print("\n===== 5. 可视化随机样本 =====")
        vis_dir = args.vis_dir if args.vis_dir else str(Path(args.final_yolo_dir) / "visual")
        visualize_random_samples(args.final_yolo_dir, vis_dir, class_names, args.vis_num)

    if args.export_json:
        print("\n===== 6. 导出 X-AnyLabeling JSON =====")
        json_output_dir = args.json_dir if args.json_dir else str(Path(args.final_yolo_dir) / "json")
        convert_yolo_to_xanylabeling(
            yolo_txt_dir=os.path.join(args.final_yolo_dir, "labels"),
            image_dir=os.path.join(args.final_yolo_dir, "images"),
            output_dir=json_output_dir,
            class_names=class_names
        )

    if not args.keep_temp:
        print("\n===== 7. 清理临时目录 =====")
        for tmp_dir in [args.coco_tmp, args.slice_coco_dir]:
            p = Path(tmp_dir)
            if p.exists():
                try:
                    safe_rmtree(p, description="临时目录")
                except Exception as e:
                    print(f"删除 {p} 失败：{e}")
            else:
                print(f"目录 {p} 不存在，跳过。")
    else:
        print("\n保留临时目录（--keep_temp 已指定）")
        print(f"  COCO 临时目录：{args.coco_tmp}")
        print(f"  切片临时目录：{args.slice_coco_dir}")

    print("\n==================== 全部处理完成 ====================")
    print(f"最终切片数据集路径：{yolo_all_root}")
    print("该目录下包含：")
    print("  - images/ : 所有切片图片")
    print("  - labels/ : 对应的 YOLO 格式标签文件（每张图片对应一个 txt，无目标则为空文件）")
    print(f"切片方式：2x2网格，重叠比例 = {args.overlap_ratio:.2f}")
    print(f"子图尺寸 = (1+{args.overlap_ratio:.2f}) * 原图尺寸 / 2，训练时请根据实际子图尺寸设置 imgsz")
    print(f"面积过滤：子图面积阈值 = {args.min_area_ratio:.4f}，原始面积保留比例 = {args.min_retain_ratio:.4f}")
    print(f"类别数：{len(class_names)}，类别名：{class_names}")
    if args.vis_num > 0:
        print(f"可视化样本保存至：{vis_dir}")
    if args.export_json:
        print(f"X-AnyLabeling JSON 已导出至：{json_output_dir}")


if __name__ == "__main__":
    main()
