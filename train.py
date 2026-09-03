import os
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

''' 

4 原图 ──在线切片──→ 16 切片    ──┐
        ──保留────→ 4 原图      ──┼→ 65 张混合样本池 → mosaic 取 4 张拼接 → 训练
        ──合成────→ 1 大图      ──┘   (2×2 拼接, 8000×6000 → 1280×1280)
    ──更改分辨率────→ 4 图片──┘
──运动模糊 + 失焦 ────→ 40 图片  ──┘

'''


if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/26/yolo26n.yaml')
    model.load(r'C:\Users\ASUS\Desktop\ultralytics-main\runs\exp-2\weights\best.pt')
    model.train(
        #---------训练参数---------------
        data='data.yaml',              
        cache=False,
        imgsz=1280,                   
        epochs=1,                      
        batch=4,                       
        close_mosaic=20,              
        workers=0,                     
        optimizer='MuSGD',
        device='0',
        patience=50,
        amp=True,
        project='ultralytics-improved/runs',
        name='exp',
        exist_ok=False,


        # ---------Mosaic在线增强---------------
        mosaic=1.0,
        # mosaic_save_dir=r"C:\Users\ASUS\Desktop\ultralytics-main\base_0_0\mosaic_save_dir",

        # ---------SAHI在线切片 (slice_*)-----------
        slice_prob=1.0,               # 在线切片概率 [0,1]; 0=关闭切片
        slice_overlap_ratio=0.2,      # 相邻切片重叠比例 [0,1); 例: 原图4000x3000+重叠0.2 -> 切片2400x1800
        slice_background_ratio=-1,    # 背景切片数=正切片数x该值; -1=全部背景保留; 0=不要背景 (废弃，与 slice_all_tiles 冲突)
        slice_all_tiles=True,         # 每张原图的 4 片子图全部参与训练
        slice_keep_origin=True,       # 默认False, 仅slice_all_tiles下生效: 每张原图额外保留 1 张未切片原图样本, len = 5N（4 切片 + 1 原图）

        # ---- 被切目标的保留判定 ----
        # slice_min_tile_area_ratio 与 slice_min_box_retain_ratio 是双重面积过滤阈值, 决定"被切片边界切到的目标, 露多少才值得保留"; AND 同时满足才丢弃
        slice_min_tile_area_ratio=0.005,  # 切片块面积下界: 切片面积 < 原图x该值 的切片丢弃
        slice_min_box_retain_ratio=0.4,   # 目标框保留下界: 目标在切片内可见面积占原框比例 < 该值则丢弃
        slice_center_constraint=True,     # 目标唯一归属: 每个目标只分配给"中心所在"切片, 防同一目标被切两半重复出现
        slice_min_center_retain_ratio=0.6,# 中心不在本片时, 若本片内可见面积占比 >= 该值仍保留; 1.0=严格只留中心片
        slice_full_box_only=False,        # 目标必须完整落在切片内才保留, 被边界切开即过滤; 开启时优先于 slice_center_constraint

        # ---- 切片保存 (人工检查切片是否正确) ----
        slice_save_annotated=True,   # 保存时画标注框+类别
        slice_save_max=0,            # 最多保存张数; 0=不限
        # slice_save_dir=r"C:\Users\ASUS\Desktop\ultralytics-main\base_0_0\sliced_save_dir",

        # ---------在线合成 (compose_*): 每 4 张原图拼 1 张 2x2 大图, 提供更大范围多目标上下文---------
        compose_keep=True, # 需开启 slice_all_tiles + slice_keep_origin; 每 4 张原图额外合成 1 张 2×2 大图进样本池
        compose_save=True, # 保存合成图 (默认False不保存); 需 compose_save_dir 或 slice_save_dir 指定目录
        # compose_save_dir=r"C:\Users\ASUS\Desktop\ultralytics-main\base_0_0\composed_save_dir",

        # ---------在线比例调整 (ratio_pad_*): 在线加边框统一宽高比---------
        ratio_pad_keep=True,          # 开启在线比例调整 (每图+1张比例对齐图)
        ratio_pad_target="auto",      # auto: 4:3↔16:9 双向 + 其他比例转最近（默认）
        ratio_pad_color="gray",       # 边框颜色 black/gray/white
        # ratio_pad_save_dir=r"C:\Users\ASUS\Desktop\ultralytics-main\base_0_0\change_proportion_save_dir",

        # ---------在线运动模糊 (blur_*): 模拟无人机运动失焦, 每张原图生成 2 张模糊副本(短+长), 标签不变---------
        blur_keep=True,               # 开启 33 样本池 (每图+2张模糊图)
        blur_short_len_min=5,         # 短模糊(轻度, 无失焦) 长度下限(像素)
        blur_short_len_max=12,        # 短模糊(轻度, 无失焦) 长度上限(像素)
        blur_long_len_min=20,         # 长模糊(重度) 长度下限(像素)
        blur_long_len_max=35,         # 长模糊(重度) 长度上限(像素)
        blur_long_defocus_sigma=1.0,  # 长模糊失焦高斯 σ 上限; 0=不加失焦
        # blur_save_dir=r"C:\Users\ASUS\Desktop\ultralytics-main\base_0_0\motion_blur_save_dir",
    )