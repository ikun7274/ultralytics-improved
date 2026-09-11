# -*- coding: utf-8 -*-
import os
import warnings

# 离线训练: True=跳过启动时 PyPI 版本检查(远程机网络不稳时避免卡住); False=恢复联网检查
TRAIN_OFFLINE = False
if TRAIN_OFFLINE:
    os.environ["YOLO_OFFLINE"] = "1"

# 限制 DataLoader worker 内部线程数, 避免 workers×内部多线程导致 CPU 切换风暴
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# 仅屏蔽上游升级路径上的 DeprecationWarning/FutureWarning 类噪音, 不再一刀切 warnings.filterwarnings('ignore')。RuntimeWarning 等真错误必须保留冒泡
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
from ultralytics import YOLO

'''

样本池 = 区段式布局, 每个区段只受自己的独立开关控制 (互不影响):
    [0, 4N)        切片    slice_prob / slice_all_tiles
    [4N, 5N)       原图    slice_keep_origin (独立开关)
    [5N, 6N)       比例    ratio_pad_keep    (独立开关)
    [6N, 8N)       模糊    blur_keep         (独立开关, 短+长)
    [8N, 8N+N/4)   合成    compose_keep      (独立开关, 2×2 大图)
    [8N+N/4, +N)   气象退化 weather_keep     (独立开关, 雨/雾/噪声)
N=4 全开 = 16 + 4 + 4 + 8 + 1 + 4 = 37 张混合样本池 → mosaic 取 4 张拼接 → 训练

'''


if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/26/yolo26n.yaml')
    model.load('weights/yolo26n.pt')
    model.train(
        #---------训练参数---------------
        data='data.yaml',
        cache=False,                
        imgsz=640,
        epochs=1,
        batch=4,
        close_mosaic=20,
        workers=0,                   
        optimizer='MuSGD',
        device='0',
        # resume=r'C:\Users\Administrator\Desktop\ultralytics-improved\runs\exp-3\weights\last.pt',  # 断点续训: 改成你本机 last.pt 路径
        # resume_extend_epochs=15,  # (int, 0=关闭) 续训自动延长: 自动修补ckpt元数据(epochs/patience), 从旧停点续训到该轮数; 需>ckpt已完成轮数
        patience=50,
        amp=True,
        fraction=0.01,
        project=r"ultralytics-improved\runs",  
        name='exp',
        exist_ok=False,


        # ---------Mosaic在线增强---------------
        mosaic=1.0,

        # ---------SAHI在线切片 (slice_*)-----------
        slice_prob=1.0,               # 在线切片概率 [0,1]; 0=关闭切片
        slice_overlap_ratio=0.2,      # 相邻切片重叠比例 [0,1); 例: 原图4000x3000+重叠0.2 -> 切片2400x1800
        slice_all_tiles=True,         # True：每张原图的 4 片子图全部参与训练；False：每张原图随机取 1 片子图参与训练
        slice_background_ratio=-1,    # 背景切片保留比例: 背景切片数=正切片数x该值; -1=全部背景保留; 0=不要背景；
                                      # 注意: slice_all_tiles=True下被拒的背景切片会退回原图。
                                      # 原因：DataLoader 每个 epoch 必须产出固定数量（4N）的样本。空片占的位置不能消失，只能有两种处理：
                                      # 1. 位置消失（真・丢弃）→ `__len__` 从 4N 变成不确定值 → "预计算 + len 不恒定" ：mosaic buffer 记账错乱、epoch 内 batch 数漂移、
                                      # 进度条和训练循环不同步。要修得改 sampler + buffer + 训练循环，工程量大且容易出隐蔽 bug。
                                      # 2. 位置保留、内容替换（丢弃空片，用别的样本补位）→ len 恒定为 4N，无副作用。

        slice_ratio=0.1,              # 每个epoch随机选 N*0.5 张原图走切片, 其余整图进池; 1.0=纯切片, 0.5=一半原图切片, 0=全整图(等效关闭切片); 每epoch重新随机. 
        slice_keep_origin=False,      # 每张原图额外保留 1 张未切片原图, 独立区段 [4N,5N); 切片关闭时自动抑制(避免重复原图); 设置slice_ratio>0时, 推荐关闭。

        # 被切目标的保留判定
        slice_min_tile_area_ratio=0.005,  # 切片块面积下界: 切片面积 < 原图x该值 的切片丢弃
        slice_min_box_retain_ratio=0.4,   # 目标框保留下界: 目标在切片内可见面积占原框比例 < 该值则丢弃
        slice_center_constraint=True,     # 目标唯一归属: 每个目标只分配给"中心所在"切片, 防同一目标被切两半重复出现
        slice_min_center_retain_ratio=0.6,# 中心不在本片时, 若本片内可见面积占比 >= 该值仍保留; 1.0=严格只留中心片
        slice_full_box_only=False,        # 目标必须完整落在切片内才保留, 被边界切开即过滤; 开启时优先于 slice_center_constraint


        # ---------在线合成 (compose_*): 每 4 张原图拼 1 张 2x2 大图, 提供更大范围多目标上下文--------- 
        compose_keep=True,  # 独立开关; 每 4 张原图额外合成 1 张 2×2 大图进样本池 (区段 +ceil(N/4))
        compose_max_side=0, # 合成2x2大图拼后降采样最长边上限(像素): 0=自动=2×imgsz(默认开启, 降内存), >0=手动指定(如2560); 不想要此优化可设 compose_max_side 为一个很大的值关闭。
        compose_ratio=0.1,  # 每epoch随机选 round(x*ceil(N/4)) 组合成(组级), 未选中组退回组内第1张原图(整图直通, len恒定); 1.0=全量合成

        # ---------在线比例调整 (ratio_pad_*): 在线加边框统一宽高比---------
        ratio_pad_keep=True,          # 独立开关, 开启在线比例调整 (每图+1张比例对齐图, 区段 +N)
        ratio_pad_ratio=0.1,          # 每epoch随机选 round(x*N) 张原图做比例调整(原图级), 未选中直接整图; 1.0=全量
        ratio_pad_target="auto",      # auto: 4:3↔16:9 双向 + 其他比例转最近（默认）
        ratio_pad_color="gray",       # 边框颜色 black/gray/white

        # ---------在线运动模糊 (blur_*): 模拟无人机运动失焦, 每张原图生成 2 张模糊副本(短+长), 标签不变---------
        blur_keep=True,              # 独立开关, 开启在线运动模糊 (每图+2张模糊图, 区段 +2N)
        blur_ratio=0.1,               # 每epoch随机选 round(x*N) 张原图做运动模糊(原图级, 短+长同命运), 未选中整图x2; 1.0=全量
        blur_short_len_min=5,         # 短模糊(轻度, 无失焦) 长度下限(像素)
        blur_short_len_max=12,        # 短模糊(轻度, 无失焦) 长度上限(像素)
        blur_long_len_min=20,         # 长模糊(重度) 长度下限(像素)
        blur_long_len_max=35,         # 长模糊(重度) 长度上限(像素)
        blur_long_defocus_sigma=1.0,  # 长模糊失焦高斯 σ 上限; 0=不加失焦

        # ---------在线气象退化 (weather_*): 每张原图生成 1 张雨/雾/噪声退化图, 提升恶劣天气鲁棒性, 标签不变---------
        weather_keep=False,           # 独立开关, 开启在线气象退化 (每图+1张退化图, 区段 +N)
        weather_ratio=0.5,            # 每epoch随机选 round(x*N) 张原图做气象退化(原图级), 未选中整图直通; 0.3~0.6 推荐, 1.0=全量
        weather_types="rain,haze,noise", # (str) 退化类型池, 逗号分隔; 每张图随机抽 1 种; 可子集如 "rain,haze"
        weather_rain_density=0.15,    # 雨线密度 = 雨线数量 / max(h,w), 越大雨越密
        weather_rain_length=15.0,     # 雨线长度上限(像素), 每根随机取 0.5~1.0 倍
        weather_haze_beta=0.4,        # 雾浓度 [0,1), 越大雾越浓
        weather_noise_std=15.0,       # 高斯噪声标准差(每通道独立), 模拟传感器/弱光噪点

        # ---------训练后期关闭在线增强 (close_aug_epoch): 与 close_mosaic 同构的时间维衰减---------
        # 训练最后 N 个 epoch 把切片/合成/比例/模糊/气象退化全部关闭, 各区段退回原图直通(len恒定), 让模型在真实分布上收敛;
        close_aug_epoch=20, # 0=关闭该调度(默认, 完全向后兼容)

        # ---------验证侧在线切片评估 (val_slice_*): 验证集切片推理 + 坐标还原 + NMS 融合 (SAHI评估)---------
        # 训练侧已在线切片, 验证侧整图直推会因小目标被降采样而低估切片训练收益.
        val_slice_enable=True,        # 总开关: 验证时把验证图切成 2x2 重叠子图独立推理, 子图框还原到原图坐标,
                                      #         跨切片重复框 NMS 融合后与原图 GT 算 mAP; False=回归原生整图验证
        val_slice_all_tiles=True,     # True=每张验证图全部 2x2 子图都推理(与训练侧对齐); False=每图随机1片(快速验证)
        val_slice_ratio=1.0,          # 每轮验证随机选 round(x*N_val) 张验证图走切片, 其余整图直通; 1.0=全部切片
        val_slice_overlap_ratio=0.2,  # 验证侧切片重叠比例 [0,1), 建议与训练侧 slice_overlap_ratio 一致
        val_slice_nms_iou=0.5,        # 跨切片重复框 NMS 融合 IoU 阈值
        val_slice_dual_metric=True,   # 双口径: 先跑切片验证(主, 驱动 fitness/早停/best.pt), 再跑整图验证(参考),
                                      #         输出 whole_* 指标并额外保存 best_whole.pt/last_whole.pt (耗时为两遍验证)

        
        # ---- 在线增强保存 (人工检查切片是否正确) ----
        # slice_save_annotated=True,   # 保存时画标注框+类别
        # slice_save_max=0,            # 最多保存张数; 0=不限
        # compose_save=True, # 保存合成图 (默认False不保存)
        # slice_save_dir=r"path/sliced_save_dir",
        # compose_save_dir=r"path/composed_save_dir",
        # ratio_pad_save_dir=r"path/change_proportion_save_dir",
        # blur_save_dir=r"path/motion_blur_save_dir",
        # weather_save_dir=r"path/weather_save_dir",
        # mosaic_save_dir=r"path/mosaic_save_dir",
    )