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

# 仅屏蔽上游升级路径上的 DeprecationWarning/FutureWarning 类噪音,
# 不再一刀切 warnings.filterwarnings('ignore')。RuntimeWarning 等真错误必须保留冒泡
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
from ultralytics import YOLO

'''

样本池 = 区段式布局, 每个区段只受自己的独立开关控制 (互不影响, 不再被 slice_prob=0 强制关闭):
    [0, 4N)        切片    slice_prob / slice_all_tiles
    [4N, 5N)       原图    slice_keep_origin (独立开关)
    [5N, 6N)       比例    ratio_pad_keep    (独立开关)
    [6N, 8N)       模糊    blur_keep         (独立开关, 短+长)
    [8N, +N/4)     合成    compose_keep      (独立开关, 2×2 大图)
N=4 全开 = 16 + 4 + 4 + 8 + 1 = 33 张混合样本池 → mosaic 取 4 张拼接 → 训练

'''


if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/26/yolo26n.yaml')
    model.load('yolo26n.pt')
    model.train(
        #---------训练参数---------------
        data='data.yaml',
        cache=False,                
        imgsz=1280,
        epochs=2,
        batch=4,
        close_mosaic=20,
        workers=0,                   
        # 远程机性能调优参考(A100 80GB + HDD + 62GB内存):
        #   - 必开: OMP_NUM_THREADS=1(已在顶部设置), cache=False(HDD大图npy更慢)
        #   - 推荐: batch=16, workers=8, prefetch=2; slice_mix_ratio=0.7(降内存+缓解过拟合)
        optimizer='MuSGD',
        device='0',
        # resume=r'<项目根>/runs/<exp>/weights/last.pt',  # 断点续训: 改成你本机 last.pt 路径
        # resume_extend_epochs=5,  # (int, 0=关闭) 续训自动延长: 自动修补ckpt元数据(epochs/patience), 从旧停点续训到该轮数; 需>ckpt已完成轮数
        patience=50,
        amp=True,
        project=r"",  
        name='exp',
        exist_ok=False,


        # ---------Mosaic在线增强---------------
        mosaic=1.0,
        # mosaic_save_dir=r"<你的输出目录>/mosaic_save_dir",

        # ---------SAHI在线切片 (slice_*)-----------
        slice_prob=1.0,               # 在线切片概率 [0,1]; 0=关闭切片
        slice_overlap_ratio=0.2,      # 相邻切片重叠比例 [0,1); 例: 原图4000x3000+重叠0.2 -> 切片2400x1800
        # slice_background_ratio=-1,  # 背景切片保留比例: 背景切片数=正切片数x该值; -1=全部背景保留; 0=不要背景; emit_all 模式下每个背景切片独立判断是否保留
        slice_all_tiles=True,         # 每张原图的 4 片子图全部参与训练
        slice_mix_ratio=1.0,          # 切片/整图混排概率 [0,1]: 每个"切片样本位"有多大概率真正切片; 1.0=纯切片, 0.5=约一半切片位变整图(缓解过拟合+降内存/CPU), 0=等效关闭在线切片(样本数不变)

        slice_keep_origin=True,       # 每张原图额外保留 1 张未切片原图, 独立区段 [4N,5N); 切片关闭时自动抑制(避免重复原图); 全开仍 33

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
        # slice_save_dir=r"<你的输出目录>/sliced_save_dir",

        # ---------在线合成 (compose_*): 每 4 张原图拼 1 张 2x2 大图, 提供更大范围多目标上下文--------- (独立开关)
        compose_keep=True, # 独立开关, 不依赖切片/keep_origin; 每 4 张原图额外合成 1 张 2×2 大图进样本池 (区段 +ceil(N/4))
        compose_max_side=0, # 合成2x2大图拼后降采样最长边上限(像素): 0=自动=2×imgsz(默认开启, 降内存), >0=手动指定(如2560); 不想要此优化可设 compose_max_side 为一个很大的值关闭。
        compose_save=True, # 保存合成图 (默认False不保存); 需 compose_save_dir 或 slice_save_dir 指定目录
        # compose_save_dir=r"<你的输出目录>/composed_save_dir",

        # ---------在线比例调整 (ratio_pad_*): 在线加边框统一宽高比---------
        ratio_pad_keep=True,          # (独立开关, 不受slice_prob影响) 开启在线比例调整 (每图+1张比例对齐图, 区段 +N)
        ratio_pad_target="auto",      # auto: 4:3↔16:9 双向 + 其他比例转最近（默认）
        ratio_pad_color="gray",       # 边框颜色 black/gray/white
        # ratio_pad_save_dir=r"<你的输出目录>/change_proportion_save_dir",

        # ---------在线运动模糊 (blur_*): 模拟无人机运动失焦, 每张原图生成 2 张模糊副本(短+长), 标签不变---------
        blur_keep=True,               # (独立开关, 不受slice_prob影响) 开启在线运动模糊 (每图+2张模糊图, 区段 +2N)
        blur_short_len_min=5,         # 短模糊(轻度, 无失焦) 长度下限(像素)
        blur_short_len_max=12,        # 短模糊(轻度, 无失焦) 长度上限(像素)
        blur_long_len_min=20,         # 长模糊(重度) 长度下限(像素)
        blur_long_len_max=35,         # 长模糊(重度) 长度上限(像素)
        blur_long_defocus_sigma=1.0,  # 长模糊失焦高斯 σ 上限; 0=不加失焦
        # blur_save_dir=r"<你的输出目录>/motion_blur_save_dir",
    )