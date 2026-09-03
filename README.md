# Ultralytics 增强版：在线 SAHI 切片 + 混合样本池

在 Ultralytics YOLO26 训练流程中集成 **SAHI 式在线切片** 与 **多路在线增强**，把大尺寸航拍/遥感图在训练时动态切成子图放大，并在**同一内存样本池**中混入原图、合成大图、比例对齐图与运动模糊图，再统一送入 Mosaic 增强，实现"让模型看到更大范围、更多样化的目标上下文"。

**核心特点：全程内存操作，不落盘**——除显式指定保存目录外，不产生任何中间文件。

---

## 1. 混合样本池构成

以每 **4 张原图** 为一组，在线生成 **33 张混合样本池**：

```text
4 张原图
├── 在线切片 ──────→ 16 张切片图   (每图 4 张，2x2 重叠切片)
├── 保留原图 ──────→  4 张原图     (提供全图上下文)
├── 在线合成 ──────→  1 张合成大图  (4 图 2x2 拼接，例 8000×6000)
├── 在线比例调整 ──→  4 张比例图    (加边框统一宽高比)
└── 在线运动模糊 ──→  8 张模糊图    (每图短+长各 1，模拟运动失焦)
                    ─────────────────
                    33 张混合样本池 → 一起进 Mosaic → 训练
```

- 数据集长度：`len = 8N + ceil(N/4)`（N = 原图数）。以本仓库 `data.yaml`（N=28）为例：**231 个训练样本**（112 切片 + 28 原图 + 28 比例图 + 56 模糊图 + 7 合成大图）。
- 训练开始时日志会打印实际构成：
  `Online slicing: 231 training samples from 28 images (4 slices + 1 origin per image) + 1 ratio + 2 blur + N/4 compose`

各模块可**独立开关**：切片、原图保留、合成、比例、模糊互不依赖（合成/比例/模糊需 `slice_all_tiles + slice_keep_origin` 前置）。

---

## 2. 快速开始

```bash
# 激活 conda 环境
conda activate computevision

# 训练（train.py 已配置完整样本池）
python train.py
```

最小启用在线切片（4 切片 + 1 原图）：

```python
from ultralytics import YOLO

model = YOLO('ultralytics/cfg/models/26/yolo26n.yaml')
model.load('yolo26n.pt')
model.train(
    data='data.yaml',
    imgsz=1280, epochs=100, batch=4, workers=0,
    # ---- 开启在线切片 ----
    slice_prob=1.0,          # 在线切片概率
    slice_all_tiles=True,    # 每图 4 子图全部参与训练
    slice_keep_origin=True,  # 每图额外保留 1 张原图
    # ---- 可选增强 ----
    compose_keep=True,       # 在线合成大图
    ratio_pad_keep=True,     # 在线比例调整
    blur_keep=True,          # 在线运动模糊
)
```

---

## 3. 参数参考

### 3.1 SAHI 在线切片 `slice_*`

| 参数 | 默认 | 说明 |
|---|---|---|
| `slice_prob` | `0.0` | 在线切片概率 [0,1]，0=关闭 |
| `slice_mix_ratio` | `1.0` | batch 中走切片的样本占比；`<1.0` 混入未切片整图，缓解小数据集对切片分布过拟合 |
| `slice_overlap_ratio` | `0.2` | 相邻切片重叠比例 [0,1)。切片尺寸 = `原图×(1+overlap)/2`（2x2 网格），例：原图 4000×3000 + overlap 0.2 → 切片 2400×1800 |
| `slice_min_tile_area_ratio` | `0.005` | 切片块面积下界：切片面积 < 原图×该值 则丢弃 |
| `slice_min_box_retain_ratio` | `0.4` | 目标框保留下界：被边界切到的目标，其在切片内可见面积占原框比例 < 该值则丢弃 |
| `slice_background_ratio` | `0.2` | 背景切片数 = 正样本切片数 × 该值；负值（如 -1）= 全部背景保留；0 = 不要背景 |
| `slice_all_tiles` | `False` | 每张原图的 4 片子图全部参与训练（数据集 ×4）；`False` = 每图随机取 1 片 |
| `slice_keep_origin` | `False` | 需 `slice_all_tiles=True`：每图额外保留 1 张未切片原图 → 数据集 ×5 |
| `slice_center_constraint` | `False` | 目标唯一归属：每个目标只分配给"中心所在"切片，防同一目标被切两半重复出现 |
| `slice_min_center_retain_ratio` | `0.6` | 配合 `center_constraint`：目标中心不在本片时，本片内可见面积占比 ≥ 该值仍保留；1.0 = 严格只留中心片 |
| `slice_full_box_only` | `False` | 目标必须完整落在切片内才保留，被边界切开即过滤；开启时优先于 `center_constraint` |

### 3.2 在线合成 `compose_*`

| 参数 | 默认 | 说明 |
|---|---|---|
| `compose_keep` | `False` | 需 `slice_all_tiles + slice_keep_origin`：每 4 张原图额外合成 1 张 2×2 拼接大图进样本池，提供更大范围多目标上下文 |
| `compose_save` | `False` | **默认不保存**。需 `compose_keep` + 指定目录：保存合成大图供人工检查 |
| `compose_save_dir` | `""` | 合成图自定义保存目录 |

### 3.3 在线比例调整 `ratio_pad_*`

| 参数 | 默认 | 说明 |
|---|---|---|
| `ratio_pad_keep` | `False` | 需 `slice_all_tiles + slice_keep_origin`：每图额外生成 1 张"加边框统一宽高比"的图进样本池 |
| `ratio_pad_target` | `auto` | `auto` = 4:3↔16:9 双向对齐，其他比例转到最近的 4:3 或 16:9；`4:3` / `16:9` = 所有图统一到该比例 |
| `ratio_pad_color` | `black` | 边框颜色：`black` / `gray` / `white`（gray=114,114,114 与 YOLO letterbox 一致） |
| `ratio_pad_save_dir` | `""` | 比例调整图保存目录；空 = 不保存 |

### 3.4 在线运动模糊 `blur_*`

| 参数 | 默认 | 说明 |
|---|---|---|
| `blur_keep` | `False` | 需 `slice_all_tiles + slice_keep_origin`：每图额外生成 2 张运动模糊图（短+长）进样本池，模拟无人机运动失焦；**标签不变** |
| `blur_short_len_min` / `blur_short_len_max` | `5` / `12` | 短模糊（轻度，无失焦）运动模糊长度范围（像素） |
| `blur_long_len_min` / `blur_long_len_max` | `20` / `35` | 长模糊（重度）运动模糊长度范围（像素） |
| `blur_long_defocus_sigma` | `1.0` | 长模糊失焦高斯 σ 上限 [0,该值]，每张随机取；0 = 长模糊不加失焦 |
| `blur_save_dir` | `""` | 模糊图保存目录；空 = 不保存 |

### 3.5 中间图保存 `slice_save_*` / `mosaic_save_*`

| 参数 | 默认 | 说明 |
|---|---|---|
| `slice_save_dir` | `""` | 切片图保存目录；空 = 不保存 |
| `slice_save_max` | `0` | 最多保存张数；0 = 不限 |
| `slice_save_annotated` | `True` | 保存时是否画标注框 + 类别 |
| `slice_save_exist_ok` | `True` | 目录已存在时是否允许继续写入；False = 抛错防覆盖 |
| `mosaic_save_dir` | `""` | Mosaic 画布图保存目录，用于验证 mosaic 是否启用；空 = 不保存 |
| `mosaic_save_max` | `0` | 最多保存的 Mosaic 画布图数量；0 = 不限 |
| `mosaic_save_annotated` | `True` | 保存的 Mosaic 画布图是否画框 + 类别 |
| `mosaic_save_exist_ok` | `True` | 目录已存在时是否允许继续写入 |

---

## 4. 保存规则（重要）

**所有中间图片默认不保存**，除非显式指定了保存目录：

| 中间图 | 保存条件 |
|---|---|
| 切片图 | 指定 `slice_save_dir` |
| 合成大图 | `compose_save=True` **且** 指定 `compose_save_dir` |
| 比例调整图 | 指定 `ratio_pad_save_dir` |
| 运动模糊图 | 指定 `blur_save_dir` |
| Mosaic 画布 | 指定 `mosaic_save_dir` |

不指定任何目录时，全部增强在内存中完成，**零文件落盘**。保存功能用于人工检查切片/合成/比例/模糊是否正确。

---

## 5. 工作原理

- **在线切片**：在 `get_image_and_label` 中读取**原分辨率**图像进行切片（切片前不缩放到训练尺寸），因此小目标被切片放大后真实增大；切片尺寸 = `原图 × (1 + slice_overlap_ratio) / 2`（2x2 网格）。
- **样本池布局**：每个原图展开为 8 个样本（4 切片 + 1 原图 + 1 比例图 + 2 模糊图），每 4 组再叠加 1 个合成样本，`index` 布局由 `get_image_and_label` 统一分发。
- **惰性生成**：合成/比例/模糊样本按 `index` 首次访问时生成，生成后进入 Mosaic 缓冲池（`dataset.buffer`），不缓存大图以控制内存。
- **混合进 Mosaic**：所有样本（切片、原图、合成、比例、模糊）都是样本池的一员，Mosaic 取 4 张拼接训练。
- **训练前打印样本数**：启动时日志显示展开后的实际训练样本数，便于核对。

---

## 6. 注意事项

- **内存**：若内存分配失败崩溃，可设置 `workers=0`。
- **合成大图**：4 张 4000×3000 拼接约 8000×6000（≈137MB），惰性生成不缓存；每 epoch 数据管线会因此变慢（完整遍历 231 样本约 18s，其中 56 张模糊卷积为主要开销）。
- **训练耗时**：开启模糊后数据加载时间上升，可降低 `blur_*` 档位或样本量平衡。
---

## 7. 目录结构（相关部分）

```text
ultralytics-main/
├── train.py                     # 训练入口（已配置 33 样本池）
├── data.yaml                    # 数据集配置（base_0_0，nc=1，human）
├── ultralytics/
│   ├── data/
│   │   ├── base.py              # 样本池分发 / 合成 / 比例 / 模糊生成
│   │   └── augment.py           # OnlineSlice 切片器 + v8_transforms 挂载
│   └── cfg/
│       ├── default.yaml         # 全部参数定义与注释
│       └── __init__.py          # 参数类型校验
├── pytools/                     # 离线版工具（在线化的参考实现）
│   ├── sahi_equal_division_slice_dataset_auto_improved.py
│   ├── compose_slice_dataset_auto_improved.py
│   ├── change_image_resolution_slice_dataset_auto_improved.py
│   └── motion_blur.py
└── sahi_test/                   # 测试与脚本
    └── test_online_slice.py
```
