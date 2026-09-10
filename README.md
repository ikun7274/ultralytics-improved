# Ultralytics 增强版：在线 SAHI 切片 + 混合样本池 + 修补续训

在原生 Ultralytics（YOLO26）训练流程上扩展**在线数据增强**与**修补续训**，不改动原生训练功能：

- **在线 SAHI 切片**：把大尺寸航拍/遥感图在训练时动态切成 2×2 重叠子图，小目标经子图缩放真实放大；
- **在线合成 / 比例调整 / 运动模糊**：与切片一起构成**混合样本池**（全程内存操作，默认不落盘）；
- **修补续训**：训练提前结束（跑满 / 早停）后，对 strip 过的 `last.pt` 自动修补元数据并续训。

```text
4 张原图 ── 在线切片 16 张 ── 保留原图 4 张 ── 在线合成 1 张(2×2)
        ── 在线比例 4 张 ── 在线模糊 8 张(短+长) ──► 33 张混合样本池 ──► Mosaic ──► 训练
```

每个增强模块是**独立开关**（`slice_prob` / `slice_keep_origin` / `compose_keep` / `ratio_pad_keep` / `blur_keep`），且支持**epoch 级精确比例控制**（`slice_ratio` / `compose_ratio` / `ratio_pad_ratio` / `blur_ratio`），可任意组合或全关（全关 = 原生 Ultralytics）。

---

## 快速开始


```python
from ultralytics import YOLO

model = YOLO('ultralytics/cfg/models/26/yolo26n.yaml')
model.load('weights/yolo26n.pt')
model.train(
    data='data.yaml', imgsz=1280, epochs=100, batch=4, workers=0,
    # ---- 开启在线切片 ----
    slice_prob=1.0,           # 在线切片概率 [0,1]，0=关闭
    slice_all_tiles=True,     # 每图 4 片子图全部参与训练
    slice_keep_origin=True,   # 每图额外保留 1 张原图（全图上下文）
    # ---- 可选增强（独立开关）----
    compose_keep=True,        # 在线合成 2×2 大图（+ceil(N/4)）
    ratio_pad_keep=True,      # 在线比例调整（+N）
    blur_keep=True,           # 在线运动模糊，短+长（+2N）
)
```

训练开始前会打印实际样本数，例如：

```text
Online augment: 231 training samples from 28 images (4 slices + 1 origin + 1 ratio + 2 blur + N/4 compose per image)
```

---

## 混合样本池

样本池采用**区段式布局**，每个区段只受自己的独立开关控制（`set_epoch` 按 epoch 重建比例掩码）：

```text
[0, 4N)                切片     slice_prob / slice_all_tiles（slice_ratio 决定哪些原图走切片）
[4N, 5N)               原图     slice_keep_origin（独立开关）
[5N, 6N)               比例     ratio_pad_keep
[6N, 8N)               模糊     blur_keep（每图短/长各 1 张）
[8N, 8N+ceil(N/4))     合成     compose_keep（每 4 张原图拼 1 张 2×2 大图）
```

- 全开总数 = `8N + ceil(N/4)`（N=4 时 = 33）；
- `len` 恒定：所有"被拒/未选中"样本位**位置保留、内容替换**（如被拒背景片退回原图、未选中组退回组内第 1 张原图），避免预计算 + len 不恒定导致的 Mosaic buffer 记账错乱；
- 训练开始时日志打印实际参与训练样本数，可确认各增强确实参与训练。

---

## 参数参考

### 在线切片 `slice_*`

| 参数 | 默认 | 说明 |
|---|---|---|
| `slice_prob` | `0.0` | 在线切片概率 [0,1]，0=关闭切片 |
| `slice_ratio` | `1.0` | 每 epoch 精确选 `round(x×N)` 张原图走切片（原图级，4 片同生共死），其余整图进池；`1.0`=纯切片，`<1.0` 混入整图（缓解过拟合+降内存），`0`=全整图；每 epoch 重新随机 |
| `slice_overlap_ratio` | `0.2` | 相邻切片重叠比例 [0,1)。切片尺寸 = `原图×(1+overlap)/2`（2×2 网格），例：4000×3000 + 0.2 → 2400×1800 |
| `slice_min_tile_area_ratio` | `0.005` | 切片块面积下界：切片面积 < 原图×该值 则丢弃 |
| `slice_min_box_retain_ratio` | `0.4` | 目标框保留下界：被边界切到的目标，片内可见面积 / 原框面积 < 该值则丢弃 |
| `slice_background_ratio` | `-1` | 背景切片数 = 正样本切片数 × 该值；`-1`=全部背景保留，`0`=不要背景；`emit_all` 下被拒背景片**退回原图**（不裁空片） |
| `slice_all_tiles` | `False` | 每图 4 片子图全部参与训练（数据集 ×4）；`False`=每图随机取 1 片 |
| `slice_center_constraint` | `False` | 目标唯一归属：只分配给"中心所在"切片，防同一目标被切两半重复 |
| `slice_min_center_retain_ratio` | `0.6` | 配合 `center_constraint`：中心不在本片时，片内可见占比 ≥ 该值仍保留；1.0=严格只留中心片 |
| `slice_full_box_only` | `False` | 目标必须完整落在片内才保留，被边界切开即过滤；优先于 `center_constraint` |
| `slice_keep_origin` | `False` | 独立开关：每图额外保留 1 张未切片原图（+N，提供全图上下文）；切片关闭时自动抑制 |
| `slice_transform` | `""` | 切片后附加的变换名（如 `"hsv_h"` / `"blur"`），与 `pytools/sahi_*.py` 同名变换对应；空=不附加 |
| `slice_raw_cache_size` | `2` | 在线增强 worker 内原图 LRU 缓存大小（原图数）；同一原图最多被 imread 8~9 次，LRU 把解码降到 ~1 次/原图 |

### 在线合成 `compose_*`

| 参数 | 默认 | 说明 |
|---|---|---|
| `compose_keep` | `False` | 独立开关（不依赖切片）：每 4 张原图拼 1 张 2×2 大图进样本池（+ceil(N/4)） |
| `compose_ratio` | `1.0` | 组级比例：每 epoch 选 `round(x×⌈N/4⌉)` 组合成，未选中组退回组内第 1 张原图（len 恒定） |
| `compose_max_side` | `0` | 拼后降采样最长边上限（像素）；`0`=自动=2×imgsz（默认开启，4×4000×3000→8000×6000 约 144MB 降到 ~18MB）；`>0`=手动指定；标签为归一化坐标不受影响 |
| `compose_save` | `False` | 保存合成图供人工检查（需 `compose_keep` + 目录） |
| `compose_save_dir` | `""` | 合成图保存目录；空=回退 `slice_save_dir/compose/` |

### 在线比例调整 `ratio_pad_*`

| 参数 | 默认 | 说明 |
|---|---|---|
| `ratio_pad_keep` | `False` | 独立开关：每图生成 1 张"加边框统一宽高比"图（+N） |
| `ratio_pad_ratio` | `1.0` | 原图级比例：每 epoch 选 `round(x×N)` 张做比例调整，未选中直接整图（len 恒定） |
| `ratio_pad_target` | `auto` | `auto`=4:3↔16:9 双向对齐，其他比例转到最近的 4:3 或 16:9；`4:3`/`16:9`=统一到该比例 |
| `ratio_pad_color` | `black` | 边框颜色 `black`/`gray`/`white`（gray=114,114,114，与 YOLO letterbox 一致） |
| `ratio_pad_save_dir` | `""` | 保存目录；空=不保存 |

### 在线运动模糊 `blur_*`

| 参数 | 默认 | 说明 |
|---|---|---|
| `blur_keep` | `False` | 独立开关：每图生成 2 张运动模糊图（短+长，标签不变，+2N） |
| `blur_ratio` | `1.0` | 原图级比例：每 epoch 选 `round(x×N)` 张做模糊，短+长同命运，未选中整图×2（len 恒定） |
| `blur_short_len_min/max` | `5`/`12` | 短模糊（轻度，无失焦）运动模糊长度范围（像素） |
| `blur_long_len_min/max` | `20`/`35` | 长模糊（重度）长度范围（像素） |
| `blur_long_defocus_sigma` | `1.0` | 长模糊失焦高斯 σ 上限 [0,该值]，每张随机取；0=不加失焦 |
| `blur_save_dir` | `""` | 保存目录；空=不保存 |

### 中间图保存（默认不落盘）

| 参数 | 默认 | 说明 |
|---|---|---|
| `slice_save_dir` | `""` | 切片图保存目录；空=不保存 |
| `slice_save_max` | `0` | 四类（切片/比例/模糊/合成）共用保存限额；0=不限；可用 `slice_save_max_{tile,ratio,blur,compose}` 单独覆盖 |
| `slice_save_annotated` | `True` | 保存时画标注框+类别 |
| `slice_save_exist_ok` | `True` | 目录已存在是否继续写入；False=抛错防覆盖 |
| `mosaic_save_dir` | `""` | Mosaic 画布图保存目录（验证 mosaic 是否启用）；空=不保存 |
| `mosaic_save_max` | `0` | 最多保存的 Mosaic 画布数；0=不限 |
| `mosaic_save_annotated` | `True` | Mosaic 画布是否画框+类别 |
| `mosaic_save_exist_ok` | `True` | 目录已存在是否继续写入 |

### 修补续训（见 `项目说明.md` §3.6 详解）

| 参数 | 说明 |
|---|---|
| `resume` | 续训 ckpt 路径（如 `runs/exp/weights/last.pt`） |
| `resume_extend_epochs` | 自动修补续训到该轮数；必须 > 已完成轮数（已完成场景自动读 ckpt 的 train_args） |

---

## 工作原理

- **索引空间扩展**：`__getitem__` 索引从 N（原图）扩展为 `4N + N + N + 2N + ceil(N/4)`；`get_image_and_label` 按区段边界路由到切片 / 原图 / 比例 / 模糊 / 合成分支，每个子样本即时生成，`buffer.append(扩展索引)` 统一记账后供 Mosaic 采样；
- **在线切片在原分辨率上进行**（切片前不缩放到训练尺寸），小目标随子图缩放真实放大；
- **epoch 级精确比例**：`set_epoch(epoch)` 主进程重建各比例掩码（`random.sample` 精确选 `round(x×N)`），worker 经 fork 继承，无 per-worker 漂移；
- **标签始终与像素对齐**：切片/合成/比例都同步换算 bbox 坐标，模糊标签原样不变；
- **纯原版兼容**：所有增强全关时，`load_image` 恢复原生 buffer 自管路径，行为与原生 Ultralytics 完全一致（含 buffer 记账守卫，防止扩展索引与原生索引混用）。

---

## 注意事项

- **内存**：合成 2×2 大图 + 多 worker 预取是内存峰值主因。推荐 `compose_max_side=0`（自动降采样）、`slice_ratio` 降到 0.5~0.7、`workers` 2~4、`cache=False`；
- **勿开 `cache='ram'`**：在线增强全开会把每张原图多次读入内存，`cache='ram'` 直接爆内存；
- **不要 `pip install -U ultralytics`**：本项目是本地深度改造版，升级会覆盖全部自定义增强与续训功能。

---

## 目录结构（相关部分）

```text
ultralytics-improved/
├── train.py                     # 训练入口（全部在线增强 + 续训参数集中配置）
├── data.yaml                    # 数据集配置（当前: nc=6 车辆数据集）
├── README.md                    # 本文件（项目总览）
├── 项目说明.md                   # 深度说明：设计思路 / 模块详解 / 续训机制 / 排障经验
├── 更新说明.md                   # 功能更改逐条变更日志
├── ultralytics/
│   ├── data/
│   │   ├── base.py              # 索引空间扩展 / set_epoch 掩码 / _origin_at / _ratio_at / _blur_at / _compose_at
│   │   └── augment.py           # OnlineSlice 切片器 / Mosaic 保存 / 独立开关装配
│   ├── engine/trainer.py        # set_epoch 钩子 / resume_training 修补块 / check_resume 白名单
│   ├── optim/muon.py            # Muon 优化器（u.view→u.reshape 修复）
│   └── cfg/
│       ├── default.yaml         # 全部参数定义与中文注释
│       └── __init__.py          # 参数类型校验注册
├── pytools/                     # 离线版工具（在线化的设计来源）
│   ├── sahi_equal_division_slice_dataset_auto_improved.py
│   ├── compose_slice_dataset_auto_improved.py
│   ├── change_image_resolution_slice_dataset_auto_improved.py
│   ├── motion_blur.py
│   └── fix_checkpoint_for_extension.py
└── weights/yolo26n.pt           # 预训练权重
```
