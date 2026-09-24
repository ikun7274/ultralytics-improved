# Ultralytics 增强版：在线数据增强 + 混合样本池 + 修补续训

在原生 Ultralytics（YOLO26）训练流程上扩展**在线数据增强**与**修补续训**，不改动原生训练功能，全关所有增强时完全等价原生 Ultralytics。

- **在线 SAHI 切片**：把大尺寸航拍/遥感图在训练时动态切成 2×2 重叠子图，小目标经子图缩放真实放大；
- **在线合成 / 比例调整 / 运动模糊 / 气象退化**：与切片一起构成**混合样本池**（全程内存操作，默认不落盘）；
- **修补续训**：训练提前结束（跑满 / 早停）后，对 strip 过的 `last.pt` 自动修补元数据并续训。

### 混合样本池

```text
4 张原图 ── 在线切片 16 张 ── 保留原图 4 张 ── 在线合成 1 张(2×2)
        ── 在线比例 4 张 ── 在线模糊 8 张(短+长) ── 气象退化 4 张(雨/雾/噪声)
        ── 在线遮挡 4 张(rect/stripe)
        ──► 41 张混合样本池(不包含原图) ──► Mosaic ──► 训练
```

```text
[0, 4N)                切片     slice_prob / slice_all_tiles（slice_ratio 决定哪些原图走切片）
[4N, 5N)               原图     slice_keep_origin（独立开关）
[5N, 6N)               比例     ratio_pad_keep
[6N, 8N)               模糊     blur_keep（每图短/长各 1 张）
[8N, 8N+ceil(N/4))     合成     compose_keep（每 4 张原图拼 1 张 2×2 大图）
[8N+ceil(N/4), +N)     气象退化 weather_keep（每图 1 张雨/雾/噪声图）
[8N+ceil(N/4)+N, +N)   遮挡     occlusion_keep（每图 1 张 rect/stripe 遮挡图）
```

- `len` 恒定：所有"被拒/未选中"样本位**位置保留、内容替换**（如被拒背景片退回原图、未选中组退回组内第 1 张原图），避免预计算 + len 不恒定导致的 Mosaic buffer 记账错乱；
- 训练开始时日志打印实际参与训练样本数，可确认各增强确实参与训练。

每个增强模块是**独立开关**（`slice_prob` / `slice_keep_origin` / `compose_keep` / `ratio_pad_keep` / `blur_keep` / `weather_keep` / `occlusion_keep`），且支持**epoch 级精确比例控制**（`slice_ratio` / `compose_ratio` / `ratio_pad_ratio` / `blur_ratio` / `weather_ratio` / `occlusion_ratio`）与**训练后期统一关闭**（`close_aug_epoch`，类似 `close_mosaic`），可任意组合或全关（全关 = 原生 Ultralytics）。验证侧另支持**切片评估**（`val_slice_*`，SAHI 式切片推理 + NMS 融合 + 双口径 mAP 与两套权重）。

---

## 快速开始

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/26/yolo26n.yaml")
model.load("weights/yolo26n.pt")
model.train(
    data="data.yaml",
    imgsz=1280,
    epochs=100,
    batch=4,
    workers=0,
    # ---- 在线切片 ----
    slice_prob=1.0,  # 在线切片概率 [0,1]，0=关闭
    slice_all_tiles=True,  # 每图 4 片子图全部参与训练
    slice_ratio=1.0,  # 每 epoch 精确选 round(x×N) 张原图走切片；<1.0 混入整图
    slice_background_ratio=0.3,  # 背景片/正片比例（被拒背景片退回原图）
    # ---- 可选增强（独立开关，互不影响）----
    compose_keep=True,  # 在线合成 2×2 大图（+ceil(N/4)）
    ratio_pad_keep=True,  # 在线比例调整（+N）
    blur_keep=True,  # 在线运动模糊，短+长（+2N）
    weather_keep=True,  # 在线气象退化 雨/雾/噪声（+N，推荐 weather_ratio=0.3~0.6）
    occlusion_keep=True,  # 在线遮挡模拟 rect/stripe（+N，推荐 occlusion_ratio=0.3~0.6）
    # ---- 训练后期关闭在线增强 ----
    close_aug_epoch=5,  # 最后 5 个 epoch 全部增强退回原图，让模型在真实分布上收敛
)
```

训练开始前会打印实际样本数，例如：

```text
Online augment: 231 training samples from 28 images (4 slices + 1 origin + 1 ratio + 2 blur + N/4 compose per image)
```

## 参数参考

### 在线切片 `slice_*`

| 参数                            | 默认    | 说明                                                                                                                                                               |
| ------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `slice_prob`                    | `0.0`   | 在线切片概率 [0,1]，0=关闭切片                                                                                                                                     |
| `slice_ratio`                   | `1.0`   | 每 epoch 精确选 `round(x×N)` 张原图走切片（原图级，4 片同生共死），其余整图进池；`1.0`=纯切片，`<1.0` 混入整图（缓解过拟合+降内存），`0`=全整图；每 epoch 重新随机 |
| `slice_overlap_ratio`           | `0.2`   | 相邻切片重叠比例 [0,1)。切片尺寸 = `原图×(1+overlap)/2`（2×2 网格），例：4000×3000 + 0.2 → 2400×1800                                                               |
| `slice_min_tile_area_ratio`     | `0.005` | 切片块面积下界：切片面积 < 原图×该值 则丢弃                                                                                                                        |
| `slice_min_box_retain_ratio`    | `0.4`   | 目标框保留下界：被边界切到的目标，片内可见面积 / 原框面积 < 该值则丢弃                                                                                             |
| `slice_background_ratio`        | `-1`    | 背景切片数 = 正样本切片数 × 该值；`-1`=全部背景保留，`0`=不要背景；配额不足的背景片**退回原图**（不裁空片）                                                        |
| `slice_all_tiles`               | `False` | 每图 4 片子图全部参与训练（数据集 ×4）；`False`=每图随机取 1 片                                                                                                    |
| `slice_center_constraint`       | `False` | 目标唯一归属：只分配给"中心所在"切片，防同一目标被切两半重复                                                                                                       |
| `slice_min_center_retain_ratio` | `0.6`   | 配合 `center_constraint`：中心不在本片时，片内可见占比 ≥ 该值仍保留；1.0=严格只留中心片                                                                            |
| `slice_full_box_only`           | `False` | 目标必须完整落在片内才保留，被边界切开即过滤；优先于 `center_constraint`                                                                                           |
| `slice_center_bias`             | `False` | 目标感知切缝：切缝按本图目标中心分布微移，落在目标最稀疏区间，减少目标被劈碎；与 `center_constraint` 互补建议同开；验证侧固定网格不受影响                          |
| `slice_bias_margin`             | `0.25`  | 切缝偏移窗口：切缝只在 `[margin, 1-margin]` 区间内微移，保证 tile 不过小                                                                                           |
| `slice_bias_jitter`             | `0.05`  | 切缝抖动幅度（相对原图边长）：每图每 epoch 只抖一次，同一图同一 epoch 的 4 片共用同一网格，跨 epoch 才变化；0=关闭                                                 |
| `slice_keep_origin`             | `False` | 独立开关：每图额外保留 1 张未切片原图（+N，提供全图上下文）；切片关闭时自动抑制                                                                                    |
| `slice_raw_cache_size`          | `2`     | 在线增强 worker 内原图 LRU 缓存大小（原图数）；同一原图最多被 imread 8~9 次，LRU 把解码降到 ~1 次/原图；实际下限为 4（合成分支一次要读 4 张原图）                  |
| `slice_grouped_sampler`         | `True`  | 分组采样：同一张原图的子样本连续（单元内轮转）采样，让上面的 LRU 真正命中；False=上游全局 shuffle                                                                  |

### 在线合成 `compose_*`

| 参数               | 默认    | 说明                                                                                                                                           |
| ------------------ | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `compose_keep`     | `False` | 独立开关（不依赖切片）：每 4 张原图拼 1 张 2×2 大图进样本池（+ceil(N/4)）                                                                      |
| `compose_ratio`    | `1.0`   | 组级比例：每 epoch 选 `round(x×⌈N/4⌉)` 组合成，未选中组退回组内第 1 张原图（len 恒定）                                                         |
| `compose_max_side` | `0`     | 拼后降采样最长边上限（像素）；`0`=自动=2×imgsz（默认开启，4×4000×3000→8000×6000 约 144MB 降到 ~18MB）；`>0`=手动指定；标签为归一化坐标不受影响 |
| `compose_save`     | `False` | 保存合成图供人工检查（需 `compose_keep` + 目录）                                                                                               |
| `compose_save_dir` | `""`    | 合成图保存目录；空=回退 `slice_save_dir/compose/`                                                                                              |

### 在线比例调整 `ratio_pad_*`

| 参数                 | 默认    | 说明                                                                                |
| -------------------- | ------- | ----------------------------------------------------------------------------------- |
| `ratio_pad_keep`     | `False` | 独立开关：每图生成 1 张"加边框统一宽高比"图（+N）                                   |
| `ratio_pad_ratio`    | `1.0`   | 原图级比例：每 epoch 选 `round(x×N)` 张做比例调整，未选中直接整图（len 恒定）       |
| `ratio_pad_target`   | `auto`  | `auto`=4:3↔16:9 双向对齐，其他比例转到最近的 4:3 或 16:9；`4:3`/`16:9`=统一到该比例 |
| `ratio_pad_color`    | `black` | 边框颜色 `black`/`gray`/`white`（gray=114,114,114，与 YOLO letterbox 一致）         |
| `ratio_pad_save_dir` | `""`    | 保存目录；空=不保存                                                                 |

### 在线运动模糊 `blur_*`

| 参数                      | 默认      | 说明                                                                                                                                                                          |
| ------------------------- | --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `blur_keep`               | `False`   | 独立开关：每图生成 2 张运动模糊图（短+长，标签不变，+2N）                                                                                                                     |
| `blur_ratio`              | `1.0`     | 原图级比例：每 epoch 选 `round(x×N)` 张做模糊，短+长同命运，未选中整图×2（len 恒定）                                                                                          |
| `blur_short_len_min/max`  | `5`/`12`  | 短模糊（轻度，无失焦）运动模糊长度范围（像素）                                                                                                                                |
| `blur_long_len_min/max`   | `20`/`35` | 长模糊（重度）长度范围（像素）                                                                                                                                                |
| `blur_long_defocus_sigma` | `1.0`     | 长模糊失焦高斯 σ 上限 [0,该值]，每张随机取；0=不加失焦                                                                                                                        |
| `blur_axis_aligned`       | `True`    | 拖影方向限制为像面轴对齐（水平/垂直二选一随机），而非 [0,180) 任意角；适用于相机安装固定、运动方向在像面上恒定映射到水平或垂直的场景（车载/航拍沿航迹）；False=任意角（回退） |
| `blur_save_dir`           | `""`      | 保存目录；空=不保存                                                                                                                                                           |

### 在线气象退化 `weather_*`

> 每张原图生成 1 张退化图（类型从 `weather_types` 随机抽 1 种），**标签不变**，用于提升航拍模型在恶劣天气（雨 / 雾 / 噪声）下的鲁棒性。权衡：增强分布过宽会拖累干净场景精度，建议 `weather_ratio` 控制在 0.3~0.6。

| 参数                   | 默认              | 说明                                                                                    |
| ---------------------- | ----------------- | --------------------------------------------------------------------------------------- |
| `weather_keep`         | `False`           | 独立开关：每图生成 1 张气象退化图（+N）                                                 |
| `weather_ratio`        | `0.5`             | 原图级比例：每 epoch 选 `round(x×N)` 张做退化，未选中整图直通（len 恒定）；0.3~0.6 推荐 |
| `weather_types`        | `rain,haze,noise` | 退化类型池（逗号分隔），每张图随机抽 1 种；可子集如 `"rain,haze"`                       |
| `weather_rain_density` | `0.15`            | 雨线密度 = 雨线数量 / max(h,w)，越大雨越密                                              |
| `weather_rain_length`  | `15.0`            | 雨线长度上限（像素），每根随机取 0.5~1.0 倍                                             |
| `weather_haze_beta`    | `0.4`             | 雾浓度 [0,1)，越大雾越浓（大气散射 `I=J×(1-β)+A×β`）                                    |
| `weather_noise_std`    | `15.0`            | 高斯噪声标准差（每通道独立），模拟传感器/弱光噪点                                       |
| `weather_save_dir`     | `""`              | 保存目录（画框/限数量/按图+类型跨 epoch 去重）；空=不保存                               |

### 在线遮挡模拟 `occlusion_*`

> 每张被选中原图生成 1 张语义遮挡图（rect=树冠/阴影随机矩形，stripe=电线/枝干细长条带），**标签不变**（超阈值目标除外），提升航拍模型对部分遮挡目标的鲁棒性。`occlusion_color=auto` 时块色采样图像暗 25% 分位均值，融入场景而非突兀黑块。

| 参数                   | 默认          | 说明                                                                                    |
| ---------------------- | ------------- | --------------------------------------------------------------------------------------- |
| `occlusion_keep`       | `False`       | 独立开关：每图生成 1 张遮挡图（+N）                                                     |
| `occlusion_ratio`      | `0.5`         | 原图级比例：每 epoch 选 `round(x×N)` 张做遮挡，未选中整图直通（len 恒定）；0.3~0.6 推荐 |
| `occlusion_types`      | `rect,stripe` | 遮挡类型池（逗号分隔），每图随机抽 1 种                                                 |
| `occlusion_blocks`     | `1`           | 每图遮挡块数（1~3）                                                                     |
| `occlusion_size_ratio` | `0.1`         | 单块面积上限（相对原图面积），防目标被完全盖住                                          |
| `occlusion_color`      | `auto`        | `auto`=采样图像暗分位均值融入场景；或 `black`/`gray` 固定色                             |
| `occlusion_max_cover`  | `0.95`        | 目标被遮挡面积占比 ≥ 阈值则从标签剔除（完全被盖住的目标=纯噪声）；1.0=标签永不变        |
| `occlusion_save_dir`   | `""`          | 保存目录（画框/按图+类型跨 epoch 去重/限数量）；空=不保存                               |

### 训练后期关闭在线增强 `close_aug_epoch`

| 参数              | 默认 | 说明                                                                                                                                                                                        |
| ----------------- | ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `close_aug_epoch` | `0`  | 与 `close_mosaic` 同构的时间维调度：训练最后 N 个 epoch 把切片/合成/比例/模糊/气象退化/遮挡**全部关闭**（各区段退回原图直通，len 恒定），让模型在真实分布上收敛；`0`=不启用（完全向后兼容） |

### 验证侧在线切片评估 `val_slice_*`

> 训练侧已在线切片，验证侧整图直推会因小目标被降采样而**低估切片训练的收益**。开启后验证阶段把验证图切成 2×2 重叠子图独立推理，子图框还原到原图坐标，跨切片重复框 NMS 融合后与原图 GT 算 mAP（SAHI 评估）；`val_slice_dual_metric=True` 时另跑一遍整图验证输出 `whole_*` 参考指标，并额外保存 `best_whole.pt`/`last_whole.pt`。**默认关闭**（回归原生整图验证，完全等价原版）。

| 参数                      | 默认    | 说明                                                                                                                    |
| ------------------------- | ------- | ----------------------------------------------------------------------------------------------------------------------- |
| `val_slice_enable`        | `False` | 总开关；False=回归原生整图验证（完全等价原版）                                                                          |
| `val_slice_all_tiles`     | `False` | True=每图全部 2×2 子图推理（与训练侧对齐）；False=每图随机 1 片（快速验证）                                             |
| `val_slice_ratio`         | `1.0`   | 每轮验证随机选 `round(x×N_val)` 张验证图走切片，其余整图直通；1.0=全部切片，0=全部整图                                  |
| `val_slice_overlap_ratio` | `0.2`   | 验证侧切片重叠比例，建议与训练侧 `slice_overlap_ratio` 一致                                                             |
| `val_slice_nms_iou`       | `0.5`   | 跨切片重复框 NMS 融合 IoU 阈值                                                                                          |
| `val_slice_dual_metric`   | `False` | 双口径：切片 mAP 为主（驱动 fitness/早停/best.pt），整图 mAP 为参考（`whole_*` 指标 + `best_whole.pt`/`last_whole.pt`） |

### 中间图保存（默认不落盘）

| 参数                    | 默认   | 说明                                                                                                                                                                                                                |
| ----------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `slice_save_dir`        | `""`   | 切片图保存目录；空=不保存                                                                                                                                                                                           |
| `slice_save_max`        | `0`    | 共用保存限额，作用于切片/比例/模糊/合成四类；0=不限；可用 `slice_save_max_{tile,ratio,blur,compose}` 单独覆盖（气象退化/遮挡各自用 `slice_save_max_weather`/`slice_save_max_occlusion`，不填回退 `slice_save_max`） |
| `slice_save_annotated`  | `True` | 保存时画标注框+类别                                                                                                                                                                                                 |
| `slice_save_exist_ok`   | `True` | 目录已存在是否继续写入；False=抛错防覆盖                                                                                                                                                                            |
| `mosaic_save_dir`       | `""`   | Mosaic 画布图保存目录（验证 mosaic 是否启用）；空=不保存                                                                                                                                                            |
| `mosaic_save_max`       | `0`    | 最多保存的 Mosaic 画布数；0=不限                                                                                                                                                                                    |
| `mosaic_save_annotated` | `True` | Mosaic 画布是否画框+类别                                                                                                                                                                                            |
| `mosaic_save_exist_ok`  | `True` | 目录已存在是否继续写入                                                                                                                                                                                              |

### 内存 / 工作分辨率（新增配置）

| 参数               | 默认   | 说明                                                                                                                                                                                |
| ------------------ | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `degrade_max_side` | `0`    | 退化分支（模糊/气象退化/遮挡/比例调整）最长边像素上限：`0`=自动=2×imgsz（默认开启）；`>0`=手动上限；`<0`=禁用（原分辨率执行）。与 `compose_max_side` 语义对齐，是内存峰值控制的关键 |
| `ims_cache_frames` | `0`    | 每 worker 的 `self.ims` 整图缓存帧数上限：`0`=自动（取 min(上游公式, ims_cache_mb 预算)）；`>0`=显式帧数；`<0`=沿用上游公式                                                         |
| `ims_cache_mb`     | `1024` | `ims_cache_frames=0`（自动档）时的每 worker 内存预算（MiB）；内存紧张时调低                                                                                                         |
| `prefetch_factor`  | `2`    | 每个 worker 预取的批次数（PyTorch 默认即 2）；在线增强开启时降到 1 可减少约一半在途内存                                                                                             |
| `fraction`         | `1.0`  | 数据子集比例；小数据集直接用 1.0（`round(N×fraction)` 取整为 0 会无法启动，代码已兜底保留 1 张并告警）                                                                              |

### 修补续训（见 `项目说明.md` §3.8 详解）

| 参数                   | 说明                                                                           |
| ---------------------- | ------------------------------------------------------------------------------ |
| `resume`               | 续训 ckpt 路径（如 `runs/exp/weights/last.pt`）                                |
| `resume_extend_epochs` | 自动修补续训到该轮数；必须 > 已完成轮数（已完成场景自动读 ckpt 的 train_args） |

## 工作原理（简述）

- **索引空间扩展**：`__getitem__` 索引从 N（原图）扩展为 7 个区段（切片/原图/比例/模糊/合成/气象退化/遮挡），每个子样本按区段边界路由、即时生成，`buffer` 统一记账后供 Mosaic 采样；
- **在线切片在原分辨率上进行**：切片前不缩放到训练尺寸，小目标随子图缩放真实放大；
- **样本池长度恒定**（硬约束）：所有"被拒/未选中"样本位**位置保留、内容替换**（被拒背景片退回原图、未选中组退回整图），避免 Mosaic buffer 记账错乱；
- **epoch 级精确比例**：`set_epoch(epoch)` 重建各掩码，经共享内存通道传播到各 DataLoader worker，各进程用确定性 RNG 重建完全一致的掩码，无 per-worker 漂移；
- **标签始终与像素对齐**：切片/合成/比例同步换算 bbox 坐标，模糊/气象/遮挡标签不变（遮挡超阈值目标除外）。

## 注意事项

- **内存**：在线增强按原分辨率解码 + 多 worker 预取是内存峰值主因。推荐：`compose_max_side=0`（自动降采样）、`degrade_max_side=0`、`slice_ratio` 降到 0.5~0.7、`workers` 2~4、`cache=False`；内存紧张时调低 `ims_cache_mb`（默认 1024 MiB/worker）或 `prefetch_factor`（默认 2→1）；
- **勿开 `cache='ram'`**：在线增强全开时每张原图多次读入内存，`cache='ram'` 直接爆内存；
- **不要 `pip install -U ultralytics`**：本项目是本地深度改造版，升级会覆盖全部自定义增强与续训功能；
- **小数据集 + 小 `fraction` 陷阱**：`round(N×fraction)` 取整为 0 会让训练无法启动（已在代码层兜底保留 1 张并告警），小数据集直接用 `fraction=1.0`。

---

## 目录结构

```text
ultralytics-improved/
├── train.py                     # 训练入口（全部在线增强 + 续训参数集中配置）
├── data.yaml                    # 数据集配置（nc=6 车辆数据集示例）
├── README.md                    # 本文件（项目总览 / 快速开始）
├── 项目说明.md                   # 深度说明：设计思路 / 模块详解 / 性能排障 / 续训机制
├── 更新说明.md                   # 变更日志：每次更新解决什么问题、如何解决
├── ultralytics/
│   ├── data/
│   │   ├── base.py              # 索引空间扩展 / set_epoch 掩码 / 各在线分支 / 缓存 / 验证侧切片数据集
│   │   ├── augment.py           # OnlineSlice 切片器 / Mosaic 保存 / 独立开关装配 / 读参兜底
│   │   ├── online_degrade.py    # 退化算子（模糊 / 气象 / 遮挡 / 比例限幅）
│   │   └── online_io.py         # 保存工具（建目录 / 画框 / 去重 / 限额）
│   ├── models/yolo/detect/val.py  # 验证侧切片评估（还原 / NMS 融合 / 双口径指标）
│   ├── engine/trainer.py        # set_epoch 钩子 / 修补续训 / 双权重保存
│   └── cfg/                     # default.yaml 参数定义 + __init__.py 类型校验注册
├── pytools/                     # 离线版工具（在线化的设计来源）+ 工程工具
│   ├── sahi_equal_division_slice_dataset_auto_improved.py   # 离线切片
│   ├── compose_slice_dataset_auto_improved.py               # 离线合成
│   ├── change_image_resolution_slice_dataset_auto_improved.py # 离线比例调整
│   ├── motion_blur_improved.py  # 离线运动模糊（每张 2 张：短+长）
│   ├── fix_checkpoint_for_extension.py  # 续训修补 CLI
│   ├── lint.py                  # 提交前 lint 自查入口
│   └── _safe_io.py              # 文件删除安全护栏
└── weights/yolo26n.pt           # 预训练权重
```

## 文档导航

| 文档          | 内容                                                             |
| ------------- | ---------------------------------------------------------------- |
| `README.md`   | 快速开始                                                         |
| `项目说明.md` | 为什么这样做（设计思路）、怎么做（模块机制）、性能排障、续训详解 |
| `更新说明.md` | 每次更新解决什么问题、如何解决、验证结果                         |

---
