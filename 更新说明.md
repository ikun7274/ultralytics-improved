# online_sahi_slice_usage — 功能更改记录

> 所有功能更改（参数、逻辑、数据结构）都会记录在本文件，便于追溯每次改动的原因、范围与验证结果。

---

## 2026-09-07：移除 .npy 磁盘缓存（slice_use_cache / cache_dir）

### 背景

远程服务器训练排障时实测：`cache='disk'` + `slice_use_cache=True` 在机械盘上不仅没有提速，反而因随机读 .npy / 缓存预热开销导致 CPU 切换风暴；关闭后 CPU 利用率恢复 90-110 正常范围。结论：这套自定义 .npy 磁盘缓存**没有用处**，整体移除。

### 改动范围（6 个文件）

| 文件 | 改动 |
|---|---|
| `ultralytics/data/base.py` | 删除 `cache_dir` 构造参数；`npy_files` 恢复为与 jpg 同目录的原生默认；`_load_image_cached` 删除 npy 读取分支（损坏删除回退逻辑一并移除），改为**直接 imread + worker 内存 LRU**；`check_cache_images` 路径简化；cache='ram' 警告文本更新 |
| `ultralytics/data/augment.py` | 删除 `dataset.slice_use_cache` 赋值 |
| `ultralytics/cfg/__init__.py` | 删除 `slice_use_cache` 键、`cache_dir` 字符串键 |
| `ultralytics/cfg/default.yaml` | 删除 `slice_use_cache`、`cache_dir` 参数 |
| `ultralytics/data/build.py` | 删除两处 `cache_dir=cfg.cache_dir or ""` 传递 |
| `train.py` | 删除 `slice_use_cache` 参数与 `cache_dir` 注释 |

### 保留项（未动）

- **原生 `cache='disk'` 代码路径**（`load_image` 的 npy 读取、`cache_images_to_disk`）：train.py 默认 `cache=False` 即完全不产生/不读 npy；如需彻底禁用可另行处理。
- **worker 内存 LRU（`slice_raw_cache_size`）**：与磁盘缓存无关，是内存机制，同一张原图被 imread 8~9 次的场景下降解到 ~1 次，保留。

### 验证

- 语法检查通过（base/augment/build/cfg/train 5 个文件 py_compile）。
- 回归测试通过：
  - 全开 33 样本池正常解码；
  - 内存 LRU 命中返回副本、不重复 imread；
  - 切片+原图 4N+N 布局完好；
  - `np.load` 全程零调用（无 npy 路径）。
- 全项目无 `slice_use_cache` / `cache_dir` 残留（仅离线工具/测试目录无关引用除外）。

---

## 2026-09-08：方案 A — emit_all 被拒背景片退回原图（替代空片）

### 背景

无人机检测场景：目标小、背景占比高，`slice_all_tiles=True`（4 片全出）时每张原图约 3/4 的片是背景。原实现中，`slice_background_ratio` 配额不足的背景片被**裁成空片**进训练——但空片与配额内背景片在训练视角完全等价（同一块裁剪区域 + 空标签），负样本占比由切片几何锁死（约 67%），`slice_background_ratio` 实际调不动它。

### 改动

`ultralytics/data/augment.py` `OnlineSlice.slice_at`：配额不足时 `_emit` 返回原图（mode A contract），**不再裁剪空片占位**，直接退回整张原图（带全量 bbox）进池。

效果（N 原图，每张 1 正 3 背，ratio=0.3）：

| 池子构成 | 原实现 | 方案 A |
|---|---|---|
| 正片 | N | N |
| 背景片（真实负样本） | 0.3N | 0.3N |
| 空片（纯负样本） | 2.7N | **0** |
| 退回原图（正样本） | 0 | 2.7N |
| 负样本占比 | **67%** | **~23%**（实测 12.5%） |

### 要点

- `len` 恒为 4N 不变（位置保留、内容替换），不引入"len 不恒定"问题。
- 退回原图重复进池 = 隐式过采样正样本（每个副本独立走随机增强）。
- `slice_background_ratio` 恢复字面语义：真正控制训练中负样本占比。
- 建议配合 `slice_keep_origin=False`（退回原图已覆盖整图上下文，keep_origin 会冗余）。

### 验证

- py_compile 通过（augment.py / train.py）。
- 回归测试通过：
  - ratio=0.3：8 样本 = 5 退回原图 + 1 背景片 + 2 正片，空片为 0；
  - neg_ratio=-1：全部背景保留、无退回原图；
  - 集成路径 `get_image_and_label` len=4N=8 恒定，bbox/cls 一一匹配。

---

## 2026-09-08：slice_ratio — 每 epoch 原图级精确比例切片（替代 slice_mix_ratio）

### 背景

原 `slice_mix_ratio` 是 per-sample 独立抛硬币（概率近似）：每 epoch 实际切片样本数随机漂移；`emit_all` 下同一原图 4 片各自独立决定，可能出现"同一张图 2 片切片、2 片不切"的怪状态。需求：**每 epoch 精确选 x 比例原图走切片**（原图级决策，4 片同生共死）。

### 改动（5 个文件）

| 文件 | 改动 |
|---|---|
| `ultralytics/data/base.py` | 新增 `set_epoch(epoch)`：重建 `_slice_mask`，精确选 `round(slice_ratio × N)` 张**原图**走切片；`get_image_and_label` 切片判断由 `random.random() < ratio` 改为查掩码（`slice_ok`）；`__init__` 加 `self._slice_mask = None` |
| `ultralytics/engine/trainer.py` | epoch 循环内、`sampler.set_epoch` 旁新增：`dataset.set_epoch(epoch)`（单机/多机均生效，`hasattr` 防御） |
| `ultralytics/data/augment.py` | `dataset.slice_mix_ratio` → `dataset.slice_ratio`（两处） |
| `ultralytics/cfg/default.yaml` | 参数 `slice_mix_ratio` → `slice_ratio`，注释改为 epoch 级精确比例语义 |
| `ultralytics/cfg/__init__.py` | CFG_FLOAT_KEYS 注册表 `slice_mix_ratio` → `slice_ratio` |
| `train.py` | 参数与推荐注释同步更新 |

### 语义

- `slice_ratio=1.0`（默认）：掩码 None = 纯切片（与旧行为一致）
- `0 < slice_ratio < 1`：每 epoch 精确选 `round(x×N)` 张原图切片，其余整图进池；**原图级**（emit_all 下 4 片同命运）
- `slice_ratio=0`：全整图（等效关闭切片）
- 掩码在主进程 `set_epoch` 生成一次，worker fork 继承 → 无 per-worker 漂移
- 未调 `set_epoch`（验证/直取样本）时首次访问惰性初始化
- `len` 恒为 4N 不变

### 验证

- py_compile 通过（base / trainer / augment / cfg / train）。
- 回归测试通过：
  - 精确比例：0.5→2、0.25→1、0→全 False、1→None（纯切片）；
  - 原图级：选中原图 4 片全切（slice 调用=4×选中数），len=16 恒定；
  - 跨 epoch 重采样变化（5 个 epoch 出现 ≥2 种掩码，N=4 时两次采样碰撞概率 1/6 已规避）；
  - 惰性初始化、trainer 钩子存在；
  - 全项目无 `slice_mix_ratio` 残留。


