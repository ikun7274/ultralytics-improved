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
