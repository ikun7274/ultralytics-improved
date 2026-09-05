import warnings

# P2-7: 收窄告警过滤 — 仅屏蔽 Deprecation / Future / PendingDeprecation 类噪音,
# 不再一刀切 ignore，RuntimeWarning 等真错误必须能冒泡 (例如 P1-5 模糊核除零)。
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
import os
from pathlib import Path
import numpy as np
from prettytable import PrettyTable
from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info

def get_weight_size(path):
    stats = os.stat(path)
    return f'{stats.st_size / 1024 / 1024:.1f}'

if __name__ == '__main__':
    # P2-6: 移除硬编码的 runs/exp-2-2-5/weights/best.pt 路径 —
    # 训练脚本默认 project 已经写到 <项目根>/runs, 这里自动取最近一次训练的最佳权重。
    # 仍可手动覆盖 model_path。
    _project_root = Path(__file__).resolve().parent / "runs"
    _latest_best = None
    if _project_root.exists():
        candidates = sorted(
            (p for p in _project_root.rglob("weights/best.pt")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        _latest_best = str(candidates[0]) if candidates else None
    if _latest_best is None:
        raise FileNotFoundError(
            f"未找到任何 runs/**/weights/best.pt (搜索根目录: {_project_root})。"
            "请先训练一次或将本脚本顶部 model_path 手动指定。"
        )
    model_path = _latest_best
    model = YOLO(model_path)
    
    result = model.val(data='data.yaml',
                        split='val', # split可以选择train、val、test 根据自己的数据集情况来选择.
                        imgsz=1280,
                        batch=8,
                        workers=0, 
                        # conf=0.38,
                        # iou=0.45,
                        # rect=False,
                        # save_json=True, # if you need to cal coco metrice
                        project=str(_project_root),
                        name='exp',
                        )
    
    if model.task == 'detect': # 仅目标检测任务适用
        length = result.box.p.size
        model_names = list(result.names.values())
        preprocess_time_per_image = result.speed['preprocess']
        inference_time_per_image = result.speed['inference']
        postprocess_time_per_image = result.speed['postprocess']
        all_time_per_image = preprocess_time_per_image + inference_time_per_image + postprocess_time_per_image
        
        n_l, n_p, n_g, flops = model_info(model.model)
        
        model_info_table = PrettyTable()
        model_info_table.title = "Model Info"
        model_info_table.field_names = ["GFLOPs", "Parameters", "前处理时间/一张图", "推理时间/一张图", "后处理时间/一张图", "FPS(推理)", "Model File Size"]
        model_info_table.add_row([f'{flops:.1f}', f'{n_p:,}', 
                                  f'{preprocess_time_per_image / 1000:.6f}s', f'{inference_time_per_image / 1000:.6f}s', 
                                  f'{postprocess_time_per_image / 1000:.6f}s',
                                  f'{1000 / inference_time_per_image:.2f}', f'{get_weight_size(model_path)}MB'])
        print(model_info_table)

        model_metrice_table = PrettyTable()
        model_metrice_table.title = "Model Metrice"
        model_metrice_table.field_names = ["Class Name", "Precision", "Recall", "F1-Score", "mAP50", "mAP75", "mAP50-95"]
        for idx in range(length):
            model_metrice_table.add_row([
                                        model_names[idx], 
                                        f"{result.box.p[idx]:.4f}", 
                                        f"{result.box.r[idx]:.4f}", 
                                        f"{result.box.f1[idx]:.4f}", 
                                        f"{result.box.ap50[idx]:.4f}", 
                                        f"{result.box.all_ap[idx, 5]:.4f}", # 50 55 60 65 70 75 80 85 90 95 
                                        f"{result.box.ap[idx]:.4f}"
                                    ])
        model_metrice_table.add_row([
                                    "all(平均数据)", 
                                    f"{result.results_dict['metrics/precision(B)']:.4f}", 
                                    f"{result.results_dict['metrics/recall(B)']:.4f}", 
                                    f"{np.mean(result.box.f1[:length]):.4f}", 
                                    f"{result.results_dict['metrics/mAP50(B)']:.4f}", 
                                    f"{np.mean(result.box.all_ap[:length, 5]):.4f}", # 50 55 60 65 70 75 80 85 90 95 
                                    f"{result.results_dict['metrics/mAP50-95(B)']:.4f}"
                                ])
        print(model_metrice_table)

        with open(result.save_dir / 'paper_data.txt', 'w+') as f:
            f.write(str(model_info_table))
            f.write('\n')
            f.write(str(model_metrice_table))
        
        print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)