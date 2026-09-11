# -*- coding: utf-8 -*-
"""Mock end-to-end validation for val_slice_* (validation-side SAHI eval).
Needs a working torch+ultralytics env (conda env 'langchain' / 'computevision').
Asserts: SliceValDataset layout, remap round-trip, class-wise NMS, and mAP=1.0
with perfect sub-tile predictions. Run from repo root:
    python pytools/_mock_val_slice.py
"""
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(r"C:\Users\Administrator\Desktop\ultralytics-improved")
os.environ["YOLO_OFFLINE"] = "1"
sys.path.insert(0, str(ROOT))

from ultralytics.data.dataset import YOLODataset
from ultralytics.data.base import SliceValDataset
from ultralytics.models.yolo.detect import DetectionValidator

PASS = []


def check(name, cond, detail=""):
    assert cond, f"FAIL: {name} {detail}"
    PASS.append(name)
    print(f"  [PASS] {name}")


def make_mini_dataset(tmp: Path, n_imgs=2, imgsz=640):
    """Build a tiny real dataset: n images with known boxes at known pixel positions."""
    img_dir = tmp / "images"
    lbl_dir = tmp / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    spec = [
        (400, 300, [(50, 50, 150, 150), (250, 200, 350, 280)], [0, 1]),
        (800, 600, [(100, 100, 300, 300), (500, 400, 700, 560)], [1, 0]),
    ][:n_imgs]
    for i, (w, h, boxes, cls) in enumerate(spec):
        img = np.full((h, w, 3), 120, dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"v{i}.jpg"), img)
        lines = []
        for (x0, y0, x1, y1), c in zip(boxes, cls):
            cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
            bw, bh = (x1 - x0) / w, (y1 - y0) / h
            lines.append(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (lbl_dir / f"v{i}.txt").write_text("\n".join(lines) + "\n")
    data = {"path": str(tmp), "nc": 2, "names": ["a", "b"]}
    return YOLODataset(
        img_path=str(img_dir), imgsz=imgsz, batch_size=2, augment=False,
        hyp={}, rect=False, cache=False, single_cls=False, stride=32,
        pad=0.5, prefix="val: ", task="detect", classes=None, data=data, fraction=1.0,
    )


def test_slice_val_dataset():
    print("== SliceValDataset ==")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        ds = make_mini_dataset(tmp, n_imgs=2)
        sv = SliceValDataset(ds, overlap_ratio=0.2, all_tiles=True, ratio=1.0)
        check("len=4N", len(sv) == 8, f"len={len(sv)}")
        sample = sv[0]
        for key in ("img", "cls", "bboxes", "batch_idx", "ori_shape", "resized_shape", "ratio_pad", "im_file", "val_slice_meta"):
            check(f"sample has {key}", key in sample, str(list(sample.keys())))
        check("img is tensor", isinstance(sample["img"], torch.Tensor) and sample["img"].ndim == 3)
        meta = sample["val_slice_meta"]
        check("meta fields", all(k in meta for k in ("orig_idx", "k", "offset", "tile_shape", "orig_shape", "n_tiles", "sliced")))
        check("meta sliced+n_tiles=4", meta["sliced"] is True and meta["n_tiles"] == 4)
        sv2 = SliceValDataset(ds, overlap_ratio=0.2, all_tiles=True, ratio=0.5)
        check("ratio=0.5 len=5", len(sv2) == 5, f"len={len(sv2)}")
        n_sliced = sum(1 for i in range(len(sv2)) if sv2[i]["val_slice_meta"]["sliced"])
        check("ratio=0.5 -> 4 sliced samples", n_sliced == 4, f"n_sliced={n_sliced}")
        sv3 = SliceValDataset(ds, overlap_ratio=0.2, all_tiles=False, ratio=1.0)
        check("all_tiles=False len=N", len(sv3) == 2, f"len={len(sv3)}")


def test_remap_and_fusion():
    print("== Remap + NMS fusion + GT pixels ==")
    args = dict(val_slice_enable=True, val_slice_overlap_ratio=0.2, val_slice_all_tiles=True,
                val_slice_ratio=1.0, val_slice_nms_iou=0.5, conf=0.001, iou=0.7, imgsz=640, task="detect")
    v = DetectionValidator(args=args)
    meta = {"orig_idx": 0, "k": 0, "offset": (0, 0), "tile_shape": (180, 240),
            "orig_shape": (300, 400), "n_tiles": 4, "sliced": True}
    pred_imgsz = torch.tensor([[300.0, 330.0, 400.0, 430.0]])  # canvas box of original (100,100)-(200,200) in tile0
    out = v._remap_boxes_imgsz_to_orig(pred_imgsz, meta, 640)
    expect = torch.tensor([[100.0, 100.0, 200.0, 200.0]])
    check("remap tile0", torch.allclose(out, expect, atol=1.0), f"got {out.tolist()} want {expect.tolist()}")
    boxes = torch.tensor([[0.0, 0.0, 100.0, 100.0], [10.0, 10.0, 110.0, 110.0], [500.0, 500.0, 600.0, 600.0]])
    conf = torch.tensor([0.9, 0.8, 0.7])
    cls = torch.tensor([0.0, 0.0, 1.0])
    keep = v._class_wise_nms(boxes, conf, cls, 0.5)
    check("nms keeps 2", keep.numel() == 2, f"keep={keep.tolist()}")
    lb = {"bboxes": np.array([[0.25, 0.25, 0.25, 0.25]]), "cls": np.array([[0]]), "shape": (400, 300)}
    gcls, gbox = v._gt_orig_pixels(lb, 400, 300)
    check("gt cls", gcls.tolist() == [0.0])
    check("gt box", torch.allclose(gbox, torch.tensor([[50.0, 37.5, 150.0, 112.5]]), atol=1e-3), str(gbox.tolist()))


def test_end_to_end_fusion():
    print("== End-to-end: sub-tile preds -> remap -> fuse -> update_stats ==")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        ds = make_mini_dataset(tmp, n_imgs=2)
        sv = SliceValDataset(ds, overlap_ratio=0.2, all_tiles=True, ratio=1.0)
        args = dict(val_slice_enable=True, val_slice_overlap_ratio=0.2, val_slice_all_tiles=True,
                    val_slice_ratio=1.0, val_slice_nms_iou=0.5, conf=0.001, iou=0.7, imgsz=640, task="detect",
                    single_cls=False, plots=False, save_json=False, save_txt=False, visualize=False)
        v = DetectionValidator(args=args)
        v._slice_base_labels = sv.base.labels
        v._slice_acc = {}
        v.device = torch.device("cpu")
        v.metrics = DetectionValidator(args=args).metrics
        samples = [sv[i] for i in range(len(sv))]
        batch = {}
        for k in samples[0].keys():
            vals = [s[k] for s in samples]
            if k == "img":
                batch[k] = torch.stack(vals)
            elif k in {"cls", "bboxes", "segments", "keypoints"}:
                batch[k] = torch.cat(vals, 0)
            elif k == "batch_idx":
                bi = []
                for i, s in enumerate(samples):
                    bi.append(s["batch_idx"] + i)
                batch[k] = torch.cat(bi, 0)
            else:
                batch[k] = vals
        preds = []
        for si in range(len(samples)):
            cls_ids = batch["cls"][batch["batch_idx"] == si].squeeze(-1)
            n = len(cls_ids)
            if n == 0:
                preds.append({"bboxes": torch.empty((0, 4)), "conf": torch.empty(0), "cls": torch.empty(0), "extra": torch.empty((0, 0))})
                continue
            bboxes_xywh = batch["bboxes"][batch["batch_idx"] == si]
            cx, cy, bw, bh = bboxes_xywh[:, 0] * 640, bboxes_xywh[:, 1] * 640, bboxes_xywh[:, 2] * 640, bboxes_xywh[:, 3] * 640
            xyxy = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
            preds.append({"bboxes": xyxy, "conf": torch.full((n,), 0.9), "cls": cls_ids.float(), "extra": torch.empty((n, 0))})
        v.update_metrics(preds, batch)
        check("slice_acc drained", len(v._slice_acc) == 0, str(v._slice_acc.keys()))
        check("seen == n_imgs", v.seen == 2, f"seen={v.seen}")
        v.metrics.process(save_dir=None, plot=False, on_plot=None)
        results = v.metrics.results_dict
        check("mAP50=1.0", results["metrics/mAP50(B)"] == 1.0, str(results["metrics/mAP50(B)"]))
        check("mAP50-95=1.0", results["metrics/mAP50-95(B)"] == 1.0, str(results["metrics/mAP50-95(B)"]))


if __name__ == "__main__":
    test_slice_val_dataset()
    test_remap_and_fusion()
    test_end_to_end_fusion()
    print(f"\nALL {len(PASS)} CHECKS PASSED")
