"""
evaluate.py - Danh gia model Object Detection (VinBigData) DAY DU.

Chay tu project_root (KHONG can cd vao scripts/):
    python3 evaluate.py --model model1 --checkpoint checkpoints/model1_seed202601_best.pth --data_dir data/processed/dataset_202601/val --num_workers 8

Danh gia bang:
    - Average Loss + Average IoU tren cac cell co object (nhu luc train - nhanh)
    - Sau decode + NMS: Precision / Recall / F1 tong the va theo tung class,
      matching GT-Prediction bang IoU >= match_iou_threshold (mac dinh 0.5)
    - Bieu do per-class F1 va histogram so box du bao/anh, luu vao output_dir
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.config import NUM_CLASSES, CLASS_NAMES, IMAGE_SIZE, get_grid_size, CONF_THRESHOLD, NMS_IOU_THRESHOLD
from scripts.src.dataset import VinBigDataDetectionDataset, collate_fn
from scripts.src.utils import CombinedLocalizationLoss, calculate_iou
from scripts.src.decode import decode_predictions
from scripts.models.model1_sequential import SequentialCNNDetector
from scripts.models.model2_residual import NonSequentialCNNDetector
from scripts.models.model3_pretrained import PretrainedDetector


def build_model(model_name: str, num_classes: int, freeze_backbone: bool = True, unfreeze_from_layer: str = "layer4"):
    if model_name == "model1":
        return SequentialCNNDetector(num_classes=num_classes)
    elif model_name == "model2":
        return NonSequentialCNNDetector(num_classes=num_classes)
    elif model_name == "model3":
        return PretrainedDetector(
            num_classes=num_classes,
            freeze_backbone=freeze_backbone,
            unfreeze_from_layer=unfreeze_from_layer,
        )
    else:
        raise ValueError(f"Model khong hop le: {model_name}")


def _box_iou_xyxy(box_a, box_b):
    """IoU giua 1 box va 1 mang box (N,4), dinh dang (x1,y1,x2,y2)."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3]

    inter_x1 = np.maximum(xa1, xb1)
    inter_y1 = np.maximum(ya1, yb1)
    inter_x2 = np.minimum(xa2, xb2)
    inter_y2 = np.minimum(ya2, yb2)
    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter_area = inter_w * inter_h

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = np.clip(xb2 - xb1, 0, None) * np.clip(yb2 - yb1, 0, None)
    union = area_a + area_b - inter_area
    return np.where(union > 0, inter_area / union, 0.0)


def _match_predictions_to_gt(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_thr=0.5):
    """Greedy matching: moi GT box chi duoc dung 1 lan. Tra ve (tp, fp, fn) cho 1 anh/1 class."""
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)

    if n_gt == 0:
        return 0, n_pred, 0
    if n_pred == 0:
        return 0, 0, n_gt

    gt_matched = np.zeros(n_gt, dtype=bool)
    tp = 0
    fp = 0

    for i in range(n_pred):
        p_box = pred_boxes[i]
        p_label = pred_labels[i]

        candidate_mask = (gt_labels == p_label) & (~gt_matched)
        if candidate_mask.sum() == 0:
            fp += 1
            continue

        ious = _box_iou_xyxy(p_box, gt_boxes[candidate_mask])
        best_idx_local = np.argmax(ious)
        best_iou = ious[best_idx_local]

        if best_iou >= iou_thr:
            global_idx = np.nonzero(candidate_mask)[0][best_idx_local]
            gt_matched[global_idx] = True
            tp += 1
        else:
            fp += 1

    fn = n_gt - gt_matched.sum()
    return tp, fp, int(fn)


def _grid_target_to_boxes(target: torch.Tensor, grid_size: int):
    """Chuyen 1 grid target tensor (G,G,5+C) thanh list box that (x1,y1,x2,y2,class_id)."""
    obj_mask = target[..., 0] == 1.0
    idxs = torch.nonzero(obj_mask, as_tuple=False)

    boxes, labels = [], []
    for gy, gx in idxs.tolist():
        tx, ty, w, h = target[gy, gx, 1:5].tolist()
        cx = (gx + tx) / grid_size
        cy = (gy + ty) / grid_size
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2
        boxes.append([x1, y1, x2, y2])
        class_id = int(torch.argmax(target[gy, gx, 5:]).item())
        labels.append(class_id)

    return np.array(boxes, dtype=np.float32).reshape(-1, 4), np.array(labels, dtype=np.int64)


def evaluate_model(
    model_name: str,
    checkpoint_path: str,
    data_dir: str = "data/processed/dataset_202601/val",
    image_size: int = IMAGE_SIZE,
    batch_size: int = 16,
    num_workers: int = 8,
    conf_threshold: float = CONF_THRESHOLD,
    nms_iou_threshold: float = NMS_IOU_THRESHOLD,
    match_iou_threshold: float = 0.5,
    output_dir: str = "outputs",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Dang su dung device: {device}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Khong tim thay checkpoint tai: {checkpoint_path}")

    print(f"[Eval] Dang tai checkpoint tu: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    freeze_backbone = checkpoint.get("freeze_backbone", True)
    unfreeze_from_layer = checkpoint.get("unfreeze_from_layer", "layer4")
    ckpt_image_size = checkpoint.get("image_size", image_size)
    grid_size = checkpoint.get("grid_size", get_grid_size(image_size=ckpt_image_size))

    model = build_model(model_name, NUM_CLASSES, freeze_backbone, unfreeze_from_layer).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = VinBigDataDetectionDataset(data_dir, image_size=ckpt_image_size, grid_size=grid_size, augment=False)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )

    criterion = CombinedLocalizationLoss()

    total_loss, total_iou, n_batches = 0.0, 0.0, 0
    tp_per_class = np.zeros(NUM_CLASSES, dtype=np.int64)
    fp_per_class = np.zeros(NUM_CLASSES, dtype=np.int64)
    fn_per_class = np.zeros(NUM_CLASSES, dtype=np.int64)
    num_preds_per_image = []
    max_obj_scores = []  # debug: objectness cao nhat (sau sigmoid) tren tung anh

    print(f"[Eval] Bat dau danh gia tren tap du lieu: {data_dir} ({len(dataset)} anh)...")

    with torch.no_grad():
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            preds = model(imgs)
            loss = criterion(preds, targets)

            total_loss += loss.item()
            total_iou += calculate_iou(preds, targets, grid_size)
            n_batches += 1

            # Debug: ghi lai objectness cao nhat/anh de kiem tra model co
            # dang du bao score qua thap so voi conf_threshold hay khong
            obj_scores = torch.sigmoid(preds[..., 0])  # (B, G, G)
            batch_max = obj_scores.view(obj_scores.shape[0], -1).max(dim=1).values
            max_obj_scores.extend(batch_max.cpu().tolist())

            decoded = decode_predictions(preds, conf_threshold=conf_threshold, nms_iou_threshold=nms_iou_threshold)

            for i in range(imgs.shape[0]):
                pred_boxes = decoded[i]["boxes"].cpu().numpy()
                pred_labels = decoded[i]["labels"].cpu().numpy()
                num_preds_per_image.append(len(pred_boxes))

                gt_boxes, gt_labels = _grid_target_to_boxes(targets[i].cpu(), grid_size)

                for c in range(NUM_CLASSES):
                    p_mask = pred_labels == c
                    g_mask = gt_labels == c
                    tp, fp, fn = _match_predictions_to_gt(
                        pred_boxes[p_mask], pred_labels[p_mask],
                        gt_boxes[g_mask], gt_labels[g_mask],
                        iou_thr=match_iou_threshold,
                    )
                    tp_per_class[c] += tp
                    fp_per_class[c] += fp
                    fn_per_class[c] += fn

    avg_loss = total_loss / max(n_batches, 1)
    avg_iou = total_iou / max(n_batches, 1)

    precision_per_class = tp_per_class / np.clip(tp_per_class + fp_per_class, 1, None)
    recall_per_class = tp_per_class / np.clip(tp_per_class + fn_per_class, 1, None)
    f1_per_class = 2 * precision_per_class * recall_per_class / np.clip(precision_per_class + recall_per_class, 1e-8, None)

    macro_precision = precision_per_class.mean()
    macro_recall = recall_per_class.mean()
    macro_f1 = f1_per_class.mean()

    print("\n" + "=" * 70)
    print(f" KET QUA DANH GIA - {model_name.upper()}  (IoU matching threshold = {match_iou_threshold})")
    print("=" * 70)
    print(f"  Checkpoint       : {checkpoint_path}")
    print(f"  Tap du lieu      : {data_dir}")
    print(f"  Average Loss     : {avg_loss:.4f}")
    print(f"  Average IoU      : {avg_iou:.4f}")
    print(f"  Macro Precision  : {macro_precision:.4f}")
    print(f"  Macro Recall     : {macro_recall:.4f}")
    print(f"  Macro F1         : {macro_f1:.4f}")
    print(f"  So prediction/anh (sau NMS): mean={np.mean(num_preds_per_image):.2f}, "
          f"median={np.median(num_preds_per_image):.1f}, max={np.max(num_preds_per_image) if num_preds_per_image else 0}")
    print("-" * 70)
    print("--- DEBUG OBJECTNESS SCORE (truoc khi loc theo conf_threshold) ---")
    max_obj_arr = np.array(max_obj_scores)
    print(f"  Objectness cao nhat/anh: mean={max_obj_arr.mean():.4f}, "
          f"median={np.median(max_obj_arr):.4f}, max={max_obj_arr.max():.4f}, min={max_obj_arr.min():.4f}")
    print(f"  conf_threshold dang dung: {conf_threshold}")
    n_above_thr = (max_obj_arr >= conf_threshold).sum()
    print(f"  So anh co it nhat 1 cell vuot nguong: {n_above_thr}/{len(max_obj_arr)} "
          f"({n_above_thr / len(max_obj_arr) * 100:.1f}%)")
    if max_obj_arr.max() < conf_threshold:
        print(
            f"  [CANH BAO] objectness cao nhat toan tap ({max_obj_arr.max():.4f}) VAN THAP HON "
            f"conf_threshold ({conf_threshold}) -> KHONG CO anh nao vuot nguong, giai thich vi sao "
            f"so prediction/anh = 0 va Precision/Recall/F1 = 0. Thu giam --conf_threshold "
            f"(vd 0.1-0.2) de kiem tra lai, hoac train them epoch / kiem tra lai loss objectness."
        )
    print("-" * 70)
    print("--- CHI TIET THEO LOP ---")
    for i, name in enumerate(CLASS_NAMES[:NUM_CLASSES]):
        print(f"  {name:<22} P={precision_per_class[i]:.3f}  R={recall_per_class[i]:.3f}  "
              f"F1={f1_per_class[i]:.3f}  (TP={tp_per_class[i]}, FP={fp_per_class[i]}, FN={fn_per_class[i]})")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 5))
    xs = np.arange(NUM_CLASSES)
    plt.bar(xs, f1_per_class)
    plt.xticks(xs, CLASS_NAMES[:NUM_CLASSES], rotation=45, ha="right", fontsize=8)
    plt.ylabel("F1-score")
    plt.title(f"Per-class F1 - {model_name} (Macro F1={macro_f1:.3f})")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"per_class_f1_{model_name}.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[Eval] Da luu bieu do F1 per-class tai: {plot_path}")

    plt.figure(figsize=(8, 5))
    plt.hist(num_preds_per_image, bins=range(0, max(num_preds_per_image, default=1) + 2))
    plt.xlabel("So luong box du bao / anh (sau NMS)")
    plt.ylabel("So luong anh")
    plt.title(f"Phan bo so box du bao - {model_name}")
    plt.tight_layout()
    hist_path = os.path.join(output_dir, f"num_preds_hist_{model_name}.png")
    plt.savefig(hist_path, dpi=200)
    plt.close()
    print(f"[Eval] Da luu histogram so box du bao tai: {hist_path}")

    return {
        "loss": avg_loss,
        "iou": avg_iou,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "precision_per_class": precision_per_class.tolist(),
        "recall_per_class": recall_per_class.tolist(),
        "f1_per_class": f1_per_class.tolist(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Danh gia model Object Detection (day du: Loss/IoU + Precision/Recall/F1)")
    parser.add_argument("--model", type=str, default="model1", choices=["model1", "model2", "model3"])
    parser.add_argument("--checkpoint", type=str, required=True, help="Duong dan toi file .pth")
    parser.add_argument("--data_dir", type=str, default="data/processed/dataset_202601/val", help="Thu muc val hoac test")
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--conf_threshold", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--nms_iou_threshold", type=float, default=NMS_IOU_THRESHOLD)
    parser.add_argument("--match_iou_threshold", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default="outputs")

    args = parser.parse_args()

    evaluate_model(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        conf_threshold=args.conf_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
        match_iou_threshold=args.match_iou_threshold,
        output_dir=args.output_dir,
    )