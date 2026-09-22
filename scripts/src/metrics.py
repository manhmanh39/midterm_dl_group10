"""mAP@0.5 va Precision/Recall/F1 that (sau decode + NMS, GT doc tu file label)."""
from __future__ import annotations

import numpy as np
import torch

from scripts.src.decode import decode_predictions


def _iou_1_to_n(box, boxes):
    ix1 = np.maximum(box[0], boxes[:, 0]); iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2]); iy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    b = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    u = a + b - inter
    return np.where(u > 0, inter / np.maximum(u, 1e-12), 0.0)


def compute_voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    """
    PASCAL VOC 2010+ continuous all-points interpolated AP.
    Interpolates precision: p_interp(r) = max_{r' >= r} p(r')
    Integrates numerically over the recall curve.
    """
    if len(rec) == 0 or len(prec) == 0:
        return 0.0
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


_voc_ap = compute_voc_ap


@torch.no_grad()
def evaluate_detections(model, dataset, loader, device, num_classes,
                        conf_threshold=0.25, nms_iou=0.45, iou_thr=0.5,
                        min_score=0.01, max_batches=None):
    """
    Danh gia detections dua tren mapping image_id chinh xac giua predictions va GT.
    - mAP@0.5: Tinh theo PASCAL VOC 2010+ all-points interpolation voi min_score=0.01.
    - Operating metrics (P/R/F1): Tinh tai conf_threshold (mac dinh 0.25).
    """
    model.eval()
    preds_dict = {}
    gts_dict = {}

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        imgs = batch[0].to(device)
        if len(batch) < 3 or batch[2] is None:
            raise RuntimeError(
                "evaluate_detections requires DataLoader/collate_fn to provide image_ids "
                "for exact GT-prediction alignment (no sequential index fallback allowed)."
            )
        image_ids = batch[2]

        dec = decode_predictions(model(imgs), min_score=min_score, nms_iou_threshold=nms_iou)
        for i, d in enumerate(dec):
            if i >= len(image_ids):
                raise IndexError(f"Prediction index {i} out of bounds for batch image_ids of size {len(image_ids)}")
            img_id = str(image_ids[i])

            preds_dict[img_id] = {k: v.cpu().numpy() for k, v in d.items()}

            if hasattr(dataset, "get_raw_boxes_by_id"):
                raw = dataset.get_raw_boxes_by_id(img_id)
            else:
                raise AttributeError("Dataset must implement get_raw_boxes_by_id(image_id) for exact sample alignment.")

            if raw:
                a = np.array(raw, dtype=np.float32)
                lab = a[:, 0].astype(np.int64)
                cx, cy, w, h = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
                bx = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
            else:
                bx, lab = np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)

            gts_dict[img_id] = (bx, lab)

    ap = np.zeros(num_classes)
    P = np.zeros(num_classes)
    R = np.zeros(num_classes)
    F1 = np.zeros(num_classes)
    n_gt_c = np.zeros(num_classes, dtype=int)
    TPs = FPs = FNs = 0

    for c in range(num_classes):
        gt_by_img = {img_id: g[0][g[1] == c] for img_id, g in gts_dict.items() if (g[1] == c).any()}
        n_gt = sum(len(v) for v in gt_by_img.values())
        n_gt_c[c] = n_gt

        det_for_c = []
        for img_id, p_ in preds_dict.items():
            boxes = p_["boxes"]
            scores = p_["scores"]
            labels = p_["labels"]
            for b, s_, l in zip(boxes, scores, labels):
                if int(l) == c:
                    det_for_c.append((float(s_), img_id, b))

        det_for_c.sort(key=lambda x: -x[0])
        used = {img_id: np.zeros(len(v), bool) for img_id, v in gt_by_img.items()}
        tp = np.zeros(len(det_for_c))

        for k, (_, img_id, b) in enumerate(det_for_c):
            if img_id not in gt_by_img:
                continue
            ious = _iou_1_to_n(b, gt_by_img[img_id])
            j = int(ious.argmax())
            if ious[j] >= iou_thr and not used[img_id][j]:
                used[img_id][j] = True
                tp[k] = 1.0

        scores = np.array([d[0] for d in det_for_c])
        ctp, cfp = np.cumsum(tp), np.cumsum(1.0 - tp)

        if n_gt > 0 and len(det_for_c) > 0:
            rec = ctp / n_gt
            prec = ctp / np.maximum(ctp + cfp, 1e-9)
            ap[c] = compute_voc_ap(rec, prec)

        n_keep = int((scores >= conf_threshold).sum()) if len(det_for_c) else 0
        TP = int(ctp[n_keep - 1]) if n_keep > 0 else 0
        FP = n_keep - TP
        FN = n_gt - TP

        P[c] = TP / max(TP + FP, 1)
        R[c] = TP / max(TP + FN, 1)
        F1[c] = 2 * P[c] * R[c] / max(P[c] + R[c], 1e-8)
        TPs += TP
        FPs += FP
        FNs += FN

    valid = n_gt_c > 0
    mp = TPs / max(TPs + FPs, 1)
    mr = TPs / max(TPs + FNs, 1)
    total_preds = sum(len(p["boxes"]) for p in preds_dict.values())
    mean_preds = float(total_preds / max(len(preds_dict), 1))

    return {
        "map50": float(ap[valid].mean()) if valid.any() else 0.0,
        "ap_per_class": ap.tolist(),
        "precision_per_class": P.tolist(),
        "recall_per_class": R.tolist(),
        "f1_per_class": F1.tolist(),
        "n_gt_per_class": n_gt_c.tolist(),
        "macro_precision": float(P[valid].mean()) if valid.any() else 0.0,
        "macro_recall": float(R[valid].mean()) if valid.any() else 0.0,
        "macro_f1": float(F1[valid].mean()) if valid.any() else 0.0,
        "micro_precision": float(mp),
        "micro_recall": float(mr),
        "micro_f1": float(2 * mp * mr / max(mp + mr, 1e-8)),
        "mean_preds_per_image": mean_preds,
    }