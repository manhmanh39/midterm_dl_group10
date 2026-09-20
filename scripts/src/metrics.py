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


def _voc_ap(rec, prec):
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


@torch.no_grad()
def evaluate_detections(model, dataset, loader, device, num_classes,
                        conf_threshold=0.1, nms_iou=0.45, iou_thr=0.5,
                        min_score=0.01, max_batches=None):
    """loader PHAI shuffle=False va tro truc tiep vao `dataset` (de khop index GT)."""
    model.eval()
    preds, gts, idx = [], [], 0
    for bi, (imgs, _) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        dec = decode_predictions(model(imgs.to(device)), min_score, nms_iou)
        for d in dec:
            preds.append({k: v.cpu().numpy() for k, v in d.items()})
            raw = dataset.get_raw_boxes(idx)
            if raw:
                a = np.array(raw, dtype=np.float32)
                lab = a[:, 0].astype(np.int64)
                cx, cy, w, h = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
                bx = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
            else:
                bx, lab = np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)
            gts.append((bx, lab))
            idx += 1

    ap = np.zeros(num_classes); P = np.zeros(num_classes); R = np.zeros(num_classes)
    F1 = np.zeros(num_classes); n_gt_c = np.zeros(num_classes, dtype=int)
    TPs = FPs = FNs = 0

    det_by_c = [[] for _ in range(num_classes)]
    for i, p_ in enumerate(preds):
        for b, s_, l in zip(p_["boxes"], p_["scores"], p_["labels"]):
            det_by_c[int(l)].append((s_, i, b))

    for c in range(num_classes):
        gt_by_img = {i: g[0][g[1] == c] for i, g in enumerate(gts) if (g[1] == c).any()}
        n_gt = sum(len(v) for v in gt_by_img.values())
        n_gt_c[c] = n_gt
        det = det_by_c[c]
        det.sort(key=lambda x: -x[0])
        used = {i: np.zeros(len(v), bool) for i, v in gt_by_img.items()}
        tp = np.zeros(len(det))
        for k, (_, i, b) in enumerate(det):
            if i not in gt_by_img:
                continue
            ious = _iou_1_to_n(b, gt_by_img[i])
            j = int(ious.argmax())
            if ious[j] >= iou_thr and not used[i][j]:
                used[i][j] = True
                tp[k] = 1
        scores = np.array([d[0] for d in det])
        ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
        if n_gt > 0 and len(det) > 0:
            ap[c] = _voc_ap(ctp / n_gt, ctp / np.maximum(ctp + cfp, 1e-9))
        n_keep = int((scores >= conf_threshold).sum()) if len(det) else 0
        TP = int(ctp[n_keep - 1]) if n_keep > 0 else 0
        FP, FN = n_keep - TP, n_gt - TP
        P[c] = TP / max(TP + FP, 1); R[c] = TP / max(TP + FN, 1)
        F1[c] = 2 * P[c] * R[c] / max(P[c] + R[c], 1e-8)
        TPs += TP; FPs += FP; FNs += FN

    valid = n_gt_c > 0
    mp, mr = TPs / max(TPs + FPs, 1), TPs / max(TPs + FNs, 1)
    return {
        "map50": float(ap[valid].mean()) if valid.any() else 0.0,
        "ap_per_class": ap.tolist(), "precision_per_class": P.tolist(),
        "recall_per_class": R.tolist(), "f1_per_class": F1.tolist(),
        "n_gt_per_class": n_gt_c.tolist(),
        "macro_precision": float(P[valid].mean()) if valid.any() else 0.0,
        "macro_recall": float(R[valid].mean()) if valid.any() else 0.0,
        "macro_f1": float(F1[valid].mean()) if valid.any() else 0.0,
        "micro_precision": mp, "micro_recall": mr,
        "micro_f1": 2 * mp * mr / max(mp + mr, 1e-8),
        "mean_preds_per_image": float(np.mean([len(p["boxes"]) for p in preds])) if preds else 0.0,
    }