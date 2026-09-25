"""VinBigDataDetectionDataset.

Target shape: (G, G, A, 5 + C)
- A = BOXES_PER_CELL prediction slots/cell
- moi GT duoc gan vao 1 slot trong cell chua center
- stride 16 + nhieu slot giam collision manh ma van giu detector anchor-free
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from scripts.config import BOXES_PER_CELL, GRID_SIZE, IMAGE_SIZE, NUM_CLASSES

cv2.setNumThreads(0)

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class VinBigDataDetectionDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        image_size: int = IMAGE_SIZE,
        grid_size: int = GRID_SIZE,
        boxes_per_cell: int = BOXES_PER_CELL,
        augment: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / "images"
        self.lbl_dir = self.data_dir / "labels"
        self.image_size = image_size
        self.grid_size = grid_size
        self.boxes_per_cell = boxes_per_cell
        self.augment = augment

        if not self.img_dir.exists():
            raise FileNotFoundError(
                f"Khong tim thay thu muc anh: {self.img_dir}. Hay chay prepare_dataset.py truoc."
            )
        self.image_files = sorted(self.img_dir.glob("*.png"))
        if not self.image_files:
            raise RuntimeError(f"Khong co anh .png nao trong {self.img_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def _read_raw_boxes(self, label_path: Path):
        boxes = []
        if label_path.exists():
            text = label_path.read_text().strip()
            if text:
                for line in text.splitlines():
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    cls_id = int(float(parts[0]))
                    cx, cy, w, h = (float(v) for v in parts[1:])
                    if 0 <= cls_id < NUM_CLASSES and w > 0 and h > 0:
                        boxes.append((cls_id, cx, cy, w, h))
        return boxes

    def get_raw_boxes(self, idx: int):
        """Full raw GT sau data preparation/WBF. Dung cho final mAP."""
        return self._read_raw_boxes(self.lbl_dir / f"{self.image_files[idx].stem}.txt")

    def class_counts(self) -> np.ndarray:
        counts = np.zeros(NUM_CLASSES, dtype=np.int64)
        for img in self.image_files:
            for cls_id, *_ in self._read_raw_boxes(self.lbl_dir / f"{img.stem}.txt"):
                counts[cls_id] += 1
        return counts

    def class_weights(self, min_w: float = 0.5, max_w: float = 3.0) -> torch.Tensor:
        """Inverse-sqrt class weights; on dinh hon inverse-frequency truc tiep."""
        counts = self.class_counts().astype(np.float64)
        nonzero = counts > 0
        weights = np.ones(NUM_CLASSES, dtype=np.float64)
        weights[nonzero] = 1.0 / np.sqrt(counts[nonzero])
        if nonzero.any():
            weights[nonzero] /= weights[nonzero].mean()
        weights = np.clip(weights, min_w, max_w)
        return torch.tensor(weights, dtype=torch.float32)

    def box_size_prior(self) -> tuple[float, float]:
        ws, hs = [], []
        for img in self.image_files:
            for _, _, _, w, h in self._read_raw_boxes(self.lbl_dir / f"{img.stem}.txt"):
                ws.append(w); hs.append(h)
        if not ws:
            return 0.2, 0.2
        return float(np.median(ws)), float(np.median(hs))

    def target_assignment_stats(self) -> dict:
        """Dem collision voi GxG va A slots. Final eval KHONG bo cac box nay."""
        total = assigned = overflow = max_same_cell = 0
        G, A = self.grid_size, self.boxes_per_cell
        for img in self.image_files:
            occupancy = {}
            boxes = self._read_raw_boxes(self.lbl_dir / f"{img.stem}.txt")
            total += len(boxes)
            for _, cx, cy, _, _ in boxes:
                gx = min(max(int(cx * G), 0), G - 1)
                gy = min(max(int(cy * G), 0), G - 1)
                key = (gy, gx)
                occupancy[key] = occupancy.get(key, 0) + 1
            for n in occupancy.values():
                max_same_cell = max(max_same_cell, n)
                assigned += min(n, A)
                overflow += max(0, n - A)
        return {
            "total_boxes": total,
            "representable_boxes": assigned,
            "overflow_boxes": overflow,
            "representable_ratio": assigned / max(total, 1),
            "max_boxes_same_cell": max_same_cell,
        }

    def make_balanced_sampler(self, negative_ratio: float = 1.0) -> WeightedRandomSampler | None:
        """Can bang tong sampling mass positive vs No-finding.

        negative_ratio=1.0 -> expected positive:no-finding mass ~1:1.
        """
        if negative_ratio < 0:
            return None
        has_box = []
        for img in self.image_files:
            has_box.append(bool(self._read_raw_boxes(self.lbl_dir / f"{img.stem}.txt")))
        has_box = np.asarray(has_box, dtype=bool)
        n_pos = int(has_box.sum())
        n_neg = len(has_box) - n_pos
        if n_pos == 0 or n_neg == 0:
            return None
        w_pos = 1.0
        w_neg = (n_pos * negative_ratio) / max(n_neg, 1)
        weights = np.where(has_box, w_pos, w_neg).astype(np.float64)
        return WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )

    def _encode_grid_target(self, boxes):
        G, A = self.grid_size, self.boxes_per_cell
        target = np.zeros((G, G, A, 5 + NUM_CLASSES), dtype=np.float32)

        # Deterministic: small boxes first, de lesion nho khong bi systematically drop
        # trong rare case > A objects co center cung cell.
        boxes = sorted(boxes, key=lambda b: b[3] * b[4])
        for cls_id, cx, cy, w, h in boxes:
            gx = min(max(int(cx * G), 0), G - 1)
            gy = min(max(int(cy * G), 0), G - 1)
            free = np.where(target[gy, gx, :, 0] == 0.0)[0]
            if free.size == 0:
                continue
            a = int(free[0])
            target[gy, gx, a, 0] = 1.0
            target[gy, gx, a, 1] = cx * G - gx
            target[gy, gx, a, 2] = cy * G - gy
            target[gy, gx, a, 3] = np.clip(w, 1e-4, 1.0)
            target[gy, gx, a, 4] = np.clip(h, 1e-4, 1.0)
            target[gy, gx, a, 5 + cls_id] = 1.0
        return target

    @staticmethod
    def _clip_box_xyxy(x1, y1, x2, y2, size):
        x1, x2 = np.clip([x1, x2], 0, size - 1)
        y1, y2 = np.clip([y1, y2], 0, size - 1)
        return float(x1), float(y1), float(x2), float(y2)

    def _random_affine(self, img, boxes):
        """Mild bbox-aware affine, phu hop X-ray anatomy."""
        if np.random.rand() >= 0.55:
            return img, boxes

        s = self.image_size
        angle = np.random.uniform(-5.0, 5.0)
        scale = np.random.uniform(0.94, 1.06)
        tx = np.random.uniform(-0.03, 0.03) * s
        ty = np.random.uniform(-0.03, 0.03) * s
        M = cv2.getRotationMatrix2D((s / 2.0, s / 2.0), angle, scale)
        M[:, 2] += [tx, ty]

        warped = cv2.warpAffine(
            img, M, (s, s), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
        )

        new_boxes = []
        for cls_id, cx, cy, w, h in boxes:
            x1, y1 = (cx - w / 2) * s, (cy - h / 2) * s
            x2, y2 = (cx + w / 2) * s, (cy + h / 2) * s
            corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            homo = np.concatenate([corners, np.ones((4, 1), dtype=np.float32)], axis=1)
            tc = homo @ M.T
            nx1, ny1 = tc[:, 0].min(), tc[:, 1].min()
            nx2, ny2 = tc[:, 0].max(), tc[:, 1].max()
            nx1, ny1, nx2, ny2 = self._clip_box_xyxy(nx1, ny1, nx2, ny2, s)
            nw, nh = nx2 - nx1, ny2 - ny1
            if nw < 2 or nh < 2:
                continue
            new_boxes.append((
                cls_id,
                (nx1 + nx2) / 2 / s,
                (ny1 + ny2) / 2 / s,
                nw / s,
                nh / s,
            ))
        return warped, new_boxes

    @staticmethod
    def _photometric(img):
        """Mild contrast/noise; khong lam anatomy phi thuc te."""
        out = img.astype(np.float32)
        if np.random.rand() < 0.45:
            alpha = np.random.uniform(0.92, 1.08)
            beta = np.random.uniform(-6.0, 6.0)
            out[..., :2] = out[..., :2] * alpha + beta
        if np.random.rand() < 0.15:
            noise = np.random.normal(0.0, 2.0, size=out.shape).astype(np.float32)
            out += noise
        return np.clip(out, 0, 255).astype(np.uint8)

    def __getitem__(self, idx: int):
        img_path = self.image_files[idx]
        label_path = self.lbl_dir / f"{img_path.stem}.txt"

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Khong doc duoc anh: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        boxes = self._read_raw_boxes(label_path)

        if self.augment:
            if np.random.rand() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1, :])
                boxes = [(c, 1.0 - cx, cy, w, h) for (c, cx, cy, w, h) in boxes]
            img, boxes = self._random_affine(img, boxes)
            img = self._photometric(img)

        target = self._encode_grid_target(boxes)
        img = (img.astype(np.float32) / 255.0 - _MEAN) / _STD
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        return img_tensor, torch.from_numpy(target)


def collate_fn(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs, dim=0), torch.stack(targets, dim=0)
