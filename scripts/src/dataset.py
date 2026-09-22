"""VinBigDataDetectionDataset - anh + nhieu bbox/anh, encode grid target kieu YOLO 1-scale."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from scripts.config import NUM_CLASSES, GRID_SIZE, IMAGE_SIZE

cv2.setNumThreads(0)  # tranh oversubscription khi dung nhieu DataLoader workers

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class VinBigDataDetectionDataset(Dataset):
    def __init__(self, data_dir: str, image_size: int = IMAGE_SIZE, grid_size: int = GRID_SIZE, augment: bool = False):
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / "images"
        self.lbl_dir = self.data_dir / "labels"
        self.image_size = image_size
        self.grid_size = grid_size
        self.augment = augment

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Khong tim thay thu muc anh: {self.img_dir}. Hay chay prepare_dataset.py truoc.")
        self.image_files = sorted(self.img_dir.glob("*.png"))
        if len(self.image_files) == 0:
            raise RuntimeError(f"Khong co anh .png nao trong {self.img_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def _read_raw_boxes(self, label_path: Path):
        if not label_path.exists():
            raise FileNotFoundError(
                f"Missing label file: {label_path}. Detection dataset requires a .txt label file "
                f"for every image (0-byte file represents No-Finding)."
            )
        text = label_path.read_text(encoding="utf-8").strip()
        boxes = []
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
        """GT that tu file label theo index (khong mat box do va cham cell)."""
        return self._read_raw_boxes(self.lbl_dir / f"{self.image_files[idx].stem}.txt")

    def get_raw_boxes_by_id(self, image_id: str):
        """GT that tu file label theo image_id (khong mat box do va cham cell). Dung cho mAP."""
        return self._read_raw_boxes(self.lbl_dir / f"{image_id}.txt")

    def _encode_grid_target(self, boxes):
        G = self.grid_size
        target = np.zeros((G, G, 5 + NUM_CLASSES), dtype=np.float32)
        assigned_area = np.zeros((G, G), dtype=np.float32)

        for cls_id, cx, cy, w, h in boxes:
            gx = min(max(int(cx * G), 0), G - 1)
            gy = min(max(int(cy * G), 0), G - 1)
            area = w * h
            if target[gy, gx, 0] == 1.0 and area <= assigned_area[gy, gx]:
                continue
            target[gy, gx, 0] = 1.0
            target[gy, gx, 1] = cx * G - gx
            target[gy, gx, 2] = cy * G - gy
            target[gy, gx, 3] = w
            target[gy, gx, 4] = h
            target[gy, gx, 5:] = 0.0
            target[gy, gx, 5 + cls_id] = 1.0
            assigned_area[gy, gx] = area
        return target

    def __getitem__(self, idx: int):
        img_path = self.image_files[idx]
        image_id = img_path.stem
        label_path = self.lbl_dir / f"{image_id}.txt"

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Khong doc duoc anh: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        boxes = self._read_raw_boxes(label_path)

        if self.augment and np.random.rand() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1, :])
            boxes = [(c, 1.0 - cx, cy, w, h) for (c, cx, cy, w, h) in boxes]

        target = self._encode_grid_target(boxes)

        img = (img.astype(np.float32) / 255.0 - _MEAN) / _STD   # chuan hoa ImageNet
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        return img_tensor, torch.from_numpy(target), image_id


def collate_fn(batch):
    imgs, targets, image_ids = zip(*batch)
    return torch.stack(imgs, dim=0), torch.stack(targets, dim=0), list(image_ids)