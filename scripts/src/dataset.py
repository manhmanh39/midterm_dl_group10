"""
VinBigDataDetectionDataset - doc anh + NHIEU bbox/anh (multi-object detection thuc su).

Doc file nhan dinh dang YOLO chuan (moi dong: cls cx cy w h, da normalize [0,1],
KHONG gioi han so dong/anh - anh co the co 0, 1, hoac nhieu chuc bbox).

Encode target thanh grid tensor kieu YOLOv1/v3 don gian (anchor-free, 1-scale):
    target shape = (GRID, GRID, 5 + NUM_CLASSES)
    5 = [objectness, tx, ty, tw, th]
        - objectness: 1 neu cell nay "chiu trach nhiem" cho 1 object (co tam
          object roi vao cell), 0 neu khong.
        - tx, ty: offset cua tam box so voi CANH TREN-TRAI cua cell, in [0,1]
        - tw, th: width/height cua box, normalized theo CA ANH (khong theo cell),
          in [0,1] - don gian hoa so voi log-scale cua YOLOv2/v3 de de hoc
          hon voi CNN tu viet (khong can anchor prior).
    NUM_CLASSES: one-hot class cho cell do (chi co nghia khi objectness=1)

Neu 2 object co tam roi vao CUNG 1 cell (hiem nhung co the xay ra voi object
nho + grid thua), object co DIEN TICH LON HON se "thang" (duoc gan vao cell),
object con lai bi bo qua trong target (han che von co cua thiet ke 1
object/cell/1-scale - neu can bat toan bo, phai dung multi-scale + anchor
nhu YOLOv3+, ngoai pham vi bai tap CNN kien truc nay).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from scripts.config import NUM_CLASSES, GRID_SIZE, IMAGE_SIZE


class VinBigDataDetectionDataset(Dataset):
    def __init__(self, data_dir: str, image_size: int = IMAGE_SIZE, grid_size: int = GRID_SIZE, augment: bool = False):
        """
        Parameters
        ----------
        data_dir : thu muc split, phai chua images/ va labels/ (labels dinh
                   dang YOLO, ghi boi scripts/data/prepare_dataset.py)
        image_size : kich thuoc anh vuong dau ra
        grid_size : so cell moi chieu cua detection head (phai KHOP voi
                    grid_size cua model dang dung, xem scripts.config.GRID_SIZE)
        augment : bat/tat flip ngang cho tap train
        """
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / "images"
        self.lbl_dir = self.data_dir / "labels"
        self.image_size = image_size
        self.grid_size = grid_size
        self.augment = augment

        if not self.img_dir.exists():
            raise FileNotFoundError(
                f"Khong tim thay thu muc anh: {self.img_dir}. "
                "Hay chay scripts/data/prepare_dataset.py truoc."
            )

        self.image_files = sorted(self.img_dir.glob("*.png"))
        if len(self.image_files) == 0:
            raise RuntimeError(f"Khong co anh .png nao trong {self.img_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def _read_raw_boxes(self, label_path: Path):
        """Doc toan bo box trong file .txt, tra ve list[(cls_id, cx, cy, w, h)]."""
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

    def _encode_grid_target(self, boxes):
        """
        Chuyen list box (normalized ca anh) thanh grid target tensor
        (GRID, GRID, 5 + NUM_CLASSES).
        """
        G = self.grid_size
        target = np.zeros((G, G, 5 + NUM_CLASSES), dtype=np.float32)
        # Luu dien tich cua box da gan cho tung cell, de xu ly xung dot
        # (2 box cung roi vao 1 cell -> giu box lon hon)
        assigned_area = np.zeros((G, G), dtype=np.float32)

        for cls_id, cx, cy, w, h in boxes:
            # Xac dinh cell chua tam box
            gx = min(int(cx * G), G - 1)
            gy = min(int(cy * G), G - 1)

            area = w * h
            if target[gy, gx, 0] == 1.0 and area <= assigned_area[gy, gx]:
                # Cell nay da co object khac lon hon -> bo qua object nay
                continue

            # Offset cua tam box so voi canh trai-tren cua cell, in [0,1]
            tx = cx * G - gx
            ty = cy * G - gy

            target[gy, gx, 0] = 1.0          # objectness
            target[gy, gx, 1] = tx
            target[gy, gx, 2] = ty
            target[gy, gx, 3] = w             # normalized theo ca anh
            target[gy, gx, 4] = h
            target[gy, gx, 5:] = 0.0
            target[gy, gx, 5 + cls_id] = 1.0  # one-hot class

            assigned_area[gy, gx] = area

        return target

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

        if self.augment and np.random.rand() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1, :])
            boxes = [(c, 1.0 - cx, cy, w, h) for (c, cx, cy, w, h) in boxes]

        target = self._encode_grid_target(boxes)

        img = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        target_tensor = torch.from_numpy(target)

        return img_tensor, target_tensor


def collate_fn(batch):
    """Ghep batch don gian (moi anh co so box khac nhau nhung target da la
    tensor grid co kich thuoc co dinh, nen stack binh thuong duoc)."""
    imgs, targets = zip(*batch)
    return torch.stack(imgs, dim=0), torch.stack(targets, dim=0)