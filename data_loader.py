from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
from PIL import Image, ImageDraw

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import config
from prepared_loader import is_prepared_dataset, build_prepared_samples
from seed_utils import seed_worker, get_generator


def get_transforms(image_size: Tuple[int, int] = config.IMAGE_SIZE):
    train_transform = T.Compose([
        T.Resize(image_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.15, contrast=0.15),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.2),
        T.ToTensor(),
        T.RandomErasing(p=0.15, scale=(0.02, 0.08)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, eval_transform


class VinBigDataDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, object]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        if config.IS_MULTILABEL:
            label_t = torch.as_tensor(label, dtype=torch.float32)
        else:
            label_t = torch.as_tensor(label, dtype=torch.long)

        return image, label_t


def generate_demo_dataset(base_dir: Path, num_samples_per_split: int = 20) -> Path:
    demo_root = base_dir / "demo_data"
    splits = ["train", "val", "test"]

    for split in splits:
        split_dir = demo_root / split
        if split_dir.exists() and any(split_dir.iterdir()):
            continue
        split_dir.mkdir(parents=True, exist_ok=True)
        count = num_samples_per_split if split == "train" else max(4, num_samples_per_split // 2)

        for class_id, class_name in enumerate(config.CLASS_NAMES):
            class_folder = split_dir / f"{class_id:02d}_{class_name.replace(' ', '_')}"
            class_folder.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                img = Image.new("L", (224, 224), color=20)
                draw = ImageDraw.Draw(img)
                draw.ellipse([30, 40, 100, 180], fill=120)
                draw.ellipse([124, 40, 194, 180], fill=120)
                if class_id != 14:
                    pos_x = 40 + (class_id * 10) % 120
                    pos_y = 60 + (class_id * 7) % 100
                    draw.ellipse([pos_x, pos_y, pos_x + 20, pos_y + 20], fill=220)
                img.save(class_folder / f"sample_{i:03d}.png")

    return demo_root


def _folder_class_id(folder_name: str, fallback: int) -> int:
    try:
        return int(folder_name.split("_")[0])
    except ValueError:
        return fallback


def load_split_samples(split_dir: Path) -> List[Tuple[str, object]]:
    samples = []
    if not split_dir.exists():
        return samples

    subdirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    for dir_idx, folder in enumerate(subdirs):
        class_id = _folder_class_id(folder.name, dir_idx)
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for img_file in folder.glob(ext):
                if config.IS_MULTILABEL:
                    label = np.zeros(config.NUM_CLASSES, dtype=np.float32)
                    if 0 <= class_id < config.NUM_CLASSES:
                        label[class_id] = 1.0
                    samples.append((str(img_file), label))
                else:
                    samples.append((str(img_file), class_id))
    return samples


def compute_pos_weight(samples: List[Tuple[str, object]]) -> Optional[torch.Tensor]:
    if not config.IS_MULTILABEL or not samples:
        return None
    labels = np.stack([lbl for _, lbl in samples], axis=0)
    pos_count = labels.sum(axis=0)
    neg_count = labels.shape[0] - pos_count
    pos_weight = neg_count / np.clip(pos_count, 1, None)
    return torch.tensor(pos_weight, dtype=torch.float32)


def get_develop_dataloaders(
    data_dir: Optional[str] = None,
    data_mode: Optional[str] = None,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    image_size: Tuple[int, int] = config.IMAGE_SIZE,
    seed: int = config.SEED,
) -> Tuple[DataLoader, DataLoader, List[str], Optional[torch.Tensor]]:
    """
    Tạo DataLoaders cho giai đoạn DEVELOP (Train và Val).
    TUYỆT ĐỐI KHÔNG đọc, truy cập hoặc nạp bất kỳ dữ liệu nào của split 'test'.
    """
    mode = data_mode if data_mode is not None else config.DEFAULT_DATA_MODE
    train_transform, eval_transform = get_transforms(image_size)

    if mode == "real":
        target_dir = Path(data_dir) if data_dir else config.PROCESSED_DATA_DIR
        if not target_dir.exists() or not is_prepared_dataset(target_dir):
            raise FileNotFoundError(
                f"[FAIL CLOSED] Real prepared dataset không tồn tại tại: {target_dir}!\n"
                f"Giao thức cấm tự động fallback sang demo khi data_mode='real'."
            )
        train_samples, val_samples = build_prepared_samples(
            target_dir, splits=("train", "val"), num_classes=config.NUM_CLASSES, strict=True
        )
    elif mode == "demo":
        target_dir = Path(data_dir) if data_dir else config.DEMO_DATA_DIR
        if not target_dir.exists() or not any(target_dir.iterdir()):
            print("[data_loader] Khởi tạo demo dataset...")
            generate_demo_dataset(config.BASE_DIR)
        train_samples = load_split_samples(target_dir / "train")
        val_samples = load_split_samples(target_dir / "val")
    else:
        raise ValueError(f"Invalid data_mode: '{mode}'. Phải là 'real' hoặc 'demo'.")

    print(f"[data_loader:develop] Train={len(train_samples)}  Val={len(val_samples)} (mode={mode})")

    pos_weight = compute_pos_weight(train_samples)
    if pos_weight is not None:
        print(f"[data_loader:develop] pos_weight (class imbalance) = {pos_weight.tolist()}")

    train_dataset = VinBigDataDataset(train_samples, transform=train_transform)
    val_dataset = VinBigDataDataset(val_samples, transform=eval_transform)

    generator = get_generator(seed)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )

    return train_loader, val_loader, config.CLASS_NAMES, pos_weight


def get_test_dataloader(
    data_dir: Optional[str] = None,
    data_mode: Optional[str] = None,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    image_size: Tuple[int, int] = config.IMAGE_SIZE,
) -> Tuple[DataLoader, List[str]]:
    """
    Tạo DataLoader DUY NHẤT cho giai đoạn FINAL-TEST (Chỉ nạp split 'test').
    Chỉ được gọi sau khi Global Preflight đã pass hoàn toàn.
    """
    mode = data_mode if data_mode is not None else config.DEFAULT_DATA_MODE
    _, eval_transform = get_transforms(image_size)

    if mode == "real":
        target_dir = Path(data_dir) if data_dir else config.PROCESSED_DATA_DIR
        if not target_dir.exists() or not is_prepared_dataset(target_dir):
            raise FileNotFoundError(
                f"[FAIL CLOSED] Real prepared dataset không tồn tại tại: {target_dir}!"
            )
        (test_samples,) = build_prepared_samples(
            target_dir, splits=("test",), num_classes=config.NUM_CLASSES, strict=True
        )
    elif mode == "demo":
        target_dir = Path(data_dir) if data_dir else config.DEMO_DATA_DIR
        if not target_dir.exists() or not (target_dir / "test").exists():
            print("[data_loader] Khởi tạo demo dataset...")
            generate_demo_dataset(config.BASE_DIR)
        test_samples = load_split_samples(target_dir / "test")
    else:
        raise ValueError(f"Invalid data_mode: '{mode}'. Phải là 'real' hoặc 'demo'.")

    print(f"[data_loader:test] Test={len(test_samples)} (mode={mode})")

    test_dataset = VinBigDataDataset(test_samples, transform=eval_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )

    return test_loader, config.CLASS_NAMES


def get_dataloaders(
    data_dir: Optional[str] = None,
    data_mode: Optional[str] = None,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    image_size: Tuple[int, int] = config.IMAGE_SIZE,
    seed: int = config.SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], Optional[torch.Tensor]]:
    """Legacy helper kết hợp cả develop và test loader."""
    mode = data_mode if data_mode is not None else config.DEFAULT_DATA_MODE
    train_loader, val_loader, class_names, pos_weight = get_develop_dataloaders(
        data_dir=data_dir, data_mode=mode, batch_size=batch_size,
        num_workers=num_workers, image_size=image_size, seed=seed,
    )
    test_loader, _ = get_test_dataloader(
        data_dir=data_dir, data_mode=mode, batch_size=batch_size,
        num_workers=num_workers, image_size=image_size,
    )
    return train_loader, val_loader, test_loader, class_names, pos_weight


if __name__ == "__main__":
    train_ld, val_ld, test_ld, classes, pos_w = get_dataloaders(batch_size=4)
    images, labels = next(iter(train_ld))
    print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")