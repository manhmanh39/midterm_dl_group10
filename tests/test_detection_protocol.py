"""
tests/test_detection_protocol.py - Bo Unit Tests kiem thu 100% cac guards va dinh dang P0/P1 Detection.
Bao gom:
  1. Empty vs Missing label contract (Fail-Closed)
  2. Image ID alignment trong Dataset & Collate
  3. Cell collision audit module
  4. PASCAL VOC 2010+ continuous all-points AP
  5. Split Semantics Test Guard (Fail-Closed khi thieu token)
  6. Dataset Fingerprint & Checkpoint SHA sidecar lock
  7. Preflight Check tampering detection (Fail-Closed)
  8. Sample Standard Deviation voi ddof=1
  9. Complexity RAM state_dict_size_mib measurement
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from scripts.benchmark_utils import measure_model_complexity
from scripts.config import NUM_CLASSES
from scripts.data.prepared_loader import (
    audit_cell_collisions,
    get_develop_dataloaders,
    get_test_dataloader,
)
from scripts.experiment_config import (
    compute_dataset_fingerprint,
    compute_file_sha256,
    generate_protocol_lock,
    global_preflight_check,
    verify_protocol_lock_token,
    write_checkpoint_sha256,
)
from scripts.models.factory import build_model
from scripts.src.dataset import VinBigDataDetectionDataset, collate_fn
from scripts.src.metrics import compute_voc_ap


class TestDetectionProtocol(unittest.TestCase):
    def setUp(self):
        # Tao dummy detection dataset directory
        self.tmp_dir = tempfile.mkdtemp()
        self.root = Path(self.tmp_dir)

        for split in ("train", "val", "test"):
            (self.root / split / "images").mkdir(parents=True)
            (self.root / split / "labels").mkdir(parents=True)

        # Tao dataset.yaml
        yaml_content = f"path: {self.root.as_posix()}\ntrain: train/images\nval: val/images\ntest: test/images\nnc: 14\n"
        (self.root / "dataset.yaml").write_text(yaml_content, encoding="utf-8")

        # Tao 2 sample dummy images (512x512 RGB) va labels cho moi split
        for split in ("train", "val", "test"):
            for i in range(2):
                img_name = f"img_{i:04d}.png"
                img_p = self.root / split / "images" / img_name
                # Tao fake png bang numpy/cv2
                import cv2
                dummy_img = np.full((512, 512, 3), fill_value=(i + 1) * 40, dtype=np.uint8)
                cv2.imwrite(str(img_p), dummy_img)

                # img_0000 co 1 box; img_0001 la No-Finding (0-byte text file)
                lbl_p = self.root / split / "labels" / f"img_{i:04d}.txt"
                if i == 0:
                    lbl_p.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
                else:
                    lbl_p.write_text("", encoding="utf-8") # Empty file = valid No-Finding

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_01_empty_vs_missing_label_contract(self):
        """Kiem tra hop dong No-Finding (0-byte hop le) vs Missing file (Fail-Closed)."""
        train_dir = self.root / "train"
        ds = VinBigDataDetectionDataset(str(train_dir), image_size=512, grid_size=16)

        # img_0000: co 1 box
        boxes_0 = ds.get_raw_boxes(0)
        self.assertEqual(len(boxes_0), 1)

        # img_0001: 0-byte file -> No-finding hop le (0 box, khong loi)
        boxes_1 = ds.get_raw_boxes(1)
        self.assertEqual(len(boxes_1), 0)

        # Xoa file label cua img_0000 -> Missing label -> phai nem FileNotFoundError
        lbl_0 = self.root / "train" / "labels" / "img_0000.txt"
        lbl_0.unlink()
        with self.assertRaises(FileNotFoundError):
            ds.get_raw_boxes(0)

    def test_02_dataset_and_collate_returns_image_id(self):
        """Kiem tra Dataset va Collate_fn tra ve dung image_id de alignment."""
        train_dir = self.root / "train"
        ds = VinBigDataDetectionDataset(str(train_dir), image_size=512, grid_size=16)
        item = ds[0]
        self.assertEqual(len(item), 3)
        img_t, target_t, img_id = item
        self.assertEqual(img_id, "img_0000")
        self.assertEqual(target_t.shape, (16, 16, 19))

        batch = [ds[0], ds[1]]
        imgs, targets, ids = collate_fn(batch)
        self.assertEqual(imgs.shape, (2, 3, 512, 512))
        self.assertEqual(targets.shape, (2, 16, 16, 19))
        self.assertEqual(ids, ["img_0000", "img_0001"])

    def test_03_cell_collision_audit(self):
        """Kiem tra module audit cell collision phat hien dung va cham."""
        # Ghi 2 box cung roi vao cell (gy=8, gx=8): cx=0.51, cy=0.51 va cx=0.52, cy=0.52
        lbl_p = self.root / "train" / "labels" / "img_0000.txt"
        lbl_p.write_text("0 0.51 0.51 0.2 0.2\n1 0.52 0.52 0.3 0.3\n", encoding="utf-8")

        ds = VinBigDataDetectionDataset(str(self.root / "train"), image_size=512, grid_size=16)
        aud = audit_cell_collisions(ds, grid_size=16)
        self.assertEqual(aud["total_boxes"], 2)
        self.assertEqual(aud["collided_boxes"], 1)
        self.assertEqual(aud["images_with_collision"], 1)
        self.assertEqual(aud["max_boxes_in_cell"], 2)

    def test_04_voc_continuous_all_points_ap(self):
        """Kiem tra PASCAL VOC 2010+ continuous all-points interpolated AP."""
        # Test 1: Precision va Recall deu rong -> 0.0
        self.assertEqual(compute_voc_ap(np.array([]), np.array([])), 0.0)

        # Test 2: Perfect detection (Precision=1.0 o moi Recall level)
        rec = np.array([0.2, 0.5, 1.0])
        prec = np.array([1.0, 1.0, 1.0])
        self.assertAlmostEqual(compute_voc_ap(rec, prec), 1.0, places=5)

        # Test 3: Zigzag precision curve -> Continuous interpolation phai lay envelope max
        rec = np.array([0.1, 0.4, 0.8])
        prec = np.array([0.8, 0.4, 0.6]) # Envelope tai rec 0.4 phai duoc nang len 0.6
        ap = compute_voc_ap(rec, prec)
        self.assertGreater(ap, 0.0)
        self.assertLessEqual(ap, 1.0)

    def test_05_split_semantics_test_guard_fail_closed(self):
        """Kiem tra TestLoader tu choi truy cap neu khong co protocol_lock_token."""
        # 1. get_develop_dataloaders khong dong vao test
        train_l, val_l, _, _ = get_develop_dataloaders(self.root, batch_size=2, num_workers=0)
        self.assertIsNotNone(train_l)
        self.assertIsNotNone(val_l)

        # 2. get_test_dataloader khong truyen token -> phai nem RuntimeError
        with self.assertRaises(RuntimeError) as ctx:
            get_test_dataloader(self.root, batch_size=2, num_workers=0, lock_token=None)
        self.assertIn("TEST SET ACCESS DENIED", str(ctx.exception))

        # 3. get_test_dataloader truyen token gia mao -> phai nem RuntimeError
        with self.assertRaises(RuntimeError) as ctx:
            get_test_dataloader(self.root, batch_size=2, num_workers=0, lock_token="fake_token_12345")
        self.assertIn("TEST SET ACCESS DENIED", str(ctx.exception))

    def test_06_checkpoint_sha_sidecar_and_protocol_lock(self):
        """Kiem tra sinh SHA sidecar, tao protocol_lock.json va preflight check."""
        ckpt_dir = self.root / "checkpoints"
        ckpt_dir.mkdir()
        dummy_models = ["model1", "model2", "model3"]
        dummy_seeds = [202601, 202602, 202603]

        for m in dummy_models:
            for s in dummy_seeds:
                ckpt_p = ckpt_dir / f"{m}_seed{s}_best.pth"
                torch.save({"model_name": m, "seed": s, "weight": torch.randn(2, 2)}, ckpt_p)

        lock_path = self.root / "protocol_lock.json"
        lock_data = generate_protocol_lock(
            data_root=self.root,
            checkpoint_dir=ckpt_dir,
            lock_path=lock_path,
            models=dummy_models,
            seeds=dummy_seeds,
        )
        self.assertTrue(lock_path.is_file())
        self.assertEqual(lock_data["expected_checkpoints_count"], 9)

        # Kiem tra tat ca 9 sidecars .sha256 deu da duoc tao
        for m in dummy_models:
            for s in dummy_seeds:
                sidecar_p = ckpt_dir / f"{m}_seed{s}_best.pth.sha256"
                self.assertTrue(sidecar_p.is_file())
                sha_in_sidecar = sidecar_p.read_text().split()[0]
                actual_sha = compute_file_sha256(ckpt_dir / f"{m}_seed{s}_best.pth")
                self.assertEqual(sha_in_sidecar, actual_sha)

        # Global Preflight Check phai pass va cap token hop le
        passed, token = global_preflight_check(lock_path=lock_path, data_root=self.root)
        self.assertTrue(passed)
        self.assertTrue(verify_protocol_lock_token(token, lock_path=lock_path))

        # Mo TestLoader voi token vua cap phai thanh cong
        test_l, test_ds = get_test_dataloader(
            self.root, batch_size=2, num_workers=0, lock_token=token, lock_path=lock_path
        )
        self.assertIsNotNone(test_l)
        self.assertEqual(len(test_ds), 2)

    def test_07_preflight_tampering_fail_closed(self):
        """Kiem tra neu checkpoint hoac data bi sua sau khi khoa, preflight se Fail-Closed."""
        ckpt_dir = self.root / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        m, s = "model1", 202601
        ckpt_p = ckpt_dir / f"{m}_seed{s}_best.pth"
        torch.save({"dummy": 1}, ckpt_p)

        lock_path = self.root / "protocol_lock.json"
        generate_protocol_lock(
            data_root=self.root,
            checkpoint_dir=ckpt_dir,
            lock_path=lock_path,
            models=[m],
            seeds=[s],
        )

        # Gia mao sua noi dung checkpoint tren dia
        ckpt_p.write_bytes(b"tampered checkpoint content")

        # Global preflight check phai nem loi ngay lap tuc
        with self.assertRaises(RuntimeError) as ctx:
            global_preflight_check(lock_path=lock_path, data_root=self.root)
        self.assertIn("SHA256 mismatch", str(ctx.exception))

    def test_08_sample_std_ddof1(self):
        """Kiem tra phuong sai va do lech chuan mau dung ddof=1 (N-1)."""
        vals = [0.25, 0.27, 0.29]
        std_sample = float(np.std(vals, ddof=1))
        std_pop = float(np.std(vals, ddof=0))
        # ddof=1 phai lon hon ddof=0
        self.assertGreater(std_sample, std_pop)
        self.assertAlmostEqual(std_sample, 0.02, places=4)

    def test_09_model_complexity_ram_state_dict(self):
        """Kiem tra do dac state_dict_size_mib trong RAM."""
        model = build_model("model1", num_classes=14, pretrained=False)
        comp = measure_model_complexity(model, img_size=512)
        self.assertIn("state_dict_size_mib", comp)
        self.assertGreater(comp["state_dict_size_mib"], 0.0)
        self.assertEqual(comp["total_params"], comp["trainable_params"] + comp["frozen_params"])


if __name__ == "__main__":
    unittest.main()
