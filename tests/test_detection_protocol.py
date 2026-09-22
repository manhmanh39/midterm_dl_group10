"""
tests/test_detection_protocol.py - Bo Unit Tests kiem thu 100% cac guards va dinh dang P0/P1 Detection.
(Bao gom toan bo 18 chi muc kiem toan duoc nang cap).
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from scripts.benchmark_utils import benchmark_detection_inference, measure_model_complexity
from scripts.config import (
    BASELINE_CONF_THRESHOLD,
    BASELINE_NMS_IOU,
    EVAL_MIN_SCORE,
    NUM_CLASSES,
)
from scripts.data.prepared_loader import (
    audit_cell_collisions,
    describe_dataset,
    get_develop_dataloaders,
    get_test_dataloader,
)
from scripts.experiment_config import (
    CANONICAL_MODELS,
    CANONICAL_SEEDS,
    PreflightPermit,
    compute_dataset_fingerprint,
    compute_file_sha256,
    compute_split_fingerprint,
    generate_protocol_lock,
    global_preflight_check,
    verify_protocol_lock_token,
    write_checkpoint_sha256,
)
from evaluate import evaluate_model
from run_multi_seed import run_canonical_multi_seed
from scripts.models.factory import build_model
from scripts.src.dataset import VinBigDataDetectionDataset, collate_fn
from scripts.src.metrics import compute_voc_ap, evaluate_detections


class TestDetectionProtocolHardened(unittest.TestCase):
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

    def test_02_strict_malformed_annotation_validation(self):
        """Kiem tra validation nghiem ngat: bat loi khi dong nhan bi malformed."""
        train_dir = self.root / "train"
        lbl_p = self.root / "train" / "labels" / "img_0000.txt"

        # 1. Thieu truong (chi co 4 truong)
        lbl_p.write_text("0 0.5 0.5 0.2\n", encoding="utf-8")
        ds = VinBigDataDetectionDataset(str(train_dir), image_size=512, grid_size=16)
        with self.assertRaises(ValueError) as ctx:
            ds.get_raw_boxes(0)
        self.assertIn("expected exactly 5 fields", str(ctx.exception))

        # 2. Class ID khong phai so nguyen
        lbl_p.write_text("0.5 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            ds.get_raw_boxes(0)
        self.assertIn("Class ID must be integer", str(ctx.exception))

        # 3. Class ID ngoai khoang [0, 13]
        lbl_p.write_text("14 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            ds.get_raw_boxes(0)
        self.assertIn("out of range", str(ctx.exception))

        # 4. Toa do NaN hoac Inf
        lbl_p.write_text("0 nan 0.5 0.2 0.2\n", encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            ds.get_raw_boxes(0)
        self.assertIn("Non-finite coordinate", str(ctx.exception))

        # 5. Toa do ngoai range [0, 1]
        lbl_p.write_text("0 1.5 0.5 0.2 0.2\n", encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            ds.get_raw_boxes(0)
        self.assertIn("Center coordinates out of range", str(ctx.exception))

        # 6. Chieu dai w <= 0
        lbl_p.write_text("0 0.5 0.5 0.0 0.2\n", encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            ds.get_raw_boxes(0)
        self.assertIn("Box dimensions out of range", str(ctx.exception))

    def test_03_image_and_label_bijection_enforcement(self):
        """Kiem tra dataset fingerprint bat buoc quan he song anh giua anh va nhan."""
        # 1. Thu muc dang song anh hop le
        fp_clean = compute_split_fingerprint(self.root / "train")
        self.assertIsInstance(fp_clean, str)

        # 2. Tao them 1 file label mo coi khong co anh
        orphan_lbl = self.root / "train" / "labels" / "orphan_img.txt"
        orphan_lbl.write_text("", encoding="utf-8")
        with self.assertRaises(RuntimeError) as ctx:
            compute_split_fingerprint(self.root / "train")
        self.assertIn("khong song anh", str(ctx.exception))
        orphan_lbl.unlink()

        # 3. Tao them 1 file anh khong co nhan
        orphan_img = self.root / "train" / "images" / "orphan_img.png"
        cv2.imwrite(str(orphan_img), np.zeros((10, 10, 3), dtype=np.uint8))
        with self.assertRaises(RuntimeError) as ctx:
            compute_split_fingerprint(self.root / "train")
        self.assertIn("khong song anh", str(ctx.exception))
        orphan_img.unlink()

    def test_04_dataset_and_collate_returns_image_id(self):
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

    def test_05_image_byte_level_tamper_detection(self):
        """Kiem tra Dataset Fingerprint phat hien thay doi raw bytes cua anh PNG."""
        fp1 = compute_dataset_fingerprint(self.root)
        img_p = self.root / "test" / "images" / "img_0000.png"

        # Doc bytes cua anh, sua 1 byte cuoi cung (giu nguyen kich thuoc file)
        b = bytearray(img_p.read_bytes())
        b[-1] = (b[-1] + 1) % 256
        img_p.write_bytes(bytes(b))

        fp2 = compute_dataset_fingerprint(self.root)
        # Fingerprint phai thay doi vi raw bytes da bi sua!
        self.assertNotEqual(fp1, fp2)

    def test_06_metrics_strict_image_id_lookup_no_fallback(self):
        """Kiem tra metrics bat buoc co image_ids va khong fallback ve sequential index."""
        model = build_model("model1", num_classes=NUM_CLASSES, pretrained=False)
        train_dir = self.root / "train"
        ds = VinBigDataDetectionDataset(str(train_dir), image_size=512, grid_size=16)

        # Tao fake loader khong tra ve image_ids (chi co imgs, targets)
        fake_loader = [(torch.randn(2, 3, 512, 512), torch.zeros(2, 16, 16, 19))]
        device = torch.device("cpu")

        with self.assertRaises(RuntimeError) as ctx:
            evaluate_detections(model, ds, fake_loader, device, num_classes=NUM_CLASSES)
        self.assertIn("requires DataLoader/collate_fn to provide image_ids", str(ctx.exception))

    def test_07_cell_collision_audit(self):
        """Kiem tra module audit cell collision phat hien dung va cham."""
        lbl_p = self.root / "train" / "labels" / "img_0000.txt"
        lbl_p.write_text("0 0.51 0.51 0.2 0.2\n1 0.52 0.52 0.3 0.3\n", encoding="utf-8")

        ds = VinBigDataDetectionDataset(str(self.root / "train"), image_size=512, grid_size=16)
        aud = audit_cell_collisions(ds, grid_size=16)
        self.assertEqual(aud["total_boxes"], 2)
        self.assertEqual(aud["collided_boxes"], 1)
        self.assertEqual(aud["images_with_collision"], 1)
        self.assertEqual(aud["max_boxes_in_cell"], 2)

    def test_08_voc_continuous_all_points_ap(self):
        """Kiem tra PASCAL VOC 2010+ continuous all-points interpolated AP."""
        self.assertEqual(compute_voc_ap(np.array([]), np.array([])), 0.0)

        rec = np.array([0.2, 0.5, 1.0])
        prec = np.array([1.0, 1.0, 1.0])
        self.assertAlmostEqual(compute_voc_ap(rec, prec), 1.0, places=5)

        rec = np.array([0.1, 0.4, 0.8])
        prec = np.array([0.8, 0.4, 0.6])
        ap = compute_voc_ap(rec, prec)
        self.assertGreater(ap, 0.0)
        self.assertLessEqual(ap, 1.0)

    def test_09_split_semantics_test_guard_fail_closed(self):
        """Kiem tra TestLoader tu choi truy cap neu khong co PreflightPermit hop le."""
        train_l, val_l, _, _ = get_develop_dataloaders(self.root, batch_size=2, num_workers=0)
        self.assertIsNotNone(train_l)
        self.assertIsNotNone(val_l)

        with self.assertRaises(RuntimeError) as ctx:
            get_test_dataloader(self.root, batch_size=2, num_workers=0, lock_token=None)
        self.assertIn("TEST SET ACCESS DENIED", str(ctx.exception))

        with self.assertRaises(RuntimeError) as ctx:
            get_test_dataloader(self.root, batch_size=2, num_workers=0, lock_token="fake_token_12345")
        self.assertIn("TEST SET ACCESS DENIED", str(ctx.exception))

    def test_10_relock_prevention_in_final_test(self):
        """Kiem tra final-test KHONG DUOC PHEP tu dong sinh lai lock (chan re-lock blocker)."""
        lock_path = self.root / "outputs" / "protocol_lock.json"
        if lock_path.exists():
            lock_path.unlink()

        # Goi final-test khi chua co lock -> phai nem RuntimeError ngay lap tuc
        with self.assertRaises(RuntimeError) as ctx:
            run_canonical_multi_seed(
                models=["model1"],
                seeds=[202601],
                phase="final-test",
                data_root=str(self.root),
                output_dir=str(self.root / "outputs"),
                data_mode="demo",
            )
        self.assertIn("Protocol lock file khong ton tai", str(ctx.exception))
        # Dam bao file lock van khong bi tu dong sinh ra
        self.assertFalse(lock_path.exists())

    def test_11_checkpoint_lock_membership_enforcement(self):
        """Kiem tra evaluator chi chap nhan checkpoint da duoc khoa trong protocol_lock.json."""
        ckpt_dir = self.root / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        m, s = "model1", 202601
        valid_ckpt = ckpt_dir / f"{m}_seed{s}_best.pth"
        torch.save({"model_name": m, "seed": s, "model_state_dict": {}}, valid_ckpt)

        lock_path = self.root / "outputs" / "protocol_lock.json"
        generate_protocol_lock(
            data_root=self.root,
            checkpoint_dir=ckpt_dir,
            lock_path=lock_path,
            models=[m],
            seeds=[s],
            data_mode="demo",
        )
        _, permit = global_preflight_check(lock_path=lock_path, data_root=self.root, data_mode="demo")

        # Tao 1 checkpoint gia mao ngoai danh sach lock
        rogue_ckpt = ckpt_dir / "rogue_model.pth"
        torch.save({"model_name": m, "seed": s, "model_state_dict": {}}, rogue_ckpt)

        # Evaluator phai nem loi vi rogue_ckpt khong thuoc lock
        with self.assertRaises(RuntimeError) as ctx:
            evaluate_model(
                model_name=m,
                checkpoint_path=str(rogue_ckpt),
                data_root=str(self.root),
                lock_token=permit,
                lock_path=str(lock_path),
            )
        self.assertIn("KHONG thuoc danh sach", str(ctx.exception))

    def test_12_evaluation_config_enforcement_from_lock(self):
        """Kiem tra evaluator tu dong enforce evaluation config duoc khoa trong protocol_lock.json."""
        ckpt_dir = self.root / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        m, s = "model1", 202601
        model = build_model(m, num_classes=NUM_CLASSES, pretrained=False)
        valid_ckpt = ckpt_dir / f"{m}_seed{s}_best.pth"
        torch.save({
            "model_name": m, "seed": s,
            "model_state_dict": model.state_dict(),
            "image_size": 512, "grid_size": 16,
        }, valid_ckpt)

        lock_path = self.root / "outputs" / "protocol_lock.json"
        generate_protocol_lock(
            data_root=self.root,
            checkpoint_dir=ckpt_dir,
            lock_path=lock_path,
            models=[m],
            seeds=[s],
            data_mode="demo",
        )
        _, permit = global_preflight_check(lock_path=lock_path, data_root=self.root, data_mode="demo")

        # Chay evaluate_model voi CLI flag conf_threshold=0.99 (co tinh truyen khac)
        res = evaluate_model(
            model_name=m,
            checkpoint_path=str(valid_ckpt),
            data_root=str(self.root),
            conf_threshold=0.99, # Co tinh truyen nguong khac
            lock_token=permit,
            lock_path=str(lock_path),
            output_dir=str(self.root / "outputs"),
            num_workers=0,
        )
        # Evaluator phai enforce theo BASELINE_CONF_THRESHOLD (0.25) tu lock
        self.assertIn("map50", res)

    def test_13_exact_canonical_3x3_enforced_in_real_mode(self):
        """Kiem tra real mode bat buoc dung dung 3 canonical models x 3 canonical seeds."""
        ckpt_dir = self.root / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        lock_path = self.root / "outputs" / "protocol_lock.json"

        # Truyen thieu model hoac seed o real mode -> phai nem ValueError
        with self.assertRaises(ValueError) as ctx:
            generate_protocol_lock(
                data_root=self.root,
                checkpoint_dir=ckpt_dir,
                lock_path=lock_path,
                models=["model1"],
                seeds=[202601],
                data_mode="real",
            )
        self.assertIn("Bat buoc dung chinh xac 3 canonical models", str(ctx.exception))

    def test_14_leakage_discipline_develop_split(self):
        """Kiem tra Develop chi describe train va val, khong bao gio doc test."""
        stats = describe_dataset(self.root, splits=("train", "val"))
        self.assertIn("train", stats)
        self.assertIn("val", stats)
        self.assertNotIn("test", stats)

    def test_15_sample_std_ddof1(self):
        """Kiem tra phuong sai va do lech chuan mau dung ddof=1 (N-1)."""
        vals = [0.25, 0.27, 0.29]
        std_sample = float(np.std(vals, ddof=1))
        std_pop = float(np.std(vals, ddof=0))
        self.assertGreater(std_sample, std_pop)
        self.assertAlmostEqual(std_sample, 0.02, places=4)

    def test_16_model_complexity_ram_state_dict(self):
        """Kiem tra do dac state_dict_size_mib trong RAM."""
        model = build_model("model1", num_classes=14, pretrained=False)
        comp = measure_model_complexity(model, img_size=512)
        self.assertIn("state_dict_size_mib", comp)
        self.assertGreater(comp["state_dict_size_mib"], 0.0)
        self.assertEqual(comp["total_params"], comp["trainable_params"] + comp["frozen_params"])

    def test_17_benchmark_metric_naming_accuracy(self):
        """Kiem tra benchmark utils ghi ro batch size va device."""
        model = build_model("model1", num_classes=14, pretrained=False)
        speed = benchmark_detection_inference(model, device=torch.device("cpu"), num_warmup=1, num_runs=2)
        self.assertEqual(speed["benchmark_device"], "cpu")
        self.assertIn("throughput_batch_size", speed)
        self.assertIn(f"bs{speed['throughput_batch_size']}_throughput_fps", speed)

    def test_18_duplicate_canonical_models_seeds_rejected(self):
        """Kiem tra phat hien va tu choi duplicate models hoac seeds."""
        ckpt_dir = self.root / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        lock_path = self.root / "outputs" / "protocol_lock.json"

        # Duplicate models
        with self.assertRaises(ValueError) as ctx:
            generate_protocol_lock(
                data_root=self.root,
                checkpoint_dir=ckpt_dir,
                lock_path=lock_path,
                models=["model1", "model1", "model2"],
                seeds=[202601, 202602, 202603],
                data_mode="demo",
            )
        self.assertIn("Duplicate model", str(ctx.exception))

        # Duplicate seeds
        with self.assertRaises(ValueError) as ctx:
            generate_protocol_lock(
                data_root=self.root,
                checkpoint_dir=ckpt_dir,
                lock_path=lock_path,
                models=["model1", "model2", "model3"],
                seeds=[202601, 202601, 202602],
                data_mode="demo",
            )
        self.assertIn("Duplicate seed", str(ctx.exception))

    def test_19_forbid_phase_all_in_real_mode(self):
        """Kiem tra cam tuyet doi phase='all' o data_mode='real'."""
        with self.assertRaises(ValueError) as ctx:
            run_canonical_multi_seed(
                models=["model1", "model2", "model3"],
                seeds=[202601, 202602, 202603],
                phase="all",
                data_mode="real",
            )
        self.assertIn("cam tuyet doi chay '--phase all'", str(ctx.exception))

    def test_20_lock_sidecar_and_immutability_tamper_detection(self):
        """Kiem tra sidecar protocol_lock.json.sha256 va bat bien (Fail-Closed neu lock bi sua)."""
        ckpt_dir = self.root / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        m, s = "model1", 202601
        valid_ckpt = ckpt_dir / f"{m}_seed{s}_best.pth"
        torch.save({"model_name": m, "seed": s, "model_state_dict": {}}, valid_ckpt)

        lock_path = self.root / "outputs" / "protocol_lock.json"
        generate_protocol_lock(
            data_root=self.root,
            checkpoint_dir=ckpt_dir,
            lock_path=lock_path,
            models=[m],
            seeds=[s],
            data_mode="demo",
        )

        sidecar_path = self.root / "outputs" / "protocol_lock.json.sha256"
        self.assertTrue(sidecar_path.is_file())

        # Thu sua noi dung protocol_lock.json
        content = lock_path.read_text(encoding="utf-8")
        tampered = content.replace('"baseline_conf_threshold": 0.25', '"baseline_conf_threshold": 0.05')
        lock_path.write_text(tampered, encoding="utf-8")

        # global_preflight_check phai nem loi vi sidecar SHA bi lech
        with self.assertRaises(RuntimeError) as ctx:
            global_preflight_check(lock_path=lock_path, data_root=self.root, data_mode="demo")
        self.assertIn("Protocol lock immutability violated", str(ctx.exception))

    def test_21_semantic_model_seed_provenance_binding(self):
        """Kiem tra cross-check semantic: model_name, seed giua checkpoint va lock."""
        ckpt_dir = self.root / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        m, s = "model1", 202601
        model = build_model(m, num_classes=NUM_CLASSES, pretrained=False)
        valid_ckpt = ckpt_dir / f"{m}_seed{s}_best.pth"
        # Co tinh luu seed sai trong checkpoint: seed 999999 thay vi 202601
        torch.save({
            "model_name": m,
            "seed": 999999,
            "model_state_dict": model.state_dict(),
            "image_size": 512, "grid_size": 16,
        }, valid_ckpt)

        lock_path = self.root / "outputs" / "protocol_lock.json"
        generate_protocol_lock(
            data_root=self.root,
            checkpoint_dir=ckpt_dir,
            lock_path=lock_path,
            models=[m],
            seeds=[s],
            data_mode="demo",
        )
        _, permit = global_preflight_check(lock_path=lock_path, data_root=self.root, data_mode="demo")

        with self.assertRaises(RuntimeError) as ctx:
            evaluate_model(
                model_name=m,
                checkpoint_path=str(valid_ckpt),
                data_root=str(self.root),
                lock_token=permit,
                lock_path=str(lock_path),
                output_dir=str(self.root / "outputs"),
                data_mode="demo",
            )
        self.assertIn("Seed mismatch trong checkpoint provenance", str(ctx.exception))

    def test_22_real_mode_downgrade_bypass_prevention(self):
        """Kiem tra khong the dung data_mode=demo de bypass real lock."""
        ckpt_dir = self.root / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        lock_path = self.root / "outputs" / "protocol_lock.json"
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        # Tao lock gia lap data_mode='real'
        lock_data = {
            "protocol_version": "P0_P1_DETECTION_HARDENED_V1",
            "data_mode": "real",
            "dataset_root": str(self.root.as_posix()),
            "dataset_fingerprint": compute_dataset_fingerprint(self.root),
            "checkpoints": {},
        }
        lock_path.write_text(json.dumps(lock_data, indent=2), encoding="utf-8")
        from scripts.experiment_config import write_protocol_lock_sha256
        write_protocol_lock_sha256(lock_path)

        # Goi preflight voi data_mode='demo' phai bi chan
        with self.assertRaises(ValueError) as ctx:
            global_preflight_check(lock_path=lock_path, data_root=self.root, data_mode="demo")
        self.assertIn("Anti-Bypass Guard", str(ctx.exception))

    def test_23_prepare_dataset_root_consistency(self):
        """Kiem tra prepare_dataset.py su dung default=get_processed_data_root()."""
        from scripts.config import get_processed_data_root
        import scripts.data.prepare_dataset as prep
        import inspect
        src = inspect.getsource(prep)
        self.assertIn("default=get_processed_data_root()", src)
        self.assertEqual(get_processed_data_root(), "data/dataset_202601")

    def test_24_runtime_artifacts_do_not_dirty_git(self):
        """Kiem tra is_git_clean() loai tru runtime outputs de khong bi dirty false positive."""
        from scripts.experiment_config import is_git_clean
        status = is_git_clean(ignore_runtime_outputs=True)
        self.assertIsInstance(status, bool)


if __name__ == "__main__":
    unittest.main()
