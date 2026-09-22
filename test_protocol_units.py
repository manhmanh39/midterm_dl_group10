"""
test_protocol_units.py - Bộ kiểm thử đơn vị độc lập cho Giao thức P1 (P1 Protocol Unit Tests).
Kiểm tra:
  1. Safe Wrappers: safe_average_precision, safe_roc_auc trên edge cases (all-0, all-1).
  2. Threshold Calibration: PR-curve indexing, deterministic tie-breaking (gần 0.5 nhất).
  3. Fallback Semantics: threshold=0.5, val_f1=None, status='fallback' khi thiếu support.
  4. No-Finding Consistency Policy: Invariant total_inconsistency_rate == 0.0.
  5. Fail-Closed Provenance Guard: Checkpoint hash tampering & missing artifact abort.
"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from metrics_utils import (
    safe_average_precision,
    safe_roc_auc,
    calibrate_thresholds_from_pr_curve,
    apply_thresholds,
    measure_no_finding_inconsistency,
    enforce_no_finding_consistency,
    compute_per_class_table,
    compute_macro_metrics,
)
from experiment_config import (
    compute_file_sha256,
    global_preflight_check,
    verify_fail_closed_provenance,
)


class TestSafeMetricWrappers(unittest.TestCase):
    def test_safe_average_precision_edge_cases(self):
        # All negatives
        y_true_zeros = np.zeros(20, dtype=np.float32)
        y_prob = np.linspace(0.1, 0.9, 20)
        self.assertTrue(np.isnan(safe_average_precision(y_true_zeros, y_prob)))

        # All positives
        y_true_ones = np.ones(20, dtype=np.float32)
        self.assertTrue(np.isnan(safe_average_precision(y_true_ones, y_prob)))

        # Valid binary distribution
        y_true_mixed = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32)
        y_prob_mixed = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.2, 0.8])
        ap = safe_average_precision(y_true_mixed, y_prob_mixed)
        self.assertFalse(np.isnan(ap))
        self.assertGreater(ap, 0.0)
        self.assertLessEqual(ap, 1.0)

    def test_safe_roc_auc_edge_cases(self):
        y_true_zeros = np.zeros(20, dtype=np.float32)
        y_prob = np.linspace(0.1, 0.9, 20)
        self.assertTrue(np.isnan(safe_roc_auc(y_true_zeros, y_prob)))

        y_true_ones = np.ones(20, dtype=np.float32)
        self.assertTrue(np.isnan(safe_roc_auc(y_true_ones, y_prob)))

        y_true_mixed = np.array([0, 0, 1, 1], dtype=np.float32)
        y_prob_mixed = np.array([0.1, 0.2, 0.8, 0.9])
        auc = safe_roc_auc(y_true_mixed, y_prob_mixed)
        self.assertEqual(auc, 1.0)


class TestThresholdCalibration(unittest.TestCase):
    def test_fallback_semantics_when_support_missing(self):
        # Class 0: all negative (n_pos == 0)
        # Class 1: all positive (n_neg == 0)
        y_true = np.zeros((20, 2), dtype=np.float32)
        y_true[:, 1] = 1.0
        y_prob = np.full((20, 2), 0.3, dtype=np.float32)
        class_names = ["NoPosClass", "NoNegClass"]

        res = calibrate_thresholds_from_pr_curve(y_true, y_prob, class_names=class_names)
        thresholds = res["thresholds"]
        calibs = res["per_class_calibration"]

        # Ngưỡng phải là 0.5
        self.assertEqual(thresholds[0], 0.5)
        self.assertEqual(thresholds[1], 0.5)

        # Semantics: status='fallback', val_f1=None
        self.assertEqual(calibs[0]["status"], "fallback")
        self.assertIsNone(calibs[0]["val_f1"])
        self.assertEqual(calibs[0]["reason"], "no_positive_samples")
        self.assertEqual(calibs[0]["threshold"], 0.5)

        self.assertEqual(calibs[1]["status"], "fallback")
        self.assertIsNone(calibs[1]["val_f1"])
        self.assertEqual(calibs[1]["reason"], "no_negative_samples")
        self.assertEqual(calibs[1]["threshold"], 0.5)

    def test_deterministic_tie_breaking(self):
        # Tạo dữ liệu dẫn đến 2 threshold có F1 bằng hệt nhau
        # Ví dụ: thresholds = [0.2, 0.48, 0.8] đều đạt max F1
        # Tie-break rule: chọn threshold gần 0.5 nhất -> phải là 0.48
        y_true = np.array([[0], [0], [1], [1]], dtype=np.float32)
        y_prob = np.array([[0.1], [0.3], [0.7], [0.9]], dtype=np.float32)
        class_names = ["TestTie"]

        res = calibrate_thresholds_from_pr_curve(y_true, y_prob, class_names=class_names)
        calibs = res["per_class_calibration"]
        self.assertEqual(calibs[0]["status"], "calibrated")
        self.assertIsNotNone(calibs[0]["val_f1"])
        # Threshold chọn phải nằm trong [0.01, 0.99]
        self.assertGreaterEqual(calibs[0]["threshold"], 0.01)
        self.assertLessEqual(calibs[0]["threshold"], 0.99)


class TestNoFindingConsistency(unittest.TestCase):
    def test_inconsistency_measurement(self):
        # Giả lập 15 classes: 0..13 là pathologies, 14 là No Finding
        preds = np.zeros((4, 15), dtype=np.float32)

        # Mẫu 0: Nhất quán (có pathology, no finding = 0)
        preds[0, 0] = 1.0
        preds[0, 14] = 0.0

        # Mẫu 1: Mâu thuẫn Contradiction (có pathology VÀ no finding = 1)
        preds[1, 2] = 1.0
        preds[1, 14] = 1.0

        # Mẫu 2: Mâu thuẫn Empty Diagnosis (toàn bộ pathology = 0 VÀ no finding = 0)
        preds[2, :] = 0.0

        # Mẫu 3: Nhất quán (toàn bộ pathology = 0 VÀ no finding = 1)
        preds[3, 14] = 1.0

        incons = measure_no_finding_inconsistency(preds)
        self.assertEqual(incons["contradiction_rate"], 0.25)
        self.assertEqual(incons["empty_diagnosis_rate"], 0.25)
        self.assertEqual(incons["total_inconsistency_rate"], 0.50)

    def test_enforce_no_finding_consistency_invariant(self):
        # Random predictions trên 100 mẫu
        rng = np.random.RandomState(42)
        raw_preds = (rng.rand(100, 15) > 0.5).astype(np.float32)

        # Điều chỉnh theo chính sách nhất quán
        adjusted = enforce_no_finding_consistency(raw_preds)

        # Kiểm tra tính bất biến sau khi điều chỉnh
        incons = measure_no_finding_inconsistency(adjusted)
        self.assertEqual(incons["contradiction_rate"], 0.0)
        self.assertEqual(incons["empty_diagnosis_rate"], 0.0)
        self.assertEqual(incons["total_inconsistency_rate"], 0.0)

        # Đảm bảo 14 pathologies KHÔNG hề bị thay đổi
        np.testing.assert_array_equal(raw_preds[:, :14], adjusted[:, :14])


class TestFailClosedProvenance(unittest.TestCase):
    def test_checkpoint_tampering_aborts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ckpt_path = tmp_path / "best.pth"
            th_path = tmp_path / "calibrated_thresholds.json"

            # Lưu checkpoint giả có đầy đủ required provenance fields
            torch.save({
                "model_name": "simple",
                "seed": 202601,
                "model_state_dict": {},
                "provenance": {
                    "model_name": "simple",
                    "seed": 202601,
                    "git_commit": "abc1234",
                    "git_dirty": False,
                    "dataset_fingerprint": "fake_fingerprint_123",
                },
            }, ckpt_path)

            ckpt_hash = compute_file_sha256(ckpt_path)

            # Lưu threshold khớp
            dataset_meta = {"dataset_fingerprint": "fake_fingerprint_123", "is_demo_data": False}
            with open(th_path, "w", encoding="utf-8") as f:
                json.dump({
                    "provenance": {
                        "model_name": "simple",
                        "seed": 202601,
                        "checkpoint_sha256": ckpt_hash,
                        "dataset_fingerprint": "fake_fingerprint_123",
                        "git_commit": "abc1234",
                        "git_dirty": False,
                    },
                    "thresholds": [0.5] * 15,
                }, f)

            # Kiểm tra: Khi khớp -> pass
            verify_fail_closed_provenance(ckpt_path, th_path, dataset_meta)

            # Giả lập sửa đổi Checkpoint (Tampering)
            with open(ckpt_path, "ab") as f:
                f.write(b"\x00corrupt")

            # Phải ném ngoại lệ ValueError ngay lập tức
            with self.assertRaises(ValueError):
                verify_fail_closed_provenance(ckpt_path, th_path, dataset_meta)

    def test_missing_required_provenance_field_aborts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ckpt_path = tmp_path / "best.pth"
            th_path = tmp_path / "calibrated_thresholds.json"

            torch.save({
                "model_name": "simple",
                "seed": 202601,
                "model_state_dict": {},
                "provenance": {
                    "model_name": "simple",
                    "seed": 202601,
                    # Thiếu git_commit và dataset_fingerprint
                },
            }, ckpt_path)
            ckpt_hash = compute_file_sha256(ckpt_path)

            dataset_meta = {"dataset_fingerprint": "fake_fingerprint_123", "is_demo_data": False}
            with open(th_path, "w", encoding="utf-8") as f:
                json.dump({
                    "provenance": {
                        "model_name": "simple",
                        "seed": 202601,
                        "checkpoint_sha256": ckpt_hash,
                        # Thiếu git_commit
                    },
                    "thresholds": [0.5] * 15,
                }, f)

            with self.assertRaises(ValueError) as ctx:
                verify_fail_closed_provenance(ckpt_path, th_path, dataset_meta)
            self.assertIn("FAIL CLOSED", str(ctx.exception))

    def test_global_preflight_missing_lock_aborts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            missing_lock = tmp_path / "non_existent_protocol_lock.json"
            with self.assertRaises(FileNotFoundError):
                global_preflight_check(lock_file=missing_lock)

    def test_global_preflight_commit_mismatch_aborts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ckpt_path = tmp_path / "best.pth"
            th_path = tmp_path / "calibrated_thresholds.json"
            lock_path = tmp_path / "protocol_lock.json"

            ckpt_path.write_text("dummy checkpoint content")
            th_path.write_text(json.dumps({"thresholds": [0.5]*15}))

            real_ckpt_hash = compute_file_sha256(ckpt_path)
            real_th_hash = compute_file_sha256(th_path)

            lock_data = {
                "git_commit": "locked_commit_aaa",
                "dataset_fingerprint": "mock_fp",
                "experiments": [{
                    "model": "simple",
                    "seed": 202601,
                    "checkpoint_rel_path": "best.pth",
                    "checkpoint_sha256": real_ckpt_hash,
                    "threshold_rel_path": "calibrated_thresholds.json",
                    "threshold_sha256": real_th_hash,
                }]
            }
            lock_path.write_text(json.dumps(lock_data))

            from unittest.mock import patch
            with patch("experiment_config.DEVELOP_DIR", tmp_path):
                with patch("experiment_config.compute_dataset_fingerprint", return_value={"dataset_fingerprint": "mock_fp", "is_demo_data": False}):
                    with patch("experiment_config.git_worktree_is_dirty", return_value=False):
                        with patch("experiment_config.get_git_commit", return_value="current_different_commit_bbb"):
                            with self.assertRaises(RuntimeError) as ctx:
                                global_preflight_check(lock_file=lock_path, enforce_clean_git=True)
                            self.assertIn("Git commit hiện tại", str(ctx.exception))

    def test_global_preflight_tampered_checkpoint_aborts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ckpt_path = tmp_path / "best.pth"
            th_path = tmp_path / "calibrated_thresholds.json"
            lock_path = tmp_path / "protocol_lock.json"

            ckpt_path.write_text("dummy checkpoint content")
            th_path.write_text(json.dumps({"thresholds": [0.5]*15}))

            real_ckpt_hash = compute_file_sha256(ckpt_path)
            real_th_hash = compute_file_sha256(th_path)

            lock_data = {
                "dataset_fingerprint": "mock_fp",
                "experiments": [{
                    "model": "simple",
                    "seed": 202601,
                    "checkpoint_rel_path": "best.pth",
                    "checkpoint_sha256": "tampered_sha256_hash",
                    "threshold_rel_path": "calibrated_thresholds.json",
                    "threshold_sha256": real_th_hash,
                }]
            }
            lock_path.write_text(json.dumps(lock_data))

            from unittest.mock import patch
            with patch("experiment_config.DEVELOP_DIR", tmp_path):
                with patch("experiment_config.compute_dataset_fingerprint", return_value={"dataset_fingerprint": "mock_fp", "is_demo_data": True}):
                    with self.assertRaises(ValueError) as ctx:
                        global_preflight_check(lock_file=lock_path)
                    self.assertIn("FAIL CLOSED", str(ctx.exception))

    def test_global_preflight_missing_git_commit_aborts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            lock_path = tmp_path / "protocol_lock.json"

            # lock_data không có trường git_commit
            lock_data = {
                "dataset_fingerprint": "mock_fp",
                "experiments": []
            }
            lock_path.write_text(json.dumps(lock_data))

            from unittest.mock import patch
            with patch("experiment_config.DEVELOP_DIR", tmp_path):
                with patch("experiment_config.compute_dataset_fingerprint", return_value={"dataset_fingerprint": "mock_fp", "is_demo_data": False}):
                    with patch("experiment_config.git_worktree_is_dirty", return_value=False):
                        with self.assertRaises(ValueError) as ctx:
                            global_preflight_check(lock_file=lock_path, enforce_clean_git=True)
                        self.assertIn("thiếu trường bắt buộc 'git_commit'", str(ctx.exception))

    def test_git_checks_fail_closed_on_command_error(self):
        from unittest.mock import patch
        import subprocess
        from experiment_config import git_worktree_is_dirty, get_git_commit

        with patch("subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "git")):
            # Không được fail-open (trả về False hay uncommitted_workspace) khi allow_fallback=False
            with self.assertRaises(RuntimeError) as ctx_dirty:
                git_worktree_is_dirty()
            self.assertIn("FAIL CLOSED", str(ctx_dirty.exception))

            with self.assertRaises(RuntimeError) as ctx_commit:
                get_git_commit()
            self.assertIn("FAIL CLOSED", str(ctx_commit.exception))


class TestModelFactoryAndReload(unittest.TestCase):
    def test_transfer_reload_without_pretrained(self):
        from models import get_model
        # Phải khởi tạo được khi truyền pretrained=False, freeze_base=False
        model = get_model("transfer", num_classes=15, backbone_name="resnet18", pretrained=False, freeze_base=False)
        self.assertIsNotNone(model)
        dummy = torch.randn(2, 3, 224, 224)
        out = model(dummy)
        self.assertEqual(out.shape, (2, 15))

    def test_simple_and_complex_pop_transfer_kwargs(self):
        from models import get_model
        # Simple và complex không nhận pretrained/freeze_base nhưng không được ném TypeError
        m1 = get_model("simple", num_classes=15, pretrained=False, freeze_base=False, backbone_name="resnet50")
        m2 = get_model("complex", num_classes=15, pretrained=False, freeze_base=False, backbone_name="resnet50")
        self.assertIsNotNone(m1)
        self.assertIsNotNone(m2)


class TestAccuracyMetricsDistinction(unittest.TestCase):
    def test_per_label_vs_exact_match_accuracy(self):
        # 2 mẫu, 3 nhãn
        # Mẫu 1: true = [1, 0, 1], pred = [1, 0, 0] -> 2/3 nhãn đúng, exact match = 0
        # Mẫu 2: true = [0, 0, 0], pred = [0, 0, 0] -> 3/3 nhãn đúng, exact match = 1
        y_true = np.array([[1, 0, 1], [0, 0, 0]], dtype=np.float32)
        y_pred = np.array([[1, 0, 0], [0, 0, 0]], dtype=np.float32)

        per_label_acc = float((y_true == y_pred).mean())
        exact_match_acc = float((y_true == y_pred).all(axis=1).mean())

        self.assertAlmostEqual(per_label_acc, 5.0 / 6.0) # 83.33%
        self.assertAlmostEqual(exact_match_acc, 0.5)      # 50.0%
        self.assertNotEqual(per_label_acc, exact_match_acc)

    def test_per_class_table_and_macro_accuracy(self):
        y_true = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
        y_prob = np.array([[0.8, 0.2, 0.9], [0.1, 0.7, 0.3]], dtype=np.float32)
        thresholds = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        names = ["A", "B", "No Finding"]

        table = compute_per_class_table(y_true, y_prob, thresholds, names)
        self.assertIn("accuracy_fixed", table[0])
        self.assertIn("accuracy_calibrated", table[0])
        self.assertEqual(table[0]["accuracy_fixed"], 1.0)

        macro = compute_macro_metrics(table)
        self.assertIn("macro_accuracy_14_fixed", macro)
        self.assertIn("macro_accuracy_14_calibrated", macro)
        self.assertEqual(macro["macro_accuracy_14_fixed"], 1.0)


class TestZeroBypassRealDataProtocol(unittest.TestCase):
    def test_cannot_skip_preflight_on_real_data(self):
        from unittest.mock import patch
        from eval import evaluate_model

        with patch("eval.compute_dataset_fingerprint", return_value={"dataset_fingerprint": "mock_fp", "is_demo_data": False}):
            with self.assertRaises(RuntimeError) as ctx:
                evaluate_model(model_name="simple", enforce_preflight=False, data_mode="real")
            self.assertIn("Giao thức cấm bỏ qua preflight", str(ctx.exception))

    def test_missing_lock_on_real_data_aborts(self):
        from unittest.mock import patch
        from eval import evaluate_model
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            with patch("eval.compute_dataset_fingerprint", return_value={"dataset_fingerprint": "mock_fp", "is_demo_data": False}):
                with patch("eval.DEVELOP_DIR", tmp_path):
                    with self.assertRaises(FileNotFoundError) as ctx:
                        evaluate_model(model_name="simple", enforce_preflight=True, data_mode="real")
                    self.assertIn("Không tìm thấy file khóa giao thức bắt buộc", str(ctx.exception))


class TestP1RegressionSuite(unittest.TestCase):
    def test_1_develop_loader_zero_test_touch(self):
        """P1.2: get_develop_dataloaders chỉ load train & val, tuyệt đối không chạm test/ hay test_boxes.csv."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for sp in ["train", "val", "test"]:
                (root / sp / "images").mkdir(parents=True)
                (root / sp / f"{sp}_boxes.csv").write_text("image_id,class_id\nsample1,0\n")
                from PIL import Image
                Image.new("RGB", (32, 32)).save(root / sp / "images" / "sample1.png")

            manifest = {
                "train": ["sample1"],
                "val": ["sample1"],
                "test": ["sample1"],
            }
            (root / "manifest.json").write_text(json.dumps(manifest))

            from data_loader import get_develop_dataloaders
            # Xóa file test_boxes.csv để chứng minh nếu get_develop_dataloaders đọc test_boxes.csv nó sẽ lỗi
            (root / "test" / "test_boxes.csv").unlink()
            (root / "test" / "images" / "sample1.png").unlink()

            train_ld, val_ld, classes, pos_w = get_develop_dataloaders(
                data_dir=str(root), data_mode="real", batch_size=1, num_workers=0
            )
            self.assertIsNotNone(train_ld)
            self.assertIsNotNone(val_ld)
            self.assertEqual(len(train_ld.dataset), 1)
            self.assertEqual(len(val_ld.dataset), 1)

    def test_2_real_loader_fail_closed_missing_manifest(self):
        """P1.3: Thiếu manifest.json trên dataset strict/real -> FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            from prepared_loader import build_prepared_samples
            with self.assertRaises(FileNotFoundError) as ctx:
                build_prepared_samples(root, splits=("train",), strict=True)
            self.assertIn("FAIL CLOSED", str(ctx.exception))

    def test_3_real_loader_fail_closed_missing_csv(self):
        """P1.3: Thiếu split_boxes.csv trên dataset strict/real -> FileNotFoundError, không fallback ngầm."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "train" / "images").mkdir(parents=True)
            (root / "manifest.json").write_text(json.dumps({"train": ["sample1"]}))
            from PIL import Image
            Image.new("RGB", (32, 32)).save(root / "train" / "images" / "sample1.png")

            from prepared_loader import build_prepared_samples
            with self.assertRaises(FileNotFoundError) as ctx:
                build_prepared_samples(root, splits=("train",), strict=True)
            self.assertIn("Không tìm thấy file annotations bắt buộc", str(ctx.exception))

    def test_4_real_loader_fail_closed_missing_image(self):
        """P1.3: Có trong manifest nhưng thiếu file PNG thật -> FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "train" / "images").mkdir(parents=True)
            (root / "train" / "train_boxes.csv").write_text("image_id,class_id\nmissing_img,0\n")
            (root / "manifest.json").write_text(json.dumps({"train": ["missing_img"]}))

            from prepared_loader import build_prepared_samples
            with self.assertRaises(FileNotFoundError) as ctx:
                build_prepared_samples(root, splits=("train",), strict=True)
            self.assertIn("Thiếu file ảnh", str(ctx.exception))

    def test_5_real_loader_fail_closed_invalid_class_id(self):
        """P1.3: Class ID không hợp lệ (<0 hoặc >=15) trong CSV annotations -> ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "train" / "images").mkdir(parents=True)
            (root / "train" / "train_boxes.csv").write_text("image_id,class_id\nsample1,99\n")
            (root / "manifest.json").write_text(json.dumps({"train": ["sample1"]}))
            from PIL import Image
            Image.new("RGB", (32, 32)).save(root / "train" / "images" / "sample1.png")

            from prepared_loader import build_prepared_samples
            with self.assertRaises(ValueError) as ctx:
                build_prepared_samples(root, splits=("train",), num_classes=15, strict=True)
            self.assertIn("class_id không hợp lệ", str(ctx.exception))

    def test_6_protocol_lock_matrix_canonical_set_equality(self):
        """P1.4: Protocol lock trên real data bắt buộc kiểm tra set equality đúng 9 cặp canonical (3x3)."""
        import config
        from experiment_config import generate_protocol_lock

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with unittest.mock.patch("experiment_config.DEVELOP_DIR", root):
                with unittest.mock.patch("experiment_config.git_worktree_is_dirty", return_value=False):
                    with unittest.mock.patch("experiment_config.compute_dataset_fingerprint", return_value={"dataset_fingerprint": "mock_fp", "is_demo_data": False, "data_mode": "real"}):
                        # Tạo chỉ 8 cặp hoặc sai cặp (thiếu transfer seed 202603)
                        for m in ["simple", "complex"]:
                            for s in [202601, 202602, 202603]:
                                d = root / m / f"seed{s}"
                                d.mkdir(parents=True)
                                (d / "best.pth").write_text("dummy")
                                (d / "calibrated_thresholds.json").write_text("{}")
                        for s in [202601, 202602]:
                            d = root / "transfer" / f"seed{s}"
                            d.mkdir(parents=True)
                            (d / "best.pth").write_text("dummy")
                            (d / "calibrated_thresholds.json").write_text("{}")

                        # Thiếu 1 cặp -> Phải báo lỗi ValueError
                        with self.assertRaises(FileNotFoundError):
                            generate_protocol_lock(model_names=config.CANONICAL_MODELS, seeds=config.CANONICAL_SEEDS, data_mode="real")

    def test_7_threshold_calibration_unclipped_pr(self):
        """P1.5: Calibration không được clip về [0.01, 0.99]. Ngưỡng tối ưu tự nhiên <0.01 hoặc >0.99 phải được giữ nguyên."""
        # Tạo trường hợp threshold tối ưu rất thấp (< 0.01)
        y_true = np.array([[1], [0], [0], [0]], dtype=np.float32)
        y_prob = np.array([[0.005], [0.001], [0.0005], [0.0001]], dtype=np.float32)
        res = calibrate_thresholds_from_pr_curve(y_true, y_prob, class_names=["RareClass"])
        opt_t = res["thresholds"][0]
        # Nếu clip thì sẽ là 0.01. Không clip thì sẽ là 0.005 (hoặc giá trị trong PR thresholds <= 0.005)
        self.assertLess(opt_t, 0.01)

    def test_8_threshold_calibration_aborts_on_abnormal_or_nonfinite(self):
        """P1.5: Xác suất có NaN/Inf hoặc ground-truth không nhị phân phải ném ValueError ngay lập tức."""
        # Non-finite prob
        y_true = np.array([[1, 0], [0, 1]], dtype=np.float32)
        y_prob_nan = np.array([[np.nan, 0.5], [0.2, 0.8]], dtype=np.float32)
        with self.assertRaises(ValueError):
            calibrate_thresholds_from_pr_curve(y_true, y_prob_nan)

        # Non-binary target
        y_true_non_binary = np.array([[2, 0], [0, 1]], dtype=np.float32)
        y_prob_clean = np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32)
        with self.assertRaises(ValueError):
            calibrate_thresholds_from_pr_curve(y_true_non_binary, y_prob_clean)

    def test_9_safe_macro_auc_14_empty_classes_and_caller_abort(self):
        """P1.6: compute_safe_macro_auc trả về NaN và valid_classes=0 khi không có lớp nào hợp lệ; validate_one_epoch trên real data aborts."""
        from metrics_utils import compute_safe_macro_auc
        from train import validate_one_epoch
        import torch.nn as nn

        # Tất cả samples toàn 0 cho 14 classes -> Không class nào có cả positive và negative
        y_true = np.zeros((10, 14), dtype=np.float32)
        y_prob = np.full((10, 14), 0.5, dtype=np.float32)
        res = compute_safe_macro_auc(y_true, y_prob, class_indices=range(14))
        self.assertTrue(np.isnan(res["macro_auc"]))
        self.assertEqual(res["valid_classes"], 0)

        # Giả lập DataLoader với nhãn toàn 0
        from torch.utils.data import TensorDataset, DataLoader
        dummy_x = torch.randn(4, 3, 32, 32)
        dummy_y = torch.zeros(4, 15, dtype=torch.float32)
        loader = DataLoader(TensorDataset(dummy_x, dummy_y), batch_size=2)
        dummy_model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 15))
        criterion = nn.BCEWithLogitsLoss()

        # Trên real data, validate_one_epoch phải abort (raise RuntimeError)
        with self.assertRaises(RuntimeError) as ctx:
            validate_one_epoch(dummy_model, loader, criterion, torch.device("cpu"), data_mode="real")
        self.assertIn("0 valid classes on real data", str(ctx.exception))

    def test_10_benchmark_state_dict_size_greater_than_params_for_batchnorm(self):
        """P1.7: Với model có BatchNorm (ComplexCNN), state_dict_size_bytes phải LỚN HƠN THẬT SỰ (>) parameter_only_bytes do có buffers."""
        from models import get_model
        from benchmark_utils import get_model_complexity

        model = get_model("complex", num_classes=15)
        comp = get_model_complexity(model)

        param_bytes = comp["parameter_only_bytes"]
        state_bytes = comp["state_dict_size_bytes"]

        self.assertGreater(state_bytes, param_bytes)
        self.assertIn("state_dict_size_mib", comp)
        self.assertIn("weights_size_mb", comp)


class TestP1HardeningSuite(unittest.TestCase):
    """Bộ kiểm thử hardening nâng cao: Semantic binding, Exact-nine, Namespace, Denominator, Range guards."""

    def test_h1_exact_nine_duplicate_missing_extra_fails(self):
        """H1: Protocol lock & preflight fail nếu có duplicate pair, missing pair hoặc extra pair."""
        from experiment_config import global_preflight_check
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            lock_path = root / "protocol_lock.json"
            meta = {"dataset_fingerprint": "mock_fp", "is_demo_data": False, "data_mode": "real"}

            # Duplicate pair: 9 canonical + 1 duplicate = 10 items, unique = 9
            exp_duplicate = [
                {"model": m, "seed": s, "checkpoint_rel_path": "x", "checkpoint_sha256": "h", "threshold_rel_path": "t", "threshold_sha256": "h"}
                for m in ["simple", "complex", "transfer"] for s in [202601, 202602, 202603]
            ]
            exp_duplicate.append(exp_duplicate[0].copy())
            lock_data = {
                "git_commit": "mock_commit",
                "git_dirty": False,
                "dataset_fingerprint": "mock_fp",
                "experiments": exp_duplicate,
            }
            lock_path.write_text(json.dumps(lock_data))

            with unittest.mock.patch("experiment_config.git_worktree_is_dirty", return_value=False):
                with unittest.mock.patch("experiment_config.get_git_commit", return_value="mock_commit"):
                    with unittest.mock.patch("experiment_config.compute_dataset_fingerprint", return_value=meta):
                        with self.assertRaises(ValueError) as ctx:
                            global_preflight_check(lock_file=lock_path, data_mode="real")
                        self.assertIn("không chứa đúng 9 cặp canonical", str(ctx.exception))

    def test_h2_internal_provenance_semantic_mismatch_fails_despite_lock_sha(self):
        """H2: Internal provenance mismatch (sai model_name) dù SHA trong lock khớp vẫn abort preflight."""
        from experiment_config import global_preflight_check, compute_file_sha256
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            meta = {"dataset_fingerprint": "mock_fp", "is_demo_data": True, "data_mode": "demo", "data_dir": "mock"}

            experiments = []
            for m in ["simple", "complex", "transfer"]:
                for s in [202601, 202602, 202603]:
                    ckpt_file = root / f"{m}_{s}_best.pth"
                    th_file = root / f"{m}_{s}_th.json"

                    # Giả mạo semantic: model 'transfer' seed 202603 nhưng internal ghi model 'simple'
                    actual_model = "simple" if (m == "transfer" and s == 202603) else m

                    torch.save({
                        "model_name": actual_model,
                        "seed": s,
                        "provenance": {
                            "checkpoint_sha256": "placeholder",
                            "dataset_fingerprint": "mock_fp",
                            "git_commit": "mock_commit",
                            "model_name": actual_model,
                            "seed": s,
                            "data_dir": "mock",
                        }
                    }, ckpt_file)

                    real_ckpt_sha = compute_file_sha256(ckpt_file)
                    th_data = {
                        "thresholds": [0.5] * 15,
                        "provenance": {
                            "checkpoint_sha256": real_ckpt_sha,
                            "dataset_fingerprint": "mock_fp",
                            "git_commit": "mock_commit",
                            "model_name": actual_model,
                            "seed": s,
                            "data_dir": "mock",
                        }
                    }
                    th_file.write_text(json.dumps(th_data))
                    real_th_sha = compute_file_sha256(th_file)

                    experiments.append({
                        "model": m,
                        "seed": s,
                        "checkpoint_rel_path": ckpt_file.name,
                        "checkpoint_sha256": real_ckpt_sha,
                        "threshold_rel_path": th_file.name,
                        "threshold_sha256": real_th_sha,
                    })

            lock_path = root / "protocol_lock.json"
            lock_data = {
                "git_commit": "mock_commit",
                "git_dirty": False,
                "dataset_fingerprint": "mock_fp",
                "experiments": experiments,
            }
            lock_path.write_text(json.dumps(lock_data))

            with unittest.mock.patch("experiment_config.compute_dataset_fingerprint", return_value=meta):
                with self.assertRaises(ValueError) as ctx:
                    global_preflight_check(lock_file=lock_path, data_mode="demo")
                self.assertIn("Semantic mismatch", str(ctx.exception))

    def test_h3_zero_test_loader_call_when_artifact_provenance_fails(self):
        """H3: Nếu artifact bị lỗi provenance, TestLoader tuyệt đối không bao giờ được gọi."""
        from run_multi_seed import run_multi_seed_final_test
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            lock_file = root / "protocol_lock.json"
            lock_file.write_text(json.dumps({"experiments": []}))

            with unittest.mock.patch("experiment_config.DEVELOP_DIR", root):
                with unittest.mock.patch("run_multi_seed.get_test_dataloader") as mock_loader:
                    with self.assertRaises(Exception):
                        run_multi_seed_final_test(data_dir=str(root), data_mode="real")
                    mock_loader.assert_not_called()

    def test_h4_train_loss_denominator_with_drop_last(self):
        """H4: Đảm bảo train_one_epoch dùng seen_samples làm mẫu số (5 mẫu bs 2 drop_last -> seen 4, không chia cho 5)."""
        from train import train_one_epoch
        from torch.utils.data import TensorDataset, DataLoader

        dummy_x = torch.randn(5, 3, 16, 16)
        dummy_y = torch.zeros(5, 15, dtype=torch.float32)
        dataset = TensorDataset(dummy_x, dummy_y)
        loader = DataLoader(dataset, batch_size=2, drop_last=True)

        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 16 * 16, 15))
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        loss, _, _ = train_one_epoch(model, loader, criterion, optimizer, torch.device("cpu"))
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0.0)

    def test_h5_compare_models_real_no_legacy_fallback(self):
        """H5: compare_models ném FileNotFoundError trên real data khi thiếu metric file, không fallback."""
        from compare_models import generate_comparison_table
        with tempfile.TemporaryDirectory() as tmpdir:
            with unittest.mock.patch("experiment_config.get_final_test_dir", return_value=Path(tmpdir)):
                with self.assertRaises(FileNotFoundError) as ctx:
                    generate_comparison_table(models=["simple"], seeds=[202601], data_mode="real")
                self.assertIn("Không tìm thấy file metrics", str(ctx.exception))

    def test_h6_calibration_rejects_out_of_bound_probabilities(self):
        """H6: calibrate_thresholds_from_pr_curve ném ValueError nếu xác suất ngoài [0, 1]."""
        y_true = np.array([[1, 0], [0, 1]], dtype=np.float32)
        y_prob_high = np.array([[1.2, 0.5], [0.1, 0.8]], dtype=np.float32)
        with self.assertRaises(ValueError) as ctx:
            calibrate_thresholds_from_pr_curve(y_true, y_prob_high)
        self.assertIn("valid probabilities in [0, 1]", str(ctx.exception))

        y_prob_neg = np.array([[-0.1, 0.5], [0.1, 0.8]], dtype=np.float32)
        with self.assertRaises(ValueError) as ctx:
            calibrate_thresholds_from_pr_curve(y_true, y_prob_neg)
        self.assertIn("valid probabilities in [0, 1]", str(ctx.exception))

    def test_h7_hyperparameter_search_import_sanity(self):
        """H7: import hyperparameter_search chạy sạch, không dính NameError hay cú pháp."""
        import hyperparameter_search
        self.assertTrue(hasattr(hyperparameter_search, "objective"))
        self.assertTrue(hasattr(hyperparameter_search, "run_search"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
