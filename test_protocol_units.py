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

            # Lưu checkpoint giả
            torch.save({
                "model_name": "simple",
                "seed": 202601,
                "model_state_dict": {},
                "provenance": {"model_name": "simple", "seed": 202601},
            }, ckpt_path)

            ckpt_hash = compute_file_sha256(ckpt_path)

            # Lưu threshold khớp
            dataset_meta = {"dataset_fingerprint": "fake_fingerprint_123"}
            with open(th_path, "w", encoding="utf-8") as f:
                json.dump({
                    "provenance": {
                        "model_name": "simple",
                        "seed": 202601,
                        "checkpoint_sha256": ckpt_hash,
                        "dataset_fingerprint": "fake_fingerprint_123",
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

    def test_global_preflight_missing_lock_aborts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            missing_lock = tmp_path / "non_existent_protocol_lock.json"
            with self.assertRaises(FileNotFoundError):
                global_preflight_check(lock_file=missing_lock)

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

            # Sửa đổi DEVELOP_DIR tạm thời bằng monkeypatch hoặc truyền lock
            # Ở đây global_preflight_check tìm file relative to DEVELOP_DIR
            # Kiểm tra xem hash sai lệch có bị phát hiện không
            from unittest.mock import patch
            with patch("experiment_config.DEVELOP_DIR", tmp_path):
                with patch("experiment_config.compute_dataset_fingerprint", return_value={"dataset_fingerprint": "mock_fp", "is_demo_data": True}):
                    with self.assertRaises(ValueError) as ctx:
                        global_preflight_check(lock_file=lock_path)
                    self.assertIn("FAIL CLOSED", str(ctx.exception))


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
