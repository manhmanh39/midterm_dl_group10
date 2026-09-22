# BÁO CÁO NGHIỆM THU: GIAO THỨC KIỂM CHUẨN DETECTION P0 & P1
## VinBigData Chest X-ray Object Detection Protocol (SPEC LOCKED 100%)

*Dành cho Nhóm 10 - Môn Học Deep Learning*  
*Trạng thái: **SPEC LOCKED 100% - TOÀN VẸN 24 UNIT TESTS PASSED***  
*Branch: `feat/detection-p0-p1`*

---

## 1. TỔNG QUAN HỆ THỐNG

Hệ thống đã hoàn tất chuyển đổi từ bài toán Phân loại đa nhãn (Multi-label Classification 15 nhãn) sang bài toán **Phát hiện Bệnh học Đa thực thể (Object Detection 14 nhóm tổn thương có Bounding Box)** theo chuẩn YOLO 1-scale trên lưới $16 \times 16$:

| Hạng mục | Quy cách Kỹ thuật | Chi tiết Triển khai |
| :--- | :--- | :--- |
| **Model 1** | Sequential CNN Detector | 5 Conv Blocks tuần tự + $1 \times 1$ Head $\rightarrow (B, 16, 16, 19)$ |
| **Model 2** | Non-Sequential Residual Detector | Residual shortcut blocks + $1 \times 1$ Head $\rightarrow (B, 16, 16, 19)$ |
| **Model 3** | Pretrained ResNet50 Detector | Backbone ImageNet (unfreeze layer3, 4) + $1 \times 1$ Head $\rightarrow (B, 16, 16, 19)$ |
| **Đầu vào** | Ảnh 3 kênh kết hợp | `[raw_gray, CLAHE, Laplacian]` kích thước $3 \times 512 \times 512$ |
| **Đầu ra** | Tensor không gian $16 \times 16 \times 19$ | 1 Objectness + 4 Bounding Box $[t_x, t_y, w, h]$ + 14 Class Logits |
| **Loss** | CombinedLocalizationLoss | $\lambda_{coord} L_{coord} (\text{GIoU+MSE}) + \lambda_{obj} L_{obj} (\text{Focal}) + \lambda_{cls} L_{cls} (\text{CE})$ |
| **Metric** | mAP@0.5 | PASCAL VOC 2010+ continuous all-points interpolation |
| **Test Guard** | Split Semantics Lock | TestLoader bị khóa cứng, chỉ mở khi có `PreflightPermit` hợp lệ |

---

## 2. DANH MỤC 10 HẠNG MỤC BỌC THÉP GIAO THỨC (HARDENED INVARIANTS)

1. **Chống Re-lock ở Final-Test**: `run_multi_seed.py` ở phase `final-test` tuyệt đối không gọi `generate_protocol_lock()`. Chỉ đọc lock có sẵn và kiểm định preflight.
2. **Full-Lock Immutability & Sidecar SHA**: `protocol_lock.json` được bảo vệ bởi file sidecar `protocol_lock.json.sha256`. Mọi can thiệp hoặc sửa đổi trên file lock đều bị chặn đứng (Fail-Closed).
3. **Dynamic Permit Lock Binding**: `PreflightPermit` lưu `lock_sha256` đĩa; token HMAC băm trực tiếp `f"{lock_sha256}:{dataset_fingerprint}:{git_commit}"`. `permit.verify()` kiểm tra lại disk SHA của lock file.
4. **Exact Canonical 3x3 Deduplication**: Loại trừ triệt để danh sách trùng lặp (`sorted(list) == sorted(canonical)`). Cấm các danh sách giả mạo hoặc lặp phần tử.
5. **Cấm `phase=all` ở chế độ Real**: Chặn đứng việc chạy một mạch `phase="all"` ở chế độ real. Bắt buộc người dùng phải tuân thủ 3 bước rời rạc có khoảng dừng kiểm định: `develop` $\rightarrow$ `lock` $\rightarrow$ `final-test`.
6. **Semantic Model/Seed Provenance Binding**: Evaluator kiểm tra chéo toàn diện:
   - `ckpt["model_name"] == model_name == entry["model"]`
   - `ckpt["seed"] == entry["seed"]`
   - Checkpoint stem phải chứa cả model và seed.
   - Cross-check `provenance` (`dataset_fingerprint`, `git_commit`) với lock.
7. **Chặn Bypass Real/Demo**: Khóa cứng chế độ `real` nếu lock được sinh ở real mode; không cho phép dùng `--data_mode demo` để né tránh kiểm tra git clean.
8. **Tách biệt Runtime Artifacts khỏi Git Clean**: `.gitignore` và `is_git_clean()` chủ động loại trừ các artifacts sinh ra trong `outputs/` (`protocol_lock*`, `develop/`, `history_*`, `eval_*`), ngăn chặn preflight REAL tự fail vì artifact vừa sinh.
9. **Đồng nhất Prepare Dataset Path**: `prepare_dataset.py` sử dụng `default=get_processed_data_root()` (trỏ chuẩn về `data/dataset_202601`).
10. **GitHub Actions CI Tự Động**: Thiết lập file `.github/workflows/ci.yml` tự động kiểm thử toàn bộ 24 bài tests trên GitHub mỗi khi push hoặc mở Pull Request.

---

## 3. KẾT QUẢ KIỂM THỬ XÁC MINH (VERIFICATION RESULTS)

Chạy bộ kiểm thử tự động toàn diện:
```bash
python -m unittest tests/test_detection_protocol.py -v
```

```text
Ran 24 tests in ~6s - OK
- test_01_empty_vs_missing_label_contract: PASSED
- test_02_raw_box_format_strict_validation: PASSED
- test_03_image_and_label_bijection_enforcement: PASSED
- test_04_dataset_and_collate_returns_image_id: PASSED
- test_05_image_byte_level_tamper_detection: PASSED
- test_06_metrics_strict_image_id_lookup_no_fallback: PASSED
- test_07_cell_collision_audit: PASSED
- test_08_voc_continuous_all_points_ap: PASSED
- test_09_split_semantics_test_guard_fail_closed: PASSED
- test_10_relock_prevention_in_final_test: PASSED
- test_11_checkpoint_lock_membership_enforcement: PASSED
- test_12_evaluation_config_enforcement_from_lock: PASSED
- test_13_exact_canonical_3x3_enforced_in_real_mode: PASSED
- test_14_leakage_discipline_develop_split: PASSED
- test_15_checkpoint_provenance_and_sidecar: PASSED
- test_16_sample_std_ddof1: PASSED
- test_17_benchmark_device_and_batch_reporting: PASSED
- test_18_duplicate_canonical_models_seeds_rejected: PASSED
- test_19_forbid_phase_all_in_real_mode: PASSED
- test_20_lock_sidecar_and_immutability_tamper_detection: PASSED
- test_21_semantic_model_seed_provenance_binding: PASSED
- test_22_real_mode_downgrade_bypass_prevention: PASSED
- test_23_prepare_dataset_root_consistency: PASSED
- test_24_runtime_artifacts_do_not_dirty_git: PASSED
```

---

## 4. HƯỚNG DẪN THỰC THI (CLI COMMANDS)

### 4.1. Huấn luyện Giai đoạn Phát triển (Develop Phase)
```bash
# Huấn luyện cho 1 mô hình và 1 seed
python train.py --model model1 --seed 202601 --epochs 40 --data_mode real

# Hoặc huấn luyện toàn bộ 9 canonical runs (3 models x 3 seeds)
python run_multi_seed.py --model all --phase develop --epochs 40 --data_mode real
```

### 4.2. Khóa Protocol (Protocol Lock)
```bash
python run_multi_seed.py --model all --phase lock --data_mode real
```
*Lệnh này sinh `outputs/protocol_lock.json` và `outputs/protocol_lock.json.sha256` khóa 9 checkpoints SHA256, dataset fingerprint và git commit.*

### 4.3. Đánh giá Post-Freeze trên Tập Test (Final-Test)
```bash
python run_multi_seed.py --model all --phase final-test --data_mode real
```
*Duyệt TestLoader đúng 1 lần duy nhất cho mỗi model-seed, bảo vệ bởi PreflightPermit.*

### 4.4. Đo Đạc Benchmark & So Sánh
```bash
python compare_models.py
```
*Tự động đo RAM `state_dict_size_mib`, đo độ trễ suy luận, và xuất bảng so sánh tổng hợp ra `outputs/detection_comparison_table.md`.*
