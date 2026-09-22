# BÁO CÁO CHI TIẾT & TAKE-NOTE MÃ NGUỒN: GIAO THỨC THỰC NGHIỆM P0 & P1
Dự án: Phân loại Đa nhãn X-quang Lồng ngực VinBigData (Chest X-ray Multi-label Classification)  
Repository: `https://github.com/manhmanh39/midterm_dl_group10.git`

---

## TỔNG QUAN HỆ THỐNG GIAO THỨC

Hệ thống được thiết kế tuân thủ nghiêm ngặt chuẩn mực nghiên cứu Machine Learning/Deep Learning, ngăn ngừa triệt để hiện tượng rò rỉ dữ liệu (Data Leakage) và đảm bảo tính tái lập (Reproducibility) tuyệt đối qua hai giai đoạn:

1. **Giai đoạn P0 (Phải sửa trước khi chạy lại - Bug Fixes & Code Correctness)**:
   - Sửa đổi cấu trúc huấn luyện, nạp siêu tham số, phân tách khái niệm accuracy và lưu trữ lịch sử học tập.
2. **Giai đoạn P1 (Chạy experiment chuẩn - Rigorous Evaluation Protocol)**:
   - Hiệu chuẩn ngưỡng theo từng lớp trên tập Validation, bổ sung PR-AUC, tách riêng 14 bệnh lý, đánh giá đóng băng Test 1-pass, đa seed và sinh bảng so sánh mẫu.

---

## BẢNG TRA CỨU NHANH (MAPPING TABLE)

| STT | Mục Giao thức | Tệp Mã Nguồn | Hàm / Vị trí triển khai | Mô tả tóm tắt |
|:---|:---|:---|:---|:---|
| **P0.1** | Cho `train()` nhận optimizer, weight_decay, dropout | [`train.py`](file:///c:/Users/admin/Downloads/MidtermDL/train.py), [`models/`](file:///c:/Users/admin/Downloads/MidtermDL/models/) | `train()`, `_build_optimizer()`, các hàm `get_model*()` | Hỗ trợ cấu hình động optimizer ('adamw', 'sgd'), L2 regularization và tỷ lệ dropout qua tham số hàm và CLI. |
| **P0.2** | Apply toàn bộ `best_hparams` | [`experiment_config.py`](file:///c:/Users/admin/Downloads/MidtermDL/experiment_config.py) | `DEFAULT_TUNED_HPARAMS`, `load_best_hparams()`, `resolve_experiment_config()` | Áp dụng đầy đủ 5 siêu tham số tối ưu (lr, batch_size, optimizer, weight_decay, dropout) cho cả 3 mô hình từ kết quả search. |
| **P0.3** | Cho `main.py` support tuned config rõ ràng | [`main.py`](file:///c:/Users/admin/Downloads/MidtermDL/main.py) | `parser.add_argument("--use_tuned")`, `run_develop_phase()` | CLI switch `--use_tuned` nạp cấu hình tối ưu, in rõ ràng nguồn gốc tham số (CLI vs Tuned vs Default). |
| **P0.4** | Eval transfer với `pretrained=False` | [`eval.py`](file:///c:/Users/admin/Downloads/MidtermDL/eval.py), [`main.py`](file:///c:/Users/admin/Downloads/MidtermDL/main.py), [`run_multi_seed.py`](file:///c:/Users/admin/Downloads/MidtermDL/run_multi_seed.py) | `evaluate_frozen_model()`, `run_develop_phase()`, `run_multi_seed_develop()` | Khi nạp checkpoint đã đóng băng, tắt tải trọng số ImageNet `pretrained=False` để nạp 100% weights từ checkpoint. |
| **P0.5** | Sửa `exact_match_accuracy` | [`train.py`](file:///c:/Users/admin/Downloads/MidtermDL/train.py), [`eval.py`](file:///c:/Users/admin/Downloads/MidtermDL/eval.py), [`compare_models.py`](file:///c:/Users/admin/Downloads/MidtermDL/compare_models.py) | `_compute_batch_metrics()`, `evaluate_frozen_model()` | Phân định rạch ròi: Exact Match Accuracy (toàn bộ 15 nhãn đồng thời đúng 100%) vs Per-Label Accuracy (tỷ lệ nhãn đúng). |
| **P0.6** | Save history + metrics JSON/CSV | [`train.py`](file:///c:/Users/admin/Downloads/MidtermDL/train.py), [`eval.py`](file:///c:/Users/admin/Downloads/MidtermDL/eval.py) | `train()`, `save_final_test_artifacts()` | Lưu lịch sử từng epoch ra `history.json`/`history.csv` và test metrics ra `test_metrics_*.json`/`csv`. |
| **P1.7** | Tune threshold theo từng class trên validation | [`metrics_utils.py`](file:///c:/Users/admin/Downloads/MidtermDL/metrics_utils.py), [`experiment_config.py`](file:///c:/Users/admin/Downloads/MidtermDL/experiment_config.py) | `calibrate_thresholds_from_pr_curve()`, `save_calibrated_thresholds()` | Quét Precision-Recall curve trên tập Val để tìm ngưỡng $T^*$ tối ưu F1-score độc lập cho từng class, lưu vào `calibrated_thresholds.json`. |
| **P1.8** | Thêm PR-AUC | [`metrics_utils.py`](file:///c:/Users/admin/Downloads/MidtermDL/metrics_utils.py), [`compare_models.py`](file:///c:/Users/admin/Downloads/MidtermDL/compare_models.py) | `safe_average_precision()`, `compute_per_class_table()`, `compute_macro_metrics()` | Tính Average Precision (AP / PR-AUC) an toàn cho từng class và tổng hợp chỉ số Macro PR-AUC (`macro_ap_14`). |
| **P1.9** | Report macro metric của 14 pathologies riêng | [`metrics_utils.py`](file:///c:/Users/admin/Downloads/MidtermDL/metrics_utils.py), [`eval.py`](file:///c:/Users/admin/Downloads/MidtermDL/eval.py) | `compute_safe_macro_auc()`, `compute_macro_metrics()` | Báo cáo riêng biệt PRIMARY (14 bệnh lý thực thể, loại trừ 'No Finding' ở index 14) kèm số lượng valid classes. |
| **P1.10** | Chạy final test đúng một lần sau model selection | [`main.py`](file:///c:/Users/admin/Downloads/MidtermDL/main.py), [`experiment_config.py`](file:///c:/Users/admin/Downloads/MidtermDL/experiment_config.py) | `protocol_lock.json`, `global_preflight_check()`, guard `--phase all` | Tách rời 2 pha: Develop -> Khóa hash bằng `protocol_lock.json` -> Preflight Check -> Chạy Test duy nhất 1 lần. Cấm `--phase all` trên dữ liệu thật. |
| **P1.11** | Chạy 3–5 seeds | [`run_multi_seed.py`](file:///c:/Users/admin/Downloads/MidtermDL/run_multi_seed.py), [`config.py`](file:///c:/Users/admin/Downloads/MidtermDL/config.py) | `CANONICAL_SEEDS`, `run_multi_seed_develop()`, `run_multi_seed_final_test()` | Tự động hóa toàn bộ quy trình trên ma trận 3 mô hình x 3-5 seeds (mặc định 3 seeds: 202601, 202602, 202603). |
| **P1.12** | Tạo comparison table | [`compare_models.py`](file:///c:/Users/admin/Downloads/MidtermDL/compare_models.py) | `generate_comparison_table()`, `compute_sample_stats()` | Tổng hợp kết quả đa seed với Mean ± Sample Std (ddof=1), độ phức tạp tham số và latency/throughput ra Markdown, CSV, JSON. |

---

## CHI TIẾT KỸ THUẬT TỪNG MỤC

### 1. P0 — Sửa đổi kỹ thuật cốt lõi trước khi chạy lại

#### [P0.1] Cho `train()` nhận optimizer, weight_decay, dropout
- **Tệp**: [`train.py`](file:///c:/Users/admin/Downloads/MidtermDL/train.py) (dòng 40-48, 172-189), [`models/model1_simple.py`](file:///c:/Users/admin/Downloads/MidtermDL/models/model1_simple.py), [`models/model2_complex.py`](file:///c:/Users/admin/Downloads/MidtermDL/models/model2_complex.py), [`models/model3.py`](file:///c:/Users/admin/Downloads/MidtermDL/models/model3.py).
- **Cách hoạt động**:
  - `train()` nhận các tham số `optimizer: str = "adamw"`, `weight_decay: float = config.WEIGHT_DECAY`, `dropout: Optional[float] = None`.
  - Hàm `_build_optimizer` khởi tạo `torch.optim.AdamW` hoặc `torch.optim.SGD(momentum=0.9)` với `lr` và `weight_decay` được truyền vào.
  - Tham số `dropout` được chuyển tiếp vào hàm khởi tạo mô hình (`get_model1_simple`, `get_model2_complex`, `get_model3_transfer`).

#### [P0.2] Apply toàn bộ `best_hparams`
- **Tệp**: [`experiment_config.py`](file:///c:/Users/admin/Downloads/MidtermDL/experiment_config.py) (dòng 488-540), [`outputs/best_hparams_*.json`](file:///c:/Users/admin/Downloads/MidtermDL/outputs/).
- **Cách hoạt động**:
  - Tích hợp từ điển `DEFAULT_TUNED_HPARAMS` lưu giữ giá trị tối ưu của cả 3 mô hình từ thực nghiệm tìm kiếm:
    - **Simple CNN**: `lr=0.001406`, `batch_size=16`, `weight_decay=2.56e-4`, `optimizer='sgd'`, `dropout=0.4188`.
    - **Complex CNN**: `lr=0.003391`, `batch_size=16`, `weight_decay=1.64e-4`, `optimizer='sgd'`, `dropout=0.3117`.
    - **Transfer Learning (ResNet-50)**: `lr=0.000660`, `batch_size=64`, `weight_decay=1.60e-5`, `optimizer='adamw'`, `dropout=0.4031`.
  - Hàm `load_best_hparams()` ưu tiên đọc file JSON và fallback về từ điển tĩnh.
  - Hàm `resolve_experiment_config()` tự động nạp toàn bộ 5 siêu tham số khi cờ `use_tuned=True`.

#### [P0.3] Cho `main.py` support tuned config rõ ràng
- **Tệp**: [`main.py`](file:///c:/Users/admin/Downloads/MidtermDL/main.py) (dòng 69-74, 243-247).
- **Cách hoạt động**:
  - Thêm cờ CLI `--use_tuned` trong `argparse`.
  - Tự động gọi `resolve_experiment_config(model_name, cli_args=args, use_tuned=args.use_tuned)`.
  - In ra console nguồn gốc từng tham số (`cli`, `tuned`, hoặc `default`) và lưu metadata vào trường `parameter_sources` của checkpoint và file provenance.

#### [P0.4] Eval transfer với `pretrained=False`
- **Tệp**: [`eval.py`](file:///c:/Users/admin/Downloads/MidtermDL/eval.py) (dòng 81), [`main.py`](file:///c:/Users/admin/Downloads/MidtermDL/main.py) (dòng 105), [`run_multi_seed.py`](file:///c:/Users/admin/Downloads/MidtermDL/run_multi_seed.py) (dòng 132).
- **Cách hoạt động**:
  - Khi đánh giá hoặc calibrate từ checkpoint đã huấn luyện, mô hình được khởi tạo với `pretrained=False` và `freeze_base=False`.
  - Tránh tải lại trọng số ban đầu của ImageNet qua mạng internet, loại bỏ nguy cơ ghi đè hoặc xung đột với `checkpoint["model_state_dict"]`.

#### [P0.5] Sửa `exact_match_accuracy`
- **Tệp**: [`train.py`](file:///c:/Users/admin/Downloads/MidtermDL/train.py) (dòng 56-74), [`eval.py`](file:///c:/Users/admin/Downloads/MidtermDL/eval.py) (dòng 123-126), [`compare_models.py`](file:///c:/Users/admin/Downloads/MidtermDL/compare_models.py) (dòng 110-114).
- **Cách hoạt động**:
  - **Per-Label Accuracy**: Đo đạc tính chính xác theo từng phần tử nhãn nhị phân: $\frac{\sum (preds == labels)}{N \times C}$.
  - **Exact Match Accuracy**: Bắt buộc toàn bộ 15 nhãn của một bệnh nhân phải khớp 100%: `(preds == labels).all(dim=1).sum() / N`.
  - Giúp phát hiện hiện tượng mô hình dự đoán đúng nhiều nhãn âm tính nhưng không dự đoán chính xác toàn diện một ca bệnh đa nhãn.

#### [P0.6] Save history + metrics JSON/CSV
- **Tệp**: [`train.py`](file:///c:/Users/admin/Downloads/MidtermDL/train.py) (dòng 294-305, 391-414), [`eval.py`](file:///c:/Users/admin/Downloads/MidtermDL/eval.py) (dòng 170-175).
- **Cách hoạt động**:
  - Trong quá trình train, ghi nhận từng epoch: `train_loss`, `train_per_label_acc`, `train_exact_match_acc`, `val_loss`, `val_per_label_acc`, `val_exact_match_acc`, `val_auc`, `learning_rate`, `duration_sec`.
  - Kết thúc huấn luyện: xuất `history.json` và `history.csv` vào thư mục lưu trữ seed và đồng bộ vào `outputs/history_{model}_seed{seed}.csv`.
  - Trong test: xuất `test_metrics_{model}_seed{seed}.json`/`csv` và `test_per_class_{model}_seed{seed}.csv`.

---

### 2. P1 — Chạy Experiment Chuẩn (Rigorous Evaluation Protocol)

#### [P1.7] Tune threshold theo từng class trên validation
- **Tệp**: [`metrics_utils.py`](file:///c:/Users/admin/Downloads/MidtermDL/metrics_utils.py) (dòng 123-238), [`experiment_config.py`](file:///c:/Users/admin/Downloads/MidtermDL/experiment_config.py) (dòng 602-618).
- **Cách hoạt động**:
  - Hàm `calibrate_thresholds_from_pr_curve` tính toán Precision-Recall curve cho từng lớp trên tập Validation.
  - Tìm ngưỡng $T^* \in (0, 1)$ làm cực đại hóa F1-score: $F1 = \frac{2 \cdot P \cdot R}{P + R}$.
  - Áp dụng cơ chế tie-breaking tiền định (chọn ngưỡng gần 0.5 nhất khi F1 bằng nhau) và lưu vào `calibrated_thresholds.json`.
  - Tại pha Test, áp dụng song song cả ngưỡng cố định (Fixed-0.5) và ngưỡng hiệu chuẩn (Calibrated $T^*$) để so sánh khách quan.

#### [P1.8] Thêm PR-AUC
- **Tệp**: [`metrics_utils.py`](file:///c:/Users/admin/Downloads/MidtermDL/metrics_utils.py) (dòng 15-26, 328, 389, 411), [`compare_models.py`](file:///c:/Users/admin/Downloads/MidtermDL/compare_models.py).
- **Cách hoạt động**:
  - Triển khai `safe_average_precision()` với sklearn `average_precision_score`.
  - Trả về NaN an toàn nếu lớp không có mẫu dương hoặc âm trên tập dữ liệu.
  - Báo cáo chỉ số Average Precision (AP) cho từng bệnh lý và tính trung bình `macro_ap_14` trên các lớp hợp lệ. Đây là chỉ số then chốt với bài toán dữ liệu y tế mất cân bằng nhãn nghiêm trọng.

#### [P1.9] Report macro metric của 14 pathologies riêng
- **Tệp**: [`metrics_utils.py`](file:///c:/Users/admin/Downloads/MidtermDL/metrics_utils.py) (dòng 41-80, 372-430).
- **Cách hoạt động**:
  - Tách riêng biệt:
    - **PRIMARY**: 14 bệnh lý thực thể (indices $0 \dots 13$: Aortic enlargement, Cardiomegaly, Pulmonary fibrosis, Pneumothorax, etc.).
    - **SECONDARY**: Toàn bộ 15 lớp (bao gồm nhãn số 14 'No finding').
  - Báo cáo kèm số lượng lớp hợp lệ (`valid_classes_auc_14`, `valid_classes_ap_14`, `valid_classes_f1_14`).
  - Đảm bảo điểm số không bị kéo lệch bởi nhãn 'No finding' vốn chiếm tỷ lệ mẫu âm/dương hoàn toàn khác biệt với các tổn thương bệnh lý.

#### [P1.10] Chạy final test đúng một lần sau model selection
- **Tệp**: [`main.py`](file:///c:/Users/admin/Downloads/MidtermDL/main.py) (dòng 4-7, 155-178, 255-263), [`experiment_config.py`](file:///c:/Users/admin/Downloads/MidtermDL/experiment_config.py) (dòng 179-265, 267-392).
- **Cách hoạt động**:
  - Chia tách tuyệt đối 2 pha chạy độc lập:
    ```bash
    # Bước 1: Huấn luyện, hiệu chuẩn ngưỡng, benchmark (KHÔNG mở Test Set)
    python main.py --phase develop --data_mode real --model all
    
    # Bước 2: Khóa toàn vẹn và đánh giá Test 1 lần duy nhất
    python main.py --phase final-test --data_mode real --model all
    ```
  - **Khóa bảo vệ chống rò rỉ**: Cấm tuyệt đối `--phase all` trên dữ liệu thật (`data_mode == "real"`).
  - **Protocol Lock**: Khóa mã SHA256 của toàn bộ checkpoints và file thresholds vào `protocol_lock.json`.
  - **Global Preflight Check**: Kiểm tra tính toàn vẹn 100% của Git working tree (clean), git commit hash, và checksums của artifacts trước khi Test DataLoader được phép khởi tạo trong bộ nhớ.

#### [P1.11] Chạy 3–5 seeds
- **Tệp**: [`run_multi_seed.py`](file:///c:/Users/admin/Downloads/MidtermDL/run_multi_seed.py), [`config.py`](file:///c:/Users/admin/Downloads/MidtermDL/config.py).
- **Cách hoạt động**:
  - Cung cấp bộ điều phối toàn diện cho ma trận thực nghiệm: `models x seeds`.
  - Mặc định sử dụng danh sách 3 canonical seeds: `[202601, 202602, 202603]`.
  - Huấn luyện và lưu trữ độc lập từng không gian thực nghiệm `outputs/develop/{model}/seed{seed}/`.
  - Tự động kiểm tra set equality đảm bảo chạy đủ 9 cặp canonical (3 models x 3 seeds) trên dữ liệu thật.

#### [P1.12] Tạo comparison table
- **Tệp**: [`compare_models.py`](file:///c:/Users/admin/Downloads/MidtermDL/compare_models.py).
- **Cách hoạt động**:
  - Tự động đọc tất cả artifacts đánh giá test từ `outputs/final_test/` và benchmark từ `outputs/benchmarks/`.
  - Tính toán thống kê mẫu khách quan: Mean ± Sample Standard Deviation với bậc tự do $N-1$ (`ddof=1`):
    $$s = \sqrt{\frac{1}{N-1} \sum_{i=1}^N (x_i - \bar{x})^2}$$
  - Phân tách rõ ràng:
    1. Hiệu năng mô hình (Macro ROC-AUC, Macro PR-AUC, Macro F1 Calibrated, Exact Match Acc).
    2. Chi phí kiến trúc (Parameters (M), Model Size (MB)).
    3. Tốc độ suy luận thực tế (BS=1 Latency (ms), BS=16 Throughput (FPS)).
  - Xuất bảng kết quả đa định dạng: `protocol_comparison_table.md`, `protocol_comparison_table.csv`, `protocol_comparison_table.json`.

---

## TỔNG KẾT
Toàn bộ 12 hạng mục (6 mục P0 và 6 mục P1) đã được kiểm chứng tính nhất quán, đánh dấu rõ ràng trong mã nguồn bằng comment chuẩn hóa `[P0 - Item X: ...]` và `[P1 - Item Y: ...]`, sẵn sàng cho đánh giá và nghiệm thu.
