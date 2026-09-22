# TÀI LIỆU HƯỚNG DẪN & GHI CHÚ GIAO THỨC KIỂM CHUẨN P0 & P1
## PHÁT HIỆN BỆNH HỌC ĐA THỰC THỂ TRÊN X-QUANG LỒNG NGỰC (VINBIGDATA OBJECT DETECTION)

*Dành cho Nhóm 10 - Môn Học Deep Learning*  
*Trạng thái: **SPEC LOCKED 100% (Đã kiểm chuẩn toàn diện qua Unit Tests)***

---

## MỤC LỤC
1. [Tổng quan Chuyển đổi từ Classification sang Detection](#1-tổng-quan-chuyển-đổi)
2. [Chi tiết Kỹ thuật P0 (Code Correctness)](#2-chi-tiết-kỹ-thuật-p0)
3. [Chi tiết Kỹ thuật P1 (Protocol Discipline & Post-Freeze Evaluation)](#3-chi-tiết-kỹ-thuật-p1)
4. [Bảng Phân định Ngưỡng P1 vs P2 (Bi-Phase Threshold Policy)](#4-bảng-phân-định-ngưỡng-p1-vs-p2)
5. [Cấu trúc Mã nguồn Đã Triển khai](#5-cấu-trúc-mã-nguồn-đã-triển-khai)
6. [Hướng dẫn Chạy Lệnh Thực nghiệm (CLI Guide)](#6-hướng-dẫn-chạy-lệnh-thực-nghiệm)

---

## 1. TỔNG QUAN CHUYỂN ĐỔI

Hệ thống đã được chuyển đổi hoàn chỉnh từ bài toán Phân loại đa nhãn (Multi-label Classification 15 nhãn) sang bài toán **Phát hiện Bệnh học Đa thực thể (Object Detection 14 nhóm tổn thương có Bounding Box)** theo chuẩn YOLO 1-scale trên lưới $16 \times 16$:

| Tiêu chí | Phân loại (Classification cũ) | Phát hiện (Object Detection mới) |
| :--- | :--- | :--- |
| **Kiến trúc Model 1** | Simple CNN Classifier | `SequentialCNNDetector` (CNN tuần tự 5 blocks + Head $1\times 1$) |
| **Kiến trúc Model 2** | Complex Multi-Branch Classifier | `NonSequentialCNNDetector` (Residual blocks + Head $1\times 1$) |
| **Kiến trúc Model 3** | ResNet50 Classifier | `PretrainedDetector` (ResNet50 Backbone ImageNet + Head $1\times 1$) |
| **Đầu ra Mạng** | Vector xác suất $(B, 15)$ | Tensor không gian $(B, 16, 16, 19)$ |
| **Hàm Mất mát** | Asymmetric Loss / BCE Loss | `CombinedLocalizationLoss` (Focal Obj + GIoU/MSE BBox + Softmax CE Class) |
| **Metric Chính** | Macro-AUC 14 bệnh học | **mAP@0.5** (PASCAL VOC 2010+ continuous all-points) |
| **No-Finding Invariant** | Ràng buộc logic $y_{14} + \max(y_{0..13}) = 1$ | **0 bounding box** (file nhãn `.txt` có kích thước 0 byte) |
| **Kỷ luật Test Split** | Khóa Develop/Test Loader | **Split Semantics Guard** Fail-Closed với `protocol_lock_token` |
| **Kích thước Mô hình** | File `.pth` trên đĩa | **`state_dict_size_mib`** (Bộ nhớ RAM thực của tham số và buffers) |
| **Độ lệch chuẩn** | Mẫu số $N$ | **Sample Std với $ddof=1$** (mẫu số $N-1$ cho $N=3$ seeds) |

---

## 2. CHI TIẾT KỸ THUẬT P0 (CODE CORRECTNESS)

### P0.1 - Ba Kiến trúc Detector Đồng nhất Đầu ra
- Cả 3 mô hình (`model1_sequential`, `model2_residual`, `model3_pretrained`) nhận đầu vào ảnh $3 \times 512 \times 512$ và xuất ra tensor $(B, 16, 16, 19)$ với stride $S=32$.
- Cấu trúc 19 kênh:
  - Kênh 0: Logit xác suất có vật thể (Objectness).
  - Kênh 1-2: Tọa độ tâm tương đối trong ô lưới $(t_x, t_y) \in (0, 1)$.
  - Kênh 3-4: Chiều rộng và chiều cao chuẩn hóa $(w, h) \in (0, 1)$.
  - Kênh 5-18: Logits phân loại cho 14 lớp bệnh học.
- Đã bổ sung hàm `trainable_parameter_summary()` trả về `(trainable_params, total_params)`.

### P0.2 - Target Encoding & Audit Cell Collision
- Mã hóa bounding box từ nhãn gốc thành grid target $(16, 16, 19)$.
- **Collision Audit (`audit_cell_collisions`)**: Do mô hình sử dụng 1-scale grid, nếu một ảnh có 2 bounding box có tâm rơi vào cùng một ô lưới $(g_y, g_x)$, ô đó sẽ ưu tiên giữ box có diện tích lớn hơn và box nhỏ hơn bị ghi đè. Module audit tự động thống kê:
  - Tổng số box, số box bị va chạm và tỷ lệ mất mát (`collision_rate_boxes`).
  - Số ảnh bị ảnh hưởng va chạm (`collision_rate_images`).
  - Số box tối đa trong 1 cell (`max_boxes_in_cell`).

### P0.3 - Loss Function Chuẩn hóa Đúng
- `CombinedLocalizationLoss = \lambda_{coord} L_{coord} + \lambda_{obj} L_{obj} + \lambda_{class} L_{class}`:
  - $L_{obj}$: Focal Loss giảm trọng số các cell nền âm tính dễ (với $\gamma=2.0, \alpha=0.25$).
  - $L_{coord}$: GIoU Loss kết hợp MSE Loss trên tọa độ sigmoid.
  - $L_{class}$: Softmax Cross-Entropy trên các cell có vật thể dương tính.
  - Chuẩn hóa nghiêm ngặt bằng `num_pos = obj_mask.sum().clamp(min=1).float()`, triệt tiêu lỗi chia cho 0 trên các ảnh No-Finding.

### P0.4 - Transfer Learning Reload Safety
- `model3_pretrained.py` và `factory.py` hỗ trợ cờ `pretrained: bool`:
  - Khi train ban đầu: `pretrained=True` để nạp weights ImageNet.
  - Khi reload checkpoint để eval / test: `pretrained=False` (`weights=None`), đảm bảo chỉ nạp trọng số đã huấn luyện từ checkpoint, không bao giờ tải lại ImageNet.

### P0.5 - Khớp Predictions $\leftrightarrow$ Ground Truth theo `image_id` & VOC All-Points AP
- **Loại bỏ hoàn toàn index tuần tự**: Dataset trả về `(img, target, image_id)`. Bộ đánh giá lưu trữ dưới dạng từ điển `{image_id: detections}` và `{image_id: ground_truth}`, đảm bảo tính đúng đắn 100% ngay cả khi DataLoader xáo trộn hoặc chạy đa luồng.
- **Metric Chuẩn**: Sử dụng thuật toán **PASCAL VOC 2010+ continuous all-points interpolation**:
  $$p_{\text{interp}}(r) = \max_{r' \ge r} p(r')$$
  Tích phân diện tích dưới đường bao cong Precision-Recall toàn bộ các điểm thay đổi, không dùng COCO 101-point hay 11-point xấp xỉ thô.

### P0.6 - Lưu Lịch sử Huấn luyện Đầy đủ
- `train.py` thiết lập mặc định `eval_every=1` trong giai đoạn canonical develop, đánh giá mAP@0.5 mỗi epoch.
- Tự động xuất lịch sử ra cả 2 định dạng: `outputs/history_{model}_seed{seed}.json` và `outputs/history_{model}_seed{seed}.csv`.
- Lưu `best_model.pth` dựa trên **val mAP@0.5 cao nhất** (thay vì val loss).

### P0.7 - Checkpoint Provenance & Tránh Nghịch lý SHA256
- Checkpoint lưu đầy đủ siêu dữ liệu: `epoch`, `model_name`, `seed`, `git_commit`, `image_size`, `grid_size`, `freeze_backbone`, `unfreeze_from_layer`, `hparams`.
- **Tuyệt đối không lưu SHA bên trong file `.pth`** (tránh nghịch lý đệ quy). SHA256 được tính từ file nhị phân trên đĩa và ghi ra file sidecar `{checkpoint}.pth.sha256` song hành.

---

## 3. CHI TIẾT KỸ THUẬT P1 (PROTOCOL DISCIPLINE)

### P1.1 & P1.3 - Hợp đồng Dữ liệu & Dataset Fingerprint
- **Empty vs Missing Label Contract**:
  - Nếu file nhãn `labels/{image_id}.txt` **không tồn tại**: Ném ngay ngoại lệ `FileNotFoundError` (Fail-Closed, không được ngầm coi là rỗng).
  - Nếu file `labels/{image_id}.txt` **tồn tại nhưng có kích thước 0 byte**: Xác nhận đây là **No-Finding hợp lệ** (ảnh không có tổn thương, 0 bounding box).
- **Canonical Dataset Fingerprint (`compute_dataset_fingerprint`)**:
  Băm mật mã danh sách sorted filenames, kích thước và nội dung nhãn của cả 3 tập `train`, `val`, `test` kết hợp danh sách tên 14 lớp bệnh học thành mã SHA-256 duy nhất. Bất kỳ sự thay đổi nào về file hay nội dung đều làm lệch fingerprint.

### P1.2 - Split Semantics Test Guard (Khóa Truy cập Tập Test)
- `get_develop_dataloaders()`: Chỉ trả về `train_loader` và `val_loader`. Không chứa bất kỳ logic nào tạo `test_loader`.
- `get_test_dataloader()`: Được bọc bởi `assert_test_access_allowed(lock_token)`. Nếu không có `protocol_lock_token` hợp lệ cấp từ `global_preflight_check()`, hàm lập tức ném ngoại lệ `RuntimeError("TEST SET ACCESS DENIED!")`.
- Ngăn chặn hoàn toàn rò rỉ tập Test trong quá trình phát triển (Develop Phase).

### P1.4 - Canonical 3 Models $\times$ 3 Seeds
- Bộ 3 mô hình: `["model1", "model2", "model3"]`.
- Bộ 3 seeds cố định: `[202601, 202602, 202603]`.
- Tổng cộng 9 thực nghiệm được huấn luyện độc lập và kiểm định.

### P1.6 - Chuẩn Benchmark Tài nguyên & Độ trễ
- Đo bộ nhớ RAM thông qua `state_dict_size_mib` = tổng số byte của parameters + buffers trong RAM, phản ánh chính xác kích thước mô hình trong bộ nhớ runtime thay vì kích thước file nén `.pth`.
- Đo độ trễ suy luận BS=1 (`bs1_latency_median_ms`, `mean`, `p95`) và thông lượng BS=16 (`bs16_throughput_fps`) có kích hoạt `torch.cuda.synchronize()`.

### P1.7 - Protocol Lock & Global Preflight Check
- `generate_protocol_lock()`: Sinh file `outputs/protocol_lock.json` khóa:
  - SHA256 của toàn bộ 9 checkpoint tốt nhất.
  - Dataset Fingerprint.
  - Git Commit SHA.
  - Tham số đánh giá (`conf_threshold=0.25`, `nms_iou=0.45`, `min_score=0.01`, `grid_size=16`, `image_size=512`).
- `global_preflight_check()`: Quét toàn diện đĩa cứng để xác minh:
  - Cả 9 checkpoint và 9 sidecar `.sha256` còn nguyên vẹn 100%.
  - Dataset không bị thay đổi so với lúc khóa.
  - Nếu hợp lệ, cấp `protocol_lock_token` có chữ ký HMAC.

### Post-Freeze Single-Pass Test Evaluation
- Mỗi cặp (model, seed) chỉ được duyệt qua `TestLoader` **đúng 1 lần duy nhất (single-pass traversal)**.
- Trong 1 vòng lặp duy nhất đó, hệ thống đồng thời tính Test Loss, IoU và giải mã Bounding Box để tính mAP@0.5.

### Thống kê Mẫu $ddof=1$
- Khi tính độ lệch chuẩn đa seed ($N=3$), bắt buộc sử dụng chuẩn mẫu:
  $$s = \sqrt{\frac{1}{N-1} \sum_{i=1}^N (x_i - \bar{x})^2}$$
  tương ứng tham số `np.std(values, ddof=1)`.

---

## 4. BẢNG PHÂN ĐỊNH NGƯỠNG P1 VS P2

| Giai đoạn | Ngưỡng | Giá trị | Mục đích & Phạm vi |
| :--- | :--- | :---: | :--- |
| **P1 Baseline** | `eval_min_score` | `0.01` | Dùng để vẽ toàn vẹn đường cong Precision-Recall tính VOC all-points mAP@0.5. |
| **P1 Baseline** | `conf_threshold` | `0.25` | Khóa cứng làm điểm hoạt động cơ sở (Baseline Operating Point) cho Precision, Recall, F1. |
| **P1 Baseline** | `nms_iou_threshold` | `0.45` | Khóa cứng ngưỡng khử trùng lặp bounding box sau NMS. |
| **P2 Tuning** | `conf_threshold` | Được tune trên Val | Được phép tìm kiếm trên tập Validation để tối ưu hóa Macro F1. |
| **P2 Tuning** | *Quy tắc khóa* | Đóng băng trước Test | Ngưỡng sau khi tune trên Val phải được cập nhật vào artifact cấu hình và ghi vào `protocol_lock.json` TRƯỚC KHI mở TestLoader. |

---

## 5. CẤU TRÚC MÃ NGUỒN ĐÃ TRIỂN KHAI

```text
MidtermDL/
├── scripts/
│   ├── config.py                      # Hằng số hệ thống (14 classes, stride 32, grid 16, img 512)
│   ├── experiment_config.py           # Provenance, Dataset Fingerprint, Protocol Lock, Preflight Check
│   ├── benchmark_utils.py             # RAM state_dict_size_mib, BS=1 Latency, BS=16 FPS
│   ├── data/
│   │   ├── prepared_loader.py         # Split Guards, Develop/Test Loaders, Cell Collision Audit
│   │   └── prepare_dataset.py         # Tiền xử lý ảnh (CLAHE, Laplacian) & chia tập dữ liệu
│   ├── models/
│   │   ├── factory.py                 # Factory xây dựng 3 kiến trúc + pretrained flag
│   │   ├── model1_sequential.py       # SequentialCNNDetector + param summary
│   │   ├── model2_residual.py         # NonSequentialCNNDetector + param summary
│   │   └── model3_pretrained.py       # PretrainedDetector (ResNet50 unfreeze layer3)
│   └── src/
│       ├── dataset.py                 # Empty vs Missing label contract, image_id alignment
│       ├── decode.py                  # Decode logits sang BBox + Batched NMS
│       ├── metrics.py                 # VOC continuous all-points AP, image_id dict matching
│       └── utils.py                   # CombinedLocalizationLoss (num_pos safe), calculate_iou
├── train.py                           # Develop phase pipeline, history JSON/CSV, sidecar SHA
├── evaluate.py                        # Single-Pass Test evaluation, bảo vệ bởi lock_token
├── main.py                            # Phân định rõ --phase develop và --phase final-test
├── run_multi_seed.py                  # Điều phối Canonical 3x3 (develop -> lock -> final-test)
├── compare_models.py                  # Tổng hợp so sánh: Benchmark + Test Metrics (ddof=1)
├── tests/
│   └── test_detection_protocol.py     # Bộ 9 Unit Tests kiểm tra toàn diện 100% các guards
└── PROTOCOL_TAKENOTES.md              # Tài liệu này
```

---

## 6. HƯỚNG DẪN CHẠY LỆNH THỰC NGHIỆM (CLI GUIDE)

### 6.1. Chạy Bộ Kiểm thử Unit Tests
```bash
python -m unittest tests/test_detection_protocol.py
```
*Kết quả kỳ vọng: 9 tests passed 100% OK.*

### 6.2. Kiểm tra Dữ liệu & Audit Va chạm Cell
```bash
python -m scripts.data.prepared_loader --root data/dataset_202601 --audit_collision
```

### 6.3. Chạy Huấn luyện Phát triển (Develop Phase - Không đụng Test)
Chạy thử nghiệm cho 1 mô hình:
```bash
python train.py --model model1 --seed 202601 --epochs 40 --eval_every 1
```
Hoặc qua `main.py`:
```bash
python main.py --model model1 --phase develop --seed 202601 --epochs 40
```

### 6.4. Chạy Toàn bộ Canonical 3 Models $\times$ 3 Seeds
Chạy giai đoạn Develop cho toàn bộ 9 lượt:
```bash
python run_multi_seed.py --model all --phase develop --epochs 40
```

Sau khi train xong, tiến hành khóa Protocol và chạy Preflight Check:
```bash
python run_multi_seed.py --model all --phase lock
```

Tiến hành đánh giá Post-Freeze trên tập Test (duyệt TestLoader 1 lần duy nhất):
```bash
python run_multi_seed.py --model all --phase final-test
```

Hoặc chạy toàn bộ tuần tự tự động:
```bash
python run_multi_seed.py --model all --phase all --epochs 40
```

### 6.5. Đo Đạc Benchmark & Xuất Bảng So sánh Tổng thể
```bash
python compare_models.py
```
*Lệnh này sẽ tự động đo RAM `state_dict_size_mib`, đo độ trễ suy luận, đọc kết quả Test của 3 mô hình và xuất bảng so sánh ra `outputs/detection_comparison_table.md`, `detection_comparison_table.csv`, và `detection_comparison_table.json`.*
