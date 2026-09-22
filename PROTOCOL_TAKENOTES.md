# TÀI LIỆU HƯỚNG DẪN & GHI CHÚ GIAO THỨC KIỂM CHUẨN P0 & P1
## PHÁT HIỆN BỆNH HỌC ĐA THỰC THỂ TRÊN X-QUANG LỒNG NGỰC (VINBIGDATA OBJECT DETECTION)

*Dành cho Nhóm 10 - Môn Học Deep Learning*  
*Trạng thái: **SPEC LOCKED 100% (Đã kiểm chuẩn toàn diện 18 invariants qua 17 Unit Tests)***  
*Branch làm việc: `feat/detection-p0-p1` (Tuyệt đối không push trực tiếp vào `main`)*

---

## MỤC LỤC
1. [Tổng quan Chuyển đổi từ Classification sang Detection](#1-tổng-quan-chuyển-đổi)
2. [Chi tiết Kỹ thuật P0 (Code Correctness)](#2-chi-tiết-kỹ-thuật-p0)
3. [Chi tiết Kỹ thuật P1 (Protocol Discipline & Post-Freeze Evaluation)](#3-chi-tiết-kỹ-thuật-p1)
4. [Chi tiết 18 Hạng mục Kiểm chuẩn & Bọc Thép Giao thức (18 Hardened Invariants)](#4-chi-tiết-18-hạng-mục-kiểm-chuẩn)
5. [Bảng Phân định Ngưỡng P1 vs P2 (Bi-Phase Threshold Policy)](#5-bảng-phân-định-ngưỡng-p1-vs-p2)
6. [Cấu trúc Mã nguồn Đã Triển khai](#6-cấu-trúc-mã-nguồn-đã-triển-khai)
7. [Hướng dẫn Chạy Lệnh Thực nghiệm (CLI Guide)](#7-hướng-dẫn-chạy-lệnh-thực-nghiệm)

---

## 1. TỔNG QUAN CHUYỂN ĐỔI

Hệ thống đã được chuyển đổi hoàn chỉnh từ bài toán Phân loại đa nhãn (Multi-label Classification 15 nhãn) sang bài toán **Phát hiện Bệnh học Đa thực thể (Object Detection 14 nhóm tổn thương có Bounding Box)** theo chuẩn YOLO 1-scale trên lưới $16 \times 16$:

| Tiêu chí | Phân loại (Classification cũ) | Phát hiện (Object Detection mới) |
| :--- | :--- | :--- |
| **Kiến trúc Model 1** | Simple CNN Classifier | `SequentialCNNDetector` (CNN tuần tự 5 blocks + Head $1\times 1$) |
| **Kiến trúc Model 2** | Complex Multi-Branch Classifier | `NonSequentialCNNDetector` (Residual blocks + Head $1\times 1$) |
| **Kiến trúc Model 3** | ResNet50 Classifier | `PretrainedDetector` (ResNet50 Backbone ImageNet + Head $1\times 1$) |
| **Đầu vào Ảnh** | RGB Chuẩn hóa thô | 3 kênh kết hợp: `[raw_gray, CLAHE, Laplacian]` |
| **Đầu ra Mạng** | Vector xác suất $(B, 15)$ | Tensor không gian $(B, 16, 16, 19)$ |
| **Hàm Mất mát** | Asymmetric Loss / BCE Loss | `CombinedLocalizationLoss` (Focal Obj + GIoU/MSE BBox + Softmax CE Class) |
| **Metric Chính** | Macro-AUC 14 bệnh học | **mAP@0.5** (PASCAL VOC 2010+ continuous all-points) |
| **No-Finding Invariant** | Ràng buộc logic $y_{14} + \max(y_{0..13}) = 1$ | **0 bounding box** (file nhãn `.txt` có kích thước 0 byte) |
| **Kỷ luật Test Split** | Khóa Develop/Test Loader | **Split Semantics Guard** Fail-Closed với `PreflightPermit` |
| **Bộ nhớ Mô hình** | File `.pth` trên đĩa | **`state_dict_size_mib`** (Bộ nhớ RAM thực của tham số và buffers) |
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
- Kết quả kiểm toán trên tập Train thực tế (13,222 ảnh): Tỷ lệ va chạm box là **21.02%** (3,679 box trên 1,681 ảnh), max box trong 1 cell là **7**.

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
- **Loại bỏ hoàn toàn index tuần tự (`idx`)**: Dataset trả về `(img, target, image_id)`. Bộ đánh giá và hàm metric trích xuất `ds.get_raw_boxes_by_id(image_id)` theo `image_id` thực tế, đảm bảo tính đúng đắn 100% ngay cả khi DataLoader xáo trộn hoặc chạy đa luồng.
- **Fail-Closed**: Mọi trường hợp thiếu `image_id` hoặc không tìm thấy nhãn lập tức ném lỗi `KeyError` / `ValueError`, không bao giờ fallback về số thứ tự lặp.
- **Metric Chuẩn**: Sử dụng thuật toán **PASCAL VOC 2010+ continuous all-points interpolation**:
  $$p_{\text{interp}}(r) = \max_{r' \ge r} p(r')$$
  Tích phân diện tích dưới đường bao cong Precision-Recall toàn bộ các điểm thay đổi, không dùng COCO 101-point hay 11-point xấp xỉ thô.

### P0.6 - Lưu Lịch sử Huấn luyện Đầy đủ
- `train.py` thiết lập mặc định `eval_every=1` trong giai đoạn canonical develop, đánh giá mAP@0.5 mỗi epoch.
- Tự động xuất lịch sử ra cả 2 định dạng: `outputs/history_{model}_seed{seed}.json` và `outputs/history_{model}_seed{seed}.csv`.
- Lưu `best_model.pth` dựa trên **val mAP@0.5 cao nhất** (thay vì val loss).

### P0.7 - Checkpoint Provenance & Tránh Nghịch lý SHA256
- Checkpoint lưu đầy đủ siêu dữ liệu ngữ nghĩa: `epoch`, `model_name`, `seed`, `git_commit`, `git_dirty`, `dataset_fingerprint`, `device`, `stride`, `preprocessing`, `max_det`, `hparams`.
- **Tuyệt đối không lưu SHA bên trong file `.pth`** (tránh nghịch lý đệ quy). SHA256 được tính từ file nhị phân trên đĩa và ghi ra file sidecar `{checkpoint}.pth.sha256` song hành.

---

## 3. CHI TIẾT KỸ THUẬT P1 (PROTOCOL DISCIPLINE)

### P1.1 & P1.3 - Hợp đồng Dữ liệu & Dataset Fingerprint
- **Empty vs Missing Label Contract**:
  - Nếu file nhãn `labels/{image_id}.txt` **không tồn tại**: Ném ngay ngoại lệ `FileNotFoundError` (Fail-Closed, không được ngầm coi là rỗng).
  - Nếu file `labels/{image_id}.txt` **tồn tại nhưng có kích thước 0 byte**: Xác nhận đây là **No-Finding hợp lệ** (ảnh không có tổn thương, 0 bounding box).
- **Canonical Dataset Fingerprint (`compute_dataset_fingerprint`)**:
  Băm mật mã SHA-256 toàn bộ byte nhị phân của ảnh `.png`, file nhãn `.txt` và đường dẫn tương đối của cả 3 tập `train`, `val`, `test` kết hợp danh sách tên 14 lớp bệnh học thành mã SHA-256 duy nhất.
- **Strict Bijection**: Mỗi tập split bắt buộc phải thỏa mãn:
  $$\text{set(image\_stems)} == \text{set(label\_stems)}$$
  Đã hoàn thiện bổ sung 1,222 file `.txt` kích thước 0 byte cho các ảnh No-Finding ở tập Train, đảm bảo 13,222 ảnh khớp 13,222 nhãn.

### P1.2 - Split Semantics Test Guard (Khóa Truy cập Tập Test)
- `get_develop_dataloaders()`: Chỉ trả về `train_loader` và `val_loader`. Không chứa bất kỳ logic nào tạo `test_loader`.
- `get_test_dataloader()`: Được bọc bởi `assert_test_access_allowed(permit, dataset_root)`. Bắt buộc phải có `PreflightPermit` hợp lệ được sinh từ `global_preflight_check()`, và tự động quét lại fingerprint trên đĩa ngay thời điểm mở loader.
- Ngăn chặn hoàn toàn rò rỉ tập Test trong quá trình phát triển (Develop Phase).

### P1.4 - Canonical 3 Models $\times$ 3 Seeds
- Bộ 3 mô hình: `["model1", "model2", "model3"]`.
- Bộ 3 seeds cố định: `[202601, 202602, 202603]`.
- Tổng cộng 9 thực nghiệm được huấn luyện độc lập và kiểm định. Trong chế độ `--data_mode real`, `generate_protocol_lock()` bắt buộc phải có đủ đúng 9 checkpoint hợp lệ.

### P1.6 - Chuẩn Benchmark Tài nguyên & Độ trễ
- Đo bộ nhớ RAM thông qua `state_dict_size_mib` = tổng số byte của parameters + buffers trong RAM.
- Đo độ trễ suy luận BS=1 (`bs1_latency_median_ms`, `mean`, `p95`) và thông lượng FPS (`throughput_fps`), ghi rõ kích thước batch đo thực tế (`bs4_throughput_fps` hoặc `bs16_throughput_fps`) cùng thiết bị (`cuda` hoặc `cpu`).

### P1.7 - Protocol Lock & Global Preflight Check
- `generate_protocol_lock()`: Sinh file `outputs/protocol_lock.json` khóa:
  - SHA256 của toàn bộ 9 checkpoint tốt nhất.
  - Dataset Fingerprint (băm nhị phân toàn bộ ảnh và nhãn).
  - Git Commit SHA và trạng thái git tree clean.
  - Cấu hình đánh giá (`conf_threshold=0.25`, `nms_iou=0.45`, `min_score=0.01`, `max_det=100`, `grid_size=16`, `image_size=512`).
- `global_preflight_check()`: Quét toàn diện đĩa cứng để xác minh:
  - Git HEAD khớp lock và working tree không bị dirty (ở chế độ real).
  - Cả 9 checkpoint và 9 sidecar `.sha256` còn nguyên vẹn 100%.
  - Dataset không bị thay đổi so với lúc khóa.
  - Nếu hợp lệ, cấp `PreflightPermit` có chữ ký mật mã HMAC.

### Post-Freeze Single-Pass Test Evaluation
- Mỗi cặp (model, seed) chỉ được duyệt qua `TestLoader` **đúng 1 lần duy nhất (single-pass traversal)**.
- Trong 1 vòng lặp duy nhất đó, hệ thống đồng thời tính Test Loss, IoU và giải mã Bounding Box để tính mAP@0.5.

### Thống kê Mẫu $ddof=1$
- Khi tính độ lệch chuẩn đa seed ($N=3$), bắt buộc sử dụng chuẩn mẫu:
  $$s = \sqrt{\frac{1}{N-1} \sum_{i=1}^N (x_i - \bar{x})^2}$$
  tương ứng tham số `np.std(values, ddof=1)`.

---

## 4. CHI TIẾT 18 HẠNG MỤC KIỂM CHUẨN (18 HARDENED INVARIANTS)

Dưới đây là 18 hạng mục kỹ thuật đã được bọc thép (harden) và kiểm chứng 100%:

| STT | Mã Invariant | Vấn đề Giải quyết | Hiện thực Kỹ thuật | File Thực thi |
| :---: | :--- | :--- | :--- | :--- |
| **1** | `P1-BLOCKER-1` | `final-test` không được tự re-lock | `run_multi_seed.py` ở phase `final-test` **tuyệt đối không gọi `generate_protocol_lock()`**. Chỉ đọc lock có sẵn và gọi `global_preflight_check()`. | [run_multi_seed.py](file:///c:/Users/admin/Downloads/MidtermDL/run_multi_seed.py) |
| **2** | `P1-BLOCKER-2` | Checkpoint phải thuộc `protocol_lock.json` | `evaluate.py` kiểm tra checkpoint path và SHA256 trên đĩa có tồn tại trong danh sách khóa của lock file hay không. | [evaluate.py](file:///c:/Users/admin/Downloads/MidtermDL/evaluate.py) |
| **3** | `P1-BLOCKER-3` | Config đánh giá phải nạp từ lock | `evaluate.py` nạp `conf_threshold`, `nms_iou`, `min_score`, `max_det` từ lock file, ghi đè cờ CLI tùy ý. | [evaluate.py](file:///c:/Users/admin/Downloads/MidtermDL/evaluate.py) |
| **4** | `P1-BLOCKER-4` | Bắt buộc đúng 9 cặp (3x3) ở chế độ real | `generate_protocol_lock()` kiểm tra `data_mode=="real"` phải có đủ 3 models $\times$ 3 seeds = 9 checkpoints. | [experiment_config.py](file:///c:/Users/admin/Downloads/MidtermDL/scripts/experiment_config.py) |
| **5** | `P1-BLOCKER-5` | Kiểm tra Git HEAD và Clean working tree | Bắt buộc `git status --porcelain` rỗng và git commit hiện tại khớp commit trong lock khi chạy kiểm chuẩn thật. | [experiment_config.py](file:///c:/Users/admin/Downloads/MidtermDL/scripts/experiment_config.py) |
| **6** | `P0-FAILCLOSED-1`| Loại bỏ hoàn toàn fallback index `idx` | `evaluate_detections()` và `evaluate.py` chỉ chấp nhận `image_id`. Thiếu hoặc lệch ID lập tức ném ngoại lệ. | [metrics.py](file:///c:/Users/admin/Downloads/MidtermDL/scripts/src/metrics.py) |
| **7** | `P0-CONTRACT-1` | Xác thực nghiêm ngặt định dạng nhãn | `_read_raw_boxes()` kiểm tra đúng 5 trường, class integer 0-13, tọa độ hữu hạn trong $[0, 1]$, $w, h > 0$. Lỗi ném `ValueError`. | [dataset.py](file:///c:/Users/admin/Downloads/MidtermDL/scripts/src/dataset.py) |
| **8** | `P1-LEAKAGE-1` | Ngăn rò rỉ tập Test khi inspect dữ liệu | `describe_dataset()` nhận tham số `splits`. Giai đoạn Develop chỉ đọc `("train", "val")`, không bao giờ đụng `test`. | [prepared_loader.py](file:///c:/Users/admin/Downloads/MidtermDL/scripts/data/prepared_loader.py) |
| **9** | `P1-FINGERPRINT-1`| Fingerprint băm byte ảnh và nhãn thực tế | `compute_split_fingerprint()` đọc từng byte nhị phân của ảnh `.png` và nhãn `.txt`, tính SHA-256 không thể giả mạo. | [experiment_config.py](file:///c:/Users/admin/Downloads/MidtermDL/scripts/experiment_config.py) |
| **10**| `P1-BIJECTION-1` | Strict Bijection giữa ảnh và file nhãn | Bắt buộc `set(image_stems) == set(label_stems)`. Đã tạo 1,222 file nhãn rỗng 0 byte cho các ca No-Finding tập train. | [experiment_config.py](file:///c:/Users/admin/Downloads/MidtermDL/scripts/experiment_config.py) |
| **11**| `P0-PROVENANCE-1`| Siêu dữ liệu ngữ nghĩa trong Checkpoint | Checkpoint lưu: `model_name`, `seed`, `dataset_fingerprint`, `git_commit`, `git_dirty`, `device`, `stride`, `preprocessing`, `max_det`, `hparams`. | [train.py](file:///c:/Users/admin/Downloads/MidtermDL/train.py) |
| **12**| `P1-LOCK-1` | Độ đầy đủ của Protocol Lock | `protocol_lock.json` lưu trữ: `max_det`, `class_names`, `stride`, `preprocessing_identity`, `data_mode`, ISO UTC timestamp. | [experiment_config.py](file:///c:/Users/admin/Downloads/MidtermDL/scripts/experiment_config.py) |
| **13**| `P1-DYNAMIC-1` | PreflightPermit tái kiểm tra tại lúc nạp test | `get_test_dataloader()` gọi `permit.verify(recheck_dataset=True, dataset_root=...)` để quét lại fingerprint ổ đĩa. | [prepared_loader.py](file:///c:/Users/admin/Downloads/MidtermDL/scripts/data/prepared_loader.py) |
| **14**| `P1-BENCH-1` | Minh bạch batch size và device benchmark | Cột thông lượng ghi rõ `bs4_throughput_fps` hoặc `bs16_throughput_fps` và thiết bị `(cpu)` hoặc `(cuda)`. | [benchmark_utils.py](file:///c:/Users/admin/Downloads/MidtermDL/scripts/benchmark_utils.py) |
| **15**| `P1-DATAMODE-1`| Hỗ trợ `--data_mode {real, demo}` đồng nhất | Toàn bộ 4 entrypoint (`train.py`, `evaluate.py`, `main.py`, `run_multi_seed.py`) đều hỗ trợ tham số `--data_mode`. | Các entrypoint |
| **16**| `P0-DATA-1` | Thống nhất thư mục dữ liệu `dataset_202601` | Cấu hình trung tâm `get_processed_data_root()` trỏ chuẩn về `data/dataset_202601`. | [config.py](file:///c:/Users/admin/Downloads/MidtermDL/scripts/config.py) |
| **17**| `P1-TESTS-1` | 17 Unit Tests tự động hóa 100% | Bao phủ toàn bộ 18 invariants: data contract, split guards, SHA verification, bijection, single-pass test. | [test_detection_protocol.py](file:///c:/Users/admin/Downloads/MidtermDL/tests/test_detection_protocol.py) |
| **18**| `P0-DECODE-1` | Tương thích tham số `min_score` / `conf_thr` | `decode_predictions()` hỗ trợ alias `min_score` song song với `conf_threshold`. | [decode.py](file:///c:/Users/admin/Downloads/MidtermDL/scripts/src/decode.py) |

---

## 5. BẢNG PHÂN ĐỊNH NGƯỠNG P1 VS P2

| Giai đoạn | Ngưỡng | Giá trị | Mục đích & Phạm vi |
| :--- | :--- | :---: | :--- |
| **P1 Baseline** | `eval_min_score` | `0.01` | Dùng để vẽ toàn vẹn đường cong Precision-Recall tính VOC all-points mAP@0.5. |
| **P1 Baseline** | `conf_threshold` | `0.25` | Khóa cứng làm điểm hoạt động cơ sở (Baseline Operating Point) cho Precision, Recall, F1. |
| **P1 Baseline** | `nms_iou_threshold` | `0.45` | Khóa cứng ngưỡng khử trùng lặp bounding box sau NMS. |
| **P1 Baseline** | `max_det` | `100` | Giới hạn tối đa số detection giữ lại cho mỗi ảnh sau NMS. |
| **P2 Tuning** | `conf_threshold` | Được tune trên Val | Được phép tìm kiếm trên tập Validation để tối ưu hóa Macro F1 theo từng lớp. |
| **P2 Tuning** | *Quy tắc khóa* | Đóng băng trước Test | Ngưỡng sau khi tune trên Val phải được cập nhật vào artifact cấu hình và ghi vào `protocol_lock.json` TRƯỚC KHI mở TestLoader. |

---

## 6. CẤU TRÚC MÃ NGUỒN ĐÃ TRIỂN KHAI

```text
MidtermDL/
├── scripts/
│   ├── config.py                      # Hằng số hệ thống (14 classes, stride 32, grid 16, img 512, dataset_202601)
│   ├── experiment_config.py           # Provenance, Dataset Fingerprint, Protocol Lock, Preflight Check, Permit
│   ├── benchmark_utils.py             # RAM state_dict_size_mib, BS=1 Latency, Adaptive FPS (CPU/CUDA)
│   ├── data/
│   │   ├── prepared_loader.py         # Split Guards, Develop/Test Loaders, Dynamic Permit Verify
│   │   └── prepare_dataset.py         # Tiền xử lý ảnh (CLAHE, Laplacian) & chia tập dữ liệu
│   ├── models/
│   │   ├── factory.py                 # Factory xây dựng 3 kiến trúc + pretrained flag
│   │   ├── model1_sequential.py       # SequentialCNNDetector + param summary
│   │   ├── model2_residual.py         # NonSequentialCNNDetector + param summary
│   │   └── model3_pretrained.py       # PretrainedDetector (ResNet50 unfreeze layer3)
│   └── src/
│       ├── dataset.py                 # Empty vs Missing label contract, strict 5-field box validation
│       ├── decode.py                  # Decode logits sang BBox + Batched NMS + min_score alias
│       ├── metrics.py                 # VOC continuous all-points AP, image_id dict matching, No-idx-fallback
│       └── utils.py                   # CombinedLocalizationLoss (num_pos safe), calculate_iou
├── train.py                           # Develop phase pipeline, rich provenance, sidecar SHA, --data_mode
├── evaluate.py                        # Single-Pass Test evaluation, enforced locked config, SHA verification
├── main.py                            # Phân định rõ --phase develop và --phase final-test, --data_mode
├── run_multi_seed.py                  # Điều phối Canonical 3x3 (develop -> lock -> final-test), no-relock guard
├── compare_models.py                  # Tổng hợp so sánh: Benchmark (device/batch size) + Test Metrics (ddof=1)
├── tests/
│   └── test_detection_protocol.py     # Bộ 17 Unit Tests kiểm tra toàn diện 100% 18 invariants
└── PROTOCOL_TAKENOTES.md              # Tài liệu này
```

---

## 7. HƯỚNG DẪN CHẠY LỆNH THỰC NGHIỆM (CLI GUIDE)

### 7.1. Chạy Toàn Bộ 17 Unit Tests Kiểm Chuẩn
```bash
python -m unittest tests/test_detection_protocol.py
```
*Kết quả kỳ vọng: 17 tests passed 100% OK trong ~5 giây.*

### 7.2. Kiểm tra Dữ liệu & Audit Va chạm Cell
```bash
python -m scripts.data.prepared_loader --root data/dataset_202601 --audit_collision
```

### 7.3. Chạy Huấn luyện Phát triển (Develop Phase - Tuyệt đối không đụng Test)
Chạy thử nghiệm nhanh (chế độ demo 30 ảnh):
```bash
python train.py --model model1 --seed 202601 --epochs 2 --data_mode demo
```
Chạy thử nghiệm chính thức (chế độ real toàn bộ tập train/val):
```bash
python train.py --model model1 --seed 202601 --epochs 40 --data_mode real
```
Hoặc qua `main.py`:
```bash
python main.py --model model1 --phase develop --seed 202601 --epochs 40 --data_mode real
```

### 7.4. Chạy Toàn bộ Canonical 3 Models $\times$ 3 Seeds (9 Runs)
#### Bước 1: Huấn luyện Develop Phase cho 9 runs
```bash
python run_multi_seed.py --model all --phase develop --epochs 40 --data_mode real
```

#### Bước 2: Khóa Protocol (Protocol Lock)
Sau khi 9 mô hình đã huấn luyện xong và có đủ checkpoint tốt nhất:
```bash
python run_multi_seed.py --model all --phase lock --data_mode real
```
Lệnh này sẽ băm SHA256 toàn bộ 9 checkpoint, băm toàn bộ dataset, xác minh git commit và ghi ra file `outputs/protocol_lock.json`.

#### Bước 3: Đánh giá Post-Freeze trên tập Test (Chỉ chạy 1 lần duy nhất)
```bash
python run_multi_seed.py --model all --phase final-test --data_mode real
```
*Giao thức đảm bảo: Bước này KHÔNG BAO GIỜ re-lock. Nó kiểm tra đĩa cứng (preflight check), cấp `PreflightPermit`, nạp đúng config từ lock và duyệt tập Test đúng 1 lần.*

### 7.5. Đo Đạc Benchmark & Xuất Bảng So sánh Tổng thể
```bash
python compare_models.py
```
*Lệnh này sẽ tự động đo RAM `state_dict_size_mib`, đo độ trễ suy luận, đọc kết quả Test của 3 mô hình và xuất bảng so sánh ra `outputs/detection_comparison_table.md`, `detection_comparison_table.csv`, và `detection_comparison_table.json`.*

### 7.6. Đánh giá Checkpoint Độc Lập
Để đánh giá 1 checkpoint cụ thể sau khi đã có lock:
```bash
python evaluate.py --checkpoint outputs/develop/model1/seed202601/best_model.pth --split test --data_mode real
```
*Lệnh này sẽ xác thực checkpoint path và SHA256 với lock file trước khi cho phép mở TestLoader.*
