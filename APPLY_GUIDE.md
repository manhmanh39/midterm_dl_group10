# Detection refactor overlay

This folder contains every file that should be replaced in branch `feature/detection-refactor` for the proposed performance-first refactor while preserving the assignment constraints.

## Hard constraints preserved

- **Model 1**: simple CNN, fully sequential backbone, trained from scratch. Main pattern is `Conv -> Pool -> Conv -> Pool ...`; no residual/parallel backbone branches.
- **Model 2**: complex CNN, trained from scratch. Sequential stages contain explicit parallel branches that are concatenated.
- **Model 3**: pretrained ResNet-50 base network with transfer learning/fine-tuning and an improved multi-scale neck.
- All three remain **object detectors** and share the same target format, detection head, loss, decoder/NMS, train/val/test data, and raw-GT evaluation protocol.

## Files replaced

1. `requirements.txt`
2. `main.py`
3. `train.py`
4. `evaluate.py`
5. `hyperparameter_search.py`
6. `scripts/config.py`
7. `scripts/data/utils.py`
8. `scripts/data/prepare_dataset.py`
9. `scripts/src/dataset.py`
10. `scripts/src/detection_head.py`
11. `scripts/src/utils.py`
12. `scripts/src/decode.py`
13. `scripts/src/metrics.py`
14. `scripts/models/model1_sequential.py`
15. `scripts/models/model2_residual.py`
16. `scripts/models/model3_pretrained.py`
17. `scripts/models/factory.py`

`run_multi_seed.py` and `scripts/data/prepared_loader.py` can remain unchanged.

## Key fixes

- stride `32 -> 16`: 512 input now gives a 32x32 grid instead of 16x16.
- target format is `(G, G, A, 5+C)` with `A=3` slots/cell, strongly reducing center-cell collisions.
- final mAP still uses **all raw GT boxes**, not a filtered/representable subset.
- balanced positive/no-finding sampling.
- inverse-sqrt class-balanced classification loss.
- CIoU box loss + focal objectness.
- bbox-aware mild X-ray augmentation.
- robust percentile DICOM normalization.
- multilabel-stratified train/val/test split, including a pseudo-label for No Finding.
- stale files from old splits are removed to prevent leakage after re-preparing data.
- improved same-class radiologist box fusion.
- Model 3 uses `/16 + /32` feature fusion and staged fine-tuning.
- Optuna filenames now exactly match `train.py`: `best_hparams_model1/2/3.json`.
- optimizer label is correctly `adamw`, not `adam` while secretly constructing AdamW.

## IMPORTANT: old checkpoints are incompatible

The output tensor changed from `(B,G,G,5+C)` to `(B,G,G,A,5+C)`, and all backbones now expose `/16` features. Retrain all three models. Do not load old checkpoints into this code.

## Recommended run order

```bash
pip install -r requirements.txt

# Required because preprocessing and split logic changed.
python -m scripts.data.prepare_dataset \
  --zip_path "<VINBIGDATA_ZIP>" \
  --data_root data/raw \
  --output_root data/processed/dataset_202601 \
  --seed 202601 \
  --overwrite

# Optional but recommended after the refactor. Old simple/complex/transfer JSONs are obsolete.
python hyperparameter_search.py --model model1 --n_trials 25 --search_epochs 6
python hyperparameter_search.py --model model2 --n_trials 25 --search_epochs 6
python hyperparameter_search.py --model model3 --n_trials 25 --search_epochs 6

# Train + final test evaluation.
python main.py --model all --mode all --epochs 40 --seed 202601
```

For the report, after architecture/HPO choices are frozen, run multiple seeds using the existing `run_multi_seed.py` and report mean +/- std.

## Validation checks already run on this overlay

- Python compile check: PASS.
- Model 1 output shape: `(B, 32, 32, 3, 19)`: PASS.
- Model 2 output shape: `(B, 32, 32, 3, 19)`: PASS.
- Model 3 output shape with a local no-download ResNet smoke test: `(B, 32, 32, 3, 19)`: PASS.
- Synthetic dataset target shape and 3-box same-cell assignment: PASS.
- Loss forward/backward finite: PASS.
- Decoder smoke test: PASS.
- Full raw-GT evaluation smoke test: PASS.

## What is not claimed

This is a materially stronger and methodologically cleaner pipeline, but no code change can guarantee the globally best mAP before training/ablation on your exact split. Use validation mAP for architecture/HPO decisions and touch the test set only for the frozen final comparison.
