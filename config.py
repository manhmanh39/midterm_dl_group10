"""
config.py - Cấu hình hệ thống và siêu tham số cho dự án VinBigData Chest X-ray.
"""

from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"   # nơi download_dataset.py tải về
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
OUTPUT_DIR = BASE_DIR / "outputs"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
    "Other lesion", "Pleural effusion", "Pleural thickening",
    "Pneumothorax", "Pulmonary fibrosis", "No finding",
]
NUM_CLASSES = len(CLASS_NAMES)
NO_FINDING_CLASS_ID = 14

IS_MULTILABEL = True

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_WORKERS = 16
DEFAULT_EPOCHS = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

MULTILABEL_THRESHOLD = 0.5

LR_SCHEDULER = "cosine"
LR_PATIENCE = 3

FREEZE_EPOCHS = 3
UNFREEZE_LR = 1e-5
BEST_METRIC = "auc"
SEED = 202601

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")