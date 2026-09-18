import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    roc_auc_score, multilabel_confusion_matrix,
)
import torch
import torch.nn as nn
from tqdm import tqdm

import config
from data_loader import get_dataloaders
from models import get_model


def _sensitivity_specificity(mcm: np.ndarray):
    tn, fp, fn, tp = mcm[:, 0, 0], mcm[:, 0, 1], mcm[:, 1, 0], mcm[:, 1, 1]
    sens = tp / np.clip(tp + fn, 1, None)
    spec = tn / np.clip(tn + fp, 1, None)
    return sens, spec


def evaluate_model(
    model_name: str = "transfer",
    checkpoint_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    batch_size: int = config.BATCH_SIZE,
    save_plot: bool = True,
    device: torch.device = config.DEVICE,
):
    if checkpoint_path is None:
        ckpt_file = config.CHECKPOINT_DIR / f"{model_name}_best.pth"
    else:
        ckpt_file = Path(checkpoint_path)

    if not ckpt_file.exists():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint tại: {ckpt_file}. "
            f"Hãy huấn luyện trước: python train.py --model {model_name}"
        )

    print("=" * 70)
    print(f"BẮT ĐẦU ĐÁNH GIÁ MÔ HÌNH: {model_name.upper()}")
    print(f"Checkpoint: {ckpt_file}")
    print("=" * 70)

    _, _, test_loader, class_names, _ = get_dataloaders(data_dir=data_dir, batch_size=batch_size)

    checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)
    num_classes = checkpoint.get("num_classes", len(class_names))
    is_multilabel = checkpoint.get("is_multilabel", config.IS_MULTILABEL)
    backbone_name = checkpoint.get("backbone_name")
    train_seed = checkpoint.get("seed")
    best_metric = checkpoint.get("best_metric")

    if train_seed is not None:
        print(f"  (checkpoint được train với seed={train_seed}, chọn best theo {best_metric})")

    model_kwargs = {"backbone_name": backbone_name} if backbone_name else {}
    model = get_model(model_name, num_classes=num_classes, **model_kwargs).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.BCEWithLogitsLoss() if is_multilabel else nn.CrossEntropyLoss()

    total_loss = 0.0
    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="[Testing]"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            if is_multilabel:
                probs = torch.sigmoid(outputs)
                preds = (probs > config.MULTILABEL_THRESHOLD).float()
                all_probs.append(probs.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
            else:
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    test_loss = total_loss / len(test_loader.dataset)

    print("\n" + "=" * 70)
    print(f"KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST  (multi-label={is_multilabel})")
    print(f"  - Test Loss: {test_loss:.4f}")

    if is_multilabel:
        y_true = np.concatenate(all_labels, axis=0)
        y_pred = np.concatenate(all_preds, axis=0)
        y_prob = np.concatenate(all_probs, axis=0)

        exact_match_acc = (y_true == y_pred).mean()
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)

        mcm = multilabel_confusion_matrix(y_true, y_pred)
        sens, spec = _sensitivity_specificity(mcm)

        try:
            auc_per_class = roc_auc_score(y_true, y_prob, average=None)
            auc_macro = float(np.nanmean(auc_per_class))
        except ValueError:
            auc_per_class, auc_macro = None, float("nan")

        print(f"  - Per-label Accuracy: {exact_match_acc*100:.2f}%  (tham khảo, KHÔNG dùng để chọn best model)")
        print(f"  - F1-Macro: {f1_macro:.4f} | F1-Micro: {f1_micro:.4f}")
        print(f"  - Mean Sensitivity: {sens.mean():.4f} | Mean Specificity: {spec.mean():.4f}")
        print(f"  - Mean AUC-ROC: {auc_macro:.4f}")
        print("=" * 70)

        print("\n--- CHI TIẾT THEO LỚP ---")
        for i, name in enumerate(class_names[:num_classes]):
            auc_str = f"{auc_per_class[i]:.4f}" if auc_per_class is not None else "N/A"
            print(f"  {name:<22} Sens={sens[i]:.3f}  Spec={spec[i]:.3f}  AUC={auc_str}")

        if save_plot:
            plot_path = config.OUTPUT_DIR / f"per_class_auc_{model_name}.png"
            plt.figure(figsize=(10, 5))
            xs = np.arange(num_classes)
            values = auc_per_class if auc_per_class is not None else np.zeros(num_classes)
            plt.bar(xs, values)
            plt.xticks(xs, class_names[:num_classes], rotation=45, ha="right", fontsize=8)
            plt.ylabel("AUC-ROC")
            plt.title(f"AUC-ROC theo lớp - {model_name} (Mean={auc_macro:.3f})")
            plt.tight_layout()
            plt.savefig(plot_path, dpi=200)
            plt.close()
            print(f"[eval] Đã lưu biểu đồ AUC per-class tại: {plot_path}")

        return {
            "loss": test_loss,
            "exact_match_accuracy": exact_match_acc,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "mean_sensitivity": float(sens.mean()),
            "mean_specificity": float(spec.mean()),
            "mean_auc": auc_macro,
        }

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    test_accuracy = (all_preds == all_labels).mean()
    print(f"  - Test Accuracy: {test_accuracy*100:.2f}%")

    present_classes = np.unique(np.concatenate([all_labels, all_preds]))
    target_names = [class_names[i] for i in present_classes]
    report = classification_report(all_labels, all_preds, labels=present_classes,
                                    target_names=target_names, digits=4, zero_division=0)
    print("\n--- CLASSIFICATION REPORT ---")
    print(report)

    cm = confusion_matrix(all_labels, all_preds, labels=present_classes)
    if save_plot:
        plot_path = config.OUTPUT_DIR / f"confusion_matrix_{model_name}.png"
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix - {model_name} (Acc: {test_accuracy*100:.1f}%)")
        plt.colorbar()
        ticks = np.arange(len(target_names))
        plt.xticks(ticks, target_names, rotation=45, ha="right", fontsize=8)
        plt.yticks(ticks, target_names, fontsize=8)
        thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], "d"), ha="center",
                          color="white" if cm[i, j] > thresh else "black")
        plt.ylabel("Nhãn thực tế")
        plt.xlabel("Nhãn dự đoán")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"[eval] Đã lưu Confusion Matrix tại: {plot_path}")

    return {"loss": test_loss, "accuracy": test_accuracy, "confusion_matrix": cm}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá mô hình phân loại X-quang VinBigData")
    parser.add_argument("--model", type=str, default="transfer", choices=["simple", "complex", "transfer"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--no_plot", action="store_true")

    args = parser.parse_args()
    evaluate_model(model_name=args.model, checkpoint_path=args.checkpoint,
                    data_dir=args.data_dir, save_plot=not args.no_plot)