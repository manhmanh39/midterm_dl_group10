import argparse
from typing import List

import config
from data_loader import get_dataloaders
from eval import evaluate_model
from train import train


def run_pipeline(model_names: List[str], epochs: int, batch_size: int,
                  learning_rate: float, mode: str = "all", data_dir: str = None,
                  backbone: str = "resnet50"):
    print("*" * 80)
    print("      DỰ ÁN PHÂN LOẠI BẤT THƯỜNG X-QUANG PHỔI (VINBIGDATA CHEST X-RAY)      ")
    print("*" * 80)
    print(f"Mô hình: {model_names} | Mode: {mode.upper()} | Epochs: {epochs} | Batch: {batch_size}")
    print(f"Multi-label: {config.IS_MULTILABEL}")
    print("*" * 80)

    print("\n>>> BƯỚC 1: KIỂM TRA VÀ NẠP DỮ LIỆU...")
    _, _, _, class_names, _ = get_dataloaders(data_dir=data_dir, batch_size=batch_size)
    print(f"[OK] {len(class_names)} lớp nhãn sẵn sàng.\n")

    summary_results = []

    for model_name in model_names:
        print("\n" + "#" * 80)
        print(f"### TIẾN TRÌNH CHO MÔ HÌNH: {model_name.upper()} ###")
        print("#" * 80)

        if mode in ("train", "all"):
            print(f"\n>>> BƯỚC 2: HUẤN LUYỆN [{model_name}]...")
            train(model_name=model_name, epochs=epochs, batch_size=batch_size,
                  learning_rate=learning_rate, data_dir=data_dir, backbone_name=backbone)

        if mode in ("eval", "all"):
            print(f"\n>>> BƯỚC 3: ĐÁNH GIÁ [{model_name}]...")
            eval_metrics = evaluate_model(model_name=model_name, data_dir=data_dir, batch_size=batch_size)
            summary_results.append({"model": model_name, **eval_metrics})

    if summary_results and len(summary_results) > 1:
        print("\n" + "=" * 80)
        print("BẢNG TỔNG KẾT SO SÁNH")
        print("=" * 80)
        for res in summary_results:
            print(res)
        print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy pipeline phân loại X-quang VinBigData")
    parser.add_argument("--model", type=str, default="all", choices=["simple", "complex", "transfer", "all"])
    parser.add_argument("--backbone", type=str, default="resnet50",
                         choices=["resnet18", "resnet50", "mobilenet_v3", "efficientnet_b0", "convnext_tiny"])
    parser.add_argument("--mode", type=str, default="all", choices=["all", "train", "eval"])
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--data_dir", type=str, default=None)

    args = parser.parse_args()
    selected_models = ["simple", "complex", "transfer"] if args.model == "all" else [args.model]

    run_pipeline(
        model_names=selected_models, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.lr, mode=args.mode, data_dir=args.data_dir, backbone=args.backbone,
    )