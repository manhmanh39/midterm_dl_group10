import argparse
import json

import numpy as np

import config
from train import train
from eval import evaluate_model


def run_multi_seed(
    model_name: str = "transfer",
    backbone_name: str = "resnet50",
    epochs: int = config.DEFAULT_EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    data_dir: str = None,
    base_seed: int = 202601,
    n_runs: int = 5,
):
    seeds = [base_seed + i for i in range(n_runs)]
    print(f"[multi_seed] Sẽ chạy {n_runs} lần với seeds: {seeds}")

    all_results = []

    for run_idx, seed in enumerate(seeds, start=1):
        print("\n" + "#" * 80)
        print(f"### LẦN CHẠY {run_idx}/{n_runs}  —  SEED = {seed} ###")
        print("#" * 80)

        save_dir = config.CHECKPOINT_DIR / f"seed_{seed}"
        save_dir.mkdir(parents=True, exist_ok=True)

        train(
            model_name=model_name, epochs=epochs, batch_size=batch_size,
            learning_rate=learning_rate, data_dir=data_dir, save_dir=save_dir,
            backbone_name=backbone_name, seed=seed,
        )

        checkpoint_path = save_dir / f"{model_name}_best.pth"
        metrics = evaluate_model(
            model_name=model_name, checkpoint_path=str(checkpoint_path),
            data_dir=data_dir, batch_size=batch_size, save_plot=False,
        )
        metrics["seed"] = seed
        all_results.append(metrics)

    print("\n" + "=" * 80)
    print(f"KẾT QUẢ TỔNG HỢP QUA {n_runs} SEED KHÁC NHAU ({model_name.upper()})")
    print("=" * 80)

    numeric_keys = [k for k, v in all_results[0].items() if isinstance(v, (int, float)) and k != "seed"]
    summary = {}
    for key in numeric_keys:
        values = [r[key] for r in all_results]
        mean_v, std_v = float(np.mean(values)), float(np.std(values))
        summary[key] = {"mean": mean_v, "std": std_v, "values": values}
        print(f"  {key:<25} mean={mean_v:.4f}  std={std_v:.4f}   ({[round(v,4) for v in values]})")

    print("=" * 80)

    report_path = config.OUTPUT_DIR / f"multi_seed_report_{model_name}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"model": model_name, "seeds": seeds, "per_run": all_results, "summary": summary},
                   f, indent=2, ensure_ascii=False)
    print(f"\n[multi_seed] Đã lưu báo cáo chi tiết tại: {report_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy training nhiều seed để kiểm tra reproducibility")
    parser.add_argument("--model", type=str, default="transfer", choices=["simple", "complex", "transfer"])
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--base_seed", type=int, default=202601)
    parser.add_argument("--n_runs", type=int, default=5)

    args = parser.parse_args()
    run_multi_seed(
        model_name=args.model, backbone_name=args.backbone,
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr,
        data_dir=args.data_dir, base_seed=args.base_seed, n_runs=args.n_runs,
    )