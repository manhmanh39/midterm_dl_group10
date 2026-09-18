import argparse
import shutil
import zipfile
from pathlib import Path
from huggingface_hub import snapshot_download


def download_and_extract(repo_id: str, output_folder: str):
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[download] Đang tải dataset từ {repo_id} (Hugging Face)...")
    download_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=["*.zip", "*.yaml", "*.json", "*.csv"],
        )
    )

    print(f"[download] Giải nén vào {output_folder}...")

    for meta in ["dataset.yaml", "manifest.json"]:
        src = download_dir / meta
        if src.exists():
            print(f"  Copy {meta}...")
            shutil.copy2(src, output_path / meta)

    for split in ["train", "val", "test"]:
        csv_file = f"{split}_boxes.csv"
        src = download_dir / csv_file
        if src.exists():
            split_dir = output_path / split
            split_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Copy {csv_file} -> {split}/")
            shutil.copy2(src, split_dir / csv_file)

    all_zips = sorted(download_dir.glob("*.zip"))
    if not all_zips:
        print("[download] CẢNH BÁO: không tìm thấy file .zip nào — kiểm tra lại repo_id.")

    for zip_path in all_zips:
        split = zip_path.stem.split("_")[0]
        target_dir = output_path / split
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Giải nén {zip_path.name} -> {split}/ ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)

    print(f"\n[download] Hoàn tất! Dataset tại: {output_path.resolve()}")
    for sub in ["train", "val", "test"]:
        img_dir = output_path / sub / "images"
        if img_dir.exists():
            count = len(list(img_dir.glob("*.png")))
            print(f"  - {sub}: {count} ảnh PNG")
        else:
            print(f"  - {sub}: (không có)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tải dataset đã prepared từ Hugging Face")
    parser.add_argument(
        "--repo_id",
        default="TheBlindMaster/VinBigData-Chest-X-ray-Prepared",
        help="Hugging Face dataset repo ID",
    )
    parser.add_argument("--output", default="data/processed", help="Thư mục đích")
    args = parser.parse_args()

    download_and_extract(args.repo_id, args.output)