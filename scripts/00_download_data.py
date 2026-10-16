# -*- coding: utf-8 -*-
"""
00_download_data.py -- Reproducible dataset downloader.

Downloads the three public datasets used by this project into data/raw/:
  1. IBM Transactions for Anti-Money Laundering (HI-Small)  [synthetic, main]
     source: HuggingFace mirror of the Kaggle CDLA-Sharing-1.0 dataset
  2. IEEE-CIS Fraud Detection (train split)                  [real e-commerce fraud]
     source: HuggingFace mirror of the Kaggle competition data
  3. Elliptic Bitcoin Dataset                                [real blockchain AML]
     source: HuggingFace mirror of the Kaggle dataset

Idempotent: already-downloaded files are skipped, so re-running is cheap.

NOTE on licensing:
  * IBM HI-Small      : CDLA-Sharing-1.0 (permissive, allows modification/use)
  * IEEE-CIS          : Kaggle competition terms, research/education use
  * Elliptic          : publicly released research dataset
  All are used here strictly for personal learning and portfolio demonstration.
"""
import os
import urllib.request
import zipfile

from utils import DATA_RAW

# (local file, remote url)
TARGETS = {
    "ibm": [
        ("HI-Small_Trans.csv.zip",
         "https://huggingface.co/datasets/eexzzm/IBM-Transactions-for-Anti-Money-Laundering-HI-Small-Trans/resolve/main/HI-Small_Trans.csv.zip?download=true"),
    ],
    "ieee": [
        ("train_transaction.csv",
         "https://huggingface.co/datasets/aliceczr/ieee-fraud-detection/resolve/main/train_transaction.csv?download=true"),
        ("train_identity.csv",
         "https://huggingface.co/datasets/aliceczr/ieee-fraud-detection/resolve/main/train_identity.csv?download=true"),
    ],
    "elliptic": [
        ("elliptic_txs_classes.csv",
         "https://huggingface.co/datasets/yhoma/elliptic-bitcoin-dataset/resolve/main/elliptic_txs_classes.csv?download=true"),
        ("elliptic_txs_edgelist.csv",
         "https://huggingface.co/datasets/yhoma/elliptic-bitcoin-dataset/resolve/main/elliptic_txs_edgelist.csv?download=true"),
        ("elliptic_txs_features.csv",
         "https://huggingface.co/datasets/yhoma/elliptic-bitcoin-dataset/resolve/main/elliptic_txs_features.csv?download=true"),
    ],
}


def download(url, dest_path):
    """Stream a file from url to dest_path with a small progress print."""
    print(f"[download] {os.path.basename(dest_path)} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest_path, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total and done % (50 * 1024 * 1024) < 1024 * 1024:
                print(f"  ... {done / 1e6:.0f} / {total / 1e6:.0f} MB")
    print(f"[ok] {os.path.basename(dest_path)} ({os.path.getsize(dest_path) / 1e6:.0f} MB)")


def main():
    for sub, files in TARGETS.items():
        sub_dir = os.path.join(DATA_RAW, sub)
        os.makedirs(sub_dir, exist_ok=True)
        for name, url in files:
            dest = os.path.join(sub_dir, name)
            if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
                print(f"[skip] {name} already present")
                continue
            download(url, dest)

    # unzip IBM archive
    zip_path = os.path.join(DATA_RAW, "ibm", "HI-Small_Trans.csv.zip")
    csv_path = os.path.join(DATA_RAW, "ibm", "HI-Small_Trans.csv")
    if os.path.exists(zip_path) and not os.path.exists(csv_path):
        print("[unzip] HI-Small_Trans.csv.zip ...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(os.path.dirname(zip_path))
        print("[ok] extracted HI-Small_Trans.csv")


if __name__ == "__main__":
    main()
