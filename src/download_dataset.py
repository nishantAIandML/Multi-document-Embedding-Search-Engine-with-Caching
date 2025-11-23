# src/download_dataset.py

import os
from sklearn.datasets import fetch_20newsgroups

SAVE_FOLDER = "data/docs"

def save_dataset():
    print("Downloading 20 Newsgroups dataset...")
    
    dataset = fetch_20newsgroups(
        subset='train',
        remove=('headers', 'footers', 'quotes'),
        download_if_missing=True
    )

    os.makedirs(SAVE_FOLDER, exist_ok=True)

    print(f"Saving documents to: {SAVE_FOLDER}")

    for idx, text in enumerate(dataset.data):
        filename = os.path.join(SAVE_FOLDER, f"doc_{idx:04d}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)

    print(f"Saved {len(dataset.data)} documents.")

if __name__ == "__main__":
    save_dataset()
