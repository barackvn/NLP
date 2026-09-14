import os
import urllib.request
import pandas as pd
import json

BASE_URL = "https://raw.githubusercontent.com/phusroyal/ViHOS/master/data/Sequence_labeling_based_version/Word/"
RAW_FILES = {
    "train": "train_BIO_Word.csv",
    "dev": "dev_BIO_Word.csv",
    "test": "test_BIO_Word.csv"
}

def download_and_process_vihos(output_dir="data"):
    raw_dir = os.path.join(output_dir, "raw")
    processed_dir = os.path.join(output_dir, "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    print("[1/3] Downloading official ViHOS Benchmark data from phusroyal/ViHOS GitHub...")
    for split, filename in RAW_FILES.items():
        url = BASE_URL + filename
        dest_path = os.path.join(raw_dir, filename)
        if not os.path.exists(dest_path):
            print(f"  Downloading {filename}...")
            urllib.request.urlretrieve(url, dest_path)
            print(f"  Saved {filename} ({os.path.getsize(dest_path) / 1024:.1f} KB)")
        else:
            print(f"  {filename} already exists.")

    print("\n[2/3] Processing into standard sequence labeling JSON format...")
    for split, filename in RAW_FILES.items():
        csv_path = os.path.join(raw_dir, filename)
        df = pd.read_csv(csv_path)

        # Chuẩn hóa nhãn B-T -> B-HOS, I-T -> I-HOS
        df['Tag'] = df['Tag'].replace({'B-T': 'B-HOS', 'I-T': 'I-HOS'})

        # Gom nhóm theo sentence_id
        grouped = df.groupby('sentence_id')
        processed_data = []

        for sentence_id, group in grouped:
            # Sắp xếp theo index nếu có
            if 'index' in group.columns:
                group = group.sort_values('index')
            
            words = group['Word'].astype(str).tolist()
            tags = group['Tag'].astype(str).tolist()

            processed_data.append({
                "sentence_id": int(sentence_id),
                "tokens": words,
                "tags": tags
            })

        out_json = os.path.join(processed_dir, f"{split}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)

        print(f"  {split.upper()}: {len(processed_data)} sentences processed -> {out_json}")

    print("\n[3/3] ViHOS benchmark dataset ready in data/processed/!")

if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "data"
    download_and_process_vihos(out)
