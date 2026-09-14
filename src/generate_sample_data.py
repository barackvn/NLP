import os
import json
import pandas as pd

# Mẫu dữ liệu chuẩn hóa theo cấu trúc ViHOS Benchmark (EACL 2023)
SAMPLE_TRAIN_DATA = [
    {
        "tokens": ["Đồ", "ngu", "nhìn", "cái", "mặt", "mày", "hãm", "thật_sự"],
        "tags": ["B-HOS", "I-HOS", "O", "O", "O", "O", "B-HOS", "O"]
    },
    {
        "tokens": ["Mày", "là", "đồ", "súc_sinh", "mất_dạy", "vô_học", "cút", "đi"],
        "tags": ["O", "O", "B-HOS", "I-HOS", "B-HOS", "I-HOS", "B-HOS", "O"]
    },
    {
        "tokens": ["Hôm_nay", "thời_tiết", "ở", "Sài_Gòn", "đẹp", "quá", "đi", "chơi", "thôi"],
        "tags": ["O", "O", "O", "O", "O", "O", "O", "O", "O"]
    },
    {
        "tokens": ["Thằng", "chó", "rác_rưởi", "đừng", "có", "ở", "đây", "mà", "xàm"],
        "tags": ["B-HOS", "I-HOS", "B-HOS", "O", "O", "O", "O", "O", "B-HOS"]
    },
    {
        "tokens": ["Bài", "hát", "này", "nghe", "rất", "hay", "và", "ý_nghĩa"],
        "tags": ["O", "O", "O", "O", "O", "O", "O", "O", "O"]
    },
    {
        "tokens": ["Cái", "thứ", "đàn_bà", "đĩ_thỏa", "lăng_loàn"],
        "tags": ["O", "O", "O", "B-HOS", "I-HOS"]
    },
    {
        "tokens": ["Quán", "cà_phê", "này", "view", "đẹp", "nhân_viên", "nhiệt_tình"],
        "tags": ["O", "O", "O", "O", "O", "O", "O"]
    },
    {
        "tokens": ["Đúng", "là", "lũ", "óc_chó", "không", "biết", "suy_nghĩ"],
        "tags": ["O", "O", "O", "B-HOS", "O", "O", "O"]
    },
    {
        "tokens": ["Cảm_ơn", "bạn", "đã", "chia_sẻ", "thông_tin", "hữu_ích"],
        "tags": ["O", "O", "O", "O", "O", "O"]
    },
    {
        "tokens": ["Mặt", "dày", "như", "thớt", "còn", "bày_đặt", "thanh_cao"],
        "tags": ["B-HOS", "I-HOS", "O", "O", "O", "O", "O"]
    }
]

SAMPLE_DEV_DATA = [
    {
        "tokens": ["Con", "mụ", "hãm_lồn", "nói_năng", "xàm_xí"],
        "tags": ["O", "O", "B-HOS", "O", "B-HOS"]
    },
    {
        "tokens": ["Chúc", "mọi", "người", "một", "ngày", "làm_việc", "vui_vẻ"],
        "tags": ["O", "O", "O", "O", "O", "O", "O"]
    },
    {
        "tokens": ["Thằng", "ranh_con", "láo_toét", "ăn_nói", "xấc_xược"],
        "tags": ["B-HOS", "I-HOS", "B-HOS", "O", "B-HOS"]
    },
    {
        "tokens": ["Tôi", "rất", "thích", "học", "môn", "Xử_lý", "Ngôn_ngữ", "Tự_nhiên"],
        "tags": ["O", "O", "O", "O", "O", "O", "O", "O"]
    }
]

SAMPLE_TEST_DATA = [
    {
        "tokens": ["Thứ", "bại_não", "như", "mày", "sống", "chật", "đất"],
        "tags": ["O", "B-HOS", "O", "O", "O", "O", "O"]
    },
    {
        "tokens": ["Xin", "chào", "thầy_cô", "và", "các", "bạn", "sinh_viên", "UIT"],
        "tags": ["O", "O", "O", "O", "O", "O", "O", "O"]
    },
    {
        "tokens": ["Lũ", "khốn_nạn", "hút", "máu", "dân", "lành"],
        "tags": ["O", "B-HOS", "O", "O", "O", "O"]
    }
]

def generate_datasets(base_dir):
    raw_dir = os.path.join(base_dir, "data", "raw")
    processed_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # 1. Lưu file JSON cho processed
    with open(os.path.join(processed_dir, "train.json"), "w", encoding="utf-8") as f:
        json.dump(SAMPLE_TRAIN_DATA, f, ensure_ascii=False, indent=2)
    with open(os.path.join(processed_dir, "dev.json"), "w", encoding="utf-8") as f:
        json.dump(SAMPLE_DEV_DATA, f, ensure_ascii=False, indent=2)
    with open(os.path.join(processed_dir, "test.json"), "w", encoding="utf-8") as f:
        json.dump(SAMPLE_TEST_DATA, f, ensure_ascii=False, indent=2)

    # 2. Lưu file CSV cho raw
    def to_csv(data, filepath):
        rows = []
        for d in data:
            rows.append({
                "tokens": " ".join(d["tokens"]),
                "tags": " ".join(d["tags"]),
                "text": " ".join([t.replace("_", " ") for t in d["tokens"]])
            })
        pd.DataFrame(rows).to_csv(filepath, index=False, encoding="utf-8")

    to_csv(SAMPLE_TRAIN_DATA, os.path.join(raw_dir, "train.csv"))
    to_csv(SAMPLE_DEV_DATA, os.path.join(raw_dir, "dev.csv"))
    to_csv(SAMPLE_TEST_DATA, os.path.join(raw_dir, "test.csv"))
    print("[Success] Created sample ViHOS datasets in data/raw/ and data/processed/ successfully.")

if __name__ == "__main__":
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else "."
    generate_datasets(base)
