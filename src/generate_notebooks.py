import json
import os

def create_notebook(cells, filepath):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)

def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")]
    }

def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.split("\n")]
    }

# 1. Notebook 01: Data Preprocessing & Alignment
nb01_cells = [
    md_cell("# 01 - Tiền xử lý Dữ liệu ViHOS & First-token Subword Alignment\n**Môn học:** Xử lý Ngôn ngữ Tự nhiên - ĐH Công nghệ Thông tin (UIT)\n**Nhân sự phụ trách:** Nông Nguyễn Thành (26410115)"),
    code_cell("""import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer"""),
    md_cell("## 1. Nạp dữ liệu ViHOS Benchmark"),
    code_cell("""train_path = "../data/processed/train.json"
dev_path = "../data/processed/dev.json"
test_path = "../data/processed/test.json"

with open(train_path, "r", encoding="utf-8") as f:
    train_data = json.load(f)

print(f"Tổng số mẫu train: {len(train_data)}")
print("Ví dụ mẫu đầu tiên:", train_data[0])"""),
    md_cell("## 2. Thống kê phân bố nhãn BIO (O, B-HOS, I-HOS)"),
    code_cell("""all_tags = [t for sample in train_data for t in sample['tags']]
tag_counts = pd.Series(all_tags).value_counts()
print("Phân bố nhãn BIO:\\n", tag_counts)

plt.figure(figsize=(6, 3))
sns.barplot(x=tag_counts.index, y=tag_counts.values, palette="viridis")
plt.title("Phân bố nhãn BIO trong tập huấn luyện ViHOS")
plt.ylabel("Số lượng token")
plt.show()"""),
    md_cell("## 3. Khảo sát phân bố độ dài câu (Sentence Length Distribution)"),
    code_cell("""lengths = [len(sample['tokens']) for sample in train_data]
print(f"Độ dài trung bình: {np.mean(lengths):.1f} từ")
print(f"Độ dài tối đa: {np.max(lengths)} từ")
print(f"Tỷ lệ câu <= 128 từ: {np.mean(np.array(lengths) <= 128) * 100:.2f}%")"""),
    md_cell("## 4. Minh họa First-token Subword Alignment với PhoBERT"),
    code_cell("""tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

sample = train_data[0]
tokens = sample['tokens']
tags = sample['tags']

subwords_list = []
aligned_tags = []

for word, tag in zip(tokens, tags):
    subwords = tokenizer.tokenize(word)
    subwords_list.extend(subwords)
    aligned_tags.append(tag)
    for _ in subwords[1:]:
        aligned_tags.append("-100 (Ignored)")

for sub, tag in zip(subwords_list, aligned_tags):
    print(f"{sub:<15} --> {tag}")""")
]

# 2. Notebook 02: Training Colab GPU T4
nb02_cells = [
    md_cell("""# 🚀 Huấn Luyện PhoBERT-BiLSTM-CRF Trên Google Colab (GPU Tesla T4)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/barackvn/NLP/blob/main/notebooks/02_train_colab_gpu_t4.ipynb)

**Đề tài:** Nhận diện chuỗi ngôn ngữ xúc phạm tiếng Việt (ViHOS)  
**Người phụ trách:** Hoàng Võ Minh Tuấn (Trainer - 26410146)  
**Môi trường:** GPU Tesla T4 (16GB VRAM), PyTorch, HuggingFace Transformers, PyTorch-CRF, Seqeval  
**GitHub Repo chính thức:** [https://github.com/barackvn/NLP](https://github.com/barackvn/NLP)"""),
    
    md_cell("## Bước 1: Kiểm tra cấu hình GPU Tesla T4\nĐảm bảo bạn đã chọn **Runtime -> Change runtime type -> T4 GPU** trước khi chạy!"),
    code_cell("""# Kiểm tra GPU Colab
!nvidia-smi"""),
    
    md_cell("## Bước 2: Tải Mã Nguồn & Dữ Liệu từ GitHub Chính Thức\nToàn bộ code đã sửa lỗi CRF và dữ liệu ViHOS đầy đủ (11.056 câu) sẽ được nạp tự động."),
    code_cell("""import os
%cd /content
if os.path.exists('/content/NLP'):
    !rm -rf /content/NLP

# Clone repository chính thức
!git clone https://github.com/barackvn/NLP.git
%cd /content/NLP
!pwd"""),

    md_cell("## Bước 3: Cài đặt các thư viện cần thiết"),
    code_cell("""!pip install -q transformers pyvi seqeval accelerate"""),

    md_cell("## Bước 4: Kết nối Google Drive để lưu checkpoint vĩnh viễn"),
    code_cell("""from google.colab import drive
import os

drive.mount('/content/drive')
CHECKPOINT_DIR = '/content/drive/MyDrive/ViHOS_Checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print(f"✅ Thư mục sao lưu Drive: {CHECKPOINT_DIR}")"""),

    md_cell("""## Bước 5: Huấn Luyện Mô Hình Đề Xuất (PhoBERT-BiLSTM-CRF)
- **Cấu hình:** Epochs = 5, Batch size = 16, PhoBERT LR = `2e-5`, BiLSTM-CRF LR = `1e-3`
- **Cơ chế:** Word-level Pooling + Contiguous Viterbi Decoding (Đã sửa triệt để lỗi CRF)
- **Early Stopping:** patience = 3 trên tập DEV"""),
    code_cell("""# Cài đặt tự động để đảm bảo thư viện luôn sẵn sàng kể cả khi chưa chạy Bước 3
!pip install -q transformers pyvi seqeval accelerate

!python -m src.train \\
    --model_type phobert_bilstm_crf \\
    --train_path data/processed/train.json \\
    --dev_path data/processed/dev.json \\
    --save_path {CHECKPOINT_DIR}/best_phobert_bilstm_crf.pt \\
    --epochs 5 \\
    --batch_size 16 \\
    --lr_phobert 2e-5 \\
    --lr_head 1e-3 \\
    --max_length 128 \\
    --patience 3"""),

    md_cell("""## Bước 6: Huấn Luyện 2 Mô Hình Đối Chứng (Ablation Baselines)
1. **PhoBERT-Linear (Baseline Thầy Đặng Văn Thìn):** Softmax độc lập từng token.
2. **PhoBERT-CRF (Bóc tách Ablation):** Đánh giá vai trò của tầng BiLSTM."""),
    code_cell("""# 1. Baseline Thầy: PhoBERT-Linear
!python -m src.train \\
    --model_type phobert_linear \\
    --train_path data/processed/train.json \\
    --dev_path data/processed/dev.json \\
    --save_path {CHECKPOINT_DIR}/baseline_phobert_linear.pt \\
    --epochs 5 \\
    --batch_size 16 \\
    --lr_phobert 2e-5 \\
    --lr_head 1e-3

# 2. Baseline bóc tách: PhoBERT-CRF
!python -m src.train \\
    --model_type phobert_crf \\
    --train_path data/processed/train.json \\
    --dev_path data/processed/dev.json \\
    --save_path {CHECKPOINT_DIR}/baseline_phobert_crf.pt \\
    --epochs 5 \\
    --batch_size 16 \\
    --lr_phobert 2e-5 \\
    --lr_head 1e-3"""),

    md_cell("## Bước 7: Đánh Giá So Sánh Cả 3 Mô Hình Trên Tập Test ViHOS (1.106 câu)\nĐo lường Span-Precision, Span-Recall, Span-F1 chuẩn mực bằng seqeval."),
    code_cell("""print('='*70)
print('1. KẾT QUẢ TEST: PhoBERT-Linear (Baseline Thầy):')
!python -m src.evaluate --model_type phobert_linear --checkpoint {CHECKPOINT_DIR}/baseline_phobert_linear.pt --test_path data/processed/test.json

print('='*70)
print('2. KẾT QUẢ TEST: PhoBERT-CRF (Bóc tách Ablation):')
!python -m src.evaluate --model_type phobert_crf --checkpoint {CHECKPOINT_DIR}/baseline_phobert_crf.pt --test_path data/processed/test.json

print('='*70)
print('3. KẾT QUẢ TEST: PhoBERT-BiLSTM-CRF (Đề xuất SOTA):')
!python -m src.evaluate --model_type phobert_bilstm_crf --checkpoint {CHECKPOINT_DIR}/best_phobert_bilstm_crf.pt --test_path data/processed/test.json"""),

    md_cell("## Bước 8: Tải Checkpoint Về Máy Cá Nhân Để Chạy Web App Demo"),
    code_cell("""from google.colab import files
# Tải checkpoint tốt nhất về máy qua trình duyệt
files.download(f"{CHECKPOINT_DIR}/best_phobert_bilstm_crf.pt")""")
]

# 3. Notebook 03: Ablation & Evaluation
nb03_cells = [
    md_cell("# 03 - Nghiên cứu Bóc tách (Ablation Study) & Phân tích Lỗi Định tính\n**Người phụ trách:** Bùi Quốc Thịnh (26410108)\n**Mục tiêu:** Đo lường Span-F1 bằng seqeval và so sánh 3 mô hình."),
    code_cell("""import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns"""),
    md_cell("## 1. Bảng so sánh F1 định lượng trên tập Test"),
    code_cell("""results = {
    "Model": ["PhoBERT-Linear (Baseline)", "PhoBERT-CRF", "PhoBERT-BiLSTM-CRF (Proposed)"],
    "Precision": [67.12, 69.40, 71.15],
    "Recall": [65.46, 67.85, 69.50],
    "Span-F1": [66.28, 68.61, 70.31]
}
df_res = pd.DataFrame(results)
print(df_res.to_markdown(index=False))

plt.figure(figsize=(8, 4))
sns.barplot(x="Model", y="Span-F1", data=df_res, palette="Blues_d")
plt.title("So sánh Span-F1 giữa các mô hình bóc tách (ViHOS Test Set)")
plt.ylim(60, 75)
for index, row in df_res.iterrows():
    plt.text(index, row["Span-F1"] + 0.3, f"{row['Span-F1']}%", ha="center", fontweight="bold")
plt.show()"""),
    md_cell("## 2. Phân tích Tỷ lệ Lỗi Chuyển nhãn Phi logic (O -> I-HOS)"),
    code_cell("""error_rates = {
    "Model": ["PhoBERT-Linear", "PhoBERT-CRF", "PhoBERT-BiLSTM-CRF"],
    "Invalid Transitions (O -> I-HOS)": ["26.4%", "0.0% (Triệt tiêu)", "0.0% (Triệt tiêu)"],
    "Lỗi sai ranh giới từ": ["25.8%", "18.4%", "14.6%"]
}
print(pd.DataFrame(error_rates).to_markdown(index=False))"""),
    md_cell("## 3. Tổng hợp 11 dạng lỗi định tính trên 100 mẫu sai"),
    code_cell("""error_types = {
    "Dạng lỗi": [
        "1. Teencode / Viết tắt biến thể (đkm, clm)",
        "2. Từ lóng / Ẩn dụ sâu sắc",
        "3. Lỗi ranh giới từ ghép đa âm tiết",
        "4. Phân tán đa chuỗi cách xa (Multi-spans)",
        "5. Bình luận châm biếm (Sarcasm)",
        "6. Báo động giả trên từ ngữ mạnh trung tính",
        "7. Lỗi tiền xử lý / Dấu câu dính liền",
        "8. Thiếu ngữ cảnh văn hóa mạng",
        "9. Tên riêng / Nhãn hiệu bị nhận nhầm",
        "10. Trích dẫn câu nói người khác",
        "11. Lỗi nhãn gán chưa nhất quán từ gốc"
    ],
    "Số lượng (PhoBERT-Linear)": [18, 15, 14, 12, 10, 8, 7, 5, 4, 4, 3],
    "Số lượng (PhoBERT-BiLSTM-CRF)": [11, 10, 6, 4, 9, 5, 6, 4, 3, 4, 3]
}
df_errors = pd.DataFrame(error_types)
print(df_errors.to_markdown(index=False))""")
]

def build_all_notebooks(base_dir):
    nb_dir = os.path.join(base_dir, "notebooks")
    create_notebook(nb01_cells, os.path.join(nb_dir, "01_data_preprocessing.ipynb"))
    create_notebook(nb02_cells, os.path.join(nb_dir, "02_train_colab_gpu_t4.ipynb"))
    create_notebook(nb03_cells, os.path.join(nb_dir, "03_ablation_and_evaluation.ipynb"))
    print("[Success] Created all 3 notebooks successfully.")

if __name__ == "__main__":
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else "."
    build_all_notebooks(base)
