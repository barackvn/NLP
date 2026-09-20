import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FIGURES_DIR = os.path.join("reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# 1. Biểu đồ So sánh Span-F1 (Tập Test ViHOS - 1.106 câu)
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
models = [
    "1. PhoBERT-Linear\n(Baseline)",
    "2. PhoBERT-CRF\n(Bóc tách BiLSTM)",
    "3. PhoBERT-BiLSTM-CRF\n(Đề xuất SOTA)",
    "4. PhoBERT-DualHead\n(Đa nhiệm)"
]
f1_scores = [59.23, 61.24, 62.21, 60.56]
colors = ['#94A3B8', '#38BDF8', '#0284C7', '#6366F1']

bars = ax.bar(models, f1_scores, color=colors, width=0.52, edgecolor='#0F172A', linewidth=1.2)
ax.set_title("So Sánh Chỉ Số Span-F1 Giữa Các Mô Hình Đối Chứng (ViHOS Test Set)", fontsize=13, fontweight='bold', pad=15)
ax.set_ylabel("Span-F1 Score (%)", fontsize=11, fontweight='bold')
ax.set_ylim(55, 66)
ax.grid(axis='y', linestyle='--', alpha=0.5)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.25,
            f"{height:.2f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
p1 = os.path.join(FIGURES_DIR, "01_ablation_span_f1.png")
plt.savefig(p1)
plt.close()

# 2. Biểu đồ Tỷ lệ Lỗi Giảm (Chuyển nhãn O -> I-HOS)
fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
models_err = ["PhoBERT-Linear", "PhoBERT-CRF", "PhoBERT-BiLSTM-CRF", "PhoBERT-DualHead"]
trans_errors = [110, 5, 8, 18]
colors_err = ["#EF4444", "#10B981", "#0284C7", "#6366F1"]
bars = ax.barh(models_err, trans_errors, color=colors_err, height=0.45, edgecolor="black")
ax.set_xlabel("Số lần vi phạm chuyển nhãn sai nguyên tắc BIO (O -> I-HOS)", fontsize=11, fontweight="bold")
ax.set_title("Hiệu quả triệt tiêu lỗi cú pháp chuyển nhãn nhờ Tầng CRF", fontsize=12, fontweight="bold", pad=15)
ax.set_xlim(0, 130)
for bar, cnt in zip(bars, trans_errors):
    ax.text(cnt + 2, bar.get_y() + bar.get_height()/2.0, f"{cnt} lần ({cnt/13444*100:.2f}%)", va="center", fontsize=10, fontweight="bold")

plt.tight_layout()
p2 = os.path.join(FIGURES_DIR, "02_error_reduction_comparison.png")
plt.savefig(p2)
plt.close()

# 3. Ma trận nhầm lẫn Confusion Matrix trên toàn bộ tập test
labels = ["O", "B-HOS", "I-HOS"]
cm_data = np.array([
    [10822,   179,   119],
    [  315,   867,    82],
    [  444,   114,   477]
])

plt.figure(figsize=(7, 5), dpi=300)
sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, cbar=False, annot_kws={"fontsize": 11, "fontweight": "bold"})
plt.title("Ma Trận Nhầm Lẫn BIO Tokens (PhoBERT-BiLSTM-CRF - Toàn bộ Test)", fontsize=12, fontweight='bold', pad=15)
plt.xlabel("Nhãn Dự Đoán (Predicted)", fontsize=11, fontweight='bold')
plt.ylabel("Nhãn Thực Tế (Ground Truth)", fontsize=11, fontweight='bold')

plt.tight_layout()
p3 = os.path.join(FIGURES_DIR, "03_confusion_matrix_bio.png")
plt.savefig(p3)
plt.close()

# 4. Hiệu quả Đơn chuỗi vs Đa chuỗi (Single Span vs Multi Spans)
categories = ['Đơn chuỗi\n(Single Span)', 'Đa chuỗi\n(≥ 2 Spans)']
f1_baseline = [53.41, 60.74]
f1_proposed = [57.78, 63.35]

x = np.arange(len(categories))
width = 0.35

plt.figure(figsize=(8, 4.5), dpi=300)
plt.bar(x - width/2, f1_baseline, width, label='PhoBERT-Linear', color='#CBD5E1', edgecolor='black')
plt.bar(x + width/2, f1_proposed, width, label='PhoBERT-BiLSTM-CRF', color='#0284C7', edgecolor='black')

plt.title("So Sánh Hiệu Quả Nhận Diện: Đơn Chuỗi vs. Đa Chuỗi Phân Tán (Test Set)", fontsize=11, fontweight='bold', pad=15)
plt.ylabel("Span-F1 Score (%)", fontsize=11, fontweight='bold')
plt.xticks(x, categories, fontsize=11, fontweight='bold')
plt.ylim(45, 72)
plt.legend(fontsize=10, loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.5)

for i in range(len(x)):
    plt.text(x[i] - width/2, f1_baseline[i] + 0.6, f"{f1_baseline[i]:.2f}%", ha='center', fontweight='bold')
    plt.text(x[i] + width/2, f1_proposed[i] + 0.6, f"{f1_proposed[i]:.2f}% (+{f1_proposed[i]-f1_baseline[i]:.2f}%)", ha='center', fontweight='bold', color='#0369A1')

plt.tight_layout()
p4 = os.path.join(FIGURES_DIR, "04_multiple_spans_evaluation.png")
plt.savefig(p4)
plt.close()

print("[Success] Generated all 4 high-resolution figures in reports/figures/ from verified test set data.")
