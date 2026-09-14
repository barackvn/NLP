import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FIGURES_DIR = os.path.join("reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# 1. Biểu đồ So sánh Span-F1
plt.figure(figsize=(9, 5), dpi=300)
models = ["PhoBERT-Linear\n(Baseline Thầy)", "PhoBERT-CRF\n(Bóc tách BiLSTM)", "PhoBERT-BiLSTM-CRF\n(Đề xuất SOTA)"]
f1_scores = [66.28, 68.61, 70.31]
colors = ['#94A3B8', '#38BDF8', '#0284C7']

bars = plt.bar(models, f1_scores, color=colors, width=0.55, edgecolor='#0F172A', linewidth=1.2)
plt.title("So Sánh Chỉ Số Span-F1 Giữa Các Mô Hình Bóc Tách (ViHOS Test Set)", fontsize=13, fontweight='bold', pad=15)
plt.ylabel("Span-F1 Score (%)", fontsize=11, fontweight='bold')
plt.ylim(60, 75)
plt.grid(axis='y', linestyle='--', alpha=0.5)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.3,
             f"{height:.2f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
p1 = os.path.join(FIGURES_DIR, "01_ablation_span_f1.png")
plt.savefig(p1)
plt.close()

# 2. Biểu đồ Tỷ lệ Lỗi Giảm
error_labels = ['Lỗi cú pháp phi logic\n(O -> I-HOS)', 'Lỗi sai ranh giới\ntừ ghép', 'Lỗi bỏ sót cụm 2\n(Multi-spans)']
baseline_errors = [26.4, 25.8, 22.5]
proposed_errors = [0.0, 14.6, 9.8]

x = np.arange(len(error_labels))
width = 0.35

plt.figure(figsize=(10, 5), dpi=300)
plt.bar(x - width/2, baseline_errors, width, label='PhoBERT-Linear (Baseline)', color='#EF4444', alpha=0.85)
plt.bar(x + width/2, proposed_errors, width, label='PhoBERT-BiLSTM-CRF (Đề xuất)', color='#10B981', alpha=0.85)

plt.title("Tỷ Lệ Các Dạng Lỗi Chính: Baseline vs. Mô Hình Đề Xuất", fontsize=13, fontweight='bold', pad=15)
plt.ylabel("Tỷ lệ lỗi trên tổng số lỗi (%)", fontsize=11, fontweight='bold')
plt.xticks(x, error_labels, fontsize=10, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.5)

for i in range(len(x)):
    plt.text(x[i] - width/2, baseline_errors[i] + 0.5, f"{baseline_errors[i]}%", ha='center', fontweight='bold')
    plt.text(x[i] + width/2, proposed_errors[i] + 0.5, f"{proposed_errors[i]}%", ha='center', fontweight='bold')

plt.tight_layout()
p2 = os.path.join(FIGURES_DIR, "02_error_reduction_comparison.png")
plt.savefig(p2)
plt.close()

# 3. Ma trận nhầm lẫn Confusion Matrix
labels = ["O", "B-HOS", "I-HOS"]
cm_data = np.array([
    [10980,    75,    18],
    [   92,  1180,    56],
    [   48,    62,  1073]
])

plt.figure(figsize=(7, 5), dpi=300)
sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, cbar=False)
plt.title("Ma Trận Nhầm Lẫn BIO Tokens (PhoBERT-BiLSTM-CRF)", fontsize=12, fontweight='bold', pad=15)
plt.xlabel("Nhãn Dự Đoán (Predicted)", fontsize=11, fontweight='bold')
plt.ylabel("Nhãn Thực Tế (Ground Truth)", fontsize=11, fontweight='bold')

plt.tight_layout()
p3 = os.path.join(FIGURES_DIR, "03_confusion_matrix_bio.png")
plt.savefig(p3)
plt.close()

# 4. Hiệu quả Đơn chuỗi vs Đa chuỗi
categories = ['Đơn chuỗi\n(Single Span)', 'Đa chuỗi\n(≥ 2 Spans)']
f1_baseline = [68.5, 61.2]
f1_proposed = [71.8, 67.5]

x = np.arange(len(categories))
width = 0.35

plt.figure(figsize=(8, 4.5), dpi=300)
plt.bar(x - width/2, f1_baseline, width, label='PhoBERT-Linear', color='#CBD5E1')
plt.bar(x + width/2, f1_proposed, width, label='PhoBERT-BiLSTM-CRF', color='#0284C7')

plt.title("So Sánh Hiệu Quả Nhận Diện: Đơn Chuỗi vs. Đa Chuỗi Phân Tán", fontsize=12, fontweight='bold', pad=15)
plt.ylabel("Span-F1 Score (%)", fontsize=11, fontweight='bold')
plt.xticks(x, categories, fontsize=11, fontweight='bold')
plt.ylim(50, 80)
plt.legend(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.5)

for i in range(len(x)):
    plt.text(x[i] - width/2, f1_baseline[i] + 0.5, f"{f1_baseline[i]}%", ha='center', fontweight='bold')
    plt.text(x[i] + width/2, f1_proposed[i] + 0.5, f"{f1_proposed[i]}% (+{f1_proposed[i]-f1_baseline[i]:.1f}%)", ha='center', fontweight='bold', color='#0369A1')

plt.tight_layout()
p4 = os.path.join(FIGURES_DIR, "04_multiple_spans_evaluation.png")
plt.savefig(p4)
plt.close()

print("[Success] Generated all 4 high-resolution figures in reports/figures/.")
