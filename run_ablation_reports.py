import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("CHƯƠNG TRÌNH SINH BÁO CÁO & BIỂU ĐỒ ABLATION STUDY (ViHOS)")
print("="*70)

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
FIG_DIR = os.path.join(ROOT_DIR, "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
print(f"📁 Thư mục lưu biểu đồ: {FIG_DIR}")

# 1. Bảng số liệu Ablation Study Benchmark - ĐỐI CHIẾU THỰC TẾ TRÊN TOÀN BỘ 1.106 CÂU TẬP TEST
benchmark_data = {
    "Mô hình": [
        "1. PhoBERT-Linear (Baseline)",
        "2. PhoBERT-CRF (Ablation)",
        "3. PhoBERT-BiLSTM-CRF",
        "4. PhoBERT-DualHead"
    ],
    "Cơ chế giải mã": [
        "Softmax độc lập",
        "Viterbi toàn cục",
        "Viterbi + Smoothing",
        "Viterbi + Multi-Task Gated"
    ],
    "Precision (%)": [60.34, 63.37, 65.00, 62.99],
    "Recall (%)": [58.15, 59.26, 59.65, 58.31],
    "Span-F1 (%)": [59.23, 61.24, 62.21, 60.56],
    "Lỗi O -> I-HOS": ["0.82% (110 lần)", "0.04% (5 lần)", "0.06% (8 lần)", "0.13% (18 lần)"],
    "Lỗi lệch ranh giới (%)": [23.89, 23.94, 22.93, 24.10]
}

df_benchmark = pd.DataFrame(benchmark_data)
print("\n[1] BẢNG SO SÁNH BENCHMARK TẬP TEST (ĐỐI CHIẾU THỰC NGHIỆM):")
print(df_benchmark.to_string(index=False))

# Xuất file markdown bảng
ablation_md_path = os.path.join(ROOT_DIR, "reports", "ablation_table.md")
with open(ablation_md_path, "w", encoding="utf-8") as f:
    f.write("# Bảng đối chứng thực nghiệm (Ablation Study) — Tập Test ViHOS (1.106 câu)\n\n")
    f.write(df_benchmark.to_markdown(index=False))
    f.write("\n\n*Nguồn: Đối chiếu trực tiếp từ 4 file checkpoint trên toàn bộ tập test ViHOS, lưu tại reports/test_evaluation_official_log.txt.*\n")
    f.write("- **PhoBERT-BiLSTM-CRF** đạt Span-F1 cao nhất (**62,21%**), cải thiện **+2,98% F1** và **+4,66% Precision** so với PhoBERT-Linear gốc.\n")
    f.write("- **Tầng CRF** triệt tiêu hơn 92% lỗi cú pháp chuyển nhãn vi phạm nguyên tắc BIO (`O -> I-HOS`), giảm từ 110 lần (Linear) xuống còn 5–8 lần.\n")
    f.write("- **Nhánh DualHead** (60,56%) cải thiện so với Linear (+1,33%) nhưng dập tắt một số nhãn span biên do ngưỡng cố định 0.5, mở ra hướng tinh chỉnh threshold tuning.\n")
print(f"✅ Đã lưu file markdown: {ablation_md_path}")

# 2. Biểu đồ so sánh Span-F1
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#94A3B8", "#38BDF8", "#0284C7", "#6366F1"]
bars = ax.bar(df_benchmark["Mô hình"], df_benchmark["Span-F1 (%)"], color=colors, width=0.52, edgecolor="black", linewidth=1.2)
ax.set_ylim(55, 66)
ax.set_ylabel("Span-F1 Score (%)", fontsize=12, fontweight="bold")
ax.set_title("So sánh Span-F1 giữa 4 mô hình đối chứng trên Toàn bộ Tập Test ViHOS", fontsize=13, fontweight="bold", pad=15)
plt.xticks(fontsize=10, fontweight="bold")
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.25, f"{yval:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
plt.tight_layout()
fig_f1_path = os.path.join(FIG_DIR, "01_ablation_span_f1.png")
plt.savefig(fig_f1_path, dpi=300)
plt.close()
print(f"✅ Đã tạo biểu đồ F1: {fig_f1_path}")

# 3. Biểu đồ so sánh tỷ lệ lỗi cú pháp chuyển nhãn phi logic (O -> I-HOS)
fig, ax = plt.subplots(figsize=(9, 4.5))
models = ["PhoBERT-Linear", "PhoBERT-CRF", "PhoBERT-BiLSTM-CRF", "PhoBERT-DualHead"]
trans_errors = [110, 5, 8, 18]
colors_err = ["#EF4444", "#10B981", "#0284C7", "#6366F1"]
bars = ax.barh(models, trans_errors, color=colors_err, height=0.45, edgecolor="black")
ax.set_xlabel("Số lần vi phạm chuyển nhãn sai nguyên tắc BIO (O -> I-HOS)", fontsize=11, fontweight="bold")
ax.set_title("Hiệu quả triệt tiêu lỗi cú pháp chuyển nhãn nhờ Tầng CRF", fontsize=12, fontweight="bold", pad=15)
ax.set_xlim(0, 130)
for bar, cnt in zip(bars, trans_errors):
    ax.text(cnt + 2, bar.get_y() + bar.get_height()/2.0, f"{cnt} lần ({cnt/13444*100:.2f}%)", va="center", fontsize=10, fontweight="bold")
plt.tight_layout()
fig_err_path = os.path.join(FIG_DIR, "02_error_reduction_comparison.png")
plt.savefig(fig_err_path, dpi=300)
plt.close()
print(f"✅ Đã tạo biểu đồ giảm lỗi chuyển nhãn: {fig_err_path}")

# 4. Biểu đồ Ma trận nhầm lẫn chuẩn xác trên TOÀN BỘ 1.106 CÂU TẬP TEST (PhoBERT-BiLSTM-CRF)
cm_data = np.array([
    [10822,   179,   119],
    [  315,   867,    82],
    [  444,   114,   477]
])
labels_order = ["O", "B-HOS", "I-HOS"]
cm_df = pd.DataFrame(cm_data, index=[f"True_{l}" for l in labels_order], columns=[f"Pred_{l}" for l in labels_order])

plt.figure(figsize=(7, 5.5))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"fontsize": 11, "fontweight": "bold"})
plt.title("Ma Trận Nhầm Lẫn BIO Tokens — PhoBERT-BiLSTM-CRF (Toàn bộ Test Set)", fontsize=11, fontweight="bold", pad=15)
plt.ylabel("Nhãn Thực Tế (Ground Truth)", fontsize=11, fontweight="bold")
plt.xlabel("Nhãn Dự Đoán (Predicted)", fontsize=11, fontweight="bold")
plt.tight_layout()
fig_cm_path = os.path.join(FIG_DIR, "03_confusion_matrix_bio.png")
plt.savefig(fig_cm_path, dpi=300)
plt.close()
print(f"✅ Đã tạo ma trận nhầm lẫn toàn bộ test set: {fig_cm_path}")

# 5. Biểu đồ Phân tích Đơn chuỗi vs Đa chuỗi (Single Span vs Multi Spans)
categories = ['Đơn chuỗi (Single Span)', 'Đa chuỗi (≥ 2 Spans)']
f1_baseline = [53.41, 60.74]
f1_proposed = [57.78, 63.35]

x = np.arange(len(categories))
width = 0.35

plt.figure(figsize=(8, 4.5), dpi=300)
plt.bar(x - width/2, f1_baseline, width, label='PhoBERT-Linear (Baseline)', color='#CBD5E1', edgecolor='black')
plt.bar(x + width/2, f1_proposed, width, label='PhoBERT-BiLSTM-CRF (Đề xuất)', color='#0284C7', edgecolor='black')

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
fig_dist_path = os.path.join(FIG_DIR, "04_multiple_spans_evaluation.png")
plt.savefig(fig_dist_path, dpi=300)
plt.close()
print(f"✅ Đã tạo biểu đồ đơn chuỗi vs đa chuỗi: {fig_dist_path}")

print("\n" + "="*70)
print("🎉 HOÀN THÀNH TẤT CẢ! TOÀN BỘ 4 BIỂU ĐỒ CHUẨN XÁC ĐÃ ĐƯỢC LƯU VÀO reports/figures/")
print("="*70)
