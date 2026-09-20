# Bảng đối chứng thực nghiệm (Ablation Study) — Tập Test ViHOS (1.106 câu)

| Mô hình                      | Cơ chế giải mã             |   Precision (%) |   Recall (%) |   Span-F1 (%) | Lỗi O -> I-HOS   |   Lỗi lệch ranh giới (%) |
|:-----------------------------|:---------------------------|----------------:|-------------:|--------------:|:-----------------|-------------------------:|
| 1. PhoBERT-Linear (Baseline) | Softmax độc lập            |           60.34 |        58.15 |         59.23 | 0.82% (110 lần)  |                    23.89 |
| 2. PhoBERT-CRF (Ablation)    | Viterbi toàn cục           |           63.37 |        59.26 |         61.24 | 0.04% (5 lần)    |                    23.94 |
| 3. PhoBERT-BiLSTM-CRF        | Viterbi + Smoothing        |           65    |        59.65 |         62.21 | 0.06% (8 lần)    |                    22.93 |
| 4. PhoBERT-DualHead          | Viterbi + Multi-Task Gated |           62.99 |        58.31 |         60.56 | 0.13% (18 lần)   |                    24.1  |

*Nguồn: Đối chiếu trực tiếp từ 4 file checkpoint trên toàn bộ tập test ViHOS, lưu tại reports/test_evaluation_official_log.txt.*
- **PhoBERT-BiLSTM-CRF** đạt Span-F1 cao nhất (**62,21%**), cải thiện **+2,98% F1** và **+4,66% Precision** so với PhoBERT-Linear gốc.
- **Tầng CRF** triệt tiêu hơn 92% lỗi cú pháp chuyển nhãn vi phạm nguyên tắc BIO (`O -> I-HOS`), giảm từ 110 lần (Linear) xuống còn 5–8 lần.
- **Nhánh DualHead** (60,56%) cải thiện so với Linear (+1,33%) nhưng dập tắt một số nhãn span biên do ngưỡng cố định 0.5, mở ra hướng tinh chỉnh threshold tuning.
