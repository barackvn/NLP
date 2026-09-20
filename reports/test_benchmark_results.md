# Kết quả Đánh giá Benchmark Chính thức — Toàn bộ 1.106 câu Test ViHOS

*Thời gian thực hiện: 20/09/2026 17:06:11*

| Mô hình                                           | Loại                        |   Epoch tốt nhất |   Precision (%) |   Recall (%) |   Span-F1 (%) |   Lỗi O -> I-HOS (%) |   Lỗi lệch ranh giới (%) |   Thời gian eval (s) |
|:--------------------------------------------------|:----------------------------|-----------------:|----------------:|-------------:|--------------:|---------------------:|-------------------------:|---------------------:|
| 1. PhoBERT-Linear (Baseline Gốc)                  | phobert_linear              |                5 |           60.34 |        58.15 |         59.23 |                 0.82 |                    23.89 |                120.5 |
| 2. PhoBERT-CRF (Ablation Bóc Tách)                | phobert_crf                 |                4 |           63.37 |        59.26 |         61.24 |                 0.04 |                    23.94 |                121.2 |
| 3. PhoBERT-BiLSTM-CRF (Smoothing Bridge)          | phobert_bilstm_crf          |                3 |           65    |        59.65 |         62.21 |                 0.06 |                    22.93 |                127.3 |
| 4. PhoBERT-DualHead-BiLSTM-CRF (Đề Xuất Đa Nhiệm) | phobert_dualhead_bilstm_crf |                5 |           62.99 |        58.31 |         60.56 |                 0.13 |                    24.1  |                132.6 |

---
### Ghi chú minh chứng:
- Dữ liệu đánh giá: `data/processed/test.json` (toàn bộ 1.106 mẫu test độc lập, không rò rỉ tập huấn luyện/validation).
- Đánh giá cấp độ Span theo chuẩn IOB2 Exact Match (Precision, Recall, Span-F1).
- File log terminal đầy đủ: `reports/test_evaluation_official_log.txt`.
- File chi tiết dự đoán từng câu: `reports/test_predictions_<model>.json`.
