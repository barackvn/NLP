# Bảng đối chứng hiện tại — cập nhật 16/09/2026

| Mô hình | Precision (%) | Recall (%) | Span-F1 (%) | Lỗi O → I-HOS (%) | Lỗi ranh giới (%) |
|---|---:|---:|---:|---:|---:|
| PhoBERT–Linear | 67,12 | 65,46 | 66,28 | 26,4 | 25,8 |
| PhoBERT–CRF | 69,40 | 67,85 | 68,61 | 0,0 | 18,4 |
| PhoBERT–BiLSTM–CRF (DualHead) | 74,82 | 68,20 | 71,35 | 0,0 | 13,8 |

Nguồn: benchmark_data trong run_ablation_reports.py và hình 01–02 hiện có. Số liệu khai báo trong mã tạo báo cáo, chưa được xác minh lại bằng log test trong lần cập nhật tài liệu.

DualHead tăng 5,07 điểm F1 so với Linear, 2,74 điểm so với CRF. Lỗi ranh giới thấp hơn Linear 12,0 điểm phần trăm, không phải mức giảm tương đối 12%. Kết quả 0% lỗi chuyển nhãn không bảo đảm mô hình luôn tuân thủ BIO trên mọi câu.

Ma trận ở hình 03 có phạm vi khác: 100 câu đầu test, 1.186 từ. Hình 04 là thống kê 11 dạng lỗi trên 100 câu phân tích, không phải F1 đơn chuỗi/đa chuỗi.
