# BÁO CÁO ĐỒ ÁN ViHOS
## Nhận diện cụm xúc phạm tiếng Việt bằng PhoBERT và mô hình DualHead–BiLSTM–CRF

**Bản cập nhật:** 16/09/2026, đồng bộ bốn hình trong `reports/figures` và bảng `ablation_table.md`.

- **Đơn vị:** Trường Đại học Công nghệ Thông tin – ĐHQG TP.HCM.
- **Giảng viên hướng dẫn theo hồ sơ nhóm:** Đặng Văn Thìn; tác giả tài liệu tham khảo: Trần Quốc Khánh.
- **Thành viên:** Dương Quốc Thương (26410127), Nông Nguyễn Thành (26410115), Hoàng Võ Minh Tuấn (26410146), Bùi Quốc Thịnh (26410108), Trần Tiến Dũng (26410024).

## Tóm tắt

Đồ án tìm vị trí các cụm xúc phạm trong bình luận tiếng Việt. Phiên bản báo cáo hiện tại so sánh ba mô hình: **PhoBERT–Linear**, **PhoBERT–CRF** và **PhoBERT–BiLSTM–CRF (DualHead)**. Tên cuối tương ứng lớp `PhoBERT_DualHead_BiLSTM_CRF` trong mã nguồn.

Theo bảng số liệu dùng tạo hình, ba mô hình có Span-F1 lần lượt **66,28%**, **68,61%** và **71,35%**. DualHead cao hơn Linear **5,07 điểm phần trăm**, cao hơn CRF **2,74 điểm phần trăm**. Bộ hình mới còn trình bày lỗi ranh giới, 11 dạng lỗi định tính và ma trận nhầm lẫn BIO trên 100 câu đầu của tập test.

**Phạm vi bằng chứng:** Bảng so sánh và thống kê 11 dạng lỗi là các giá trị khai báo trong `run_ablation_reports.py`. Phần tạo ma trận của tệp này chạy checkpoint DualHead trên `Subset(test_ds, range(100))`. Lần cập nhật tài liệu này đọc hình và đối chiếu dữ liệu, không chạy lại huấn luyện hoặc suy luận. Không dùng ma trận 100 câu để xác nhận Span-F1 toàn tập test.

---

## 1. Bài toán và lý do thực hiện

Khi có nhiều bình luận, kiểm tra thủ công từng câu mất thời gian. Từ lóng, viết tắt và ngữ cảnh khiến việc xác định nội dung xúc phạm khó hơn việc tìm một danh sách từ khóa.

Phân loại cấp câu trả lời bình luận có xúc phạm hay không. Đồ án tập trung vào mức chi tiết hơn: **tìm đúng từ hoặc cụm xúc phạm**, hạn chế bỏ sót và đánh dấu nhầm. Đầu ra giúp người kiểm duyệt biết phần nào cần xem lại hoặc có thể che bằng `***`. Việc xóa cả câu là lựa chọn của ứng dụng, không phải điều bắt buộc của phương pháp phân loại câu.

- **Đầu vào:** một bình luận tiếng Việt.
- **Đầu ra mô hình:** nhãn BIO theo từ.
- **Đầu ra ứng dụng:** cụm được phát hiện và văn bản đã đánh dấu hoặc che.

### 1.1. Nhãn BIO

| Nhãn | Ý nghĩa |
|---|---|
| B-HOS | Từ đầu của một cụm xúc phạm |
| I-HOS | Từ tiếp theo trong cùng cụm |
| O | Từ ngoài cụm |

Ví dụ minh họa: `Bạn / nói / thật / ngu` có thể biểu diễn bằng `O / O / O / B-HOS`. Một câu có thể có nhiều cụm riêng biệt; mỗi cụm bắt đầu bằng B-HOS. Đây là ví dụ giải thích định dạng nhãn, không phải phép đo mô hình.

## 2. Dữ liệu

| Tập | Số câu | Số từ trong JSON | Vai trò |
|---|---:|---:|---|
| Train | 8.844 | 108.430 | Học trọng số |
| Dev | 1.106 | 13.948 | Chọn cấu hình và checkpoint |
| Test | 1.106 | 13.424 | Đánh giá cuối cùng |

Tổng cộng có 11.056 câu. Nhãn O chiếm khoảng 82,37% số từ trong ba tệp JSON. Vì vậy, độ đúng cấp từ có thể cao ngay cả khi mô hình bỏ sót cụm; cần thêm thước đo ở cấp cụm.

Mã hiện tại lưu vị trí subword đầu tiên của mỗi từ bằng `word_indices`, gom biểu diễn tại các vị trí này và dùng `word_mask` cho chuỗi từ. Giới hạn mặc định là 128 token đầu vào, gồm token đặc biệt. Câu dài có thể bị cắt; không đồng nhất số từ với số subword.

Riêng **100 câu đầu của test.json có 1.186 từ**, gồm 936 O, 124 B-HOS và 126 I-HOS. Các tổng này khớp tổng hàng của ma trận mới.

## 3. Phương pháp

### 3.1. Ba mô hình trong bộ so sánh hiện tại

| Mô hình | Cách dự đoán |
|---|---|
| PhoBERT–Linear | PhoBERT tạo biểu diễn có ngữ cảnh; Linear chọn nhãn ở từng vị trí |
| PhoBERT–CRF | Thêm điểm chuyển nhãn và giải mã cả chuỗi bằng Viterbi |
| PhoBERT–BiLSTM–CRF (DualHead) | Nhánh tìm cụm dùng BiLSTM–Linear–CRF; nhánh cấp câu hỗ trợ lọc nhãn |

PhoBERT đã cung cấp ngữ cảnh trong cả ba mô hình. Không mô tả Linear là mô hình hoàn toàn không biết từ trước hoặc sau. Bộ so sánh mới không có một dòng riêng cho BiLSTM–CRF đơn đầu, nên không dùng nó để tách riêng đóng góp của BiLSTM và Intent Head. Muốn làm điều đó cần thêm đối chứng.

### 3.2. Chức năng từng tầng

| Phần | Thành phần | Chức năng dễ hiểu | Mục tiêu |
|---|---|---|---|
| Tầng 1 | PhoBERT | Đọc từ trong ngữ cảnh, tạo vector 768 chiều | Phân biệt nghĩa theo câu |
| Tầng 2, đầu 1 | BiLSTM | Xử lý chuỗi theo hai hướng | Kết nối thông tin trước và sau |
| Tầng 3, đầu 1 | Linear và CRF | Linear tạo điểm nhãn; CRF chọn chuỗi có điểm cao | Hỗ trợ dự đoán ranh giới và quan hệ nhãn |
| Đầu 2 | Intent Head | Dùng vector vị trí đầu qua MLP để kiểm tra câu có cụm xúc phạm | Bổ sung quyết định cấp câu |
| Cổng kết hợp | Gated Intent Fusion | P < 0,5: đổi nhãn về O; P ≥ 0,5: giữ nhãn | Lọc cụm nghi bị đánh dấu nhầm |

CRF học điểm chuyển nhãn; mã hiện tại chưa đặt ràng buộc BIO cứng. Vì vậy, kết quả 0% lỗi chuyển nhãn trong bảng không có nghĩa mọi đầu vào mới đều hợp lệ. Cổng cũng có thể bỏ sót cụm thật nếu đầu cấp câu dự đoán sai.

### 3.3. Huấn luyện dual head

Hai đầu dùng chung PhoBERT. Nhãn của đầu cấp câu mặc định được suy từ chuỗi BIO: có ít nhất một B hoặc I thì nhãn câu là 1. Đây là nhãn **có cụm xúc phạm**, chưa phải phép đo trực tiếp ý định tâm lý của người viết.

`L_total = L_CRF + 0,5 × L_BCE`

Hệ số 0,5 trong hàm học và ngưỡng 0,5 khi dự đoán có hai vai trò khác nhau. Mã huấn luyện dùng AdamW; mặc định learning rate PhoBERT 2×10⁻⁵, phần đầu 10⁻³, batch 16, tối đa 5 epoch, seed 42, dropout 0,3, warmup 10%, weight decay 0,01. Checkpoint được chọn theo Span-F1 dev. Đây là cấu hình mặc định trong mã, không thay thế cấu hình của một lần chạy đã lưu.

### 3.4. Ví dụ ngữ cảnh và phạm vi triển khai

Với câu “Con chó này đẹp, Mày đúng là con chó”, mục tiêu là giữ phần nói về thú cưng và phát hiện phần công kích người khác.

Để giải thích cổng khi xử lý riêng từng vế, có thể **giả sử** vế đầu có P = 0,05 và vế sau P = 0,98. Cổng sẽ bỏ nhãn ở vế đầu và giữ nhãn tìm được ở vế sau. Đây là minh họa; không khẳng định Intent Head đã đo hai xác suất đó.

Trong backend hiện tại, `apply_clause_gated_fusion` dùng luật từ khóa ở từng vế và gán các điểm cố định như 0,05 hoặc 0,98. Cổng của mạng DualHead áp dụng cho mỗi đầu vào. Muốn mạng đánh giá riêng từng vế phải tách vế, chạy mạng và ghép lại đúng vị trí. Khi đo hiệu quả cần tách tác dụng của mạng với tác dụng của luật.

## 4. Kết quả và diễn giải bốn hình mới

### 4.1. Thước đo

- **Precision:** tỷ lệ cụm đúng trong các cụm đã dự đoán.
- **Recall:** tỷ lệ cụm thật mà mô hình tìm đúng.
- **Span-F1:** `2PR/(P+R)`. Cụm phải khớp ranh giới và loại nhãn theo tiêu chí đánh giá.
- **Ma trận BIO:** số đếm ở cấp từ, không phải số cụm và không trực tiếp cho Span-F1.

F1 không phải tỷ lệ câu hoàn toàn đúng. Các điểm phần trăm là hiệu giữa hai tỷ lệ phần trăm; mức giảm tương đối phải chia thêm cho giá trị ban đầu.

### 4.2. Bảng đối chứng cập nhật

| Mô hình | Precision (%) | Recall (%) | Span-F1 (%) | Lỗi chuyển nhãn (%) | Lỗi ranh giới (%) |
|---|---:|---:|---:|---:|---:|
| PhoBERT–Linear | 67,12 | 65,46 | 66,28 | 26,4 | 25,8 |
| PhoBERT–CRF | 69,40 | 67,85 | 68,61 | 0,0 | 18,4 |
| PhoBERT–BiLSTM–CRF (DualHead) | 74,82 | 68,20 | 71,35 | 0,0 | 13,8 |

Nguồn: `ablation_table.md` và `benchmark_data` trong `run_ablation_reports.py`. Các giá trị hiện được khai báo trong mã tạo báo cáo; cần log và dự đoán test để xác nhận kết quả checkpoint trên toàn bộ test. Mẫu số của hai tỷ lệ lỗi cần được ghi rõ trong hồ sơ đánh giá, không tự coi là số câu test.

DualHead cao hơn Linear 5,07 điểm F1 và cao hơn CRF 2,74 điểm. So với CRF, Precision tăng 5,42 điểm và Recall tăng 0,35 điểm. Không còn dùng nhận xét “Recall giảm so với BiLSTM–CRF” trong bộ đối chứng ba mô hình này vì không có dòng đơn đầu tương ứng.

### 4.3. Hình 1 — Span-F1 của ba mô hình hiện tại

![Span-F1](figures/01_ablation_span_f1.png)

Ba cột lần lượt là 66,28%; 68,61%; 71,35%. Cột cuối đã là **DualHead**, không phải BiLSTM–CRF đơn đầu. Trục tung bắt đầu tại 60%; đọc mức chênh lệch bằng số, không suy ra mức tăng từ tỷ lệ chiều cao cột.

### 4.4. Hình 2 — Lỗi ranh giới từ ghép

![Lỗi ranh giới](figures/02_error_reduction_comparison.png)

Tỷ lệ trên hình là 25,8% ở Linear, 18,4% ở CRF và 13,8% ở DualHead. DualHead thấp hơn Linear **12,0 điểm phần trăm**, tương đương giảm tương đối khoảng **46,5%** nếu cùng mẫu số và cách đo. Nhãn “Giảm 12,0%” trên hình đang biểu diễn phép trừ hai tỷ lệ; trong báo cáo và lời nói phải đọc là **12,0 điểm phần trăm**. Với CRF, chênh lệch tương ứng là 7,4 điểm.

Hình mới chỉ trình bày lỗi ranh giới. Không dùng nó để diễn giải ba nhóm lỗi chuyển nhãn, ranh giới và bỏ sót cụm thứ hai như bản hình trước.

### 4.5. Hình 3 — Ma trận BIO trên 100 câu đầu của test

![Ma trận BIO](figures/03_confusion_matrix_bio.png)

| Nhãn thật / Dự đoán | O | B-HOS | I-HOS | Tổng |
|---|---:|---:|---:|---:|
| O | 908 | 14 | 14 | 936 |
| B-HOS | 25 | 89 | 10 | 124 |
| I-HOS | 34 | 15 | 77 | 126 |
| Tổng | 967 | 118 | 101 | 1.186 |

Mã sinh hình dùng checkpoint `best_phobert_dualhead_bilstm_crf.pt` và **100 câu đầu**, không phải mẫu ngẫu nhiên hoặc toàn bộ 1.106 câu test. Tổng hàng khớp dữ liệu JSON của 100 câu này.

- Có 25 + 34 = **59 từ thuộc cụm** bị dự đoán thành O.
- Có 14 + 14 = **28 từ O** bị dự đoán thành B hoặc I.
- Có 10 + 15 = **25 từ bị nhầm giữa B và I**.
- Đường chéo có 1.074 từ đúng trong 1.186 từ, tương đương **90,56% độ đúng cấp từ trên phần dữ liệu này**. Đây không phải Span-F1 hoặc độ đúng toàn câu.

Không suy ra F1 71,35% từ ma trận này, vì đơn vị và phạm vi khác nhau.

### 4.6. Hình 4 — 11 dạng lỗi định tính

![11 dạng lỗi](figures/04_multiple_spans_evaluation.png)

Tên tệp cũ vẫn là `04_multiple_spans_evaluation.png`, nhưng **nội dung mới là 11 dạng lỗi trên 100 câu phân tích**, không phải F1 đơn chuỗi/đa chuỗi.

| Dạng lỗi | Linear | DualHead | Giảm số mẫu lỗi |
|---|---:|---:|---:|
| Teencode / viết tắt | 18 | 11 | 7 |
| Từ lóng / ẩn dụ | 15 | 10 | 5 |
| Ranh giới từ ghép | 14 | 6 | 8 |
| Đa chuỗi cách xa | 12 | 4 | 8 |
| Châm biếm | 10 | 8 | 2 |
| Đánh dấu nhầm từ động vật lành tính | 8 | 1 | 7 |
| Tiền xử lý / dấu câu | 7 | 6 | 1 |
| Thiếu ngữ cảnh văn hóa mạng | 5 | 4 | 1 |
| Tên riêng / nhãn hiệu | 4 | 3 | 1 |
| Trích dẫn lời người khác | 4 | 4 | 0 |
| Nhãn gốc chưa nhất quán | 3 | 3 | 0 |

Nguồn là `error_analysis_data` trong mã tạo hình. Chưa có danh sách mẫu để xác nhận đây là cùng 100 câu của ma trận hoặc các nhóm có loại trừ nhau hay không; không cộng các hàng thành tỷ lệ lỗi toàn bộ test.

Nhóm lỗi đánh dấu nhầm từ động vật giảm 8 còn 1, tức giảm tương đối **87,5% trong nhóm này**, nhưng vẫn còn một mẫu lỗi. Do đó, không kết luận “loại bỏ hoàn toàn báo động giả”. Trích dẫn và nhãn gốc chưa nhất quán chưa giảm trong thống kê này.

### 4.7. Ca từ lóng ở bản đơn đầu trước đây

Báo cáo cũ ghi câu “Món ăn của quán này bình thường nhưng giá cả hơi đắt như con ket”: Linear và CRF tìm được cụm, còn BiLSTM–CRF **chưa có DualHead** bỏ sót. Ca này được giữ làm ví dụ về giới hạn của điểm tổng hợp, không phải kết quả thử DualHead mới. Giả thuyết làm mượt quá mức chưa có thí nghiệm xác nhận cơ chế.

## 5. Ứng dụng

Giao diện React gửi bình luận đến FastAPI. Engine nạp checkpoint, trả nhãn và cụm để hiển thị hoặc che bằng `***`. Backend chọn lớp DualHead khi trọng số có phần `intent_head`; cần hiển thị rõ phiên bản mô hình thực tế.

Ứng dụng còn có bộ lọc từng vế bằng luật và chế độ dự phòng khi không tải được checkpoint. Các nhóm như INSULT hoặc THREAT do hậu xử lý bổ sung, không phải các lớp BIO của mô hình hiện tại. Khi báo cáo độ trễ cần đo trên nhiều câu, ghi máy chạy và phân biệt lần tải đầu với suy luận sau khi tải; không lấy một ca làm độ trễ trung bình.

## 6. Hạn chế và hướng cải thiện

1. **Kết quả cần truy vết:** lưu cấu hình, checkpoint, nhãn dự đoán và log trên toàn bộ test. Bảng khai báo và hình minh họa không thay thế bằng chứng chạy.
2. **Từ lóng và teencode:** bổ sung biến thể từ train, kiểm tra lại nhãn, đánh giá riêng câu thông thường và câu biến thể.
3. **Cổng DualHead:** thử ngưỡng trên dev, so sánh bật/tắt cổng, đo cả Precision và Recall. Với nhiều vế cần bảo toàn vị trí khi tách và ghép.
4. **Tách tác dụng mô hình và luật:** so sánh cùng dữ liệu khi bật/tắt lọc luật. Thêm BiLSTM–CRF đơn đầu nếu muốn đo riêng đóng góp Intent Head.
5. **Châm biếm và trích dẫn:** bổ sung ngữ cảnh, thống nhất tiêu chí gán nhãn. Hình mới vẫn ghi 8 lỗi châm biếm và 4 lỗi trích dẫn.
6. **Nhãn dữ liệu:** rà soát các trường hợp chưa nhất quán; không kỳ vọng một tầng mô hình tự sửa được nhãn gốc sai.

## 7. Kết luận

Bộ đối chứng mới gồm Linear, CRF và BiLSTM–CRF có DualHead. Theo số liệu báo cáo, DualHead có Span-F1 71,35%, cao hơn Linear 5,07 điểm phần trăm; lỗi ranh giới giảm từ 25,8% còn 13,8%. Thống kê 11 dạng lỗi cho thấy một số nhóm cải thiện, trong khi trích dẫn và nhãn chưa nhất quán chưa giảm.

Ma trận mới mô tả riêng 100 câu đầu test với 1.186 từ. Kết quả này hữu ích để đọc loại lỗi, không thay thế phép đánh giá Span-F1 toàn test. Hệ thống hướng đến hỗ trợ người kiểm duyệt và vẫn cần xác minh kết quả, cải thiện dữ liệu, phân biệt rõ mô hình học với bộ lọc luật.

## Nguồn và các tệp đồng bộ

- `reports/figures/01_ablation_span_f1.png`: F1 ba mô hình hiện tại.
- `reports/figures/02_error_reduction_comparison.png`: lỗi ranh giới.
- `reports/figures/03_confusion_matrix_bio.png`: ma trận trên 100 câu đầu test.
- `reports/figures/04_multiple_spans_evaluation.png`: 11 dạng lỗi, dù tên tệp giữ từ phiên bản cũ.
- `run_ablation_reports.py`: số liệu khai báo và phạm vi tạo ma trận.
- `src/model.py`, `src/dataset.py`, `src/train.py`, `src/evaluate.py`: triển khai mô hình và đánh giá.
- `backend/inference.py`, `backend/main.py`: triển khai ứng dụng và bộ lọc luật.
- `SLIDES_THUYET_TRINH_18_TRANG.md`: nội dung trình chiếu đồng bộ.

**Tài liệu tham khảo trong hồ sơ dự án:** Tran và cộng sự (2023), *ViHOS: Vietnamese Hate and Offensive Spans Detection*; Nguyen và Nguyen (2020), *PhoBERT*; Lample và cộng sự (2016), *Neural Architectures for Named Entity Recognition*; Lafferty và cộng sự (2001), *Conditional Random Fields*. Không sử dụng danh sách này để suy ra tuyên bố mới hoàn toàn hoặc SOTA khi chưa có đối chứng tương đương.

