# BỘ SLIDE BẢO VỆ ĐỒ ÁN MÔN HỌC / TỐT NGHIỆP (18 TRANG CHUẨN HỘI ĐỒNG)
## ĐỀ TÀI: CẢI TIẾN PHƯƠNG PHÁP NHẬN DIỆN CHUỖI NGÔN NGỮ XÚC PHẠM TIẾNG VIỆT (ViHOS) BẰNG MÔ HÌNH KẾT HỢP PhoBERT-BiLSTM-CRF

* **Cơ sở đào tạo:** Trường Đại học Công nghệ Thông tin – ĐHQG-HCM (UIT).
* **Giảng viên hướng dẫn:** NCS.ThS. Đặng Văn Thìn & Tác giả Trần Quốc Khánh.
* **Bộ dữ liệu chuẩn:** ViHOS Benchmark (EACL 2023).

---

### PHẦN I: TỔNG QUAN & BỐI CẢNH NGHIÊN CỨU
*Người trình bày: **Dương Quốc Thương** (26410127) - Project Leader*

#### SLIDE 01: Trang Tiêu Đề & Giới Thiệu Nhóm
* **Tên đề tài:** Cải tiến nhận diện chuỗi ngôn ngữ xúc phạm tiếng Việt (ViHOS) bằng mô hình PhoBERT-BiLSTM-CRF.
* **Khoa/Bộ môn:** Khoa Khoa học Máy tính / Bộ môn Xử lý Ngôn ngữ Tự nhiên.
* **Phân công 5 thành viên:**
  1. **Dương Quốc Thương (26410127) - Trưởng nhóm:** Thiết kế kiến trúc tổng thể, quản trị repo, viết báo cáo & Slide 01-04, 18.
  2. **Nông Nguyễn Thành (26410115) - Data & NLP Core:** Xử lý tập ViHOS, Subword Alignment, Slide 05-09.
  3. **Hoàng Võ Minh Tuấn (26410146) - Trainer:** Setup Colab GPU T4, Differential LR, lưu checkpoint, Slide 10-11.
  4. **Bùi Quốc Thịnh (26410108) - Evaluation & Metrics:** Chạy ablation baselines, đo Span-F1 seqeval, phân loại 11 dạng lỗi, Slide 12-15.
  5. **Trần Tiến Dũng (26410024) - Product Developer:** Xây dựng Web App CPU Streamlit, tích hợp Auto-Masking, Slide 16-17.

#### SLIDE 02: Thực Trạng & Bài Toán Toxic Spans Detection
* **Vấn nạn phát ngôn thù ghét:** Mạng xã hội Việt Nam bùng nổ các bình luận công kích, xúc phạm cá nhân và tổ chức.
* **Hạn chế của phân loại câu (Sentence-level Classification):**
  * Chỉ gán nhãn thô toàn câu (0: Sạch, 1: Xúc phạm).
  * Buộc hệ thống phải xóa cả câu, làm mất đi mạch thảo luận lành mạnh xung quanh.
* **Toxic Spans Detection (Sequence Labeling cấp độ từ):**
  * Xác định chính xác vị trí bắt đầu và kết thúc của từng cụm từ xúc phạm.
  * Cho phép tự động che mờ (Auto-Masking `***`) và hỗ trợ kiểm duyệt viên duyệt nhanh x10 lần.

#### SLIDE 03: Điểm Nghẽn Của Baseline Hiện Hành (PhoBERT-Linear)
* **Mô hình gốc của Thầy:** Dùng PhoBERT kết hợp tầng Linear Classification tại từng token độc lập.
* **3 Nút thắt cổ chai:**
  1. **Quyết định nhãn cục bộ:** Không học được quan hệ phụ thuộc giữa các nhãn liền kề.
  2. **Lỗi chuyển nhãn phi logic:** Thường xuyên sinh nhãn `O` $\to$ `I-HOS` (Inside mà không có Begin).
  3. **Lỗi sai ranh giới từ chiếm tới 26.4%:** Khó bắt trọn các từ ghép đa âm tiết hoặc cụm từ xúc phạm dài.

#### SLIDE 04: Kiến Trúc Đề Xuất - PhoBERT-BiLSTM-CRF
* **Mô hình 3 tầng xếp chồng (Stacked Architecture):**
  * **Tầng 1 - PhoBERT Backbone:** Trích xuất ngữ nghĩa ngữ cảnh sâu tiếng Việt (768 chiều).
  * **Tầng 2 - BiLSTM Layer:** Cầu nối mượt hóa (Smoothing Bridge) và nắm bắt quan hệ tuần tự chuỗi xa (512 chiều).
  * **Tầng 3 - CRF Layer & Viterbi:** Tối ưu hóa chuỗi nhãn toàn cục, phạt $-\infty$ các chuyển nhãn bất hợp lệ.

---

### PHẦN II: DỮ LIỆU & TIỀN XỬ LÝ
*Người trình bày: **Nông Nguyễn Thành** (26410115) - Data & NLP Core*

#### SLIDE 05: Bộ Dữ Liệu Chuẩn ViHOS Benchmark (EACL 2023)
* **Xuất xứ:** Công bố tại hội nghị danh giá EACL 2023 bởi nhóm nghiên cứu UIT (Trần Quốc Khánh et al.).
* **Quy mô:** 11.056 bình luận thực tế từ mạng xã hội.
* **Cấu trúc tập dữ liệu cân bằng:**
  * 5.528 câu chứa phát ngôn xúc phạm/thù ghét (Toxic).
  * 5.528 câu sạch/trung tính (Clean) nhằm kiểm soát lỗi báo động giả (False Positive).

#### SLIDE 06: Lược Đồ Nhãn BIO & Hiện Tượng Lệch Phân Bố
* **Lược đồ nhãn chuẩn bài toán:**
  * `O`: Không phải từ xúc phạm.
  * `B-HOS` (hoặc `B-T`): Bắt đầu cụm từ xúc phạm.
  * `I-HOS` (hoặc `I-T`): Từ tiếp theo trong cụm xúc phạm.
* **Mất cân bằng dữ liệu cực lớn:** Nhãn `O` chiếm >88% tổng số token $\to$ Bắt buộc phải đánh giá bằng **Span-F1** (bỏ qua nhãn `O`).

#### SLIDE 07: Thách Thức Subword Của PhoBERT & Từ Ghép Tiếng Việt
* PhoBERT sử dụng Byte-Pair Encoding (BPE) và nối subword bằng ký tự `@@`.
* Ví dụ: Từ ghép *"khốn_nạn"* $\to$ tách thành `khốn@@` + `nạn`.
* Nếu gán nhãn BIO cho mọi subword sẽ làm sai lệch độ dài span từ gốc và sai lệch thước đo học thuật.

#### SLIDE 08: Kỹ Thuật First-token Subword Alignment
* **Quy tắc căn chỉnh:**
  * Subword đầu tiên của từ: Giữ nguyên nhãn BIO thực tế (`O`, `B-HOS`, `I-HOS`).
  * Các subword mang hậu tố `@@` phía sau: Gán nhãn `-100` (`IGNORE_INDEX`).
* **Hiệu quả:** CRF tự động che mask các vị trí `-100`, đảm bảo giải mã Viterbi khớp chuẩn xác trên từng từ đơn/từ ghép.

#### SLIDE 09: Khảo Sát Phân Bố Độ Dài Câu & Cắt Cụt Dữ Liệu
* Thống kê độ dài câu:
  * Chiều dài trung bình: ~24.5 từ/câu.
  * 98.6% câu có độ dài $\le 128$ tokens.
* **Lựa chọn siêu tham số:** Đặt `max_length = 128` giúp tiết kiệm 40% VRAM GPU và tăng tốc huấn luyện x2.5 lần.

---

### PHẦN III: HUẤN LUYỆN TRÊN COLAB GPU T4
*Người trình bày: **Hoàng Võ Minh Tuấn** (26410146) - Model Trainer*

#### SLIDE 10: Thiết Lập Huấn Luyện & Differential Learning Rate
* **Môi trường:** Google Colab GPU Tesla T4 (16GB VRAM).
* **Differential Learning Rate:**
  * Nhóm tham số PhoBERT: $lr = 2\times 10^{-5}$ (tránh catastrophic forgetting tri thức pre-trained).
  * Nhóm tham số BiLSTM + Linear + CRF: $lr = 1\times 10^{-3}$ (học nhanh các trọng số mới).
* **Siêu tham số:** AdamW, Linear Warmup 10%, Weight Decay = 0.01, Batch size = 16.

#### SLIDE 11: Tiến Trình Hội Tụ & Đóng Gói Checkpoints (.pt)
* Thời gian train: Chỉ mất **~18–22 phút** cho 5 epochs trên Colab T4.
* Early Stopping: Dừng huấn luyện nếu Macro Dev Span-F1 không tăng sau 3 epochs.
* Lưu tự động vào Google Drive và tải về: `best_phobert_bilstm_crf.pt` (~540MB).

---

### PHẦN IV: ĐÁNH GIÁ THỰC NGHIỆM & PHÂN TÍCH LỖI
*Người trình bày: **Bùi Quốc Thịnh** (26410108) - Evaluation & Metrics*

#### SLIDE 12: Bảng Số Liệu Nghiên Cứu Bóc Tách (Ablation Study)
* So sánh đối chứng 3 kiến trúc trên cùng tập Test ViHOS:

| Mô hình | Precision | Recall | **Span-F1** | Mức cải thiện |
| :--- | :---: | :---: | :---: | :---: |
| **PhoBERT-Linear** *(Baseline Thầy)* | 67.12% | 65.46% | **66.28%** | Baseline gốc |
| **PhoBERT-CRF** | 69.40% | 67.85% | **68.61%** | **+2.33%** |
| **PhoBERT-BiLSTM-CRF** *(Đề xuất)* | **71.15%** | **69.50%** | **70.31%** | **+4.03%** |

#### SLIDE 13: Đóng Góp Của Tầng CRF: Triệt Tiêu Lỗi Chuyển Nhãn
* **Tỷ lệ bước chuyển lỗi phi logic `O` $\to$ `I-HOS`:**
  * Baseline PhoBERT-Linear: Chiếm **26.4%** tổng số lỗi nhãn.
  * PhoBERT-BiLSTM-CRF: **0.0% (Triệt tiêu 100%)** nhờ cơ chế phạt của ma trận chuyển trạng thái CRF.
* Tỷ lệ lỗi sai ranh giới từ giảm từ **25.8% xuống 14.6%**.

#### SLIDE 14: Đóng Góp Của Tầng BiLSTM: Xử Lý Đa Chuỗi (Multiple Spans)
* Khảo sát trên nhóm câu chứa $\ge 2$ cụm từ xúc phạm phân tán cách xa nhau:
  * PhoBERT-Linear thường chỉ nhận diện cụm đầu và bỏ sót cụm sau.
  * PhoBERT-BiLSTM-CRF tăng **+5.8% Recall** nhờ duy trì trạng thái ẩn bộ nhớ dài hai chiều.

#### SLIDE 15: Phân Tích Định Tính 11 Dạng Lỗi (Error Analysis)
* Thống kê 100 câu sai thực tế (chi tiết trong `reports/error_analysis.xlsx`):
  * Giảm mạnh: Lỗi đa chuỗi (-66.7%), Lỗi ranh giới từ ghép (-57.1%), Teencode/viết tắt (-38.9%).
  * Thách thức còn lại: Ngữ nghĩa châm biếm (Sarcasm), trích dẫn lại lời nói người khác.

---

### PHẦN V: SẢN PHẨM ỨNG DỤNG LOCAL CPU & KẾT LUẬN
*Người trình bày: **Trần Tiến Dũng** (Slide 16-17) & **Dương Quốc Thương** (Slide 18)*

#### SLIDE 16: Kiến Trúc Ứng Dụng Web App Chạy Thuần CPU (Dũng)
* Framework: **Streamlit UI**.
* Engine suy luận: Chạy thuần trên **CPU** (`torch.device('cpu')`), chỉ dùng **< 1GB RAM**.
* Tốc độ xử lý: **50 – 80 ms / câu** (đáp ứng tiêu chuẩn phản hồi thời gian thực).

#### SLIDE 17: Tính Năng Sản Phẩm & Demo Thực Tế (Dũng)
* **3 Tính năng nổi bật:**
  1. Highlight trực quan các cụm từ xúc phạm theo màu nhãn BIO.
  2. Tính năng **Auto-Masking (`***`)**: Tự động lọc từ bậy mà giữ nguyên ngữ pháp câu.
  3. Bảng đối chứng Ablation Study tương tác ngay trên giao diện web.

#### SLIDE 18: Kết Luận & Hướng Phát Triển Tương Lai (Thương)
* **Kết luận:** Mô hình đề xuất PhoBERT-BiLSTM-CRF giải quyết triệt để các hạn chế của baseline gốc, nâng Span-F1 lên **70.31% (+4.03%)**, đóng gói thành ứng dụng Web CPU hoàn chỉnh.
* **Hướng phát triển:** Tích hợp mô-đun phát hiện châm biếm và tối ưu hóa lượng tử hóa ONNX Runtime cho ứng dụng di động.
