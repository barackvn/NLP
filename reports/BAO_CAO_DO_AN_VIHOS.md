# BÁO CÁO THUYẾT MINH ĐỒ ÁN KỸ THUẬT
## CẢI TIẾN PHƯƠNG PHÁP NHẬN DIỆN CHUỖI NGÔN NGỮ XÚC PHẠM TIẾNG VIỆT (ViHOS) BẰNG MÔ HÌNH KẾT HỢP PhoBERT-BiLSTM-CRF

* **Đơn vị đào tạo:** Trường Đại học Công nghệ Thông tin – Đại học Quốc gia TP.HCM
* **Bộ môn:** Xử lý Ngôn ngữ Tự nhiên (NLP)
* **Giảng viên hướng dẫn:** NCS.ThS. Đặng Văn Thìn & Tác giả Trần Quốc Khánh
* **Nhóm sinh viên thực hiện:**
  1. Dương Quốc Thương (26410127) - Nhóm trưởng
  2. Nông Nguyễn Thành (26410115)
  3. Hoàng Võ Minh Tuấn (26410146)
  4. Bùi Quốc Thịnh (26410108)
  5. Trần Tiến Dũng (26410024)

---

## CHƯƠNG 1: TỔNG QUAN & ĐẶT VẤN ĐỀ

### 1.1. Bối cảnh nghiên cứu
Sự bùng nổ của mạng xã hội (Facebook, YouTube, TikTok) tại Việt Nam tạo môi trường trao đổi thuận lợi nhưng cũng làm gia tăng đáng kể các hành vi phát ngôn xúc phạm, công kích thù ghét (Hate & Offensive Speech).

### 1.2. Hạn chế của bài toán Phân loại câu (Sentence-level Classification)
Hầu hết các nghiên cứu trước đây chỉ tiếp cận dưới góc độ phân loại nhị phân toàn câu: câu có chứa yếu tố xúc phạm (1) hay câu sạch (0). Phương pháp này gặp hạn chế lớn trong môi trường kiểm duyệt thực tế:
* Buộc hệ thống phải xóa bỏ hoàn toàn câu bình luận, gây gián đoạn luồng thảo luận.
* Không thể chỉ ra chính xác cụm từ nào mang tính vi phạm để giải thích lý do cho người dùng.

### 1.3. Bài toán Nhận diện chuỗi ngôn ngữ xúc phạm (Toxic Spans Detection)
Nhiệm vụ là xác định chính xác vị trí (offsets) hoặc chuỗi các từ ngữ mang tính xúc phạm trong câu văn theo bài toán Sequence Labeling:
* `O` (Outside): Từ ngữ trung tính.
* `B-HOS`: Từ bắt đầu chuỗi xúc phạm.
* `I-HOS`: Từ tiếp theo nằm trong chuỗi xúc phạm.

---

## CHƯƠNG 2: BỘ DỮ LIỆU CHUẨN ViHOS BENCHMARK (EACL 2023)

* **Quy mô:** 11.056 bình luận được gán nhãn thủ công bởi con người, công bố tại hội nghị EACL 2023 (Trần Quốc Khánh et al.).
* **Cấu trúc phân chia:**
  * Tập huấn luyện (Train): 8.844 câu.
  * Tập kiểm định (Dev): 1.106 câu.
  * Tập kiểm thử (Test): 1.106 câu.
* **Tính cân bằng:** Gồm 5.528 câu thù ghét/xúc phạm và 5.528 câu sạch trung tính nhằm kiểm soát tỷ lệ báo động giả (False Positive).

---

## CHƯƠNG 3: THIẾT KẾ KIẾN TRÚC MÔ HÌNH ĐỀ XUẤT

Kiến trúc **PhoBERT-BiLSTM-CRF** được thiết kế dưới dạng **3 tầng xếp chồng (Stacked Architecture)**:

```
[Câu bình luận tiếng Việt]
       │
       ▼
┌──────────────────────────────┐
│   TẦNG 1: PhoBERT Backbone   │ ──► Trích xuất vector ẩn ngữ cảnh (768 chiều)
└──────────────────────────────┘
       │ H ∈ R^(B × T × 768)
       ▼
┌──────────────────────────────┐
│   TẦNG 2: BiLSTM Layer       │ ──► Smoothing Bridge & Kết nối tuần tự chuỗi xa (512 chiều)
└──────────────────────────────┘
       │ H_lstm ∈ R^(B × T × 512)
       ▼
┌──────────────────────────────┐
│   TẦNG 3: CRF Layer          │ ──► Tối ưu hóa chuỗi nhãn toàn cục & Giải mã Viterbi
└──────────────────────────────┘
       │
       ▼
[Chuỗi nhãn BIO tối ưu: O, B-HOS, I-HOS]
```

### 3.1. Tầng PhoBERT Backbone (`vinai/phobert-base-v2`)
* Đã được huấn luyện sẵn trên 20GB văn bản tiếng Việt chất lượng cao.
* Cung cấp biểu diễn ngữ cảnh mạnh mẽ cho từng subword.

### 3.2. Tầng BiLSTM Layer
* `input_size = 768`, `hidden_size = 256`, `bidirectional = True` $\to$ Đầu ra 512 chiều.
* Đóng vai trò là "cầu nối mượt hóa" (smoothing bridge), kết nối ngữ cảnh hai chiều giúp khắc phục hiện tượng mất thông tin ở các câu có nhiều cụm từ xúc phạm phân tán cách xa nhau (Multiple Spans).

### 3.3. Tầng CRF Layer & Viterbi Decoding
* Duy trì ma trận chuyển đổi trạng thái nhãn $A_{i,j}$.
* Huấn luyện thông qua hàm Negative Log-Likelihood Loss toàn cục.
* Giải mã Viterbi triệt tiêu hoàn toàn các bước chuyển nhãn sai cú pháp (như $O \to I\text{-HOS}$).

### 3.4. Kỹ thuật First-token Subword Alignment
* Subword đầu tiên của từ: Gán nhãn BIO thật.
* Các subword mang hậu tố `@@` phía sau: Gán nhãn `-100` (`IGNORE_INDEX`).
* Tầng CRF tự động che mask các vị trí `-100`, đảm bảo việc tính toán và đánh giá luôn chuẩn xác theo ranh giới từ vựng gốc.

---

## CHƯƠNG 4: THỰC NGHIỆM & KẾT QUẢ

### 4.1. Thiết lập thực nghiệm
* **Môi trường:** Google Colab GPU Tesla T4 (16GB VRAM).
* **Differential Learning Rate:**
  * PhoBERT backbone: $lr = 2\times 10^{-5}$
  * BiLSTM + Linear + CRF: $lr = 1\times 10^{-3}$
* **Optimizer:** AdamW, Warmup 10% steps, Weight Decay = 0.01.
* **Thước đo đánh giá:** Span-Precision, Span-Recall, Span-F1 (thư viện `seqeval`).

### 4.2. Kết quả Nghiên cứu Bóc tách (Ablation Study)

| Cấu trúc Mô hình | Precision | Recall | **Span-F1** | Lỗi $O \to I\text{-HOS}$ | Lỗi ranh giới từ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PhoBERT-Linear** *(Baseline gốc)* | 67.12% | 65.46% | **66.28%** | 26.4% | 25.8% |
| **PhoBERT-CRF** | 69.40% | 67.85% | **68.61%** *(+2.33%)* | **0.0%** | 18.4% |
| **PhoBERT-BiLSTM-CRF** *(Đề xuất)* | **71.15%** | **69.50%** | **70.31%** *(+4.03%)* | **0.0%** | **14.6%** |

### 4.3. Phân tích định tính 11 dạng lỗi trên 100 mẫu
* Giảm lỗi đa chuỗi cách xa từ 12 câu xuống 4 câu (-66.7%).
* Giảm lỗi sai ranh giới từ ghép từ 14 câu xuống 6 câu (-57.1%).
* Giảm lỗi nhận diện teencode/viết tắt từ 18 câu xuống 11 câu (-38.9%).

---

## CHƯƠNG 5: SẢN PHẨM ỨNG DỤNG LOCAL CPU (STREAMLIT APP)

* Nhóm đã đóng gói toàn bộ mô hình thành ứng dụng tương tác chạy thuần trên **CPU** (`app/app.py`).
* Tốc độ phản hồi đạt **~50–80 ms/câu** trên CPU tiêu chuẩn với mức tiêu thụ RAM < 1GB.
* Tích hợp tính năng **Auto-Masking (`***`)**: Tự động che giấu các cụm từ xúc phạm mà không làm thay đổi các từ trung tính còn lại.

---

## CHƯƠNG 6: KẾT LUẬN & HƯỚNG PHÁT TRIỂN
Đồ án đã giải quyết thành công bài toán Toxic Spans Detection trên tiếng Việt với mô hình PhoBERT-BiLSTM-CRF đạt F1 vượt trội **70.31%**, triệt tiêu toàn bộ lỗi chuyển nhãn cú pháp và hiện thực hóa thành sản phẩm Web App hoàn chỉnh.
