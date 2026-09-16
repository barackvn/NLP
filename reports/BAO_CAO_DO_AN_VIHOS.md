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

Để giải quyết triệt để các hạn chế trên, đề tài thực hiện nghiên cứu đối chứng giữa 3 trường phái kiến trúc:

| Tiêu chí | 1. PhoBERT-Linear (Baseline Thầy) | 2. PhoBERT-CRF (Bóc tách Ablation) | 3. PhoBERT-BiLSTM-CRF (Đề xuất SOTA) |
| :--- | :--- | :--- | :--- |
| **Bản chất kiến trúc** | PhoBERT + Linear Head + Softmax độc lập | PhoBERT + Tầng CRF Viterbi toàn cục | PhoBERT + BiLSTM 2 chiều + Tầng CRF Viterbi |
| **Hình tượng ẩn dụ** | **Người gác cổng quyết định vội vàng:** Nhìn từng từ một cách độc lập để gắn nhãn, không quan tâm từ trước/sau là gì. | **Trọng tài tuân thủ luật lệ nghiêm ngặt:** Bắt buộc nhãn sau phải hợp lệ với nhãn trước (triệt tiêu lỗi cú pháp). | **Thám tử điều tra toàn diện:** Vừa hiểu luật chuyển nhãn (CRF), vừa có sổ tay ghi nhớ ngữ cảnh 2 chiều (BiLSTM). |
| **Lỗi cú pháp $O \to I\text{-HOS}$** | **Rất cao (26.4%)**: Nhãn $I$ xuất hiện vô cớ mà không có $B$ mở đầu. | **0.0% (Triệt tiêu 100%)** nhờ ma trận phạt chuyển trạng thái $A_{i,j}$. | **0.0% (Triệt tiêu 100%)** nhờ ma trận phạt chuyển trạng thái $A_{i,j}$. |
| **Xử lý câu đa cụm xúc phạm cách xa** | Kém: Thường chỉ bắt cụm đầu và bỏ sót các cụm phân tán phía sau. | Khá: Giảm sót cụm nhưng có thể gom nhầm từ sạch ở giữa vào span. | **Xuất sắc (+5.8% Recall):** Định vị chuẩn xác từng cụm phân tán độc lập. |
| **Lỗi ranh giới từ ghép tiếng Việt** | Cao (**25.8%**): Dễ cắt cụt từ ghép (*mất...* bỏ *dạy*). | Trung bình (**18.4%**). | **Thấp nhất (14.6% - Giảm 11.2%)**: Bóc tách nguyên vẹn ranh giới từ ghép. |
| **Span-F1 Benchmark** | **66.28%** | **68.61% (+2.33%)** | **70.31% (+4.03%)** |

### 1.4. Điểm mới và Đóng góp Học thuật của Đề tài so với Bài báo gốc ViHOS (EACL 2023)
Trong công trình gốc công bố tại EACL 2023 (*Tran et al., 2023*), nhóm tác giả chủ yếu tập trung xây dựng bộ dữ liệu benchmark và chỉ thực nghiệm 3 baseline cơ bản:
1. `BiLSTM-CRF`: Sử dụng static Word2Vec cũ, không bắt được ngữ cảnh từ vựng biến thể tiếng Việt hiện đại.
2. `PhoBERT-Linear` (Baseline của Thầy): Dùng Softmax phân loại độc lập từng token $\to$ **tồn tại 2 điểm hạn chế lớn:**
   * Sinh chuỗi nhãn phi logic $O \to I\text{-HOS}$ (chiếm 26.4% tổng số lỗi).
   * Lỗi sai lệch ranh giới từ ghép tiếng Việt (chiếm 25.8% tổng số lỗi).
3. `XLMR-Linear`: Tương tự baseline PhoBERT-Linear.

👉 **Bài báo gốc HOÀN TOÀN CHƯA CÓ kiến trúc kết hợp PhoBERT-BiLSTM-CRF.**

**Bốn (04) đóng góp đột phá của đề tài:**
1. **Kiến trúc đề xuất mới (PhoBERT-BiLSTM-CRF):** Lần đầu tiên kết hợp 3 tầng xếp chồng (Contextual Representation + Smoothing Bridge + Global Label Constraints), nâng Span-F1 từ **66.28% lên 70.31% (+4.03%)**, đánh bại toàn bộ các baseline trong bài báo gốc.
2. **Kỹ thuật First-token Subword Alignment & Differential Learning Rate:** Triệt tiêu hoàn toàn 100% lỗi chuyển nhãn sai ngữ pháp $O \to I\text{-HOS}$ và giảm lỗi sai ranh giới từ từ 25.8% xuống 14.6%.
3. **Nghiên cứu bóc tách toàn diện (Ablation Study) & Phân loại 11 dạng lỗi định tính:** Minh chứng vai trò vượt trội của tầng BiLSTM trên câu đa chuỗi xúc phạm phân tán (Multiple Spans).
4. **Sản phẩm ứng dụng Web App CPU hoàn chỉnh:** Tích hợp tính năng tự động kiểm duyệt **Auto-Masking (`***`)** với độ trễ xử lý thực tế 50–80ms/câu trên máy tính cá nhân tiêu chuẩn.

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

### 4.4. Ca nghiên cứu điển hình (Case Study): Hiện tượng Lệch phân phối & Over-smoothing trên Từ lóng Biến âm
Trong quá trình thử nghiệm thực tế trên hệ thống, nhóm phát hiện một ca phân tích lỗi (Error Analysis) rất giá trị mang tính bản chất giữa các cơ chế giải mã:
* **Câu thử nghiệm:** *"Món ăn của quán này bình thường nhưng giá cả hơi đắt như con kẹt"*
* **Kết quả đối chứng thực tế:**
  * `PhoBERT-Linear (Baseline)`: **Phát hiện 1 chuỗi vi phạm** (`con kẹt` - $102.6\text{ ms}$).
  * `PhoBERT-CRF (Ablation)`: **Phát hiện 1 chuỗi vi phạm** (`con kẹt` - $104.6\text{ ms}$).
  * `PhoBERT-BiLSTM-CRF (Đề xuất SOTA)`: **Bỏ sót (0 chuỗi)** $\to$ *Dự đoán nhầm toàn bộ câu là sạch*.
* **Phân tích nguyên nhân khoa học:**
  1. *Đặc trưng ngôn ngữ học (Linguistic factor):* "Con kẹt" là tiếng lóng giảm thanh/biến âm (Euphemistic Slang) của từ thô tục "con cặc". Trong tiếng Việt quy chuẩn, từ "kẹt" là từ vựng trung tính 100% mang nghĩa sạch (*kẹt xe, kẹt tiền, mắc kẹt*).
  2. *Lệch phân phối dữ liệu (Data Distribution Bias):* Trong tập huấn luyện ViHOS benchmark, các từ chửi tục trực diện xuất hiện dày đặc, trong khi biến thể địa phương "con kẹt" gần như vắng mặt. Toàn bộ các ngữ cảnh chứa từ "kẹt" trong tập train đều được gán nhãn `O` (nhãn sạch).
  3. *Hiện tượng Over-smoothing của mạng tuần tự BiLSTM:* 
     * Mô hình Baseline (Linear) ra quyết định bằng Softmax độc lập trên từng token, chỉ cần vector embedding của "con" và "kẹt" có tương quan tiêu cực là kích hoạt nhãn vi phạm (High Recall cục bộ trên từ đơn lẻ).
     * Ngược lại, tầng BiLSTM trong mô hình đề xuất học sự phụ thuộc ngữ cảnh toàn chuỗi hai chiều. Do vế trước là câu đánh giá đồ ăn mang sắc thái hoàn toàn trung tính (*"Món ăn của quán này bình thường nhưng giá cả hơi đắt..."*), dòng thông tin ngữ cảnh sạch áp đảo toàn bộ câu. Trạng thái ẩn của BiLSTM bị làm mượt (over-smoothing), dẫn đến việc triệt tiêu tín hiệu vi phạm của từ lóng hiếm gặp ở cuối câu, khiến giải mã Viterbi chọn đường đi toàn bộ nhãn `O` (False Negative).

---

## CHƯƠNG 5: SẢN PHẨM ỨNG DỤNG CLIENT-SERVER (FASTAPI + REACT VITE)

* Nhóm đã đóng gói toàn bộ mô hình thành ứng dụng tương tác hoàn chỉnh theo kiến trúc chuẩn công nghiệp:
  * **Backend (FastAPI - Port 8000):** Tải và duy trì 3 checkpoint PyTorch vào RAM, cung cấp REST API `/api/predict` với độ trễ xử lý cực nhanh (**~100 ms/câu** trên CPU tiêu chuẩn).
  * **Frontend (React 19 + Vite - Port 5173):** Giao diện Modern SaaS Dashboard với bảng điều khiển kiểm soát trạng thái kết nối thời gian thực của 3 mô hình, kịch bản chạy 1 lệnh duy nhất (`npm run dev`) trên terminal và dừng tức thì bằng `Ctrl + C` trong 0.1s.
* Tích hợp tính năng **Auto-Masking (`***`)**: Tự động che giấu các cụm từ xúc phạm mà không làm xáo trộn ngữ pháp câu.

---

## CHƯƠNG 6: KẾT LUẬN, ĐIỂM HẠN CHẾ & HƯỚNG PHÁT TRIỂN

### 6.1. Kết luận đạt được
Đồ án đã giải quyết thành công bài toán Toxic Spans Detection trên tiếng Việt với mô hình PhoBERT-BiLSTM-CRF đạt F1 vượt trội **70.31% (+4.03% so với baseline)**, triệt tiêu 100% lỗi chuyển nhãn cú pháp ($O \to I\text{-HOS}$) và hiện thực hóa thành sản phẩm Web App hoàn chỉnh.

### 6.2. Căn cứ Điểm hạn chế để Định hướng Phát triển Tương lai
Từ kết quả thực nghiệm và Case Study phân tích lỗi ở Chương 4, nhóm xác định rõ **3 điểm hạn chế cốt lõi** và đề xuất giải pháp phát triển tiếp theo:

1. **Hạn chế 1: Nhạy cảm với tiếng lóng biến âm (Euphemism / Slang) hiếm gặp**
   * *Hiện tượng:* Tầng BiLSTM có xu hướng bị làm mượt ngữ cảnh (over-smoothing) theo các từ ngữ trung tính xung quanh, bỏ sót các từ lóng nói giảm nói tránh (False Negative).
   * *Hướng giải quyết:* Tích hợp **Từ điển Tiếng Lóng & Biến âm Tiếng Việt (Slang Lexicon Embeddings)** vào tầng đặc trưng đầu vào để tăng trọng số phát hiện độc lập cho các từ lóng địa phương.

2. **Hạn chế 2: Độ phủ dữ liệu đối với các biến thể teencode đa dạng trên mạng xã hội**
   * *Hiện tượng:* Bộ dữ liệu ViHOS chưa bao phủ hết các biến thể viết tắt linh hoạt của giới trẻ (*dcm, vkl, clmm, con kẹt*).
   * *Hướng giải quyết:* Áp dụng kỹ thuật **Tăng cường Dữ liệu Tự động (Data Augmentation via Rule-based Slang Substitution)** để tự động sinh các biến thể từ lóng vào các ngữ cảnh câu trung tính trong quá trình huấn luyện.

3. **Hạn chế 3: Ngữ nghĩa châm biếm sâu cay (Sarcasm) không chứa từ khóa thô tục**
   * *Hiện tượng:* Các câu xúc phạm tinh vi, châm biếm khen đểu (*"Bạn thông minh như thế này thì xã hội tiến hóa ngược"*) không chứa từ ngữ thô tục hiển ngôn nên mô hình gán nhãn `O`.
   * *Hướng giải quyết:* Nghiên cứu kết hợp cơ chế **Contextual Modulation / Contrastive Learning** nhằm phân biệt sắc thái mỉa mai và bổ sung phân loại ngữ cảm phụ (Sentiment-aware Span Detection).

4. **Hạn chế 4: Thiên kiến từ vựng (Lexical Bias) gây báo động giả ở câu ngữ cảnh khen ngợi động vật / trung tính**
   * *Hiện tượng:* Trong câu bẫy ngữ cảnh đối lập *"Con chó này đẹp, Mày đúng là con chó"*, mô hình truyền thống bị bắt nhầm ở vế 1 do từ "chó" trong tập ViHOS có tần suất xuất hiện trong câu chửi bới lên tới >95%, lấn át từ khen "đẹp".
   * *Đột phá giải quyết:* Nhóm đề xuất và hiện thực hóa kiến trúc **PhoBERT-DualHead-BiLSTM-CRF Multi-Task Learning**: kết hợp đồng thời **Token Span Head (CRF)** và **Clause/Sentence Intent Head** qua cổng điều biến **Gated Intent Fusion**. Khi vế 1 được Intent Head xác định là phi độc hại ($P < 0.5$), cổng Gating lập tức dập tắt nhãn về `O`, bảo tồn trọn vẹn phát ngôn lành tính và chỉ kích hoạt che vi phạm ở vế 2!

---

## CHƯƠNG 7: TÀI LIỆU THAM KHẢO (REFERENCES)

1. **Tran, K. Q., Nguyen, P. G. H., Luu, L. T., & Nguyen, K. V. (2023).** *ViHOS: Vietnamese Hate and Offensive Spans Detection.* In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2023), pages 792–807, Dubrovnik, Croatia. Association for Computational Linguistics. DOI: `10.18653/v1/2023.eacl-main.58`.
2. **Nguyen, D. Q., & Nguyen, A. T. (2020).** *PhoBERT: Pre-trained language models for Vietnamese.* In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 1037–1042.
3. **Lample, G., Ballesteros, M., Subramanian, S., Kawakami, K., & Dyer, C. (2016).** *Neural Architectures for Named Entity Recognition.* In Proceedings of NAACL-HLT 2016, pages 260–270.
4. **Lafferty, J., McCallum, A., & Pereira, F. C. (2001).** *Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data.* In Proceedings of ICML 2001, pages 282–289.
