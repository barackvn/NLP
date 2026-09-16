# KỊCH BẢN THUYẾT TRÌNH 15 SLIDE TINH GỌN (NGÔN NGỮ ĐỜI THƯỜNG, DỄ NÓI)
## ĐỒ ÁN: NHẬN DIỆN CHUỖI TỪ NGỮ XÚC PHẠM TIẾNG VIỆT (ViHOS) BẰNG PhoBERT-BiLSTM-CRF
**Trường:** Đại học Công nghệ Thông tin – ĐHQG TP.HCM (UIT)  
**Thời lượng chuẩn:** 10 – 12 phút (Khoảng 45 – 50 giây / slide)  
**File PowerPoint:** `Bao_Cao_Do_An_ViHOS_15_Slide.pptx`

---

### PHẦN I: MỞ ĐẦU & ĐẶT VẤN ĐỀ (Slide 01 – 03)
*Người trình bày: **Dương Quốc Thương** (26410127) - Trưởng nhóm*

#### 🎙️ SLIDE 01: Bìa Báo Cáo
* **Lời thoại mở màn:**
  > "Kính thưa quý Thầy trong Hội đồng chấm đồ án môn Xử lý Ngôn ngữ Tự nhiên. Hôm nay nhóm chúng em gồm 5 thành viên xin phép báo cáo đề tài: **Cải tiến phương pháp nhận diện chuỗi từ ngữ xúc phạm tiếng Việt bằng mô hình PhoBERT-BiLSTM-CRF trên bộ dữ liệu chuẩn ViHOS EACL 2023**.  
  > Nhóm gồm em là Dương Quốc Thương - trưởng nhóm, bạn Nông Nguyễn Thành phụ trách dữ liệu, bạn Hoàng Võ Minh Tuấn phụ trách huấn luyện, bạn Bùi Quốc Thịnh phụ trách thực nghiệm và bạn Trần Tiến Dũng phụ trách phát triển ứng dụng Web."

#### 🎙️ SLIDE 02: Mục Lục Báo Cáo
* **Lời thoại:**
  > "Bài báo cáo của tụi em được gói gọn trong **15 slide tinh gọn qua 5 phần chính**:  
  > 1. Đặt vấn đề và ý tưởng cốt lõi.  
  > 2. Dữ liệu ViHOS và mẹo giải quyết từ ghép tiếng Việt.  
  > 3. Mô hình 3 tầng và cách huấn luyện khôn ngoan trên GPU.  
  > 4. Bảng vàng so sánh 3 trường phái và ca nghiên cứu từ lóng thực tế.  
  > 5. Demo sản phẩm Web App chạy thực tế và tổng kết."

#### 🎙️ SLIDE 03: Ý Tưởng Cốt Lõi: "Nhặt Quả Táo Sâu Thay Vì Vứt Cả Giỏ"
* **Lời thoại:**
  > "Kính thưa Thầy, phát ngôn thù ghét trên mạng xã hội hiện nay rất phổ biến. Cách làm truyền thống trước đây là **phân loại cả câu**, tức là máy thấy trong câu có từ bậy thì nó gán nhãn 1 và **xóa luôn cả câu văn**.  
  > Điều này giống như việc *'trong một giỏ táo có 1 quả bị sâu mà chúng ta vứt đi cả giỏ táo lành'*. Người dùng có khi viết một bình luận dài 50 từ rất tâm huyết nhưng lỡ chêm 1 từ bậy, nếu xóa cả câu thì mất đi mạch tranh luận văn minh.  
  > Vì vậy, nhóm em chọn bài toán **Toxic Spans Detection**: Tụi em nhặt đúng quả táo sâu để khoét bỏ (tự động che `***` cụm từ bậy), giữ lại 99% nội dung hữu ích còn lại của người dùng."

---

### PHẦN II: DỮ LIỆU & TIỀN XỬ LÝ (Slide 04 – 06)
*Người trình bày: **Nông Nguyễn Thành** (26410115) - Data & NLP Core*

#### 🎙️ SLIDE 04: Bộ Dữ Liệu Chuẩn ViHOS Benchmark (EACL 2023)
* **Lời thoại:**
  > "Kính chào Thầy, em là Thành phụ trách dữ liệu. Tụi em sử dụng bộ dữ liệu chuẩn **ViHOS của Hội nghị quốc tế EACL 2023** gồm hơn **11.000 câu bình luận thực tế**, chia theo tỷ lệ vàng: 80% Train, 10% Dev và 10% Test.  
  > Khó khăn lớn nhất tụi em gặp phải là **sự mất cân bằng dữ liệu cực lớn**: Từ sạch bình thường (nhãn O) chiếm tới **88.5%**, trong khi từ chửi (B-HOS, I-HOS) chỉ có hơn **11%**.  
  > Nếu mô hình lười biếng đoán bừa tất cả là từ sạch thì vẫn đúng 88%, nhưng lại bỏ lọt 100% từ bậy. Điều này đòi hỏi mô hình phải có khả năng hiểu ngữ cảnh rất sâu."

#### 🎙️ SLIDE 05: Khó Khăn Tiếng Việt: Từ Ghép & Teencode Né Lọc
* **Lời thoại:**
  > "Tiếng Việt trên mạng xã hội có 3 cái bẫy rất khó:  
  > 1. **Bẫy từ ghép:** Ví dụ từ *'mất dạy'*, nếu máy chém cụt chữ *'mất'* (động từ) bỏ chữ *'dạy'* (giáo dục) thì mất hẳn ý nghĩa xúc phạm.  
  > 2. **Teencode né lọc:** Người dùng cố tình viết tắt hay chệch âm như *'đm'* thành *'đcm'*, *'vcl'* thành *'vkl'*, *'lồn'* viết thành *'l0n'* khiến các bộ lọc từ điển tĩnh hoàn toàn bất lực.  
  > 3. **Tiếng lóng giảm thanh:** Dùng từ sạch để chửi xéo, ví dụ như câu *'đắt như con kẹt'*. Từ *'kẹt'* vốn dĩ là từ sạch (kẹt xe), nhưng ở đây là từ bậy nói tránh."

#### 🎙️ SLIDE 06: Mẹo Xử Lý Từ Chẻ Nhỏ: First-Token Subword Alignment
* **Lời thoại:**
  > "Khi đưa tiếng Việt vào mô hình PhoBERT, PhoBERT dùng thuật toán BPE nên nó **chẻ từ ghép ra như chẻ củi**. Ví dụ từ *'mất_dạy'* bị chẻ thành 2 mẩu: *'mất@@'* và *'dạy'*.  
  > Trong khi bộ dữ liệu gốc chỉ có 1 nhãn, làm máy bị lệch kích thước và báo lỗi ngay.  
  > Nhóm em giải quyết bằng mẹo **First-token Alignment**: Mẩu đầu tiên (*'mất@@'*) tụi em dán đúng nhãn gốc; còn mẩu đuôi (*'dạy'*), tụi em dán nhãn đặc biệt là **-100** để máy **tự động bỏ qua không tính điểm**. Nhờ mẹo này, mô hình học êm ru mà không bao giờ bị lệch ranh giới từ."

---

### PHẦN III: MÔ HÌNH & HUẤN LUYỆN (Slide 07 – 08)
*Người trình bày: **Hoàng Võ Minh Tuấn** (26410146) - Model Trainer*

#### 🎙️ SLIDE 07: Mô Hình 3 Tầng: 3 'Chuyên Gia' Phối Hợp
* **Lời thoại:**
  > "Kính thưa Thầy, em là Tuấn phụ trách huấn luyện. Kiến trúc của nhóm kết hợp nhịp nhàng 3 'chuyên gia':  
  > 1. **Tầng 1 - PhoBERT:** Là 'chuyên gia tiếng Việt', đã đọc hàng triệu câu văn trên mạng nên hiểu sâu sắc ngữ cảnh từ vựng.  
  > 2. **Tầng 2 - BiLSTM:** Là 'chuyên gia trí nhớ dai', đọc câu xuôi rồi đọc ngược để nhớ cả câu trước lẫn câu sau, không bao giờ bỏ sót các cụm chửi ở xa.  
  > 3. **Tầng 3 - CRF:** Là 'trọng tài giữ luật', bắt buộc từ sau phải đi đúng luật với từ trước, tuyệt đối cấm nhãn I nhảy xổ ra khi chưa có nhãn B mở đầu."

#### 🎙️ SLIDE 08: Chiến Lược Huấn Luyện Khôn Ngoan Trên Colab GPU
* **Lời thoại:**
  > "Để huấn luyện mô hình hiệu quả trên GPU Tesla T4 của Colab, nhóm dùng **mẹo học phân tầng (Differential Learning Rate)**:  
  > * Phần PhoBERT vốn đã rất giỏi tiếng Việt rồi, nên tụi em chỉ cho học thật chậm (LR = 2e-5) để không bị 'quên gốc'.  
  > * Phần BiLSTM và CRF mới thêm vào chưa biết gì, nên tụi em cho học nhanh gấp 50 lần (LR = 1e-3) để mau thuộc bài.  
  > Đồng thời, tụi em cắt độ dài câu ở mức 128 từ (bao phủ 98.7% số câu), giúp máy chạy nhanh gấp 2.5 lần và không bao giờ bị tràn VRAM."

---

### PHẦN IV: KẾT QUẢ THỰC NGHIỆM & CA 'CON KẸT' (Slide 09 – 12)
*Người trình bày: **Bùi Quốc Thịnh** (26410108) - Evaluation & Metrics*

#### 🎙️ SLIDE 09: Bảng Vàng So Sánh Bản Chất 3 Trường Phái
* **Lời thoại:**
  > "Kính thưa Hội đồng, em là Thịnh phụ trách đánh giá. Đây là bảng so sánh đắt giá nhất của đồ án trên toàn bộ 1.106 câu kiểm thử:  
  > 1. **Mô hình 1 - PhoBERT-Linear (Baseline của Thầy):** Giống như *'Người gác cổng vội vàng'*, nhìn từng từ rồi đoán mò độc lập, không quan sát xung quanh, nên mắc tới 26.4% lỗi cú pháp vô lý.  
  > 2. **Mô hình 2 - PhoBERT-CRF:** Giống như *'Trọng tài nghiêm ngặt'*, bắt đúng luật chuyển nhãn nên **triệt tiêu 100% lỗi cú pháp**, F1 tăng lên 68.61%.  
  > 3. **Mô hình 3 - PhoBERT-BiLSTM-CRF (Đề xuất của nhóm):** Giống như *'Thám tử toàn diện'*, vừa nhớ dai vừa thuộc luật, bóc tách chuẩn xác ranh giới từ ghép và đạt **Span-F1 cao nhất là 70.31% (tăng hơn 4% so với baseline)**."

#### 🎙️ SLIDE 10: Đột Phá Của Tầng CRF: Triệt Tiêu 100% Lỗi Vô Lý
* **Lời thoại:**
  > "Nhìn vào biểu đồ bên phải, Thầy sẽ thấy điều kỳ diệu của tầng CRF:  
  > Ở mô hình cũ, lỗi phi lý 'nhãn I tự nhiên xuất hiện mà không có nhãn B' chiếm tới hơn một phần tư (26.4%).  
  > Nhờ CRF đặt mức phạt cực nặng vào thuật toán Viterbi, **tỷ lệ lỗi này rớt thẳng từ 26.4% về đúng 0.0%**. Đồng thời lỗi chém cụt từ ghép giảm từ 25.8% xuống 14.6%."

#### 🎙️ SLIDE 11: Đột Phá Của Tầng BiLSTM: Bắt Trọn Câu Đa Cụm Xúc Phạm
* **Lời thoại:**
  > "Trong thực tế, hơn 30% câu chửi trên mạng có từ 2 cụm vi phạm cách xa nhau.  
  > Các mô hình cũ thường chỉ bắt được cụm đầu rồi quên mất cụm sau. Nhờ BiLSTM có bộ nhớ hai chiều xuôi ngược, mô hình của nhóm **tăng thêm 5.8% độ bắt trúng (Recall)** trên các câu đa cụm phức tạp này."

#### 🎙️ SLIDE 12: Ca Thực Tế Thú Vị: Vì Sao Mô Hình Sót Từ 'Con Kẹt'?
* **Lời thoại:**
  > "Tụi em xin chia sẻ một ca phân tích lỗi thực tế rất thú vị:  
  > Khi test câu: *'Món ăn của quán này bình thường nhưng giá cả hơi đắt như con kẹt'*, mô hình cũ của Thầy lại bắt được từ *'con kẹt'*, nhưng mô hình hiện đại của tụi em lại bỏ sót!  
  > **Vì sao lại như vậy?**  
  > Vì vế trước câu là lời khen chê món ăn quá đàng hoàng, lịch sự, nên trí nhớ dài của BiLSTM bị 'lừa', tưởng cả câu đều là câu văn minh, làm mờ đi tín hiệu từ bậy ở cuối câu.  
  > Việc trung thực chỉ ra ca lỗi này chứng minh tụi em hiểu sâu bản chất mô hình và đây là căn cứ để tụi em xây dựng từ điển tiếng lóng sau này."

---

### PHẦN V: SẢN PHẨM WEB APP & TỔNG KẾT (Slide 13 – 15)
*Người trình bày: **Trần Tiến Dũng** (26410024) & **Dương Quốc Thương** (26410127)*

#### 🎙️ SLIDE 13: Web App Client-Server Bật 1 Lệnh 'npm run dev' (Dũng)
* **Lời thoại:**
  > "Kính thưa Thầy, em là Dũng phụ trách sản phẩm. Tụi em đã loại bỏ hoàn toàn Streamlit vì hay bị đơ máy trên Windows để xây dựng hệ thống **Client-Server công nghiệp**:  
  > * Backend dùng **FastAPI**: nạp sẵn 3 mô hình AI vào RAM, trả kết quả cực nhanh chỉ trong **0.1 giây**.  
  > * Frontend dùng **React 19**: giao diện hiện đại toàn màn hình.  
  > * Đột phá vận hành: Tụi em gom chung lại, chỉ cần gõ đúng 1 lệnh **'npm run dev'** là cả Web và AI cùng chạy, khi tắt chỉ bấm `Ctrl + C` là tắt sạch trong chớp mắt."

#### 🎙️ SLIDE 14: Demo Tính Năng Web App Thực Tế (Dũng)
* **Lời thoại:**
  > "Trên màn hình demo của tụi em có 3 tính năng đắt giá:  
  > 1. Vừa nhập câu vào là máy **bôi đỏ ngay lập tức các từ xúc phạm** để người duyệt nhìn thấy ngay.  
  > 2. Khung **Auto-Masking** bên cạnh tự động thay từ bậy bằng dấu `***`, giữ nguyên các từ sạch xung quanh.  
  > 3. Nút trạng thái **'🟢 3/3 Mô hình Sẵn sàng'** trên góc màn hình cho phép bấm vào kiểm tra sức khỏe của cả 3 mô hình AI nạp thật 100%."

#### 🎙️ SLIDE 15: Tổng Kết Đồ Án & Lời Cảm Ơn (Thương)
* **Lời thoại:**
  > "Kính thưa quý Thầy trong Hội đồng, tóm lại đồ án của nhóm đã đạt được 3 thành quả:  
  > 1. Nâng Span-F1 lên **70.31% (+4.03%)**.  
  > 2. Triệt tiêu 100% lỗi cú pháp vô lý.  
  > 3. Đóng gói thành sản phẩm Web App hoàn chỉnh chạy siêu nhanh trên máy tính thông thường.  
  > Nhóm cũng chỉ ra hướng phát triển tiếp theo là làm giàu từ điển tiếng lóng và xử lý câu nói mỉa mai.  
  > **Nhóm chúng em xin chân thành cảm ơn quý Thầy và xin lắng nghe các câu hỏi nhận xét từ Hội đồng ạ!**"
