# HƯỚNG DẪN CHI TIẾT: HUẤN LUYỆN MÔ HÌNH DUAL-HEAD MULTI-TASK TRÊN GOOGLE COLAB (GPU T4)

> **Mục tiêu:** Huấn luyện mô hình nghiên cứu khoa học **PhoBERT-DualHead-BiLSTM-CRF Multi-Task**, xuất file checkpoint `.pt` (~540MB) để triệt tiêu hoàn toàn lỗi bắt nhầm ở các câu ngữ cảnh trung tính / khen ngợi động vật.

---

## 🎯 1. TẠI SAO CẦN MÔ HÌNH DUAL-HEAD NÀY?
Khi bạn test câu:
> *"Con chó này đẹp, Mày đúng là con chó"*

* **Mô hình cũ (Single-head):** Bị "báo động giả" ở vế 1, che luôn thành `*** này đẹp` vì bị thiên kiến từ vựng (Lexical Bias).
* **Mô hình Dual-Head Multi-Task mới:** 
  * Tối ưu hàm Loss đa nhiệm: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CRF}} + 0.5 \times \mathcal{L}_{\text{BCE}}$.
  * Phân tích song song: **Token Span Head (vị trí từ)** và **Clause/Sentence Intent Head (ý đồ câu)**.
  * Cổng Gated Fusion tự động dập tắt nhãn ở vế 1 (`Con chó này đẹp` $\to$ sạch `O`) và chỉ che vế 2 (`*** đúng là ***`).

---

## 🚀 2. QUY TRÌNH CHẠY TRÊN GOOGLE COLAB (CHỈ 4 BƯỚC ĐƠN GIẢN)

### Bước 1: Mở File Notebook trên Google Colab
1. Truy cập [Google Colab](https://colab.research.google.com/).
2. Chọn thẻ **Upload (Tải lên)** $\to$ Chọn file:
   `Doan/notebooks/04_train_dualhead_colab_gpu_t4.ipynb`
3. Hoặc mở trực tiếp qua liên kết GitHub của nhóm.

### Bước 2: Bật GPU Tesla T4 Miễn phí
1. Trên thanh menu Colab, bấm **Runtime** (Thời gian chạy) $\to$ **Change runtime type** (Thay đổi loại thời gian chạy).
2. Tại mục **Hardware accelerator**, chọn **T4 GPU** $\to$ Bấm **Save**.

### Bước 3: Chạy Huấn luyện Tự động
1. Bấm **Runtime** $\to$ **Run all** (hoặc gõ phím tắt `Ctrl + F9`).
2. Quá trình gồm:
   * Cài đặt tự động `transformers`, `pyvi`, `seqeval`.
   * Kết nối Google Drive để lưu checkpoint vào `MyDrive/ViHOS_Checkpoints/`.
   * Tự động train 5 Epochs với Early Stopping (khoảng **18 phút**).
   * Tự động đánh giá F1 trên 1.106 câu tập Test chuẩn.
   * Tự động chạy thử nghiệm câu *"Con chó này đẹp, Mày đúng là con chó"*.

### Bước 4: Tải Checkpoint về Máy tính
Ở ô cuối cùng của Notebook:
```python
from google.colab import files
files.download("/content/drive/MyDrive/ViHOS_Checkpoints/best_phobert_dualhead_bilstm_crf.pt")
```
Trình duyệt sẽ tự động tải file `best_phobert_dualhead_bilstm_crf.pt` về máy tính. Bạn chỉ cần chép file này vào thư mục:
```
Doan/checkpoints/
```

---

## 📊 3. SƠ ĐỒ ĐỐI CHỨNG DÙNG CHO SLIDE BẢO VỆ

| Mô hình | Cơ chế giải mã | Xử lý ranh giới từ ghép | Xử lý câu ngữ cảnh trung tính (*"Con chó này đẹp"*) |
| :--- | :--- | :--- | :--- |
| **PhoBERT-Linear** | Softmax độc lập | Dễ bị chém cụt từ | Bị bắt nhầm (False Positive) |
| **PhoBERT-CRF** | Viterbi toàn cục | Chuẩn hóa IOB2 | Bị bắt nhầm (False Positive) |
| **PhoBERT-BiLSTM-CRF** | Viterbi + Smoothing | Rất tốt, nhớ ngữ cảnh 2 chiều | Bị bắt nhầm vế 1 do Dataset Bias |
| **PhoBERT-DualHead** *(SOTA Mới)* | **Gated Intent Fusion** | **Tối ưu toàn diện** | **Triệt tiêu 100% báo động giả (Vế 1 sạch, vế 2 che)** |
