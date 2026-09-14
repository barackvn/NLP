# HƯỚNG DẪN CHI TIẾT: HUẤN LUYỆN MÔ HÌNH TRÊN GOOGLE COLAB (GPU T4) & XUẤT CHECKPOINT
> **Dành cho:** Hoàng Võ Minh Tuấn (Trainer - 26410146) & Nhóm nghiên cứu ViHOS  
> **Mục tiêu:** Huấn luyện mô hình Deep Learning SOTA `PhoBERT-BiLSTM-CRF`, xuất file trọng số `.pt` (~540MB) đưa vào Web App và thu thập số liệu Ablation Study cho Slide 10–15.

---

## 🎯 KẾT QUẢ ĐẦU RA BẮT BUỘC CẦN BÀN GIAO (DELIVERABLES)
1. **File trọng số chính:** `best_phobert_bilstm_crf.pt` (Đưa vào thư mục `Doan/checkpoints/` trên máy local để kích hoạt Web App AI thật).
2. **Hai file trọng số đối chứng (Ablation):**
   * `baseline_phobert_linear.pt` (Baseline gốc của Thầy).
   * `baseline_phobert_crf.pt` (Bóc tách vai trò của tầng BiLSTM).
3. **Ảnh chụp màn hình (Screenshots):** Log quá trình train, biểu đồ Loss và F1 qua từng epoch để làm tư liệu đưa vào Slide bảo vệ (Slide 10–11).

---

## ⏱️ THỜI GIAN & CHI PHÍ TÀI NGUYÊN
* **Phần cứng:** GPU Tesla T4 (16GB VRAM) – **Miễn phí 100%** trên Google Colab.
* **Thời gian huấn luyện:** Khoảng **18 – 20 phút / mô hình** (5 Epochs trên tập dữ liệu 8.844 câu).

---

## 🚀 QUY TRÌNH THỰC HIỆN TỪNG BƯỚC (STEP-BY-STEP)

### BƯỚC 1: Tải thư mục dự án lên Google Drive
1. Mở Google Drive cá nhân của bạn.
2. Tải toàn bộ thư mục `Doan` (đang có trên máy tính) lên Google Drive tại thư mục gốc:
   👉 Đường dẫn chuẩn trên Drive: `My Drive/Doan`
   *(Đảm bảo trong thư mục `Doan/data/processed/` đã có đủ 3 file: `train.json`, `dev.json`, `test.json`)*.

---

### BƯỚC 2: Mở Google Colab & Bật GPU Tesla T4
1. Truy cập: [Google Colab](https://colab.research.google.com/).
2. Chọn thẻ **Upload** (Tải lên) $\to$ Chọn file: `Doan/notebooks/02_train_colab_gpu_t4.ipynb`.
3. **BẬT GPU:**
   * Trên thanh menu Colab, chọn: **Runtime** $\to$ **Change runtime type** (Thay đổi loại thời gian chạy).
   * Tại mục **Hardware accelerator** (Bộ tăng tốc phần cứng): Chọn **T4 GPU**.
   * Nhấn **Save** (Lưu).

---

### BƯỚC 3: Kết nối Drive & Cài đặt thư viện (Chạy trên Colab)
Tạo hoặc chạy các cell lệnh sau trong Notebook:

```python
# 1. Kiểm tra thông tin GPU Tesla T4
!nvidia-smi

# 2. Kết nối Google Drive để lưu checkpoint vĩnh viễn (không sợ mất khi rớt mạng)
from google.colab import drive
import os

drive.mount('/content/drive')

# Chuyển vào thư mục dự án trên Drive
%cd /content/drive/MyDrive/Doan

# 3. Cài đặt các thư viện cần thiết
!pip install -q transformers pyvi pytorch-crf seqeval accelerate
```

---

### BƯỚC 4: Huấn luyện Mô hình Đề xuất (PhoBERT-BiLSTM-CRF)
Chạy lệnh sau để bắt đầu huấn luyện. Quá trình này chạy tự động 5 epochs và tự động lưu checkpoint có điểm Span-F1 cao nhất:

```bash
!python -m src.train \
    --model_type phobert_bilstm_crf \
    --train_path data/processed/train.json \
    --dev_path data/processed/dev.json \
    --save_path checkpoints/best_phobert_bilstm_crf.pt \
    --epochs 5 \
    --batch_size 16 \
    --lr_phobert 2e-5 \
    --lr_head 1e-3 \
    --max_length 128 \
    --patience 3
```

* **Dấu hiệu hoàn thành:** Sau khoảng 18–20 phút, log màn hình sẽ báo:  
  `[SUCCESS] Training completed! Best Dev Span-F1: ~70.xx%`  
  File `best_phobert_bilstm_crf.pt` (~540MB) đã nằm an toàn trong thư mục `Doan/checkpoints/` trên Google Drive.

---

### BƯỚC 5: Huấn luyện 2 Mô hình Đối chứng (Ablation Baselines)
Để có số liệu lập Bảng so sánh nộp Thầy và Hội đồng, chạy tiếp 2 lệnh sau:

```bash
# 1. Train Baseline gốc của Thầy (PhoBERT-Linear)
!python -m src.train \
    --model_type phobert_linear \
    --train_path data/processed/train.json \
    --dev_path data/processed/dev.json \
    --save_path checkpoints/baseline_phobert_linear.pt \
    --epochs 5 \
    --batch_size 16 \
    --lr_phobert 2e-5 \
    --lr_head 1e-3

# 2. Train Mô hình bóc tách BiLSTM (PhoBERT-CRF)
!python -m src.train \
    --model_type phobert_crf \
    --train_path data/processed/train.json \
    --dev_path data/processed/dev.json \
    --save_path checkpoints/baseline_phobert_crf.pt \
    --epochs 5 \
    --batch_size 16 \
    --lr_phobert 2e-5 \
    --lr_head 1e-3
```

---

### BƯỚC 6: Tải Checkpoint về Máy Local & Kích hoạt Web App
1. Vào Google Drive cá nhân trên máy tính:
   * Vào thư mục: `MyDrive/Doan/checkpoints/`
   * Tải file: **`best_phobert_bilstm_crf.pt`** về máy tính cá nhân.
2. Đặt file vừa tải vào đúng thư mục checkpoints trong dự án:  
   👉 `checkpoints/best_phobert_bilstm_crf.pt` (hoặc `Doan/checkpoints/` tùy thư mục gốc của bạn).
3. Mở lại trình duyệt Web App Streamlit (`http://localhost:8501`) và nhấn **F5**:
   * ✅ Banner vàng biến mất $\to$ chuyển thành **Banner Xanh lá cây**.
   * ✅ Trạng thái chuyển thành: **`🟢 MÔ HÌNH ĐÃ TRAIN (AI ACTIVE)`**.
   * ✅ Độ trễ xử lý thực tế trên CPU hiển thị: **50 – 80 ms / câu**.

---

## 🛠️ XỬ LÝ LỖI THƯỜNG GẶP (TROUBLESHOOTING)

| Lỗi | Nguyên nhân | Cách khắc phục |
| :--- | :--- | :--- |
| `CUDA out of memory` | VRAM bị đầy do batch size lớn | Đổi `--batch_size 16` thành `--batch_size 8` trong câu lệnh train. |
| `FileNotFoundError: data/processed/train.json` | Sai đường dẫn làm việc trên Colab | Kiểm tra xem lệnh `%cd /content/drive/MyDrive/Doan` đã chạy đúng thư mục chưa bằng cách gõ `!ls`. |
| Mất mạng / Colab ngắt kết nối giữa chừng | Giới hạn phiên của Colab | Không lo lắng! Code đã lưu checkpoint sau mỗi epoch vào Google Drive, bạn chỉ cần mở lại Colab và mount lại Drive. |
