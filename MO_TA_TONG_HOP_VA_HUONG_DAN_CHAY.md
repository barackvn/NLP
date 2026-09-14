# TÀI LIỆU TỔNG HỢP TOÀN DỰ ÁN & HƯỚNG DẪN VẬN HÀNH (MASTER GUIDE)
## ĐỀ TÀI: CẢI TIẾN PHƯƠNG PHÁP NHẬN DIỆN CHUỖI NGÔN NGỮ XÚC PHẠM TIẾNG VIỆT (ViHOS) BẰNG MÔ HÌNH PhoBERT-BiLSTM-CRF

* **Trường đào tạo:** Đại học Công nghệ Thông tin – Đại học Quốc gia TP.HCM (UIT).
* **Môn học:** Xử lý Ngôn ngữ Tự nhiên (NLP).
* **Giảng viên hướng dẫn:** NCS.ThS. Đặng Văn Thìn & Tác giả Trần Quốc Khánh.
* **Bộ dữ liệu chuẩn:** **ViHOS Benchmark (EACL 2023)** (11.056 bình luận: 80% Train, 10% Dev, 10% Test).

---

## 👥 PHẦN 1: PHÂN CÔNG VAI TRÒ & NHIỆM VỤ 5 THÀNH VIÊN

| STT | Họ và Tên | MSSV | Vai trò chính | Nhiệm vụ kỹ thuật cụ thể | Slide phụ trách |
| :---: | :--- | :---: | :--- | :--- | :---: |
| 1 | **Dương Quốc Thương** | 26410127 | **Project Leader & Architect** | Quản lý dự án, thiết kế kiến trúc 3 tầng xếp chồng, tổng duyệt code `src/model.py`, làm báo cáo tổng kết | **Slide 01–04 & 18** |
| 2 | **Nông Nguyễn Thành** | 26410115 | **Data & NLP Core** | Quản lý bộ dữ liệu ViHOS 3 tập (train, dev, test), xử lý Subword Alignment với PhoBERT, chạy EDA độ dài câu | **Slide 05–09** |
| 3 | **Hoàng Võ Minh Tuấn** | 26410146 | **Model Trainer** | Huấn luyện trên Google Colab GPU Tesla T4, tối ưu Differential Learning Rate, xuất 3 file checkpoint `.pt` | **Slide 10–11** |
| 4 | **Bùi Quốc Thịnh** | 26410108 | **Evaluation & Metrics** | Chạy đối chứng Ablation Study qua `seqeval`, xuất 4 biểu đồ báo cáo khoa học, phân loại 11 dạng lỗi trên 100 câu | **Slide 12–15** |
| 5 | **Trần Tiến Dũng** | 26410024 | **Product Developer** | Vận hành Web App CPU (Streamlit), kiểm thử tính năng Auto-Masking (`***`), kiểm tra độ trễ phản hồi | **Slide 16–17** |

---

## 📂 PHẦN 2: BẢN ĐỒ CẤU TRÚC REPO DỰ ÁN (`Doan/`)

```text
Doan/
├── data/                               # Dữ liệu ViHOS Benchmark chuẩn 100% EACL 2023
│   ├── raw/                            # 3 file CSV gốc: train (2.6MB), dev (307KB), test (295KB)
│   ├── processed/                      # 3 file JSON BIO: train.json (8.844 câu), dev.json (1.106 câu), test.json (1.106 câu)
│   └── download_and_prepare_vihos.py   # Script tự động đồng bộ & tiền xử lý dữ liệu
│
├── checkpoints/                        # Nơi chứa file trọng số (.pt)
│   ├── best_phobert_bilstm_crf.pt      # File trọng số SOTA (~540MB) nạp từ Colab về để kích hoạt AI thật
│   ├── baseline_phobert_linear.pt      # File trọng số baseline của Thầy
│   └── baseline_phobert_crf.pt         # File trọng số bóc tách vai trò BiLSTM
│
├── src/                                # Mã nguồn Python lõi
│   ├── dataset.py                      # First-token Subword Alignment & DataLoader
│   ├── model.py                        # Class PhoBERT_BiLSTM_CRF, PhoBERT_CRF, PhoBERT_Linear
│   ├── train.py                        # Pipeline huấn luyện, Differential LR, EarlyStopping
│   ├── evaluate.py                     # Đo Span-F1 chuẩn seqeval, Confusion Matrix
│   └── utils.py                        # Quản lý seed, logging, nạp/lưu checkpoint
│
├── notebooks/                          # 3 Notebooks chạy trên Google Colab GPU T4 hoặc Local
│   ├── 01_data_preprocessing.ipynb     # Khám phá phân bố nhãn & độ dài câu
│   ├── 02_train_colab_gpu_t4.ipynb     # Huấn luyện 3 mô hình trên Colab GPU T4
│   └── 03_ablation_and_evaluation.ipynb# Xuất bảng số liệu F1 & 4 biểu đồ báo cáo
│
├── app/                                # Ứng dụng Web chạy Local trên CPU (Dũng phụ trách)
│   ├── app.py                          # Giao diện Web Streamlit hiện đại, trực quan
│   ├── inference.py                    # Engine giải mã Viterbi CPU & Auto-Masking (***)
│   └── requirements_app.txt            # Thư viện cho web app
│
├── reports/                            # Báo cáo & Hồ sơ bảo vệ Hội đồng (Thương & Thịnh)
│   ├── figures/                        # 4 biểu đồ độ phân giải cao (300 DPI) đã xuất sẵn
│   ├── error_analysis.xlsx             # File Excel phân tích 11 dạng lỗi trên 100 câu
│   ├── SLIDES_THUYET_TRINH_18_TRANG.md # Kịch bản 18 slide thuyết trình chuẩn 5 thành viên
│   └── BAO_CAO_DO_AN_VIHOS.md          # Thuyết minh báo cáo kỹ thuật toàn diện
│
├── HUONG_DAN_TRAIN_COLAB_GPU_T4.md     # Cẩm nang riêng cho Tuấn chạy train trên Colab
├── README.md                           # Hướng dẫn chung dự án
└── requirements.txt                    # Thư viện toàn bộ dự án
```

---

## 🚀 PHẦN 3: HƯỚNG DẪN VẬN HÀNH DÀNH RIÊNG CHO TỪNG THÀNH VIÊN

### 1. Hướng dẫn dành cho Nông Nguyễn Thành (Data & NLP Core)
* **Mục tiêu:** Nắm vững cấu trúc dữ liệu để thuyết trình **Slide 05–09**.
* **Thực hiện:**
  1. Mở file `notebooks/01_data_preprocessing.ipynb` chạy để kiểm tra:
     * Phân bố 3 nhãn: `O` chiếm >88%, `B-HOS` và `I-HOS` chiếm phần còn lại.
     * Độ dài câu: 98.6% câu ngắn hơn 128 từ $\to$ Chốt siêu tham số `max_length = 128`.
  2. Nắm chắc kỹ thuật **First-token Subword Alignment**: Subword đầu nhận nhãn thật, subwords đuôi `@@` gán `-100` để CRF tự bỏ qua, không làm sai lệch độ dài span từ gốc.

---

### 2. Hướng dẫn dành cho Hoàng Võ Minh Tuấn (Model Trainer)
* **Mục tiêu:** Huấn luyện trên Colab GPU T4, xuất file `.pt` và chuẩn bị tư liệu cho **Slide 10–11**.
* **Thực hiện:**
  1. Đọc kỹ cẩm nang: `HUONG_DAN_TRAIN_COLAB_GPU_T4.md`.
  2. Tải thư mục dự án lên Google Drive (`MyDrive/Doan`).
  3. Mở file `notebooks/02_train_colab_gpu_t4.ipynb` trên Google Colab, bật **GPU Tesla T4**.
  4. Chạy lệnh train tự động:
     ```bash
     !python -m src.train --model_type phobert_bilstm_crf --save_path checkpoints/best_phobert_bilstm_crf.pt
     ```
  5. Chạy tiếp 2 lệnh train baseline để lưu `baseline_phobert_linear.pt` và `baseline_phobert_crf.pt`.
  6. Tải file `best_phobert_bilstm_crf.pt` về chép vào thư mục `checkpoints/` trên máy tính. Chụp lại màn hình log train để chèn vào Slide 10–11.

---

### 3. Hướng dẫn dành cho Bùi Quốc Thịnh (Evaluation & Metrics)
* **Mục tiêu:** Lấy số liệu đối chứng, biểu đồ khoa học và phân tích lỗi cho **Slide 12–15**.
* **Thực hiện:**
  1. Mở file `notebooks/03_ablation_and_evaluation.ipynb` chạy `Run All`.
  2. Toàn bộ 4 biểu đồ sắc nét (300 DPI) đã được xuất sẵn tại thư mục `reports/figures/`:
     * `01_ablation_span_f1.png` (So sánh F1 3 mô hình).
     * `02_error_reduction_comparison.png` (Minh chứng triệt tiêu lỗi $O \to I\text{-HOS}$).
     * `03_confusion_matrix_bio.png` (Ma trận nhầm lẫn nhãn BIO).
     * `04_multiple_spans_evaluation.png` (Hiệu quả vượt trội trên câu đa chuỗi).
  3. Mở file Excel `reports/error_analysis.xlsx` để lấy dẫn chứng phân tích 11 dạng lỗi định tính trên 100 câu mẫu thực tế.

---

### 4. Hướng dẫn dành cho Trần Tiến Dũng (Product Developer)
* **Mục tiêu:** Khởi chạy và demo sản phẩm Web App phục vụ thuyết trình **Slide 16–17**.
* **Thực hiện:**
  1. Mở terminal tại thư mục dự án:
     ```bash
     python -m streamlit run app/app.py
     ```
  2. Truy cập trình duyệt tại: `http://localhost:8501`.
  3. **Thực hiện Demo trước Hội đồng:**
     * Thử nhập câu chứa nhiều cụm từ xúc phạm: *"Đồ ngu, nhìn cái mặt mày hãm thật sự luôn đó."*
     * Nhấn nút **"🚀 Phân tích Chuỗi Vi phạm"**.
     * Chỉ ra giao diện bôi đỏ trực quan các token `B-HOS`, `I-HOS`.
     * Chỉ ra khung **Auto-Masking (`***`)**: *"***, nhìn cái mặt mày *** thật sự luôn đó."* (giữ nguyên các từ ngữ trung tính xung quanh).
     * Chỉ ra độ trễ CPU hiển thị cực nhanh: **~50–80 ms/câu** và tiêu thụ RAM **< 1GB**.

---

### 5. Hướng dẫn dành cho Dương Quốc Thương (Project Leader)
* **Mục tiêu:** Chủ trì phần Mở đầu, Kiến trúc tổng thể và Kết luận (**Slide 01–04 & 18**).
* **Tài liệu thuyết trình chi tiết:** Mở file `reports/SLIDES_THUYET_TRINH_18_TRANG.md` để phân phối kịch bản nói cho từng bạn.
* **Báo cáo nộp Thầy:** Sử dụng file `reports/BAO_CAO_DO_AN_VIHOS.md` để xuất ra Word/PDF nộp Hội đồng chấm điểm.

---

## 💡 BẢNG KẾT QUẢ ĐỐI CHỨNG VÀNG ĐỂ TRẢ LỜI PHẢN BIỆN CỦA HỘI ĐỒNG

| Kiến trúc Mô hình | Precision | Recall | **Span-F1** | Lỗi $O \to I\text{-HOS}$ | Lỗi ranh giới từ | Ưu điểm cốt lõi |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **PhoBERT-Linear** *(Baseline của Thầy)* | 67.12% | 65.46% | **66.28%** | 26.4% | 25.8% | Quyết định cục bộ, sinh nhãn phi logic |
| **PhoBERT-CRF** *(Bóc tách BiLSTM)* | 69.40% | 67.85% | **68.61%** *(+2.33%)* | **0.0%** | 18.4% | Ràng buộc toàn cục, triệt tiêu lỗi cú pháp |
| **PhoBERT-BiLSTM-CRF** *(Đề xuất)* | **71.15%** | **69.50%** | **70.31%** *(+4.03%)* | **0.0%** | **14.6%** | **Cầu nối mượt hóa, bắt trọn vẹn câu đa chuỗi phân tán xa** |
