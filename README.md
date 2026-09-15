# 🛡️ ViHOS Toxic Spans Guard: PhoBERT-BiLSTM-CRF
> **Đồ án môn học:** Xử lý Ngôn ngữ Tự nhiên (Natural Language Processing)  
> **Trường:** Đại học Công nghệ Thông tin – Đại học Quốc gia TP.HCM (UIT)  
> **Giảng viên hướng dẫn:** NCS.ThS. Đặng Văn Thìn & Tác giả Trần Quốc Khánh  
> **Bộ dữ liệu chuẩn:** [ViHOS Benchmark (EACL 2023)](https://github.com/phusroyal/ViHOS) (11.056 bình luận mạng xã hội Việt Nam)  
> **Kiến trúc ứng dụng:** Client-Server công nghiệp (FastAPI Backend + React 19 / Vite Frontend)

---

## 👥 Danh Sách Nhóm & Phân Công Nhiệm Vụ 5 Thành Viên

| STT | Họ và Tên | MSSV | Vai trò chính | Nhiệm vụ kỹ thuật cụ thể | Slide phụ trách |
| :---: | :--- | :---: | :--- | :--- | :---: |
| 1 | **Dương Quốc Thương** | 26410127 | **Project Leader & Architect** | Quản lý dự án, thiết kế kiến trúc mô hình, code lõi `src/model.py`, làm báo cáo tổng kết | **Slide 01–04 & 18** |
| 2 | **Nông Nguyễn Thành** | 26410115 | **Data & NLP Core** | Quản lý bộ dữ liệu ViHOS 3 tập (train, dev, test), xử lý Subword Alignment với PhoBERT, chạy EDA độ dài câu | **Slide 05–09** |
| 3 | **Hoàng Võ Minh Tuấn** | 26410146 | **Model Trainer** | Huấn luyện trên Google Colab GPU Tesla T4, tối ưu Differential Learning Rate, xuất 3 file checkpoint `.pt` | **Slide 10–11** |
| 4 | **Bùi Quốc Thịnh** | 26410108 | **Evaluation & Metrics** | Chạy đối chứng Ablation Study qua `seqeval`, xuất 4 biểu đồ báo cáo khoa học, phân loại 11 dạng lỗi trên 100 câu | **Slide 12–15** |
| 5 | **Trần Tiến Dũng** | 26410024 | **Product Developer** | Xây dựng kiến trúc Client-Server: Backend FastAPI + Frontend React 19 Vite, kiểm soát kết nối 3 mô hình, Auto-Masking (`***`) | **Slide 16–17** |

---

## 🔬 Kết Quả Nghiên Cứu Bóc Tách & Đối Chứng 3 Trường Phái

Nghiên cứu đối chứng thực nghiệm được thực hiện trên toàn bộ tập Test **1.106 câu** chuẩn của ViHOS Benchmark:

| Tiêu chí | 1. PhoBERT-Linear (Baseline Thầy) | 2. PhoBERT-CRF (Bóc tách Ablation) | 3. PhoBERT-BiLSTM-CRF (Đề xuất SOTA) |
| :--- | :--- | :--- | :--- |
| **Bản chất kiến trúc** | Softmax quyết định độc lập trên từng từ | CRF ràng buộc chuyển nhãn toàn cục | BiLSTM nhớ dài 2 chiều + CRF Viterbi |
| **Hình tượng ẩn dụ** | **Người gác cổng vội vàng:** Nhìn từng từ đơn lẻ để gắn nhãn, không quan tâm từ trước/sau là gì. | **Trọng tài tuân thủ luật lệ nghiêm ngặt:** Bắt buộc nhãn sau phải hợp lệ với nhãn trước (triệt tiêu lỗi cú pháp). | **Thám tử điều tra toàn diện:** Vừa thuộc luật chuyển nhãn (CRF), vừa có sổ tay ghi nhớ ngữ cảnh 2 chiều (BiLSTM). |
| **Lỗi cú pháp $O \to I\text{-HOS}$** | **Rất cao (26.4%)**: Nhãn $I$ xuất hiện vô cớ mà không có $B$ mở đầu. | **0.0% (Triệt tiêu 100%)** nhờ ma trận phạt chuyển trạng thái $A_{i,j}$. | **0.0% (Triệt tiêu 100%)** nhờ ma trận phạt chuyển trạng thái $A_{i,j}$. |
| **Lỗi ranh giới từ ghép tiếng Việt** | Cao (**25.8%**): Dễ cắt cụt từ ghép (*mất...* bỏ *dạy*). | Trung bình (**18.4%**). | **Thấp nhất (14.6% - Giảm 11.2%)**: Bóc tách nguyên vẹn ranh giới từ ghép. |
| **Xử lý câu đa cụm xúc phạm cách xa** | Kém: Thường chỉ bắt cụm đầu và bỏ sót các cụm phân tán phía sau. | Khá: Giảm sót cụm nhưng có thể gom nhầm từ sạch ở giữa vào span. | **Xuất sắc (+5.8% Recall):** Định vị chuẩn xác từng cụm phân tán độc lập. |
| **Span-Precision** | 67.12% | 69.40% | **71.15%** |
| **Span-Recall** | 65.46% | 67.85% | **69.50%** |
| **Span-F1 Benchmark** | **66.28%** | **68.61% (+2.33%)** | **70.31% (+4.03%)** |

---

## 📊 Biểu Đồ Báo Cáo Thực Nghiệm (Reports Figures)

### 1. So sánh F1 & Triệt tiêu Lỗi Chuyển nhãn
| So sánh Span-F1 giữa 3 mô hình | Tỷ lệ giảm các dạng lỗi chính |
| :---: | :---: |
| ![Ablation Span F1](reports/figures/01_ablation_span_f1.png) | ![Error Reduction](reports/figures/02_error_reduction_comparison.png) |

### 2. Ma trận Nhầm lẫn & Đa chuỗi (Multiple Spans)
| Ma trận nhầm lẫn BIO Tokens | Hiệu quả trên câu Đa chuỗi phân tán |
| :---: | :---: |
| ![Confusion Matrix](reports/figures/03_confusion_matrix_bio.png) | ![Multiple Spans Evaluation](reports/figures/04_multiple_spans_evaluation.png) |

---

## 📁 Cấu Trúc Thư Mục Repository Phân Tách (FE / BE)

```text
Doan/
├── backend/                            # [BE] Backend API Server (Python FastAPI)
│   ├── main.py                         # REST API endpoints (/api/predict, /api/health, /static)
│   ├── inference.py                    # Engine giải mã Viterbi CPU, Auto-Masking & phân loại nhãn
│   └── package.json                    # Cấu hình lệnh npm run dev cho backend
│
├── frontend/                           # [FE] Modern Web Dashboard (React 19 + Vite)
│   ├── src/
│   │   ├── App.jsx                     # Giao diện Modern SaaS Dashboard chuẩn 100% Mockup
│   │   ├── App.css                     # Styling chi tiết Full Width, Badges, Auto-masking
│   │   └── index.css                   # Base fonts (Inter) và layout
│   ├── package.json                    # React 19, Lucide-react, Vite
│   └── vite.config.js                  # Proxy cấu hình chuyển tiếp API sang port 8000
│
├── data/                               # Dữ liệu ViHOS Benchmark chuẩn 100% EACL 2023
│   ├── raw/                            # 3 file CSV gốc: train_BIO_Word.csv, dev_BIO_Word.csv, test_BIO_Word.csv
│   └── processed/                      # 3 file JSON BIO: train.json, dev.json, test.json
│
├── checkpoints/                        # 3 file trọng số PyTorch (.pt)
│   ├── baseline_phobert_linear.pt      # 1.61 GB
│   ├── baseline_phobert_crf.pt         # 1.61 GB
│   └── best_phobert_bilstm_crf.pt      # 1.64 GB
│
├── src/                                # Mã nguồn huấn luyện lõi
│   ├── dataset.py                      # First-token Subword Alignment & DataLoader
│   ├── model.py                        # Kiến trúc PhoBERT_BiLSTM_CRF, PhoBERT_CRF, PhoBERT_Linear
│   ├── train.py                        # Pipeline huấn luyện, Differential LR, EarlyStopping
│   └── evaluate.py                     # Đánh giá Span-F1 chuẩn seqeval
│
├── notebooks/                          # Notebooks huấn luyện & phân tích trên Google Colab
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_train_colab_gpu_t4.ipynb
│   └── 03_ablation_and_evaluation.ipynb
│
├── reports/                            # Báo cáo khoa học & Kịch bản Slide bảo vệ
│   ├── figures/                        # 4 biểu đồ khoa học 300 DPI
│   ├── error_analysis.xlsx             # File Excel phân tích 11 dạng lỗi trên 100 câu mẫu
│   ├── BAO_CAO_DO_AN_VIHOS.md          # Toàn văn báo cáo khoa học 7 chương
│   └── SLIDES_THUYET_TRINH_18_TRANG.md # Kịch bản 18 slide thuyết trình chuẩn 5 thành viên
│
├── run_backend.bat                     # Chạy riêng Backend (Port 8000)
├── run_frontend.bat                    # Chạy riêng Frontend (Port 5173)
├── run_all.bat                         # Khởi động 1-click cả FE và BE
└── package.json                        # Điều khiển hệ thống qua npm (npm run dev)
```

---

## 🚀 Hướng Dẫn Cài Đặt & Vận Hành Nhanh

### 1. Cài đặt Môi trường
Yêu cầu: **Python 3.10+** và **Node.js 18+** (máy tính đã có Node v22 & Python).

```bash
# Cài đặt thư viện Python
pip install -r requirements.txt

# Cài đặt thư viện Frontend
cd frontend && npm install && cd ..

# Cài đặt concurrently ở thư mục gốc
npm install
```

### 2. Khởi động Toàn bộ Hệ thống (1 Lệnh Duy Nhất)
Tại thư mục gốc `Doan/`, mở terminal và chạy:
```bash
npm run dev
```

* Cả **Backend (FastAPI - Port 8000)** và **Frontend (React/Vite - Port 5173)** sẽ cùng khởi động ngay trong 1 terminal duy nhất.
* Mở trình duyệt tại: **`http://localhost:5173`** để trải nghiệm giao diện.
* Xem tài liệu API Swagger tại: **`http://localhost:8000/docs`**.

### 3. Dừng Hệ thống (Stop)
Bấm tổ hợp phím **`Ctrl + C`** ngay tại terminal. Tiến trình sẽ dừng ngay lập tức trong **0.1 giây** mà không gặp bất kỳ hiện tượng treo hay giữ RAM nào.

---

## 🎯 Điểm Nhấn Đột Phá So Với Bài Báo Gốc (EACL 2023)
1. **Kiến trúc đề xuất SOTA (PhoBERT-BiLSTM-CRF):** Tăng F1 từ 66.28% lên **70.31% (+4.03%)**, triệt tiêu 100% lỗi cú pháp $O \to I\text{-HOS}$.
2. **Kỹ thuật First-token Subword Alignment:** Giải quyết triệt để vấn đề lệch ranh giới subword `@@` của BPE Tokenizer.
3. **Phân tích lỗi sâu sắc (Error Analysis):** Xác định bản chất hiện tượng over-smoothing ngữ cảnh của BiLSTM đối với từ lóng biến âm hiếm gặp làm tiền đề mở rộng từ điển tiếng lóng (Slang Lexicon Embeddings).
4. **Sản phẩm Web App Client-Server chuẩn công nghiệp:** Phân tách hoàn toàn FE và BE, thời gian phản hồi thời gian thực **~100 ms/câu** trên CPU thông thường.
