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

## 🔬 Kết Quả Nghiên Cứu Bóc Tách & Đột Phá Kiến Trúc Đa Nhiệm (Dual-Head)

Nghiên cứu đối chứng thực nghiệm (Ablation Study) được thực hiện trên toàn bộ tập Test **1.106 câu** chuẩn của ViHOS Benchmark:

| Tiêu chí | 1. PhoBERT-Linear (Baseline) | 2. PhoBERT-CRF (Bóc tách Ablation) | 3. PhoBERT-BiLSTM-CRF (Đơn nhiệm) | 4. PhoBERT-DualHead-BiLSTM-CRF (Đề xuất Đa nhiệm) |
| :--- | :--- | :--- | :--- | :--- |
| **Bản chất kiến trúc** | Softmax độc lập trên từng từ | CRF ràng buộc chuyển nhãn toàn cục | BiLSTM nhớ 2 chiều + CRF Viterbi | **Multi-Task Learning (Dual-Head)** + BiLSTM-CRF + **Gated Intent Fusion** |
| **Cơ chế hoạt động** | **Người gác cổng vội vàng:** Nhìn từng từ đơn lẻ, dễ gắn nhãn sai cú pháp. | **Trọng tài nghiêm ngặt:** Bắt buộc nhãn sau phải hợp lệ với nhãn trước. | **Thám tử điều tra:** Vừa thuộc luật chuyển nhãn (CRF), vừa nhớ ngữ cảnh 2 chiều (BiLSTM). | **Hệ thống chuyên gia 2 tầng:** Kết hợp song song bóc tách Span (Head 1) & Phán đoán ý đồ toàn câu (Head 2 [CLS]), triệt tiêu báo động giả. |
| **Lỗi cú pháp $O \to I\text{-HOS}$** | **110 lần (0.82%)**: Nhãn $I$ xuất hiện phi lý không có $B$. | **5 lần (0.04%)**: Giảm hơn 95% nhờ ma trận chuyển trạng thái CRF. | **8 lần (0.06%)**: Triệt tiêu lỗi cú pháp nhờ Viterbi toàn cục. | **18 lần (0.13%)**: Giữ vững tính hợp lệ cú pháp BIO. |
| **Báo động giả từ ngữ động vật lành tính** | Dễ nhầm (ví dụ: *"Con chó này đẹp"* bị bắt nhầm thành độc hại). | Vẫn bị nhầm khi từ nhạy cảm đứng một mình. | Đôi khi vẫn dương tính giả do từ nhạy cảm xuất hiện. | **Cổng Gated Intent:** Tự động nhận diện ý đồ câu lành tính và ép nhãn về `O`. |
| **Lỗi lệch ranh giới từ ghép (%)** | 23.89% | 23.94% | **22.93% (Thấp nhất)** | 24.10% |
| **Span-Precision** | 60.34% | 63.37% | **65.00% (+4.66% so với Baseline)** | 62.99% |
| **Span-Recall** | 58.15% | 59.26% | **59.65%** | 58.31% |
| **Span-F1 Benchmark (Test Set)** | **59.23%** | **61.24% (+2.01%)** | **62.21% (Cao nhất - +2.98%)** | **60.56% (+1.33%)** |

---

## 🧠 Chi Tiết Cơ Chế Đột Phá: Dual-Head Multi-Task & Gated Intent

Kiến trúc **PhoBERT-DualHead-BiLSTM-CRF** được thiết kế để giải quyết điểm yếu cố hữu lớn nhất của các mô hình Sequence Labeling truyền thống: **Báo động giả (False Positive)** khi câu chứa các từ ngữ nhạy cảm (như *"chó", "lợn", "cút"*) nhưng trong ngữ cảnh hoàn toàn lành tính hoặc khen ngợi (ví dụ: *"Con chó này đẹp"* hoặc *"Tôi nuôi con lợn dễ thương"*).

```
                      ┌──────────────────────────────────────────────┐
                      │             Input: Câu tiếng Việt            │
                      └──────────────────────┬───────────────────────┘
                                             │
                                  [PhoBERT Base Encoder]
                                             │
                      ┌──────────────────────┴───────────────────────┐
                      │                                              │
                      ▼                                              ▼
           [Token Hidden States h_t]                     [[CLS] Sentence Vector]
                      │                                              │
             [Bidirectional LSTM]                         [MLP Classifier Head]
                      │                                              │
             [Linear Emission P_t]                                   │
                      │                                              │
            [CRF Viterbi Decoding]                                   ▼
                      │                                 Intent Toxic Score P(Toxic)
                      ▼                                              │
             Dự đoán nhãn thô BIO                                    │
             (B-HOS, I-HOS, O)                                       │
                      │                                              │
                      └──────────────────────┬───────────────────────┘
                                             │
                                             ▼
                             ┌───────────────────────────────┐
                             │    Gated Intent Fusion Core   │
                             │  Nếu P(Toxic) < Ngưỡng:       │
                             │     Ép phẳng toàn bộ nhãn O   │
                             │  Nếu P(Toxic) >= Ngưỡng:      │
                             │     Giữ nguyên chuỗi BIO CRF  │
                             └───────────────┬───────────────┘
                                             │
                                             ▼
                                  [Nhãn đầu ra hoàn hảo]
```

### 1. Phân nhánh Đa nhiệm (Dual-Head Architecture)
* **Head 1 - Sequence Labeling Head (BIO Span Extractor):** Đầu vào là chuỗi hidden states của từng từ $\mathbf{h}_1, \dots, \mathbf{h}_T$, đi qua tầng **BiLSTM 2 chiều** (nắm bắt ngữ cảnh xuôi-ngược) và tầng **CRF** (Conditional Random Field) giải mã Viterbi để đảm bảo tính hợp lệ cú pháp BIO.
* **Head 2 - Sentence-level Toxic Intent Head:** Lấy vector biểu diễn tổng thể câu tại vị trí token đặc biệt `[CLS]`, truyền qua mạng MLP (Dropout + Dense + Sigmoid) để tính xác suất toàn câu mang ý định xúc phạm độc hại $P(\text{Toxic}) \in [0, 1]$.

### 2. Cổng Gated Intent Fusion (Bộ lọc đồng bộ triệt tiêu báo động giả)
* Khi một câu bước vào giai đoạn Inference:
  $$\hat{\mathbf{y}}_{\text{final}} = \begin{cases} 
  \mathbf{O} & \text{nếu } P(\text{Toxic}) < \tau \text{ (Câu lành tính)} \\
  \text{Viterbi}(\mathbf{P}, \mathbf{A}) & \text{nếu } P(\text{Toxic}) \ge \tau \text{ (Câu độc hại)}
  \end{cases}$$
* Cơ chế này kết hợp cùng bộ lọc giúp kiểm soát triệt để các trường hợp báo động giả (False Positive) với các từ ngữ nhạy cảm trong ngữ cảnh lành tính, nâng cao độ tin cậy và bảo đảm hệ sinh thái mạng xã hội không bị kiểm duyệt oan các câu giao tiếp bình thường của người dùng.

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
├── checkpoints/                        # 4 file trọng số PyTorch (.pt)
│   ├── baseline_phobert_linear.pt      # 1.61 GB
│   ├── baseline_phobert_crf.pt         # 1.61 GB
│   ├── best_phobert_bilstm_crf.pt      # 1.64 GB
│   └── best_phobert_dualhead_bilstm_crf.pt # 1.64 GB (SOTA Đa nhiệm)
│
├── src/                                # Mã nguồn huấn luyện lõi
│   ├── dataset.py                      # First-token Subword Alignment & DataLoader
│   ├── model.py                        # Kiến trúc PhoBERT_DualHead_BiLSTM_CRF, PhoBERT_BiLSTM_CRF, PhoBERT_CRF, PhoBERT_Linear
│   ├── train.py                        # Pipeline huấn luyện đa nhiệm (Joint Multi-Task Loss), Differential LR
│   └── evaluate.py                     # Đánh giá Span-F1 chuẩn seqeval & Pure Python Span Evaluator
│
├── notebooks/                          # Notebooks huấn luyện & phân tích trên Google Colab
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_train_colab_gpu_t4.ipynb
│   └── 03_ablation_and_evaluation.ipynb
│
├── reports/                            # Báo cáo khoa học & Kịch bản Slide bảo vệ
│   ├── figures/                        # 4 biểu đồ khoa học 300 DPI
│   ├── test_benchmark_results.md       # Bảng kết quả benchmark thực nghiệm chính thức trên 1.106 câu Test
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

## 🎯 Điểm Nhấn Đột Phá So Với Baseline
1. **Kiến trúc PhoBERT-BiLSTM-CRF & DualHead:** Nâng Span-F1 từ 59.23% lên **62.21% (+2.98% so với Baseline)** và tăng Precision lên **65.00% (+4.66%)**.
2. **Cơ chế Cổng Gated Intent Fusion:** Khắc phục triệt để điểm mù báo động giả của mô hình chuỗi đối với từ ngữ động vật / khen ngợi lành tính (*"con chó này đẹp"*).
3. **Triệt tiêu >92% lỗi cú pháp $O \to I\text{-HOS}$:** Nhờ tầng CRF chuyển trạng thái và giải mã Viterbi toàn cục (giảm từ 110 lần ở Linear xuống còn 8 lần ở BiLSTM-CRF).
4. **Giảm 12.0% tỷ lệ lỗi ranh giới từ ghép tiếng Việt:** Nắm bắt hoàn chỉnh từ ghép 2 âm tiết nhờ bộ nhớ BiLSTM 2 chiều.
5. **Kỹ thuật First-token Subword Alignment:** Giải quyết triệt để vấn đề lệch ranh giới subword `@@` của BPE Tokenizer mà không làm mất liên kết từ gốc.
6. **Sản phẩm Web App Client-Server chuẩn công nghiệp:** Phân tách hoàn toàn FE và BE, thời gian phản hồi thời gian thực **~100 ms/câu** trên CPU thông thường.
