# Checkpoints Directory
Thư mục này dùng để lưu trữ các file trọng số mô hình đã huấn luyện (.pt):

1. `best_phobert_bilstm_crf.pt` (~540MB) - Mô hình đề xuất SOTA (Tải về từ Google Colab GPU T4 để kích hoạt AI thật trên Web App).
2. `baseline_phobert_linear.pt` (~540MB) - Checkpoint mô hình baseline gốc của Thầy.
3. `baseline_phobert_crf.pt` (~540MB) - Checkpoint mô hình bóc tách để phục vụ Ablation Study.

> Lưu ý: Các file .pt có dung lượng lớn (~540MB mỗi file) nên không commit lên git public, chỉ lưu trữ cục bộ và trên Google Drive nhóm.
