import os
import sys
import io
import json
import time
from datetime import datetime

# Đảm bảo UTF-8 encoding trên console Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Thêm thư mục gốc dự án vào sys.path để import module src
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.dataset import ViHOSDataset, vihos_collate_fn
from src.model import PhoBERT_Linear, PhoBERT_CRF, PhoBERT_BiLSTM_CRF, PhoBERT_DualHead_BiLSTM_CRF
from src.evaluate import evaluate_model, get_spans, compute_span_metrics
from src.utils import load_checkpoint, get_device, ID2LABEL

def count_transition_and_boundary_errors(true_tags_list, pred_tags_list):
    """
    Đếm số lỗi chuyển nhãn sai nguyên tắc BIO (O -> I-HOS)
    và số lỗi ranh giới từ (span trùng một phần nhưng không khớp exact boundary).
    """
    total_tokens = 0
    illegal_transitions = 0
    
    total_true_spans = 0
    total_pred_spans = 0
    exact_matches = 0
    overlap_matches = 0 # Trùng ít nhất 1 token nhưng lệch ranh giới
    
    for true_seq, pred_seq in zip(true_tags_list, pred_tags_list):
        total_tokens += len(pred_seq)
        
        # 1. Đếm lỗi O -> I-HOS
        prev_tag = "O"
        for tag in pred_seq:
            if tag == "I-HOS" and prev_tag == "O":
                illegal_transitions += 1
            prev_tag = tag
            
        # 2. Đếm lỗi ranh giới
        t_spans = get_spans(true_seq) # set of (start, end)
        p_spans = get_spans(pred_seq)
        total_true_spans += len(t_spans)
        total_pred_spans += len(p_spans)
        
        for p_start, p_end in p_spans:
            if (p_start, p_end) in t_spans:
                exact_matches += 1
            else:
                # Kiểm tra có giao với span thật nào không
                is_overlap = False
                for t_start, t_end in t_spans:
                    if max(p_start, t_start) <= min(p_end, t_end):
                        is_overlap = True
                        break
                if is_overlap:
                    overlap_matches += 1
                    
    trans_err_rate = (illegal_transitions / total_tokens * 100) if total_tokens > 0 else 0.0
    boundary_err_rate = (overlap_matches / total_pred_spans * 100) if total_pred_spans > 0 else 0.0
    
    return {
        "illegal_o_to_i_count": illegal_transitions,
        "illegal_o_to_i_pct": trans_err_rate,
        "boundary_error_count": overlap_matches,
        "boundary_error_pct": boundary_err_rate,
        "total_true_spans": total_true_spans,
        "total_pred_spans": total_pred_spans,
        "exact_matches": exact_matches
    }

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    reports_dir = os.path.join(root_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    log_file_path = os.path.join(reports_dir, "test_evaluation_official_log.txt")
    log_file = open(log_file_path, "w", encoding="utf-8")
    
    def log_both(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()
        
    start_time_all = time.time()
    log_both("=" * 80)
    log_both(f"CHƯƠNG TRÌNH ĐỐI CHIẾU & ĐÁNH GIÁ CHÍNH THỨC TRÊN TẬP TEST (ViHOS)")
    log_both(f"Thời gian bắt đầu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_both("=" * 80)
    
    test_path = os.path.join(root_dir, "data", "processed", "test.json")
    pretrained_name = "vinai/phobert-base-v2"
    batch_size = 32
    max_length = 128
    
    device = get_device()
    log_both(f"Device: {device}")
    log_both(f"Tập test: {test_path}")
    log_both(f"PhoBERT backbone: {pretrained_name}")
    log_both(f"Batch size: {batch_size}, Max length: {max_length}")
    
    # Nạp Tokenizer & Dataset
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    test_ds = ViHOSDataset(test_path, tokenizer=tokenizer, max_length=max_length)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: vihos_collate_fn(b, pad_token_id=tokenizer.pad_token_id)
    )
    log_both(f"Tổng số mẫu test đã nạp: {len(test_ds)} câu\n")
    
    configs = [
        {
            "name": "1. PhoBERT-Linear (Baseline Gốc)",
            "model_type": "phobert_linear",
            "checkpoint": os.path.join(root_dir, "checkpoints", "baseline_phobert_linear.pt"),
            "model_cls": PhoBERT_Linear
        },
        {
            "name": "2. PhoBERT-CRF (Ablation Bóc Tách)",
            "model_type": "phobert_crf",
            "checkpoint": os.path.join(root_dir, "checkpoints", "baseline_phobert_crf.pt"),
            "model_cls": PhoBERT_CRF
        },
        {
            "name": "3. PhoBERT-BiLSTM-CRF (Smoothing Bridge)",
            "model_type": "phobert_bilstm_crf",
            "checkpoint": os.path.join(root_dir, "checkpoints", "best_phobert_bilstm_crf.pt"),
            "model_cls": PhoBERT_BiLSTM_CRF
        },
        {
            "name": "4. PhoBERT-DualHead-BiLSTM-CRF (Đề Xuất Đa Nhiệm)",
            "model_type": "phobert_dualhead_bilstm_crf",
            "checkpoint": os.path.join(root_dir, "checkpoints", "best_phobert_dualhead_bilstm_crf.pt"),
            "model_cls": PhoBERT_DualHead_BiLSTM_CRF
        }
    ]
    
    results = []
    
    for cfg in configs:
        ckpt_path = cfg["checkpoint"]
        if not os.path.exists(ckpt_path):
            log_both(f"[BỎ QUA] Không tìm thấy checkpoint: {ckpt_path}")
            continue
            
        log_both("-" * 80)
        log_both(f"Đang đánh giá mô hình: {cfg['name']}")
        log_both(f"Checkpoint: {os.path.basename(ckpt_path)}")
        t0 = time.time()
        
        # Khởi tạo mô hình
        model = cfg["model_cls"](pretrained_name=pretrained_name)
        saved_metrics, epoch = load_checkpoint(ckpt_path, model, map_location=str(device))
        model.to(device)
        model.eval()
        
        log_both(f"Đã nạp checkpoint thành công (Được lưu tại Epoch {epoch})")
        
        # Thực hiện đánh giá
        eval_metrics = evaluate_model(model, test_loader, device)
        eval_time = time.time() - t0
        
        true_tags = eval_metrics["true_tags"]
        pred_tags = eval_metrics["pred_tags"]
        
        # Phân tích lỗi
        err_stats = count_transition_and_boundary_errors(true_tags, pred_tags)
        
        p = eval_metrics["span_precision"] * 100
        r = eval_metrics["span_recall"] * 100
        f1 = eval_metrics["span_f1"] * 100
        
        log_both(f"Thời gian đánh giá: {eval_time:.1f} giây")
        log_both(f"Kết quả Span-Level (IOB2 Exact Match):")
        log_both(f"   + Precision: {p:.2f}%")
        log_both(f"   + Recall:    {r:.2f}%")
        log_both(f"   + Span-F1:   {f1:.2f}%")
        log_both(f"Lỗi cú pháp chuỗi:")
        log_both(f"   + Lỗi O -> I-HOS: {err_stats['illegal_o_to_i_count']} lần ({err_stats['illegal_o_to_i_pct']:.2f}%)")
        log_both(f"   + Lỗi lệch ranh giới span: {err_stats['boundary_error_count']} / {err_stats['total_pred_spans']} ({err_stats['boundary_error_pct']:.2f}%)")
        log_both(f"Confusion Matrix (Token level):")
        log_both(eval_metrics["confusion_matrix"].to_string())
        log_both("")
        
        # Lưu file dự đoán chi tiết
        pred_filename = os.path.join(reports_dir, f"test_predictions_{cfg['model_type']}.json")
        pred_records = []
        for idx in range(len(true_tags)):
            sample_item = test_ds.examples[idx]
            tokens = sample_item.get("tokens", [])
            text_str = sample_item.get("text", " ".join(tokens) if isinstance(tokens, list) else str(tokens))
            pred_records.append({
                "index": idx,
                "text": text_str,
                "tokens": tokens,
                "true_tags": true_tags[idx],
                "pred_tags": pred_tags[idx],
                "true_spans": list(get_spans(true_tags[idx])),
                "pred_spans": list(get_spans(pred_tags[idx]))
            })
        with open(pred_filename, "w", encoding="utf-8") as pf:
            json.dump(pred_records, pf, ensure_ascii=False, indent=2)
        log_both(f"Đã xuất file dự đoán chi tiết từng câu ra: {os.path.basename(pred_filename)}")
        
        results.append({
            "Mô hình": cfg["name"],
            "Loại": cfg["model_type"],
            "Epoch tốt nhất": epoch,
            "Precision (%)": round(p, 2),
            "Recall (%)": round(r, 2),
            "Span-F1 (%)": round(f1, 2),
            "Lỗi O -> I-HOS (%)": round(err_stats["illegal_o_to_i_pct"], 2),
            "Lỗi lệch ranh giới (%)": round(err_stats["boundary_error_pct"], 2),
            "Thời gian eval (s)": round(eval_time, 1)
        })
        
    # Tạo bảng tổng kết
    df_summary = pd.DataFrame(results)
    log_both("=" * 80)
    log_both("BẢNG TỔNG KẾT ĐỐI CHIẾU CHÍNH THỨC TRÊN TOÀN BỘ 1.106 CÂU TẬP TEST:")
    log_both("=" * 80)
    log_both(df_summary.to_string(index=False))
    
    # Xuất file json và markdown
    json_path = os.path.join(reports_dir, "test_benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(results, jf, ensure_ascii=False, indent=2)
        
    md_path = os.path.join(reports_dir, "test_benchmark_results.md")
    with open(md_path, "w", encoding="utf-8") as mf:
        mf.write(f"# Kết quả Đánh giá Benchmark Chính thức — Toàn bộ 1.106 câu Test ViHOS\n\n")
        mf.write(f"*Thời gian thực hiện: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}*\n\n")
        mf.write(df_summary.to_markdown(index=False))
        mf.write("\n\n---\n")
        mf.write("### Ghi chú minh chứng:\n")
        mf.write("- Dữ liệu đánh giá: `data/processed/test.json` (toàn bộ 1.106 mẫu test độc lập, không rò rỉ tập huấn luyện/validation).\n")
        mf.write("- Đánh giá cấp độ Span theo chuẩn IOB2 Exact Match (Precision, Recall, Span-F1).\n")
        mf.write("- File log terminal đầy đủ: `reports/test_evaluation_official_log.txt`.\n")
        mf.write("- File chi tiết dự đoán từng câu: `reports/test_predictions_<model>.json`.\n")
        
    log_both(f"\n✅ Đã lưu file Markdown chính thức: {md_path}")
    log_both(f"✅ Đã lưu file JSON chính thức: {json_path}")
    log_both(f"Tổng thời gian toàn bộ quá trình: {time.time() - start_time_all:.1f} giây")
    log_both("=" * 80)
    log_file.close()

if __name__ == "__main__":
    main()
