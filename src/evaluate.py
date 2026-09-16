import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd

from .utils import ID2LABEL, IGNORE_INDEX

try:
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
    from seqeval.scheme import IOB2
    HAS_SEQEVAL = True
except ImportError:
    HAS_SEQEVAL = False

def get_spans(tags):
    """Trích xuất danh sách các thực thể span theo chuẩn IOB2."""
    spans = set()
    start = None
    for i, t in enumerate(tags):
        if t == 'B-HOS':
            if start is not None:
                spans.add((start, i - 1))
            start = i
        elif t == 'I-HOS':
            if start is None:
                start = None
        else: # 'O'
            if start is not None:
                spans.add((start, i - 1))
                start = None
    if start is not None:
        spans.add((start, len(tags) - 1))
    return spans

def compute_span_metrics(true_list, pred_list):
    """Tính toán Precision, Recall, Span-F1 chuẩn IOB2 exact match không cần thư viện ngoài."""
    total_tp, total_pred, total_true = 0, 0, 0
    for t_seq, p_seq in zip(true_list, pred_list):
        t_spans = get_spans(t_seq)
        p_spans = get_spans(p_seq)
        total_true += len(t_spans)
        total_pred += len(p_spans)
        total_tp += len(t_spans & p_spans)
    p = total_tp / total_pred if total_pred > 0 else 0.0
    r = total_tp / total_true if total_true > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1

def evaluate_model(model, dataloader, device):
    """
    Đánh giá mô hình trên tập validation/test sử dụng seqeval.
    Chỉ lấy các vị trí valid_mask (subword đầu tiên của mỗi từ) để so sánh word-level span F1.
    """
    model.eval()
    all_true_tags = []
    all_pred_tags = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            word_indices = batch.get("word_indices")
            if word_indices is not None:
                word_indices = word_indices.to(device)
            word_mask = batch.get("word_mask")
            if word_mask is not None:
                word_mask = word_mask.to(device)
            labels = batch["labels"].to(device)
            
            # Giải mã nhãn dự đoán qua decode method cấp độ word
            predictions = model.decode(input_ids, attention_mask, word_indices=word_indices, word_mask=word_mask)
            
            labels_np = labels.cpu().numpy()
            mask_np = word_mask.cpu().numpy() if word_mask is not None else batch.get("valid_mask", torch.ones_like(labels)).cpu().numpy()
            
            for i in range(len(predictions)):
                pred_seq = predictions[i]
                
                # Trích xuất nhãn thực tế tại các vị trí word hợp lệ
                true_seq = []
                for label_id, is_valid in zip(labels_np[i], mask_np[i]):
                    if is_valid and label_id != IGNORE_INDEX:
                        true_seq.append(ID2LABEL.get(label_id, "O"))
                
                # Chuẩn hóa độ dài dự đoán khớp 1:1 với nhãn thực tế
                pred_seq_labels = [ID2LABEL.get(p, "O") for p in pred_seq[:len(true_seq)]]
                if len(pred_seq_labels) < len(true_seq):
                    pred_seq_labels.extend(["O"] * (len(true_seq) - len(pred_seq_labels)))
                
                all_true_tags.append(true_seq)
                all_pred_tags.append(pred_seq_labels)

    # Tính toán các chỉ số Span-Level (chuẩn IOB2 Exact Match)
    if HAS_SEQEVAL:
        span_p = precision_score(all_true_tags, all_pred_tags, mode='strict', scheme=IOB2)
        span_r = recall_score(all_true_tags, all_pred_tags, mode='strict', scheme=IOB2)
        span_f1 = f1_score(all_true_tags, all_pred_tags, mode='strict', scheme=IOB2)
        report = classification_report(all_true_tags, all_pred_tags, mode='strict', scheme=IOB2)
    else:
        span_p, span_r, span_f1 = compute_span_metrics(all_true_tags, all_pred_tags)
        report = f"Span-Precision: {span_p*100:.2f}%\nSpan-Recall: {span_r*100:.2f}%\nSpan-F1: {span_f1*100:.2f}%"

    # Token-level confusion matrix
    flat_true = [t for seq in all_true_tags for t in seq]
    flat_pred = [p for seq in all_pred_tags for p in seq]
    labels_order = ["O", "B-HOS", "I-HOS"]
    cm = confusion_matrix(flat_true, flat_pred, labels=labels_order)
    cm_df = pd.DataFrame(cm, index=[f"True_{l}" for l in labels_order], columns=[f"Pred_{l}" for l in labels_order])

    metrics = {
        "span_precision": span_p,
        "span_recall": span_r,
        "span_f1": span_f1,
        "classification_report": report,
        "confusion_matrix": cm_df,
        "true_tags": all_true_tags,
        "pred_tags": all_pred_tags
    }
    return metrics


if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from .dataset import ViHOSDataset, vihos_collate_fn
    from .model import PhoBERT_BiLSTM_CRF, PhoBERT_CRF, PhoBERT_Linear, PhoBERT_DualHead_BiLSTM_CRF
    from .utils import load_checkpoint, get_device

    parser = argparse.ArgumentParser(description="Đánh giá mô hình ViHOS Toxic Spans Detection")
    parser.add_argument("--model_type", type=str, required=True, choices=["phobert_dualhead_bilstm_crf", "phobert_bilstm_crf", "phobert_crf", "phobert_linear"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_path", type=str, default="data/processed/test.json")
    parser.add_argument("--pretrained_name", type=str, default="vinai/phobert-base-v2")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_name)

    test_ds = ViHOSDataset(args.test_path, tokenizer=tokenizer, max_length=args.max_length)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: vihos_collate_fn(b, pad_token_id=tokenizer.pad_token_id)
    )

    if args.model_type == "phobert_dualhead_bilstm_crf":
        model = PhoBERT_DualHead_BiLSTM_CRF(pretrained_name=args.pretrained_name)
    elif args.model_type == "phobert_bilstm_crf":
        model = PhoBERT_BiLSTM_CRF(pretrained_name=args.pretrained_name)
    elif args.model_type == "phobert_crf":
        model = PhoBERT_CRF(pretrained_name=args.pretrained_name)
    else:
        model = PhoBERT_Linear(pretrained_name=args.pretrained_name)

    load_checkpoint(args.checkpoint, model, map_location=str(device))
    model.to(device)

    print(f"\nEvaluating {args.model_type} on {args.test_path}...")
    metrics = evaluate_model(model, test_loader, device)
    print(f"Precision: {metrics['span_precision']*100:.2f}%")
    print(f"Recall:    {metrics['span_recall']*100:.2f}%")
    print(f"Span-F1:   {metrics['span_f1']*100:.2f}%\n")
    print("Classification Report:\n", metrics["classification_report"])
