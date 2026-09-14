import torch
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from seqeval.scheme import IOB2
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd

from .utils import ID2LABEL, IGNORE_INDEX

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
            labels = batch["labels"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            
            # Giải mã nhãn dự đoán qua decode method
            predictions = model.decode(input_ids, attention_mask, valid_mask=valid_mask)
            
            labels_np = labels.cpu().numpy()
            valid_mask_np = valid_mask.cpu().numpy()
            
            for i in range(len(predictions)):
                pred_seq = predictions[i]
                
                # Trích xuất nhãn thực tế tại các vị trí valid
                true_seq = []
                for label_id, is_valid in zip(labels_np[i], valid_mask_np[i]):
                    if is_valid and label_id != IGNORE_INDEX:
                        true_seq.append(ID2LABEL.get(label_id, "O"))
                
                # Chuẩn hóa độ dài dự đoán khớp với nhãn thực tế
                pred_seq_labels = [ID2LABEL.get(p, "O") for p in pred_seq[:len(true_seq)]]
                if len(pred_seq_labels) < len(true_seq):
                    pred_seq_labels.extend(["O"] * (len(true_seq) - len(pred_seq_labels)))
                
                all_true_tags.append(true_seq)
                all_pred_tags.append(pred_seq_labels)

    # Tính toán các chỉ số Span-Level bằng seqeval (chuẩn IOB2)
    span_p = precision_score(all_true_tags, all_pred_tags, mode='strict', scheme=IOB2)
    span_r = recall_score(all_true_tags, all_pred_tags, mode='strict', scheme=IOB2)
    span_f1 = f1_score(all_true_tags, all_pred_tags, mode='strict', scheme=IOB2)
    report = classification_report(all_true_tags, all_pred_tags, mode='strict', scheme=IOB2)

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
