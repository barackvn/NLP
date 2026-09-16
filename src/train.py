import argparse
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from .utils import set_seed, get_device, setup_logger, save_checkpoint
from .dataset import ViHOSDataset, vihos_collate_fn
from .model import PhoBERT_BiLSTM_CRF, PhoBERT_CRF, PhoBERT_Linear, PhoBERT_DualHead_BiLSTM_CRF
from .evaluate import evaluate_model

def get_optimizer_and_scheduler(model, model_type, lr_phobert=2e-5, lr_head=1e-3, weight_decay=0.01, total_steps=1000):
    """
    Thiết lập Differential Learning Rate:
    - PhoBERT Backbone: lr nhỏ (2e-5) để bảo toàn tri thức pre-trained.
    - Head Layers (BiLSTM, Classifier, CRF, Intent Head): lr lớn hơn (1e-3) để học biểu diễn bài toán mới.
    """
    if model_type in ("phobert_bilstm_crf", "phobert_dualhead_bilstm_crf"):
        optimizer_grouped_parameters = [
            {"params": model.phobert.parameters(), "lr": lr_phobert, "weight_decay": weight_decay},
            {"params": model.bilstm.parameters(), "lr": lr_head, "weight_decay": weight_decay},
            {"params": model.classifier.parameters(), "lr": lr_head, "weight_decay": weight_decay},
            {"params": model.crf.parameters(), "lr": lr_head, "weight_decay": weight_decay}
        ]
        if hasattr(model, "intent_head"):
            optimizer_grouped_parameters.append(
                {"params": model.intent_head.parameters(), "lr": lr_head, "weight_decay": weight_decay}
            )
    elif model_type == "phobert_crf":
        optimizer_grouped_parameters = [
            {"params": model.phobert.parameters(), "lr": lr_phobert, "weight_decay": weight_decay},
            {"params": model.classifier.parameters(), "lr": lr_head, "weight_decay": weight_decay},
            {"params": model.crf.parameters(), "lr": lr_head, "weight_decay": weight_decay}
        ]
    else: # phobert_linear
        optimizer_grouped_parameters = [
            {"params": model.phobert.parameters(), "lr": lr_phobert, "weight_decay": weight_decay},
            {"params": model.classifier.parameters(), "lr": lr_head, "weight_decay": weight_decay}
        ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    return optimizer, scheduler

def train_pipeline(args):
    set_seed(args.seed)
    logger = setup_logger("ViHOS_Train")
    device = get_device()
    logger.info(f"Using device: {device} | Model: {args.model_type}")

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_name)

    # 1. Nạp Dataset
    logger.info("Loading Datasets...")
    train_dataset = ViHOSDataset(args.train_path, tokenizer=tokenizer, max_length=args.max_length)
    dev_dataset = ViHOSDataset(args.dev_path, tokenizer=tokenizer, max_length=args.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: vihos_collate_fn(b, pad_token_id=tokenizer.pad_token_id)
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: vihos_collate_fn(b, pad_token_id=tokenizer.pad_token_id)
    )

    # 2. Khởi tạo Mô hình
    logger.info(f"Initializing model architecture: {args.model_type}...")
    if args.model_type == "phobert_bilstm_crf":
        model = PhoBERT_BiLSTM_CRF(pretrained_name=args.pretrained_name, dropout_p=args.dropout)
    elif args.model_type == "phobert_dualhead_bilstm_crf":
        model = PhoBERT_DualHead_BiLSTM_CRF(pretrained_name=args.pretrained_name, dropout_p=args.dropout)
    elif args.model_type == "phobert_crf":
        model = PhoBERT_CRF(pretrained_name=args.pretrained_name, dropout_p=args.dropout)
    elif args.model_type == "phobert_linear":
        model = PhoBERT_Linear(pretrained_name=args.pretrained_name, dropout_p=args.dropout)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    model.to(device)

    # 3. Optimizer & Scheduler
    total_steps = len(train_loader) * args.epochs
    optimizer, scheduler = get_optimizer_and_scheduler(
        model, args.model_type,
        lr_phobert=args.lr_phobert,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        total_steps=total_steps
    )

    best_f1 = 0.0
    patience_counter = 0

    # 4. Training Loop
    logger.info("Starting training loop...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")

        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            word_indices = batch["word_indices"].to(device)
            word_mask = batch["word_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            loss, _ = model(input_ids, attention_mask, word_indices=word_indices, word_mask=word_mask, labels=labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / len(train_loader)
        logger.info(f"Epoch {epoch}/{args.epochs} - Avg Train Loss: {avg_train_loss:.4f}")

        # Đánh giá trên tập Dev
        logger.info(f"Evaluating on Dev Set for Epoch {epoch}...")
        dev_metrics = evaluate_model(model, dev_loader, device)
        dev_f1 = dev_metrics["span_f1"]
        dev_p = dev_metrics["span_precision"]
        dev_r = dev_metrics["span_recall"]
        
        logger.info(f"Dev Precision: {dev_p*100:.2f}% | Recall: {dev_r*100:.2f}% | Span-F1: {dev_f1*100:.2f}%")

        # Lưu checkpoint tốt nhất (Early Stopping)
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            patience_counter = 0
            logger.info(f"--> Found new best model with Span-F1: {best_f1*100:.2f}%. Saving to {args.save_path}...")
            save_checkpoint(model, optimizer, epoch, dev_metrics, args.save_path)
        else:
            patience_counter += 1
            logger.info(f"Early stopping counter: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                logger.info("Early stopping triggered! Training finished.")
                break

    logger.info(f"Training completed! Best Dev Span-F1: {best_f1*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình ViHOS Toxic Spans Detection")
    parser.add_argument("--model_type", type=str, default="phobert_dualhead_bilstm_crf", choices=["phobert_dualhead_bilstm_crf", "phobert_bilstm_crf", "phobert_crf", "phobert_linear"])
    parser.add_argument("--train_path", type=str, default="data/processed/train.json")
    parser.add_argument("--dev_path", type=str, default="data/processed/dev.json")
    parser.add_argument("--pretrained_name", type=str, default="vinai/phobert-base-v2")
    parser.add_argument("--save_path", type=str, default="checkpoints/best_phobert_bilstm_crf.pt")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr_phobert", type=float, default=2e-5)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_pipeline(args)
