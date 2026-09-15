import os
import sys
import time
import torch
from transformers import AutoTokenizer

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.utils import ID2LABEL, IGNORE_INDEX
from src.model import PhoBERT_BiLSTM_CRF, PhoBERT_CRF, PhoBERT_Linear

# Danh mục từ ngữ xúc phạm mẫu để hỗ trợ fallback khi chưa tải file checkpoint nặng
FALLBACK_TOXIC_LEXICON = {
    "ngu", "chó", "súc_sinh", "đĩ", "mất_dạy", "khốn_nạn", "rác_rưởi", "đm", "đkm", "clm",
    "hãm", "óc_chó", "chó_đẻ", "vô_học", "đồ_điên", "đồ_ngu", "hãm_lồn", "con_đĩ", "thằng_ngu"
}

class ViHOSInferenceEngine:
    model_name: str = "PhoBERT-BiLSTM-CRF (Đề xuất SOTA)"
    checkpoint_loaded: bool = False

    def __init__(self, checkpoint_path: str = None, pretrained_name: str = "vinai/phobert-base-v2"):
        self.device = torch.device("cpu")
        self.pretrained_name = pretrained_name
        self.checkpoint_path = checkpoint_path
        self.tokenizer = None
        self.model = None
        self.model_name = "Chế độ Giả lập (Mock Rule)"
        self.checkpoint_loaded = False
        self.checkpoint_metrics = {}
        
        self.load_error = None
        # Nạp checkpoint và tokenizer nếu có file .pt
        if not checkpoint_path:
            self.load_error = "Chưa chỉ định checkpoint_path."
        elif not os.path.exists(checkpoint_path):
            self.load_error = f"Không tìm thấy file checkpoint tại: {checkpoint_path}"
        else:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                except TypeError:
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)

                state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
                self.checkpoint_metrics = checkpoint.get("metrics", {})

                # Tự động nhận diện kiến trúc dựa trên trọng số
                has_bilstm = any("bilstm" in k for k in state_dict.keys())
                has_crf = any("crf" in k for k in state_dict.keys())

                if has_bilstm and has_crf:
                    self.model = PhoBERT_BiLSTM_CRF(pretrained_name=pretrained_name)
                    self.model_name = "🌟 PhoBERT-BiLSTM-CRF (Đề xuất SOTA)"
                elif has_crf:
                    self.model = PhoBERT_CRF(pretrained_name=pretrained_name)
                    self.model_name = "🔬 PhoBERT-CRF (Bóc tách Ablation)"
                else:
                    self.model = PhoBERT_Linear(pretrained_name=pretrained_name)
                    self.model_name = "📌 PhoBERT-Linear (Baseline Thầy)"

                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                self.checkpoint_loaded = True
                try:
                    print(f"[Success] Model loaded successfully: {self.model_name}")
                except Exception:
                    pass
            except Exception as e:
                import traceback
                self.load_error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                try:
                    print(f"[Warning] Error loading checkpoint: {e}")
                except Exception:
                    pass

    def tokenize_vietnamese(self, text: str):
        """Tách từ tiếng Việt chuẩn hóa."""
        try:
            from pyvi import ViTokenizer
            segmented = ViTokenizer.tokenize(text)
            words = segmented.split()
        except:
            words = text.split()
        return words

    def predict(self, text: str):
        """
        Dự đoán các chuỗi toxic spans trong câu văn.
        Đo độ trễ (latency ms) trên CPU.
        """
        start_time = time.time()
        words = self.tokenize_vietnamese(text)
        
        if not words:
            return {"words": [], "tags": [], "spans": [], "latency_ms": 0.0, "is_toxic": False}

        # Nếu mô hình checkpoint đã nạp và tokenizer sẵn sàng
        if self.checkpoint_loaded and self.model and self.tokenizer:
            input_ids = [self.tokenizer.bos_token_id]
            valid_mask = [0]
            word_to_token_idx = []

            for word in words:
                subwords = self.tokenizer.tokenize(word)
                if not subwords:
                    subwords = [self.tokenizer.unk_token]
                sub_ids = self.tokenizer.convert_tokens_to_ids(subwords)
                
                word_to_token_idx.append(len(input_ids))
                input_ids.append(sub_ids[0])
                valid_mask.append(1)
                
                for s in sub_ids[1:]:
                    input_ids.append(s)
                    valid_mask.append(0)

            input_ids.append(self.tokenizer.eos_token_id)
            valid_mask.append(0)

            t_input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
            t_attn_mask = torch.tensor([[1] * len(input_ids)], dtype=torch.long, device=self.device)
            t_valid_mask = torch.tensor([valid_mask], dtype=torch.bool, device=self.device)

            with torch.no_grad():
                preds = self.model.decode(t_input_ids, t_attn_mask, valid_mask=t_valid_mask)[0]

            tags = [ID2LABEL.get(p, "O") for p in preds[:len(words)]]
            if len(tags) < len(words):
                tags.extend(["O"] * (len(words) - len(tags)))
        else:
            # Fallback mô phỏng dự đoán dựa trên từ điển toxic + phân tích cú pháp BIO
            tags = []
            prev_tag = "O"
            for w in words:
                clean_w = w.lower().replace("_", "")
                if clean_w in FALLBACK_TOXIC_LEXICON or any(tox in clean_w for tox in ["ngu", "chó", "đĩ", "hãm"]):
                    curr_tag = "I-HOS" if prev_tag in ["B-HOS", "I-HOS"] else "B-HOS"
                else:
                    curr_tag = "O"
                tags.append(curr_tag)
                prev_tag = curr_tag

        latency_ms = (time.time() - start_time) * 1000.0

        # Trích xuất spans
        spans = []
        current_span = []
        for i, (w, t) in enumerate(zip(words, tags)):
            if t == "B-HOS":
                if current_span:
                    spans.append(" ".join(current_span))
                    current_span = []
                current_span.append(w)
            elif t == "I-HOS":
                if current_span:
                    current_span.append(w)
                else:
                    current_span.append(w)
            else:
                if current_span:
                    spans.append(" ".join(current_span))
                    current_span = []
        if current_span:
            spans.append(" ".join(current_span))

        is_toxic = len(spans) > 0

        return {
            "words": words,
            "tags": tags,
            "spans": spans,
            "latency_ms": round(latency_ms, 2),
            "is_toxic": is_toxic,
            "model_name": self.model_name,
            "engine_mode": self.model_name if self.checkpoint_loaded else "MOCK_DEMO_RULE",
            "checkpoint_loaded": self.checkpoint_loaded
        }

    def auto_mask(self, words: list, tags: list, mask_symbol: str = "***") -> str:
        """
        Tự động che giấu (Auto-Masking) các từ mang nhãn B-HOS, I-HOS bằng ***
        giữ nguyên cấu trúc ngữ pháp của các từ trung tính còn lại.
        """
        masked_words = []
        in_mask = False
        for w, t in zip(words, tags):
            clean_word = w.replace("_", " ")
            if t in ["B-HOS", "I-HOS"]:
                if not in_mask:
                    masked_words.append(mask_symbol)
                    in_mask = True
            else:
                in_mask = False
                masked_words.append(clean_word)
        return " ".join(masked_words)
