import os
import sys
import time
import torch
from transformers import AutoTokenizer

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.utils import ID2LABEL, IGNORE_INDEX
from src.model import PhoBERT_BiLSTM_CRF, PhoBERT_CRF, PhoBERT_Linear, PhoBERT_DualHead_BiLSTM_CRF

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
                has_intent = any("intent_head" in k for k in state_dict.keys())
                has_bilstm = any("bilstm" in k for k in state_dict.keys())
                has_crf = any("crf" in k for k in state_dict.keys())

                if has_intent:
                    self.model = PhoBERT_DualHead_BiLSTM_CRF(pretrained_name=pretrained_name)
                    self.model_name = "🌟 PhoBERT-DualHead-BiLSTM-CRF (Đề xuất SOTA Đa nhiệm)"
                elif has_bilstm and has_crf:
                    self.model = PhoBERT_BiLSTM_CRF(pretrained_name=pretrained_name)
                    self.model_name = "🌟 PhoBERT-BiLSTM-CRF (Đề xuất SOTA)"
                elif has_crf:
                    self.model = PhoBERT_CRF(pretrained_name=pretrained_name)
                    self.model_name = "🔬 PhoBERT-CRF (Bóc tách Ablation)"
                else:
                    self.model = PhoBERT_Linear(pretrained_name=pretrained_name)
                    self.model_name = "📌 PhoBERT-Linear (Baseline)"

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

    def apply_clause_gated_fusion(self, words: list, tags: list):
        """
        Module Cổng Điều Biến Ngữ Cảnh Mệnh Đề (Clause Gated Intent Fusion):
        Tách câu thành các mệnh đề độc lập. Đánh giá ý định độc hại (Toxic Intent) của từng mệnh đề.
        Nếu mệnh đề là ngữ cảnh khen ngợi thú cưng hoặc sự vật lành tính,
        dập tắt nhãn vi phạm (B-HOS, I-HOS) về 'O'.
        """
        delimiters = {",", ";", ".", "!", "?", "nhưng", "mà", "còn", "tuy_nhiên", "thế_mà", "song"}
        clause_bounds = []
        start_idx = 0
        for i, w in enumerate(words):
            clean_w = w.lower().strip(",;.!?")
            if w in delimiters or clean_w in delimiters or any(p in w for p in [",", ";", ".", "!", "?"]):
                clause_bounds.append((start_idx, i + 1))
                start_idx = i + 1
        if start_idx < len(words):
            clause_bounds.append((start_idx, len(words)))
        if not clause_bounds:
            clause_bounds = [(0, len(words))]

        new_tags = list(tags)
        gated_applied = False
        gated_explanations = []

        animal_keywords = {"chó", "con_chó", "cún", "cún_con", "mèo", "con_mèo", "lợn", "heo", "cẩu"}
        praise_keywords = {
            "đẹp", "xinh", "dễ_thương", "đáng_yêu", "khôn", "ngoan", "tốt", "hiền",
            "cưng", "nuôi", "chăm_sóc", "cho_ăn", "dắt", "tắm", "mua", "bán",
            "thuần_chủng", "thú_cưng", "thương", "yêu", "thích", "mê", "tuyệt_vời"
        }
        attack_pronouns = {"mày", "tao", "đồ", "thứ", "lũ", "bọn", "thằng", "con_đĩ", "mất_dạy", "khốn_nạn", "chết_tiệt", "cút"}

        has_any_toxic_clause = False
        for c_start, c_end in clause_bounds:
            c_words = [w.lower() for w in words[c_start:c_end]]
            c_tags = new_tags[c_start:c_end]

            has_toxic_tags = any(t in ("B-HOS", "I-HOS") for t in c_tags)
            if not has_toxic_tags:
                continue

            has_animal = any(any(ak in cw for ak in animal_keywords) for cw in c_words)
            has_praise = any(any(pk in cw for pk in praise_keywords) for cw in c_words)
            has_attack = any(any(at in cw for at in attack_pronouns) for cw in c_words)

            if has_animal and has_praise and not has_attack:
                for k in range(c_start, c_end):
                    if new_tags[k] in ("B-HOS", "I-HOS"):
                        new_tags[k] = "O"
                gated_applied = True
                clause_str = " ".join(words[c_start:c_end])
                gated_explanations.append(f"Mệnh đề khen ngợi động vật: '{clause_str}'")
            else:
                has_any_toxic_clause = True

        overall_intent = 0.98 if has_any_toxic_clause else (0.05 if gated_applied else (0.92 if any(t != 'O' for t in new_tags) else 0.02))
        note = " | ".join(gated_explanations) if gated_applied else "Không cần điều biến"

        return new_tags, gated_applied, overall_intent, note

    def predict(self, text: str, enable_gated_intent: bool = True):
        """
        Dự đoán các chuỗi toxic spans trong câu văn.
        Tích hợp Module Clause Gated Intent Fusion nhằm triệt tiêu báo động giả.
        Đo độ trễ (latency ms) trên CPU.
        """
        start_time = time.time()
        words = self.tokenize_vietnamese(text)
        
        if not words:
            return {
                "words": [], "tags": [], "spans": [], "latency_ms": 0.0, "is_toxic": False,
                "intent_score": 0.0, "gated_filter_applied": False, "gated_note": ""
            }

        # Nếu mô hình checkpoint đã nạp và tokenizer sẵn sàng
        if self.checkpoint_loaded and self.model and self.tokenizer:
            input_ids = [self.tokenizer.bos_token_id]
            word_indices = []

            for word in words:
                subwords = self.tokenizer.tokenize(word)
                if not subwords:
                    subwords = [self.tokenizer.unk_token]
                sub_ids = self.tokenizer.convert_tokens_to_ids(subwords)
                
                word_indices.append(len(input_ids))
                input_ids.append(sub_ids[0])
                for s in sub_ids[1:]:
                    input_ids.append(s)

            input_ids.append(self.tokenizer.eos_token_id)
            attention_mask = [1] * len(input_ids)
            word_mask = [1] * len(word_indices)

            t_input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
            t_attn_mask = torch.tensor([attention_mask], dtype=torch.long, device=self.device)
            t_word_idx = torch.tensor([word_indices], dtype=torch.long, device=self.device)
            t_word_mask = torch.tensor([word_mask], dtype=torch.bool, device=self.device)

            with torch.no_grad():
                res_decode = self.model.decode(t_input_ids, t_attn_mask, word_indices=t_word_idx, word_mask=t_word_mask)
                preds = res_decode[0] if isinstance(res_decode, tuple) else res_decode[0]

            raw_tags = [ID2LABEL.get(p, "O") for p in preds[:len(words)]]
            if len(raw_tags) < len(words):
                raw_tags.extend(["O"] * (len(words) - len(raw_tags)))
        else:
            # Fallback mô phỏng dự đoán dựa trên từ điển toxic + phân tích cú pháp BIO
            raw_tags = []
            prev_tag = "O"
            for w in words:
                clean_w = w.lower().replace("_", "")
                if clean_w in FALLBACK_TOXIC_LEXICON or any(tox in clean_w for tox in ["ngu", "chó", "đĩ", "hãm"]):
                    curr_tag = "I-HOS" if prev_tag in ["B-HOS", "I-HOS"] else "B-HOS"
                else:
                    curr_tag = "O"
                raw_tags.append(curr_tag)
                prev_tag = curr_tag

        # Áp dụng Cổng Gated Intent Fusion
        if enable_gated_intent:
            tags, gated_applied, intent_score, gated_note = self.apply_clause_gated_fusion(words, raw_tags)
        else:
            tags = raw_tags
            gated_applied = False
            gated_note = "Chưa kích hoạt cổng"
            intent_score = 0.95 if any(t != "O" for t in tags) else 0.03

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
            "raw_tags": raw_tags,
            "spans": spans,
            "latency_ms": round(latency_ms, 2),
            "is_toxic": is_toxic,
            "intent_score": round(intent_score, 2),
            "gated_filter_applied": gated_applied,
            "gated_note": gated_note,
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
