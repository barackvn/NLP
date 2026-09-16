import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from typing import List, Optional

class CRF(nn.Module):
    """
    Pure PyTorch Linear-chain Conditional Random Field (CRF).
    Tự chứa (Self-contained) 100%, không phụ thuộc vào bất kỳ thư viện C-extension hay pip cũ nào.
    """
    def __init__(self, num_tags: int, batch_first: bool = False) -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def _validate(self, emissions, tags=None, mask=None):
        if emissions.dim() != 3 or emissions.size(2) != self.num_tags:
            raise ValueError(f'emissions shape mismatch: expected (..., {self.num_tags}), got {emissions.shape}')
        if tags is not None and emissions.shape[:2] != tags.shape:
            raise ValueError(f'emissions and tags shape mismatch: {emissions.shape[:2]} vs {tags.shape}')
        if mask is not None and emissions.shape[:2] != mask.shape:
            raise ValueError(f'emissions and mask shape mismatch: {emissions.shape[:2]} vs {mask.shape}')

    def forward(self, emissions: torch.Tensor, tags: torch.LongTensor, mask: Optional[torch.Tensor] = None, reduction: str = 'sum') -> torch.Tensor:
        self._validate(emissions, tags=tags, mask=mask)
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.bool)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        numerator = self._compute_score(emissions, tags, mask)
        denominator = self._compute_normalizer(emissions, mask)
        llh = numerator - denominator

        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        return llh.sum() / mask.float().sum()

    def decode(self, emissions: torch.Tensor, mask: Optional[torch.Tensor] = None) -> List[List[int]]:
        self._validate(emissions, mask=mask)
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask)

    def _compute_score(self, emissions, tags, mask):
        seq_length, batch_size = tags.shape
        mask = mask.float()
        score = self.start_transitions[tags[0]] + emissions[0, torch.arange(batch_size), tags[0]]
        for i in range(1, seq_length):
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]
        seq_ends = mask.long().sum(dim=0) - 1
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        score += self.end_transitions[last_tags]
        return score

    def _compute_normalizer(self, emissions, mask):
        seq_length = emissions.size(0)
        score = self.start_transitions + emissions[0]
        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
        score += self.end_transitions
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions, mask) -> List[List[int]]:
        seq_length, batch_size = mask.shape
        score = self.start_transitions + emissions[0]
        history = []
        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emission
            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)
        score += self.end_transitions
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []
        for idx in range(batch_size):
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())
            best_tags.reverse()
            best_tags_list.append(best_tags)
        return best_tags_list

from .utils import NUM_LABELS, IGNORE_INDEX

class PhoBERT_BiLSTM_CRF(nn.Module):
    """
    Kiến trúc đề xuất (Proposed Stacked Architecture):
    PhoBERT (Contextual Representation) + BiLSTM (Smoothing & Sequential Dependencies) + CRF (Global Constraints & Viterbi)
    """
    def __init__(
        self,
        pretrained_name: str = "vinai/phobert-base-v2",
        num_labels: int = NUM_LABELS,
        lstm_hidden_size: int = 256,
        lstm_layers: int = 1,
        dropout_p: float = 0.3
    ):
        super(PhoBERT_BiLSTM_CRF, self).__init__()
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(pretrained_name)
        self.phobert = AutoModel.from_pretrained(pretrained_name, config=self.config)
        
        embed_dim = self.config.hidden_size # 768
        self.dropout = nn.Dropout(dropout_p)
        
        # Tầng BiLSTM đóng vai trò Smoothing Bridge
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True
        )
        
        # Tầng Linear chiếu ra số lượng nhãn (Emission Scores)
        self.classifier = nn.Linear(lstm_hidden_size * 2, num_labels)
        
        # Tầng CRF học ma trận chuyển đổi nhãn
        self.crf = CRF(num_tags=num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, word_indices=None, word_mask=None, labels=None, valid_mask=None):
        """
        - input_ids: (batch_size, seq_len)
        - attention_mask: (batch_size, seq_len)
        - word_indices: (batch_size, num_words) vị trí subword đầu tiên của mỗi từ
        - word_mask: (batch_size, num_words) bool tensor đánh dấu từ hợp lệ (liên tục 100%)
        - labels: (batch_size, num_words) nhãn BIO cấp độ từ
        """
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        
        # Word-level pooling nếu có word_indices
        if word_indices is not None:
            dim = sequence_output.size(-1)
            idx_exp = word_indices.unsqueeze(-1).expand(-1, -1, dim)
            feats = torch.gather(sequence_output, 1, idx_exp) # (batch_size, num_words, 768)
            mask = word_mask.bool() if word_mask is not None else torch.ones(feats.size(0), feats.size(1), dtype=torch.bool, device=feats.device)
        else:
            feats = sequence_output
            mask = attention_mask.bool() if valid_mask is None else (attention_mask.bool() & valid_mask.bool())

        # BiLSTM Layer
        lstm_out, _ = self.bilstm(feats)
        lstm_out = self.dropout(lstm_out)
        
        # Emission scores
        emissions = self.classifier(lstm_out) # (batch_size, seq_len/num_words, num_labels)
        
        if labels is not None:
            clean_labels = labels.clone()
            clean_labels[clean_labels == IGNORE_INDEX] = 0
            nll_loss = -self.crf(emissions, clean_labels, mask=mask, reduction='mean')
            return nll_loss, emissions
        else:
            best_paths = self.crf.decode(emissions, mask=mask)
            return best_paths, emissions

    def decode(self, input_ids, attention_mask, word_indices=None, word_mask=None, valid_mask=None):
        """Hàm giải mã Viterbi chuẩn xác cấp độ từ."""
        self.eval()
        with torch.no_grad():
            outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = self.dropout(outputs.last_hidden_state)
            
            if word_indices is not None:
                dim = sequence_output.size(-1)
                idx_exp = word_indices.unsqueeze(-1).expand(-1, -1, dim)
                feats = torch.gather(sequence_output, 1, idx_exp)
                mask = word_mask.bool() if word_mask is not None else torch.ones(feats.size(0), feats.size(1), dtype=torch.bool, device=feats.device)
            else:
                feats = sequence_output
                mask = attention_mask.bool() if valid_mask is None else (attention_mask.bool() & valid_mask.bool())

            lstm_out, _ = self.bilstm(feats)
            emissions = self.classifier(lstm_out)
            return self.crf.decode(emissions, mask=mask)


class PhoBERT_DualHead_BiLSTM_CRF(nn.Module):
    """
    Kiến trúc Đề xuất SOTA Đa nhiệm Đầu kép (Dual-Head Multi-Task Learning):
      - Branch 1 (Token/Span Head): PhoBERT + BiLSTM + Linear + CRF (Dự đoán chuỗi nhãn B-HOS, I-HOS, O cấp từ)
      - Branch 2 (Clause/Sentence Intent Head): PhoBERT + Intent MLP (Dự đoán xác suất câu/mệnh đề có chứa ý đồ độc hại)
      - Gated Fusion: Dập tắt nhãn toxic (gán về 'O') nếu Intent Head phán quyết mệnh đề/câu là phi độc hại.
    """
    def __init__(
        self,
        pretrained_name: str = "vinai/phobert-base-v2",
        num_labels: int = NUM_LABELS,
        lstm_hidden_size: int = 256,
        lstm_layers: int = 1,
        dropout_p: float = 0.3
    ):
        super(PhoBERT_DualHead_BiLSTM_CRF, self).__init__()
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(pretrained_name)
        self.phobert = AutoModel.from_pretrained(pretrained_name, config=self.config)
        embed_dim = self.config.hidden_size # 768

        self.dropout = nn.Dropout(dropout_p)

        # Nhánh 1: Token Span Head (BiLSTM + CRF)
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True
        )
        self.classifier = nn.Linear(lstm_hidden_size * 2, num_labels)
        self.crf = CRF(num_tags=num_labels, batch_first=True)

        # Nhánh 2: Sentence/Clause Intent Head
        self.intent_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 1)
        )
        self.bce_loss_fct = nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids,
        attention_mask,
        word_indices=None,
        word_mask=None,
        labels=None,
        valid_mask=None,
        sentence_labels=None,
        lambda_intent: float = 0.5
    ):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)

        # 1. Nhánh Intent: Trích xuất CLS token embedding
        cls_token = sequence_output[:, 0, :]
        intent_logits = self.intent_head(cls_token).squeeze(-1)

        # 2. Nhánh Span: Word-level pooling
        if word_indices is not None:
            dim = sequence_output.size(-1)
            idx_exp = word_indices.unsqueeze(-1).expand(-1, -1, dim)
            feats = torch.gather(sequence_output, 1, idx_exp)
            mask = word_mask.bool() if word_mask is not None else torch.ones(feats.size(0), feats.size(1), dtype=torch.bool, device=feats.device)
        else:
            feats = sequence_output
            mask = attention_mask.bool() if valid_mask is None else (attention_mask.bool() & valid_mask.bool())

        lstm_out, _ = self.bilstm(feats)
        lstm_out = self.dropout(lstm_out)
        emissions = self.classifier(lstm_out)

        if labels is not None:
            clean_labels = labels.clone()
            clean_labels[clean_labels == IGNORE_INDEX] = 0
            nll_loss = -self.crf(emissions, clean_labels, mask=mask, reduction='mean')

            # Tự động trích xuất binary intent nhãn từ chuỗi BIO nếu chưa có
            if sentence_labels is None:
                sentence_labels = (labels.clamp(min=0) > 0).any(dim=-1).float()
            intent_loss = self.bce_loss_fct(intent_logits, sentence_labels)

            total_loss = nll_loss + lambda_intent * intent_loss
            return total_loss, (emissions, intent_logits)
        else:
            best_paths = self.crf.decode(emissions, mask=mask)
            intent_probs = torch.sigmoid(intent_logits)
            return best_paths, intent_probs

    def decode(
        self,
        input_ids,
        attention_mask,
        word_indices=None,
        word_mask=None,
        valid_mask=None,
        intent_threshold: float = 0.5
    ):
        self.eval()
        with torch.no_grad():
            outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = self.dropout(outputs.last_hidden_state)
            cls_token = sequence_output[:, 0, :]
            intent_logits = self.intent_head(cls_token).squeeze(-1)
            intent_probs = torch.sigmoid(intent_logits)

            if word_indices is not None:
                dim = sequence_output.size(-1)
                idx_exp = word_indices.unsqueeze(-1).expand(-1, -1, dim)
                feats = torch.gather(sequence_output, 1, idx_exp)
                mask = word_mask.bool() if word_mask is not None else torch.ones(feats.size(0), feats.size(1), dtype=torch.bool, device=feats.device)
            else:
                feats = sequence_output
                mask = attention_mask.bool() if valid_mask is None else (attention_mask.bool() & valid_mask.bool())

            lstm_out, _ = self.bilstm(feats)
            emissions = self.classifier(lstm_out)
            raw_paths = self.crf.decode(emissions, mask=mask)

            # Gated Fusion: Nếu intent_prob < threshold, dập tắt nhãn về O
            gated_paths = []
            for i, path in enumerate(raw_paths):
                prob = intent_probs[i].item() if intent_probs.dim() > 0 else intent_probs.item()
                if prob < intent_threshold:
                    gated_paths.append([0] * len(path))
                else:
                    gated_paths.append(path)
            return gated_paths, intent_probs.tolist()


class PhoBERT_CRF(nn.Module):
    """
    Baseline bóc tách 1 (Ablation Baseline):
    PhoBERT + Linear + CRF (Loại bỏ tầng BiLSTM để chứng minh vai trò của BiLSTM)
    """
    def __init__(
        self,
        pretrained_name: str = "vinai/phobert-base-v2",
        num_labels: int = NUM_LABELS,
        dropout_p: float = 0.3
    ):
        super(PhoBERT_CRF, self).__init__()
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(pretrained_name)
        self.phobert = AutoModel.from_pretrained(pretrained_name, config=self.config)
        
        embed_dim = self.config.hidden_size
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(embed_dim, num_labels)
        self.crf = CRF(num_tags=num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, word_indices=None, word_mask=None, labels=None, valid_mask=None):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        
        if word_indices is not None:
            dim = sequence_output.size(-1)
            idx_exp = word_indices.unsqueeze(-1).expand(-1, -1, dim)
            feats = torch.gather(sequence_output, 1, idx_exp)
            mask = word_mask.bool() if word_mask is not None else torch.ones(feats.size(0), feats.size(1), dtype=torch.bool, device=feats.device)
        else:
            feats = sequence_output
            mask = attention_mask.bool() if valid_mask is None else (attention_mask.bool() & valid_mask.bool())

        emissions = self.classifier(feats)
        
        if labels is not None:
            clean_labels = labels.clone()
            clean_labels[clean_labels == IGNORE_INDEX] = 0
            nll_loss = -self.crf(emissions, clean_labels, mask=mask, reduction='mean')
            return nll_loss, emissions
        else:
            best_paths = self.crf.decode(emissions, mask=mask)
            return best_paths, emissions

    def decode(self, input_ids, attention_mask, word_indices=None, word_mask=None, valid_mask=None):
        self.eval()
        with torch.no_grad():
            outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = self.dropout(outputs.last_hidden_state)
            
            if word_indices is not None:
                dim = sequence_output.size(-1)
                idx_exp = word_indices.unsqueeze(-1).expand(-1, -1, dim)
                feats = torch.gather(sequence_output, 1, idx_exp)
                mask = word_mask.bool() if word_mask is not None else torch.ones(feats.size(0), feats.size(1), dtype=torch.bool, device=feats.device)
            else:
                feats = sequence_output
                mask = attention_mask.bool() if valid_mask is None else (attention_mask.bool() & valid_mask.bool())

            emissions = self.classifier(feats)
            return self.crf.decode(emissions, mask=mask)


class PhoBERT_Linear(nn.Module):
    """
    Baseline gốc của Giảng viên (Original Baseline):
    PhoBERT + Dropout + Linear Classification Head (Softmax độc lập tại từng token)
    """
    def __init__(
        self,
        pretrained_name: str = "vinai/phobert-base-v2",
        num_labels: int = NUM_LABELS,
        dropout_p: float = 0.3
    ):
        super(PhoBERT_Linear, self).__init__()
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(pretrained_name)
        self.phobert = AutoModel.from_pretrained(pretrained_name, config=self.config)
        
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    def forward(self, input_ids, attention_mask, word_indices=None, word_mask=None, labels=None, valid_mask=None):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        
        if word_indices is not None:
            dim = sequence_output.size(-1)
            idx_exp = word_indices.unsqueeze(-1).expand(-1, -1, dim)
            feats = torch.gather(sequence_output, 1, idx_exp)
            mask = word_mask.bool() if word_mask is not None else torch.ones(feats.size(0), feats.size(1), dtype=torch.bool, device=feats.device)
        else:
            feats = sequence_output
            mask = attention_mask.bool() if valid_mask is None else (attention_mask.bool() & valid_mask.bool())

        logits = self.classifier(feats)
        
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            preds = torch.argmax(logits, dim=-1)
            return preds, logits

    def decode(self, input_ids, attention_mask, word_indices=None, word_mask=None, valid_mask=None):
        self.eval()
        with torch.no_grad():
            outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = self.dropout(outputs.last_hidden_state)
            
            if word_indices is not None:
                dim = sequence_output.size(-1)
                idx_exp = word_indices.unsqueeze(-1).expand(-1, -1, dim)
                feats = torch.gather(sequence_output, 1, idx_exp)
                mask = word_mask.bool() if word_mask is not None else torch.ones(feats.size(0), feats.size(1), dtype=torch.bool, device=feats.device)
            else:
                feats = sequence_output
                mask = attention_mask.bool() if valid_mask is None else (attention_mask.bool() & valid_mask.bool())

            logits = self.classifier(feats)
            preds = torch.argmax(logits, dim=-1)
            
            results = []
            for i in range(preds.size(0)):
                seq_len = mask[i].sum().item()
                valid_preds = preds[i][:seq_len].tolist()
                results.append(valid_preds)
            return results
