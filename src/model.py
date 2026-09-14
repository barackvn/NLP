import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

try:
    from torchcrf import CRF
except ImportError:
    from TorchCRF import CRF

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

    def forward(self, input_ids, attention_mask, labels=None, valid_mask=None):
        """
        - input_ids: (batch_size, seq_len)
        - attention_mask: (batch_size, seq_len)
        - labels: (batch_size, seq_len) với các nhãn subword phụ là -100
        - valid_mask: (batch_size, seq_len) bool tensor đánh dấu token hợp lệ cho CRF
        """
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_len, 768)
        sequence_output = self.dropout(sequence_output)
        
        # BiLSTM Layer
        lstm_out, _ = self.bilstm(sequence_output)   # (batch_size, seq_len, 512)
        lstm_out = self.dropout(lstm_out)
        
        # Emission scores
        emissions = self.classifier(lstm_out)       # (batch_size, seq_len, num_labels)
        
        if valid_mask is None:
            mask = attention_mask.bool()
        else:
            mask = (attention_mask.bool()) & (valid_mask.bool())

        if labels is not None:
            # Chuẩn hóa nhãn -100 thành 0 để truyền vào CRF, các vị trí này sẽ bị che bởi mask
            clean_labels = labels.clone()
            clean_labels[clean_labels == IGNORE_INDEX] = 0
            
            # CRF forward trả về log-likelihood -> Negative log-likelihood làm loss
            nll_loss = -self.crf(emissions, clean_labels, mask=mask, reduction='mean')
            return nll_loss, emissions
        else:
            # Giải mã Viterbi trả về danh sách các nhãn dự đoán tốt nhất
            best_paths = self.crf.decode(emissions, mask=mask)
            return best_paths, emissions

    def decode(self, input_ids, attention_mask, valid_mask=None):
        """Hàm giải mã Viterbi thuận tiện khi suy luận (Inference)."""
        self.eval()
        with torch.no_grad():
            outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = self.dropout(outputs.last_hidden_state)
            lstm_out, _ = self.bilstm(sequence_output)
            emissions = self.classifier(lstm_out)
            
            if valid_mask is None:
                mask = attention_mask.bool()
            else:
                mask = (attention_mask.bool()) & (valid_mask.bool())
                
            predictions = self.crf.decode(emissions, mask=mask)
            return predictions


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

    def forward(self, input_ids, attention_mask, labels=None, valid_mask=None):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        emissions = self.classifier(sequence_output)
        
        if valid_mask is None:
            mask = attention_mask.bool()
        else:
            mask = (attention_mask.bool()) & (valid_mask.bool())

        if labels is not None:
            clean_labels = labels.clone()
            clean_labels[clean_labels == IGNORE_INDEX] = 0
            nll_loss = -self.crf(emissions, clean_labels, mask=mask, reduction='mean')
            return nll_loss, emissions
        else:
            best_paths = self.crf.decode(emissions, mask=mask)
            return best_paths, emissions

    def decode(self, input_ids, attention_mask, valid_mask=None):
        self.eval()
        with torch.no_grad():
            outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = self.dropout(outputs.last_hidden_state)
            emissions = self.classifier(sequence_output)
            
            if valid_mask is None:
                mask = attention_mask.bool()
            else:
                mask = (attention_mask.bool()) & (valid_mask.bool())
                
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

    def forward(self, input_ids, attention_mask, labels=None, valid_mask=None):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)  # (batch_size, seq_len, num_labels)
        
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            preds = torch.argmax(logits, dim=-1)
            return preds, logits

    def decode(self, input_ids, attention_mask, valid_mask=None):
        self.eval()
        with torch.no_grad():
            outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = self.dropout(outputs.last_hidden_state)
            logits = self.classifier(sequence_output)
            preds = torch.argmax(logits, dim=-1) # (batch_size, seq_len)
            
            mask = attention_mask.bool() if valid_mask is None else (attention_mask.bool() & valid_mask.bool())
            
            # Chuyển đổi thành list of lists tương đương decode của CRF
            results = []
            for i in range(preds.size(0)):
                seq_len = mask[i].sum().item()
                valid_preds = preds[i][mask[i]][:seq_len].tolist()
                results.append(valid_preds)
            return results
