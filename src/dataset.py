import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .utils import LABEL2ID, IGNORE_INDEX

class ViHOSDataset(Dataset):
    """
    Dataset class cho bài toán ViHOS Toxic Spans Detection.
    Hỗ trợ kỹ thuật First-token Subword Alignment cho PhoBERT Tokenizer.
    """
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer = None,
        pretrained_name: str = "vinai/phobert-base-v2",
        max_length: int = 128
    ):
        self.max_length = max_length
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(pretrained_name)
        self.examples = self._load_data(data_path)

    def _load_data(self, data_path: str):
        """Đọc dữ liệu từ file csv hoặc json."""
        examples = []
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            for _, row in df.iterrows():
                tokens = row['tokens']
                tags = row['tags']
                if isinstance(tokens, str):
                    try:
                        tokens = json.loads(tokens)
                    except:
                        tokens = tokens.split()
                if isinstance(tags, str):
                    try:
                        tags = json.loads(tags)
                    except:
                        tags = tags.split()
                examples.append({"tokens": tokens, "tags": tags})
        elif data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                examples = json.load(f)
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        tokens = item["tokens"]
        tags = item["tags"]

        input_ids = [self.tokenizer.bos_token_id]
        word_indices = []
        word_labels = []

        for word, tag in zip(tokens, tags):
            if word is None:
                continue
            if not isinstance(word, str):
                word_str = str(word).strip()
                if word_str.lower() in ("nan", "null", "none", ""):
                    continue
                word = word_str
            else:
                word = word.strip()
                if not word:
                    continue

            word_subwords = self.tokenizer.tokenize(word)
            if len(word_subwords) == 0:
                continue
            subword_ids = self.tokenizer.convert_tokens_to_ids(word_subwords)
            label_id = LABEL2ID.get(tag, LABEL2ID["O"])

            # Ghi nhận vị trí subword đầu tiên của từ
            first_subword_pos = len(input_ids)
            
            # Kiểm tra nếu vị trí vượt quá max_length - 1 (dành chỗ cho </s>)
            if first_subword_pos >= self.max_length - 1:
                break

            word_indices.append(first_subword_pos)
            word_labels.append(label_id)

            for sub_id in subword_ids:
                if len(input_ids) < self.max_length - 1:
                    input_ids.append(sub_id)
                else:
                    break

        # Thêm </s> token
        input_ids.append(self.tokenizer.eos_token_id)
        attention_mask = [1] * len(input_ids)
        word_mask = [1] * len(word_indices)

        # Đảm bảo có ít nhất 1 word token để không làm sập tensor batch
        if not word_indices:
            word_indices = [1]
            word_labels = [LABEL2ID["O"]]
            word_mask = [1]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "word_indices": word_indices,
            "word_labels": word_labels,
            "word_mask": word_mask,
            "original_tokens": tokens[:len(word_indices)],
            "original_tags": tags[:len(word_labels)]
        }


def vihos_collate_fn(batch, pad_token_id=1):
    """
    Hàm gom batch và đệm (padding) động theo độ dài lớn nhất trong batch.
    - Subwords được pad theo max_subword_len.
    - Words và Word Masks được pad theo max_word_len (word_mask liên tục 100%).
    """
    max_subword_len = max(len(item["input_ids"]) for item in batch)
    max_word_len = max(len(item["word_indices"]) for item in batch)
    
    batch_input_ids = []
    batch_attention_mask = []
    batch_word_indices = []
    batch_word_labels = []
    batch_word_mask = []
    original_tokens = [item["original_tokens"] for item in batch]
    original_tags = [item["original_tags"] for item in batch]

    for item in batch:
        pad_sub_len = max_subword_len - len(item["input_ids"])
        pad_word_len = max_word_len - len(item["word_indices"])
        
        batch_input_ids.append(item["input_ids"] + [pad_token_id] * pad_sub_len)
        batch_attention_mask.append(item["attention_mask"] + [0] * pad_sub_len)
        
        # Word indices pad với 0 (vị trí <s> an toàn)
        batch_word_indices.append(item["word_indices"] + [0] * pad_word_len)
        batch_word_labels.append(item["word_labels"] + [IGNORE_INDEX] * pad_word_len)
        batch_word_mask.append(item["word_mask"] + [0] * pad_word_len)

    return {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        "word_indices": torch.tensor(batch_word_indices, dtype=torch.long),
        "labels": torch.tensor(batch_word_labels, dtype=torch.long),
        "word_mask": torch.tensor(batch_word_mask, dtype=torch.bool),
        "original_tokens": original_tokens,
        "original_tags": original_tags
    }
