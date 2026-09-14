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
        aligned_labels = [IGNORE_INDEX]
        valid_mask = [0] # <s> không gán nhãn

        for word, tag in zip(tokens, tags):
            word_subwords = self.tokenizer.tokenize(word)
            if len(word_subwords) == 0:
                continue
            subword_ids = self.tokenizer.convert_tokens_to_ids(word_subwords)
            label_id = LABEL2ID.get(tag, LABEL2ID["O"])

            # First-token subword alignment:
            # Subword đầu tiên nhận nhãn BIO thật
            input_ids.append(subword_ids[0])
            aligned_labels.append(label_id)
            valid_mask.append(1)

            # Các subword tiếp theo nhận IGNORE_INDEX (-100)
            for sub_id in subword_ids[1:]:
                input_ids.append(sub_id)
                aligned_labels.append(IGNORE_INDEX)
                valid_mask.append(0)

        # Thêm </s> token
        input_ids.append(self.tokenizer.eos_token_id)
        aligned_labels.append(IGNORE_INDEX)
        valid_mask.append(0)

        # Cắt bớt nếu vượt quá max_length
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length - 1] + [self.tokenizer.eos_token_id]
            aligned_labels = aligned_labels[:self.max_length - 1] + [IGNORE_INDEX]
            valid_mask = valid_mask[:self.max_length - 1] + [0]

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": aligned_labels,
            "valid_mask": valid_mask,
            "original_tokens": tokens,
            "original_tags": tags
        }


def vihos_collate_fn(batch, pad_token_id=1):
    """
    Hàm gom batch và đệm (padding) động theo độ dài lớn nhất trong batch.
    PhoBERT pad_token_id mặc định là 1 (<pad>).
    """
    max_len = max(len(item["input_ids"]) for item in batch)
    
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []
    batch_valid_mask = []
    original_tokens = [item["original_tokens"] for item in batch]
    original_tags = [item["original_tags"] for item in batch]

    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        
        batch_input_ids.append(item["input_ids"] + [pad_token_id] * pad_len)
        batch_attention_mask.append(item["attention_mask"] + [0] * pad_len)
        batch_labels.append(item["labels"] + [IGNORE_INDEX] * pad_len)
        batch_valid_mask.append(item["valid_mask"] + [0] * pad_len)

    return {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long),
        "valid_mask": torch.tensor(batch_valid_mask, dtype=torch.bool),
        "original_tokens": original_tokens,
        "original_tags": original_tags
    }
