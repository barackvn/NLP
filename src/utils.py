import os
import random
import logging
import torch
import numpy as np

# Định nghĩa nhãn chuẩn bài toán ViHOS
LABEL2ID = {"O": 0, "B-HOS": 1, "I-HOS": 2}
ID2LABEL = {0: "O", 1: "B-HOS", 2: "I-HOS"}
NUM_LABELS = len(LABEL2ID)
IGNORE_INDEX = -100

def set_seed(seed: int = 42):
    """Cố định seed ngẫu nhiên cho tính tái lập kết quả (Reproducibility)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_device() -> torch.device:
    """Xác định device tối ưu (CUDA nếu có GPU, ngược lại CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

def setup_logger(name: str = "ViHOS", log_file: str = None) -> logging.Logger:
    """Khởi tạo logger theo chuẩn format."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            
    return logger

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: dict, filepath: str):
    """Lưu trữ checkpoint mô hình cùng optimizer và metrics."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "metrics": metrics,
    }
    torch.save(state, filepath)

def load_checkpoint(filepath: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None, map_location: str = "cpu"):
    """Nạp trọng số mô hình từ checkpoint."""
    checkpoint = torch.load(filepath, map_location=map_location)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    if optimizer and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"]:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("metrics", {}), checkpoint.get("epoch", 0)
