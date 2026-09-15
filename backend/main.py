import os
import sys
import time
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Thiết lập đường dẫn import
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from inference import ViHOSInferenceEngine

app = FastAPI(
    title="ViHOS Toxic Spans Guard API",
    description="Backend API phục vụ phát hiện chuỗi ngôn ngữ xúc phạm tiếng Việt bằng mô hình PhoBERT-BiLSTM-CRF",
    version="1.0.0"
)

# Cho phép CORS cho frontend (mặc định Vite port 5173 và các port local khác)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Phục vụ file biểu đồ tĩnh từ reports/figures
figures_dir = os.path.join(BASE_DIR, "reports", "figures")
if os.path.exists(figures_dir):
    app.mount("/static/figures", StaticFiles(directory=figures_dir), name="figures")

# Cấu hình 3 mô hình
MODEL_CONFIGS = {
    "linear": {
        "key": "PhoBERT-Linear",
        "file": "baseline_phobert_linear.pt",
        "name": "PhoBERT-Linear (Baseline Thầy)",
        "num": "1",
        "f1_val": "0.663",
        "color": "red",
        "eval_type": "warning",
        "eval_title": "Hiệu quả ở mức cơ bản",
        "eval_desc": "Phát hiện được các chuỗi xúc phạm rõ ràng, dễ dính lỗi cú pháp O → I-HOS và chưa tốt với ngữ cảnh phức tạp."
    },
    "crf": {
        "key": "PhoBERT-CRF",
        "file": "baseline_phobert_crf.pt",
        "name": "PhoBERT-CRF (Bóc tách Ablation)",
        "num": "2",
        "f1_val": "0.686",
        "color": "blue",
        "eval_type": "info",
        "eval_title": "Cải thiện so với baseline",
        "eval_desc": "Bóc tách ranh giới chuỗi tốt hơn, giảm nhiễu, triệt tiêu lỗi cú pháp nhờ Viterbi giải mã toàn cục."
    },
    "bilstm_crf": {
        "key": "PhoBERT-BiLSTM-CRF",
        "file": "best_phobert_bilstm_crf.pt",
        "name": "PhoBERT-BiLSTM-CRF (Đề xuất SOTA)",
        "num": "3",
        "f1_val": "0.703",
        "color": "green",
        "eval_type": "success",
        "eval_title": "Hiệu quả tốt nhất",
        "eval_desc": "Độ chính xác cao, xử lý tốt ngữ cảnh và ranh giới chuỗi, ghi nhớ ngữ cảnh 2 chiều."
    }
}

# Cache lưu trữ các engine đã tải vào RAM
loaded_engines: Dict[str, ViHOSInferenceEngine] = {}

def get_engine(model_alias: str) -> ViHOSInferenceEngine:
    if model_alias not in loaded_engines:
        cfg = MODEL_CONFIGS[model_alias]
        ckpt_path = os.path.join(BASE_DIR, "checkpoints", cfg["file"])
        engine = ViHOSInferenceEngine(checkpoint_path=ckpt_path)
        engine.model_name = cfg["name"]
        loaded_engines[model_alias] = engine
    return loaded_engines[model_alias]

def classify_categories(words: List[str], spans: List[str]) -> List[str]:
    """Phân loại cụm từ xúc phạm thành 5 nhóm nhãn phổ biến."""
    detected = set()
    insult_kw = ["ngu", "đần", "óc", "chó", "súc sinh", "mặt lợn", "điên", "dốt", "hãm", "vô học", "rác rưởi", "phế vật", "con điếm", "đĩ", "mặt dày", "đồ khùng", "mất dạy"]
    profanity_kw = ["đụ", "địt", "lồn", "cặc", "vãi", "mẹ kiếp", "cút", "đéo", "éo", "lol", "buồi", "thằng chó", "con mẹ", "đm", "dcm", "vcl", "xàm"]
    threat_kw = ["giết", "đập", "đánh", "chém", "bẻ gãy", "bắn", "đốt", "bóp cổ", "cho ăn đòn", "xử đẹp", "khử", "ra khỏi đây", "cút xéo"]
    discrim_kw = ["bắc kỳ", "nam kỳ", "mọi rợ", "bê đê", "nhà quê", "bần nông", "mọi", "nam cầy", "bắc cầy", "phản quốc"]

    for sp in spans:
        sp_clean = sp.replace("_", " ").lower()
        if any(kw in sp_clean for kw in insult_kw):
            detected.add("INSULT")
        if any(kw in sp_clean for kw in profanity_kw):
            detected.add("PROFANITY")
        if any(kw in sp_clean for kw in threat_kw):
            detected.add("THREAT")
        if any(kw in sp_clean for kw in discrim_kw):
            detected.add("DISCRIMINATION")

    if not detected and spans:
        detected.add("OTHER")

    # Thứ tự chuẩn hiển thị
    order = ["INSULT", "PROFANITY", "THREAT", "DISCRIMINATION", "OTHER"]
    return [cat for cat in order if cat in detected]

def format_masked_text(words: List[str], tags: List[str]) -> str:
    """Tạo chuỗi văn bản đã che ký tự độc hại với ***"""
    masked_tokens = []
    for w, t in zip(words, tags):
        if t in ("B-HOS", "I-HOS"):
            masked_tokens.append("***")
        else:
            masked_tokens.append(w.replace("_", " "))

    merged = []
    in_star = False
    for token in masked_tokens:
        if token == "***":
            if not in_star:
                merged.append("***")
                in_star = True
        else:
            merged.append(token)
            in_star = False
    return " ".join(merged)

class PredictRequest(BaseModel):
    text: str
    model_key: Optional[str] = "all"  # "linear", "crf", "bilstm_crf", or "all"

@app.get("/")
def index():
    return {
        "status": "online",
        "service": "ViHOS Toxic Spans Guard API",
        "version": "1.0.0",
        "docs_url": "/docs"
    }

@app.get("/api/health")
def health_check():
    models_status = {}
    for alias, cfg in MODEL_CONFIGS.items():
        is_loaded = alias in loaded_engines and getattr(loaded_engines[alias], "checkpoint_loaded", False)
        ckpt_exists = os.path.exists(os.path.join(BASE_DIR, "checkpoints", cfg["file"]))
        models_status[alias] = {
            "name": cfg["name"],
            "file": cfg["file"],
            "checkpoint_exists": ckpt_exists,
            "loaded_in_memory": is_loaded
        }
    return {
        "status": "ok",
        "platform": sys.platform,
        "models": models_status
    }

@app.post("/api/predict")
def predict_endpoint(req: PredictRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Vui lòng nhập văn bản cần phân tích.")

    aliases = ["linear", "crf", "bilstm_crf"] if req.model_key == "all" else [req.model_key]
    
    start_total = time.time()
    results = []

    for alias in aliases:
        if alias not in MODEL_CONFIGS:
            continue
        cfg = MODEL_CONFIGS[alias]
        eng = get_engine(alias)
        pred = eng.predict(text)
        
        detected_cats = classify_categories(pred["words"], pred["spans"])
        masked_txt = format_masked_text(pred["words"], pred["tags"])

        # Kiểm tra bất thường cú pháp nhãn
        has_syntax_err = any(p == "O" and c == "I-HOS" for p, c in zip(pred["tags"][:-1], pred["tags"][1:]))

        results.append({
            "alias": alias,
            "key": cfg["key"],
            "name": cfg["name"],
            "num": cfg["num"],
            "f1_val": cfg["f1_val"],
            "color": cfg["color"],
            "is_toxic": pred["is_toxic"],
            "spans_count": len(pred["spans"]),
            "spans": pred["spans"],
            "latency_ms": round(pred["latency_ms"], 1),
            "categories": detected_cats,
            "masked_text": masked_txt,
            "has_syntax_error": has_syntax_err,
            "eval_type": cfg["eval_type"],
            "eval_title": cfg["eval_title"],
            "eval_desc": cfg["eval_desc"],
            "words": pred["words"],
            "tags": pred["tags"]
        })

    total_time_s = round(time.time() - start_total, 2)

    return {
        "text": text,
        "total_latency_seconds": total_time_s,
        "models": results
    }

@app.get("/api/reports/download-excel")
def download_excel():
    excel_path = os.path.join(BASE_DIR, "reports", "error_analysis.xlsx")
    if os.path.exists(excel_path):
        return FileResponse(excel_path, filename="ViHOS_Error_Analysis.xlsx")
    raise HTTPException(status_code=404, detail="File báo cáo excel chưa được sinh.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
