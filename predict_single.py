import json, re, torch
from transformers import AutoTokenizer
from model_multitask import MultiTaskIndoBERTBiLSTM, MODEL_NAME

MODEL_PATH = "artifacts/best_model.pt"
LABEL_PATH = "artifacts/label_space.json" 
MAX_LEN = 160
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def preprocess_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r'https?://\S+|www\.\S+',' ', s)
    s = re.sub(r'[^0-9a-zà-ÿ\s]', ' ', s)
    s = re.sub(r'\s+',' ', s).strip()
    return s

# load label space
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    ls = json.load(f)

# gunakan IDX2LABEL kalau tersedia, fallback ke MAP
emo_idx2label = ls.get("EMO_IDX2LABEL")
urg_idx2label = ls.get("URG_IDX2LABEL")
if emo_idx2label is None:
    emo_map = ls["EMO_MAP"]; emo_idx2label = [None]*len(emo_map)
    for k,v in emo_map.items(): emo_idx2label[v]=k
if urg_idx2label is None:
    urg_map = {int(k):v for k,v in ls["URG_MAP"].items()}
    urg_idx2label = [None]*len(urg_map)
    for k,v in urg_map.items(): urg_idx2label[v]=k

# load model & tokenizer
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model = MultiTaskIndoBERTBiLSTM(hidden_lstm=ckpt.get("hidden_lstm",256))
model.load_state_dict(ckpt["model_state"])
model.to(DEVICE).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

@torch.no_grad()
def predict_text(text: str):
    text = preprocess_text(text)
    enc = tokenizer(text, max_length=MAX_LEN, truncation=True,
                    padding="max_length", return_tensors="pt")
    enc = {k:v.to(DEVICE) for k,v in enc.items()}
    emo_logits, urg_logits = model(enc["input_ids"], enc["attention_mask"])
    emo_idx = int(torch.argmax(emo_logits, dim=-1).item())
    urg_idx = int(torch.argmax(urg_logits, dim=-1).item())
    return emo_idx2label[emo_idx], urg_idx2label[urg_idx]
