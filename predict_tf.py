import re, json, tensorflow as tf
from transformers import AutoTokenizer
from model_tf import IndoBERTBiLSTMClassifier, MAX_LEN

# paths
EMO_WEIGHTS = "artifacts/model_emosi_8020.weights.h5"
URG_WEIGHTS = "artifacts/model_urgensi_8020.weights.h5"
LABEL_PATH  = "artifacts/label_space.json"
TOK_PATH    = "artifacts/tokenizer"

# load label
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    LS = json.load(f)
EMO_IDX2LABEL = LS["EMO_IDX2LABEL"]
URG_IDX2LABEL = LS["URG_IDX2LABEL"]

# load tokenizer
tok = AutoTokenizer.from_pretrained(TOK_PATH)

def clean_text(s):
    s = s.lower().strip()
    s = re.sub(r'https?://\S+|www\.\S+', ' ', s)
    s = re.sub(r'[^0-9a-zà-ÿ\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# load models
emo_model = IndoBERTBiLSTMClassifier(num_classes=len(EMO_IDX2LABEL))
emo_model.load_weights(EMO_WEIGHTS)
urg_model = IndoBERTBiLSTMClassifier(num_classes=len(URG_IDX2LABEL))
urg_model.load_weights(URG_WEIGHTS)

def predict_text(text):
    """
    Input:
        text (str) - teks ulasan
    Output:
        emosi_label (str), urg_label (str), conf_emosi (float), conf_urg (float)
    """
    # tokenisasi + preprocessing
    enc = tok(clean_text(text),
              max_length=MAX_LEN,
              truncation=True,
              padding="max_length",
              return_attention_mask=True,
              return_tensors="tf")

    # raw model outputs (logits atau probabilities tergantung implementasi model)
    emo_logits = emo_model([enc["input_ids"], enc["attention_mask"]])
    urg_logits = urg_model([enc["input_ids"], enc["attention_mask"]])

    # pastikan menjadi numpy arrays
    emo_logits_np = emo_logits.numpy()
    urg_logits_np = urg_logits.numpy()

    # konversi ke probabilitas dengan softmax (aman baik logits maupun sudah probs)
    emo_probs = tf.nn.softmax(emo_logits_np, axis=-1).numpy()
    urg_probs = tf.nn.softmax(urg_logits_np, axis=-1).numpy()

    # ambil indeks prediksi dan labelnya
    emo_idx = int(tf.argmax(emo_probs, axis=-1).numpy()[0])
    urg_idx = int(tf.argmax(urg_probs, axis=-1).numpy()[0])

    emosi_label = EMO_IDX2LABEL[emo_idx]
    urg_label = URG_IDX2LABEL[urg_idx]

    # confidence: probabilitas tertinggi untuk setiap prediksi (skala 0..1)
    conf_emosi = float(emo_probs[0, emo_idx])
    conf_urg = float(urg_probs[0, urg_idx])

    return emosi_label, urg_label, conf_emosi, conf_urg


if __name__ == "__main__":
    teks = "Barangnya rusak parah, tidak sesuai deskripsi dan baterainya cepat habis"
    emosi, urg, conf_emosi, conf_urg = predict_text(teks)
    print("Emosi:", emosi, f"({conf_emosi*100:.2f}% confidence)")
    print("Urgensi:", urg, f"({conf_urg*100:.2f}% confidence)")
