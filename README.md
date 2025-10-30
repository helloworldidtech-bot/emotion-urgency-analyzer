# Emotion & Urgency Analyzer (IndoBERT + BiLSTM)

## Struktur
- `artifacts/best_model.pt` — weight model terbaik
- `artifacts/label_space.json` — mapping label & idx2label
- `model_multitask.py` — arsitektur model
- `predict_single.py` — fungsi prediksi
- `app_streamlit.py` — UI demo & visualisasi
- `dataset_labeled_awal.csv` — (opsional) untuk grafik distribusi

## Cara jalan
```bash
pip install -r requirements.txt
streamlit run app_streamlit.py
