import torch
import torch.nn as nn
from transformers import AutoModel

MODEL_NAME = "indobenchmark/indobert-base-p1"

class MultiTaskIndoBERTBiLSTM(nn.Module):
    def __init__(self, hidden_lstm=256, num_emo=5, num_urg=3, dropout=0.2, model_name=MODEL_NAME):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=hidden_lstm,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.head_emo = nn.Linear(hidden_lstm*2, num_emo)
        self.head_urg = nn.Linear(hidden_lstm*2, num_urg)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq = out.last_hidden_state                      # (B,T,H)
        lstm_out, _ = self.lstm(seq)                     # (B,T,2H)
        mask = attention_mask.unsqueeze(-1).float()      # (B,T,1)
        pooled = (lstm_out * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
        x = self.dropout(pooled)
        emo_logits = self.head_emo(x)
        urg_logits = self.head_urg(x)
        return emo_logits, urg_logits
