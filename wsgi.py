import os
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import gdown

# ── Load everything BEFORE gunicorn workers start ────────────────
device = torch.device("cpu")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "bert_lstm_best_v2.pt")
if not os.path.exists(MODEL_PATH):
    os.makedirs("models", exist_ok=True)
    print("Downloading model from Google Drive...")
    gdown.download(id="13lXPPs2Swgcx8QduF6s6g9GtNj1xU00M", output=MODEL_PATH, quiet=False)
    print("Model downloaded successfully!")

class BertLSTMClassifier(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased",
                 hidden_dim=256, num_layers=2,
                 num_classes=3, dropout=0.3):
        super(BertLSTMClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout, bidirectional=True)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        bert_out        = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_out.last_hidden_state
        lstm_out, _     = self.lstm(sequence_output)
        lstm_final      = lstm_out[:, -1, :]
        out             = self.dropout(lstm_final)
        return self.classifier(out)

print("Loading tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
print("Loading BERT-LSTM model...")
sentiment_model = BertLSTMClassifier().to(device)
sentiment_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
sentiment_model.eval()
print("Model loaded successfully!")

from app import app

if __name__ == "__main__":
    app.run()
