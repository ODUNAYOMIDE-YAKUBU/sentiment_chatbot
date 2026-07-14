import streamlit as st
import torch
import torch.nn as nn
import random
import os
import gdown
from transformers import BertModel, BertTokenizer

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment-Aware Conversational System",
    page_icon="🧠",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #09090f; }
    .stApp { background-color: #09090f; color: #e8e8f0; }
    .chat-message-user {
        background: linear-gradient(135deg, #5b21b6, #7c3aed);
        color: white;
        padding: 12px 16px;
        border-radius: 18px;
        border-bottom-right-radius: 4px;
        margin: 8px 0;
        margin-left: 20%;
    }
    .chat-message-bot {
        background: #1a1a2e;
        color: #e0e0e0;
        padding: 12px 16px;
        border-radius: 18px;
        border-bottom-left-radius: 4px;
        border: 1px solid #252540;
        margin: 8px 0;
        margin-right: 20%;
    }
    .badge-positive {
        background: #052e16;
        color: #86efac;
        border: 1px solid #16a34a;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 12px;
    }
    .badge-negative {
        background: #3b0a0a;
        color: #fca5a5;
        border: 1px solid #dc2626;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 12px;
    }
    .badge-neutral {
        background: #1e1b4b;
        color: #a5b4fc;
        border: 1px solid #4f46e5;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ── Model definition ──────────────────────────────────────────────
class BertLSTMClassifier(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased",
                 hidden_dim=256, num_layers=2,
                 num_classes=3, dropout=0.3):
        super(BertLSTMClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.lstm = nn.LSTM(
            input_size=768, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout, bidirectional=True
        )
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        bert_out        = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_out.last_hidden_state
        lstm_out, _     = self.lstm(sequence_output)
        lstm_final      = lstm_out[:, -1, :]
        out             = self.dropout(lstm_final)
        return self.classifier(out)

# ── Load model (cached so it only loads once) ─────────────────────
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    MODEL_PATH = "models/bert_lstm_best_v2.pt"

    if not os.path.exists(MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        with st.spinner("Downloading model from Google Drive..."):
            gdown.download(
                id="13lXPPs2Swgcx8QduF6s6g9GtNj1xU00M",
                output=MODEL_PATH,
                quiet=False
            )

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertLSTMClassifier().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, tokenizer, device

# ── Response templates ────────────────────────────────────────────
RESPONSES = {
    "positive": {
        "default": [
            "That is wonderful to hear! I am really glad things are going well for you. 😊",
            "Fantastic! Keep that positive energy going — you truly deserve it!",
            "Amazing! It sounds like things are really working out for you. 🌟",
            "I love hearing that! You seem to be in a really good place right now.",
            "That warms my heart! You deserve every bit of happiness coming your way.",
            "God don show up for you! That is really beautiful to hear. 🙏",
            "E good to hear say things dey go well — you deserve am!"
        ],
        "follow_up": [
            "You seem to be on a great streak! What has been the highlight for you?",
            "That positive energy keeps building! Tell me more. 🎉",
            "You are really thriving and I am genuinely happy to hear that.",
            "Things seem to keep getting better for you — that is beautiful to see!",
            "You dey shine! Wetin you do different wey things dey go like this?"
        ]
    },
    "negative": {
        "default": [
            "I am really sorry to hear that. That sounds genuinely tough. 💙",
            "That must be really hard for you right now. I hear you and I am here.",
            "I understand — it is okay to feel this way. You are not alone.",
            "That sounds really frustrating. Would you like to talk about it more?",
            "I no go lie, that one heavy. But I dey here with you through am. 💙",
            "E go better — this pain you dey feel now no go last forever. Hold on.",
            "Abeg no let it weigh you down — you strong pass this thing.",
            "Na so life be sometimes, but you go scale through. I dey here for you."
        ],
        "escalating": [
            "I can see this has been really weighing on you. Please know that you matter. 💙",
            "It sounds like things have been really difficult. Have you talked to someone you trust?",
            "I am genuinely concerned. You do not have to face this alone. 🤝",
            "Na you strong pass this thing — but you no need face am alone abeg."
        ]
    },
    "neutral": {
        "default": [
            "I see, thanks for sharing. Feel free to tell me more.",
            "Got it. How are you feeling about everything overall?",
            "I hear you. Is there anything specific on your mind?",
            "I am here and listening. Take your time.",
            "I dey hear you. Anything wey you wan talk about — I dey here."
        ]
    }
}

def get_response(sentiment, turn_count, sentiment_trail):
    escalating = (
        len(sentiment_trail) >= 2 and
        all(s == "negative" for s in sentiment_trail[-2:])
    )
    if sentiment == "negative" and escalating:
        pool = RESPONSES["negative"]["escalating"]
    elif sentiment == "negative":
        pool = RESPONSES["negative"]["default"]
    elif sentiment == "positive" and turn_count > 2:
        pool = RESPONSES["positive"]["follow_up"]
    elif sentiment == "positive":
        pool = RESPONSES["positive"]["default"]
    else:
        pool = RESPONSES["neutral"]["default"]
    return random.choice(pool)

def predict_sentiment(text, model, tokenizer, device):
    encoding = tokenizer(
        text, max_length=128, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred   = torch.argmax(logits, dim=1).item()
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    return label_map[pred], float(probs[pred]), {
        "negative": float(probs[0]),
        "neutral":  float(probs[1]),
        "positive": float(probs[2])
    }

# ── Initialize session state ──────────────────────────────────────
if "messages"        not in st.session_state: st.session_state.messages        = []
if "sentiment_trail" not in st.session_state: st.session_state.sentiment_trail = []
if "last_scores"     not in st.session_state: st.session_state.last_scores     = None
if "last_sentiment"  not in st.session_state: st.session_state.last_sentiment  = None

# ── Load model ────────────────────────────────────────────────────
with st.spinner("Loading BERT-LSTM model... please wait"):
    model, tokenizer, device = load_model()

# ── Header ────────────────────────────────────────────────────────
st.markdown("## 🧠 Sentiment-Aware Conversational System")
st.markdown("**Nigerian Context NLP · BERT-LSTM Architecture**")
st.divider()

# ── Layout ────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])

with col2:
    st.markdown("### 📊 Live Sentiment")
    if st.session_state.last_scores:
        scores = st.session_state.last_scores
        sentiment = st.session_state.last_sentiment
        emoji = "😊" if sentiment == "positive" else "😔" if sentiment == "negative" else "😐"
        st.markdown(f"**Detected:** {emoji} {sentiment.capitalize()}")
        st.progress(scores["positive"],  text=f"Positive: {scores['positive']*100:.1f}%")
        st.progress(scores["negative"],  text=f"Negative: {scores['negative']*100:.1f}%")
        st.progress(scores["neutral"],   text=f"Neutral:  {scores['neutral']*100:.1f}%")
    else:
        st.info("Send a message to see live sentiment scores")

    st.divider()
    st.markdown("### 💡 Try these:")
    examples = [
        "I am feeling great today!",
        "Everything don spoil for my life",
        "I dey manage sha",
        "God don bless me today!",
        "E don be for me today"
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state.quick_send = ex

    if st.button("🔄 Reset Conversation", use_container_width=True, type="secondary"):
        st.session_state.messages        = []
        st.session_state.sentiment_trail = []
        st.session_state.last_scores     = None
        st.session_state.last_sentiment  = None
        st.rerun()

with col1:
    # Display chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-message-user">{msg["content"]}</div>', unsafe_allow_html=True)
            badge_class = f"badge-{msg['sentiment']}"
            emoji = "😊" if msg['sentiment'] == "positive" else "😔" if msg['sentiment'] == "negative" else "😐"
            st.markdown(f'<span class="{badge_class}">{emoji} {msg["sentiment"].capitalize()} · {msg["confidence"]:.1f}% confidence</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message-bot">{msg["content"]}</div>', unsafe_allow_html=True)

    # Input
    user_input = st.chat_input("Type your message in English or Nigerian Pidgin...")

    # Handle quick send buttons
    if "quick_send" in st.session_state:
        user_input = st.session_state.quick_send
        del st.session_state.quick_send

    if user_input:
        sentiment, confidence, scores = predict_sentiment(
            user_input, model, tokenizer, device
        )
        st.session_state.sentiment_trail.append(sentiment)
        response = get_response(
            sentiment,
            len(st.session_state.messages) // 2,
            st.session_state.sentiment_trail
        )
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "sentiment": sentiment,
            "confidence": confidence * 100
        })
        st.session_state.messages.append({
            "role": "bot",
            "content": response
        })
        st.session_state.last_scores    = scores
        st.session_state.last_sentiment = sentiment
        st.rerun()
