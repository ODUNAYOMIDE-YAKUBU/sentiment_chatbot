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
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide default streamlit elements */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* Main background */
.stApp {
    background: #0a0a12;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #11111e;
    border-right: 1px solid #1e1e35;
}

section[data-testid="stSidebar"] * {
    color: #c8c8e0 !important;
}

/* Chat messages */
.user-bubble {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    margin: 12px 0;
    animation: slideIn 0.2s ease;
}

.bot-bubble {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    margin: 12px 0;
    animation: slideIn 0.2s ease;
}

@keyframes slideIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

.user-text {
    background: linear-gradient(135deg, #4c1d95, #7c3aed);
    color: #ffffff;
    padding: 13px 18px;
    border-radius: 20px 20px 4px 20px;
    max-width: 75%;
    font-size: 14px;
    line-height: 1.65;
    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.25);
}

.bot-text {
    background: #1a1a2e;
    color: #e0e0f0;
    padding: 13px 18px;
    border-radius: 20px 20px 20px 4px;
    max-width: 75%;
    font-size: 14px;
    line-height: 1.65;
    border: 1px solid #2a2a45;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}

.bot-label {
    font-size: 11px;
    color: #5a5a7a;
    margin-bottom: 4px;
    margin-left: 4px;
    font-weight: 500;
    letter-spacing: 0.03em;
}

.user-label {
    font-size: 11px;
    color: #5a5a7a;
    margin-bottom: 4px;
    margin-right: 4px;
    font-weight: 500;
    letter-spacing: 0.03em;
    text-align: right;
}

/* Sentiment badges */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    margin-top: 6px;
    letter-spacing: 0.04em;
}

.badge-positive {
    background: #052e16;
    color: #86efac;
    border: 1px solid #166534;
}

.badge-negative {
    background: #3b0a0a;
    color: #fca5a5;
    border: 1px solid #991b1b;
}

.badge-neutral {
    background: #1e1b4b;
    color: #a5b4fc;
    border: 1px solid #3730a3;
}

/* Score bars */
.score-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 4px 0;
}

.score-label {
    font-size: 11px;
    color: #6b7280;
    width: 60px;
    font-weight: 500;
}

.score-bar-bg {
    flex: 1;
    height: 5px;
    background: #1e1e35;
    border-radius: 3px;
    overflow: hidden;
}

.score-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.4s ease;
}

.score-val {
    font-size: 11px;
    color: #9ca3af;
    width: 38px;
    text-align: right;
    font-family: 'JetBrains Mono', monospace;
}

/* Welcome screen */
.welcome-wrap {
    text-align: center;
    padding: 60px 20px 40px;
}

.welcome-icon {
    font-size: 52px;
    margin-bottom: 16px;
}

.welcome-title {
    font-size: 22px;
    font-weight: 600;
    color: #e0e0f0;
    margin-bottom: 10px;
}

.welcome-sub {
    font-size: 14px;
    color: #6b7280;
    max-width: 420px;
    margin: 0 auto 28px;
    line-height: 1.7;
}

/* Chip buttons */
.stButton > button {
    background: #1a1a2e !important;
    border: 1px solid #2a2a45 !important;
    color: #a0a0c0 !important;
    border-radius: 20px !important;
    font-size: 12px !important;
    padding: 6px 14px !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.15s !important;
    font-weight: 400 !important;
}

.stButton > button:hover {
    border-color: #7c3aed !important;
    color: #a78bfa !important;
    background: #1e1a2e !important;
}

/* Primary send button */
div[data-testid="stButton"]:last-child > button {
    background: linear-gradient(135deg, #5b21b6, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 500 !important;
}

/* Input box */
.stTextArea textarea {
    background: #11111e !important;
    border: 1px solid #2a2a45 !important;
    border-radius: 12px !important;
    color: #e0e0f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
}

.stTextArea textarea:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.15) !important;
}

/* Metric cards */
.metric-card {
    background: #11111e;
    border: 1px solid #1e1e35;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 10px;
}

.metric-title {
    font-size: 11px;
    color: #5a5a7a;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
}

.metric-value {
    font-size: 20px;
    font-weight: 600;
    color: #e0e0f0;
}

/* Divider */
hr {
    border: none;
    border-top: 1px solid #1e1e35;
    margin: 16px 0;
}

/* Sidebar stat pill */
.stat-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #1a1a2e;
    border: 1px solid #2a2a45;
    border-radius: 20px;
    padding: 5px 12px;
    font-size: 12px;
    color: #a0a0c0;
    margin: 3px 2px;
}

.online-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #22c55e;
    display: inline-block;
    box-shadow: 0 0 5px #22c55e88;
    animation: blink 2s ease-in-out infinite;
}

@keyframes blink {
    0%,100% { opacity: 1; }
    50%      { opacity: 0.4; }
}

/* History items */
.history-item {
    background: #11111e;
    border: 1px solid #1e1e35;
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 12px;
}

.history-sentiment {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-bottom: 3px;
}

.history-text {
    color: #8888aa;
    line-height: 1.5;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
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


# ── Load model (cached) ───────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    device    = torch.device("cpu")
    MODEL_PATH = "models/bert_lstm_best_v2.pt"
    if not os.path.exists(MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        gdown.download(
            id="13lXPPs2Swgcx8QduF6s6g9GtNj1xU00M",
            output=MODEL_PATH, quiet=False
        )
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model     = BertLSTMClassifier().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, tokenizer, device


# ── Response bank ─────────────────────────────────────────────────
RESPONSES = {
    "positive": {
        "default": [
            "That is wonderful to hear! I am really glad things are going well for you. 😊",
            "Fantastic! Keep that positive energy going — you truly deserve it!",
            "Amazing! It sounds like things are really working out for you. 🌟",
            "I love hearing that! You seem to be in a really good place right now.",
            "That warms my heart! You deserve every bit of happiness coming your way.",
            "God don show up for you! That is really beautiful to hear. 🙏",
            "E good to hear say things dey go well — you deserve am!",
            "See you thriving! This is so good to hear — keep going strong."
        ],
        "follow_up": [
            "You seem to be on a great streak! What has been the highlight for you?",
            "That positive energy keeps building! Tell me more. 🎉",
            "You are really thriving and I am genuinely happy to hear that.",
            "Things seem to keep getting better for you — that is beautiful to see!",
            "You dey shine! Wetin you do different wey things dey go like this?"
        ],
        "achievement": [
            "That is a massive achievement! You put in the work and it paid off. 🏆",
            "You earned that! Nobody can take that away from you.",
            "Na your hustle carry you reach here — celebrate yourself well well!",
            "You should be incredibly proud of yourself right now. Well done! 🎊"
        ]
    },
    "negative": {
        "default": [
            "I am really sorry to hear that. That sounds genuinely tough. 💙",
            "That must be really hard for you right now. I hear you and I am here.",
            "I understand — it is okay to feel this way. You are not alone.",
            "That sounds really frustrating. Would you like to talk about it more?",
            "I am sorry you are going through this. How can I support you right now?",
            "I no go lie, that one heavy. But I dey here with you through am. 💙",
            "E go better — this pain you dey feel now no go last forever. Hold on.",
            "Abeg no let it weigh you down — you strong pass this thing.",
            "Na so life be sometimes, but you go scale through. I dey here for you.",
            "You do not have to pretend everything is fine. I am here to listen."
        ],
        "escalating": [
            "I can see this has been really weighing on you. Please know that you matter. 💙",
            "It sounds like things have been really difficult. Have you talked to someone you trust?",
            "I am genuinely concerned. You do not have to face this alone. 🤝",
            "Na you strong pass this thing — but you no need face am alone abeg.",
            "You have been carrying a lot. Please be gentle with yourself."
        ],
        "nigerian_context": [
            "E go better, I promise you. This kind situation no go last forever. 💪",
            "Even when e dark like this, e no mean say light no dey come. E go bright again.",
            "Naija people strong — and you be part of that strength. This one no go finish you.",
            "Oya breathe small. You dey do better than you think, even if e no feel like am."
        ]
    },
    "neutral": {
        "default": [
            "I see, thanks for sharing. Feel free to tell me more.",
            "Got it. How are you feeling about everything overall?",
            "I hear you. Is there anything specific on your mind?",
            "I am here and listening. Take your time.",
            "I dey hear you. Anything wey you wan talk about — I dey here.",
            "Sometimes things just are what they are. How are you sitting with it?"
        ],
        "checking_in": [
            "How are you really doing today? Sometimes the honest answer is different.",
            "Beyond the surface, how are you feeling inside?",
            "I dey check on you — how you really dey? No need to form strong."
        ]
    }
}

QUICK_MESSAGES = [
    "I am feeling great today! 😊",
    "Everything don spoil for my life 😔",
    "I dey manage sha",
    "God don bless me today!",
    "E don be for me today",
    "I just got promoted at work!"
]


def get_response(sentiment, turn_count, sentiment_trail):
    escalating = (
        len(sentiment_trail) >= 2 and
        all(s == "negative" for s in sentiment_trail[-2:])
    )
    if sentiment == "negative" and escalating:
        pool = RESPONSES["negative"]["escalating"]
    elif sentiment == "negative":
        pool = RESPONSES["negative"]["default"] + RESPONSES["negative"]["nigerian_context"]
    elif sentiment == "positive" and turn_count > 2:
        pool = RESPONSES["positive"]["follow_up"] + RESPONSES["positive"]["achievement"]
    elif sentiment == "positive":
        pool = RESPONSES["positive"]["default"]
    elif sentiment == "neutral" and turn_count > 1:
        pool = RESPONSES["neutral"]["default"] + RESPONSES["neutral"]["checking_in"]
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
        "positive": float(probs[2]),
        "negative": float(probs[0]),
        "neutral":  float(probs[1])
    }


def badge_html(sentiment, confidence):
    emoji = "😊" if sentiment == "positive" else "😔" if sentiment == "negative" else "😐"
    cls   = f"badge-{sentiment}"
    return f'<span class="badge {cls}">{emoji} {sentiment.capitalize()} · {confidence:.1f}%</span>'


def score_bars_html(scores):
    bars = ""
    colors = {"positive": "#22c55e", "negative": "#ef4444", "neutral": "#818cf8"}
    for label, color in colors.items():
        val = scores[label] * 100
        bars += f"""
        <div class="score-row">
            <span class="score-label">{label.capitalize()}</span>
            <div class="score-bar-bg">
                <div class="score-bar-fill" style="width:{val:.1f}%;background:{color}"></div>
            </div>
            <span class="score-val">{val:.1f}%</span>
        </div>"""
    return bars


# ── Session state ─────────────────────────────────────────────────
if "messages"        not in st.session_state: st.session_state.messages        = []
if "sentiment_trail" not in st.session_state: st.session_state.sentiment_trail = []
if "last_scores"     not in st.session_state: st.session_state.last_scores     = None
if "last_sentiment"  not in st.session_state: st.session_state.last_sentiment  = None
if "total_turns"     not in st.session_state: st.session_state.total_turns     = 0
if "quick_msg"       not in st.session_state: st.session_state.quick_msg       = None


# ── Load model ────────────────────────────────────────────────────
with st.spinner("Loading BERT-LSTM model — please wait a moment..."):
    model, tokenizer, device = load_model()


# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:4px 0 16px'>
        <div style='font-size:20px;font-weight:600;color:#e0e0f0;margin-bottom:4px'>🧠 SentiChat</div>
        <div style='font-size:12px;color:#5a5a7a'>Nigerian Context NLP System</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="stat-pill"><span class="online-dot"></span> Model active</div>', unsafe_allow_html=True)
    st.markdown('<div class="stat-pill">⚡ BERT-LSTM v2</div>', unsafe_allow_html=True)
    st.markdown('<div class="stat-pill">🇳🇬 Nigerian context</div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Live scores
    st.markdown('<div style="font-size:11px;font-weight:600;color:#5a5a7a;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:10px">Live sentiment scores</div>', unsafe_allow_html=True)

    if st.session_state.last_scores:
        scores    = st.session_state.last_scores
        sentiment = st.session_state.last_sentiment
        emoji     = "😊" if sentiment == "positive" else "😔" if sentiment == "negative" else "😐"
        color     = "#22c55e" if sentiment == "positive" else "#ef4444" if sentiment == "negative" else "#818cf8"

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Detected sentiment</div>
            <div class="metric-value" style="color:{color}">{emoji} {sentiment.capitalize()}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(score_bars_html(scores), unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-size:13px;color:#5a5a7a;padding:8px 0">Send a message to see live scores</div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Stats
    total = len([m for m in st.session_state.messages if m["role"] == "user"])
    if total > 0:
        trail  = st.session_state.sentiment_trail
        pos    = trail.count("positive")
        neg    = trail.count("negative")
        neu    = trail.count("neutral")
        st.markdown('<div style="font-size:11px;font-weight:600;color:#5a5a7a;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:10px">Session stats</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total turns</div>
            <div class="metric-value">{total}</div>
        </div>
        <div style="display:flex;gap:8px;margin-bottom:10px">
            <div class="metric-card" style="flex:1;text-align:center">
                <div class="metric-title">Positive</div>
                <div style="font-size:16px;font-weight:600;color:#22c55e">{pos}</div>
            </div>
            <div class="metric-card" style="flex:1;text-align:center">
                <div class="metric-title">Negative</div>
                <div style="font-size:16px;font-weight:600;color:#ef4444">{neg}</div>
            </div>
            <div class="metric-card" style="flex:1;text-align:center">
                <div class="metric-title">Neutral</div>
                <div style="font-size:16px;font-weight:600;color:#818cf8">{neu}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

    # Quick messages
    st.markdown('<div style="font-size:11px;font-weight:600;color:#5a5a7a;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:10px">Try these</div>', unsafe_allow_html=True)
    for qm in QUICK_MESSAGES:
        if st.button(qm, key=f"q_{qm}", use_container_width=True):
            st.session_state.quick_msg = qm
            st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    if st.button("🔄 Reset conversation", use_container_width=True):
        st.session_state.messages        = []
        st.session_state.sentiment_trail = []
        st.session_state.last_scores     = None
        st.session_state.last_sentiment  = None
        st.session_state.total_turns     = 0
        st.rerun()


# ── Main area ─────────────────────────────────────────────────────
st.markdown("""
<div style='padding:8px 0 20px'>
    <div style='font-size:22px;font-weight:600;color:#e0e0f0'>Sentiment-Aware Conversational System</div>
    <div style='font-size:13px;color:#5a5a7a;margin-top:3px'>Detecting emotion · Responding with empathy · Optimized for Nigerian context</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# Chat area
chat_area = st.container()

with chat_area:
    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome-wrap">
            <div class="welcome-icon">💬</div>
            <div class="welcome-title">How are you feeling today?</div>
            <div class="welcome-sub">
                This system detects your sentiment in real time and responds with genuine empathy.<br>
                You can type in <strong style="color:#a78bfa">English</strong> or <strong style="color:#a78bfa">Nigerian Pidgin</strong>.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="user-bubble">
                    <div class="user-label">You</div>
                    <div class="user-text">{msg["content"]}</div>
                    {badge_html(msg["sentiment"], msg["confidence"])}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="bot-bubble">
                    <div class="bot-label">SentiChat</div>
                    <div class="bot-text">{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# Input area
col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.text_area(
        label="message",
        placeholder="Type your message in English or Nigerian Pidgin...",
        height=70,
        label_visibility="collapsed",
        key="chat_input"
    )
with col2:
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    send = st.button("Send 💬", use_container_width=True, type="primary")

# Handle quick message
if st.session_state.quick_msg:
    user_input = st.session_state.quick_msg
    st.session_state.quick_msg = None
    send = True

# Process message
if send and user_input and user_input.strip():
    text = user_input.strip()

    sentiment, confidence, scores = predict_sentiment(text, model, tokenizer, device)
    st.session_state.sentiment_trail.append(sentiment)

    response = get_response(
        sentiment,
        len(st.session_state.messages) // 2,
        st.session_state.sentiment_trail
    )

    st.session_state.messages.append({
        "role":       "user",
        "content":    text,
        "sentiment":  sentiment,
        "confidence": confidence * 100
    })
    st.session_state.messages.append({
        "role":    "bot",
        "content": response
    })
    st.session_state.last_scores    = scores
    st.session_state.last_sentiment = sentiment
    st.session_state.total_turns   += 1
    st.rerun()
