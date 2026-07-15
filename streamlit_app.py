import streamlit as st
import torch
import torch.nn as nn
import random
import os
import gdown
from transformers import BertModel, BertTokenizer
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment-Aware Conversational System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
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

/* Hide sidebar completely */
section[data-testid="stSidebar"] {
    display: none !important;
}

/* Main container - center the chat */
.block-container {
    max-width: 900px !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    padding-top: 0 !important;
}

/* ── Header ─────────────────────────────────────── */
.header-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px 0 12px;
    border-bottom: 1px solid #1e1e35;
    margin-bottom: 0;
}

.header-icon {
    width: 40px;
    height: 40px;
    background: linear-gradient(135deg, #7c3aed, #a855f7);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
}

.header-text {
    flex: 1;
}

.header-title {
    font-size: 16px;
    font-weight: 600;
    color: #e0e0f0;
    margin: 0;
    line-height: 1.3;
}

.header-subtitle {
    font-size: 12px;
    color: #5a5a7a;
    margin: 0;
    line-height: 1.3;
}

/* ── Live Scores Bar ────────────────────────────── */
.scores-bar {
    display: flex;
    align-items: center;
    gap: 20px;
    padding: 10px 0;
    border-bottom: 1px solid #1e1e35;
    margin-bottom: 16px;
}

.scores-bar-label {
    font-size: 11px;
    font-weight: 600;
    color: #5a5a7a;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.score-item {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
}

.score-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    display: inline-block;
}

.score-dot-positive { background: #22c55e; }
.score-dot-negative { background: #ef4444; }
.score-dot-neutral  { background: #818cf8; }

.score-mini-bar {
    width: 60px;
    height: 4px;
    background: #1e1e35;
    border-radius: 2px;
    overflow: hidden;
}

.score-mini-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.4s ease;
}

.score-mini-val {
    font-size: 11px;
    font-weight: 600;
    color: #e0e0f0;
    font-family: 'JetBrains Mono', monospace;
    min-width: 36px;
}

/* ── Chat Area ──────────────────────────────────── */
.chat-container {
    max-width: 700px;
    margin: 0 auto;
}

/* Chat messages */
.message-wrapper {
    margin: 16px 0;
    animation: slideIn 0.25s ease;
}

@keyframes slideIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

.message-meta {
    font-size: 11px;
    color: #5a5a7a;
    margin-bottom: 6px;
    font-weight: 500;
}

.message-meta-user {
    text-align: right;
}

.message-meta-bot {
    text-align: left;
}

.user-bubble-wrap {
    display: flex;
    justify-content: flex-end;
}

.bot-bubble-wrap {
    display: flex;
    justify-content: flex-start;
}

.user-text {
    background: linear-gradient(135deg, #5b21b6, #7c3aed);
    color: #ffffff;
    padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    max-width: 80%;
    font-size: 14px;
    line-height: 1.6;
    box-shadow: 0 2px 12px rgba(124, 58, 237, 0.2);
}

.bot-text {
    background: #1a1a2e;
    color: #e0e0f0;
    padding: 12px 18px;
    border-radius: 18px 18px 18px 4px;
    max-width: 80%;
    font-size: 14px;
    line-height: 1.6;
    border: 1px solid #2a2a45;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}

/* Sentiment badges */
.badge-wrap-user {
    display: flex;
    justify-content: flex-end;
    margin-top: 6px;
}

.badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
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

.confidence-text {
    font-size: 11px;
    color: #5a5a7a;
    margin-top: 3px;
    text-align: right;
    font-family: 'JetBrains Mono', monospace;
}

/* Welcome screen */
.welcome-wrap {
    text-align: center;
    padding: 80px 20px 60px;
}

.welcome-icon {
    font-size: 48px;
    margin-bottom: 16px;
}

.welcome-title {
    font-size: 20px;
    font-weight: 600;
    color: #e0e0f0;
    margin-bottom: 10px;
}

.welcome-sub {
    font-size: 13px;
    color: #6b7280;
    max-width: 400px;
    margin: 0 auto 32px;
    line-height: 1.7;
}

/* Quick chips */
.quick-chips {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 8px;
    max-width: 500px;
    margin: 0 auto;
}

/* Input area */
.input-area {
    position: sticky;
    bottom: 0;
    background: #0a0a12;
    padding: 12px 0 20px;
    border-top: 1px solid #1e1e35;
    margin-top: 20px;
}

.stTextArea textarea {
    background: #11111e !important;
    border: 1px solid #2a2a45 !important;
    border-radius: 14px !important;
    color: #e0e0f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    resize: none !important;
}

.stTextArea textarea:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.15) !important;
}

/* Send button */
.send-btn > button {
    background: linear-gradient(135deg, #5b21b6, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 500 !important;
    height: 44px !important;
    width: 100% !important;
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

/* Reset button */
.reset-btn > button {
    background: transparent !important;
    border: 1px solid #2a2a45 !important;
    color: #5a5a7a !important;
    font-size: 12px !important;
}

.reset-btn > button:hover {
    border-color: #ef4444 !important;
    color: #fca5a5 !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-track {
    background: #0a0a12;
}
::-webkit-scrollbar-thumb {
    background: #2a2a45;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #3a3a55;
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
            "That is really great news! What else has been making your day good?",
            "Amazing! It sounds like things are really working out for you. 🌟",
            "I love hearing that! You seem to be in a really good place right now.",
            "That warms my heart! You deserve every bit of happiness coming your way.",
            "Wow, that is such good news! I am genuinely happy for you. 🎉",
            "See you thriving! This is so good to hear — keep going strong.",
            "You are doing amazingly well and that is something to be proud of.",
            "That kind of positivity is contagious! Thank you for sharing it with me.",
            "God don show up for you! That is really beautiful to hear. 🙏",
            "E good to hear say things dey go well — you deserve am!"
        ],
        "follow_up": [
            "You seem to be on a great streak! What has been the highlight for you?",
            "That positive energy keeps building! Tell me more about what is going on. 🎉",
            "You are really thriving and I am genuinely happy to hear that.",
            "Things seem to keep getting better for you — that is beautiful to see!",
            "You have been carrying so much good energy lately. What is your secret?",
            "I have noticed things keep going well for you — that is not by accident. Keep it up!",
            "You dey shine! Wetin you do different wey things dey go like this?",
            "Every time we talk you bring good news — I really appreciate that energy!"
        ],
        "achievement": [
            "That is a massive achievement! You put in the work and it paid off. 🏆",
            "You earned that! Nobody can take that away from you.",
            "See the results of your hard work! This is just the beginning.",
            "You should be incredibly proud of yourself right now. Well done! 🎊",
            "Na your hustle carry you reach here — celebrate yourself well well!",
            "This kind win no come easy — you deserve to enjoy every moment of it."
        ]
    },

    "negative": {
        "default": [
            "I am really sorry to hear that. That sounds genuinely tough. 💙",
            "That must be really hard for you right now. I hear you and I am here.",
            "I understand — it is completely okay to feel this way. You are not alone.",
            "That sounds really frustrating. Would you like to talk about it more?",
            "I am sorry you are going through this. How can I support you right now?",
            "That is a lot to carry. Please know that your feelings are completely valid.",
            "I hear the pain in your words and I want you to know I genuinely care.",
            "Sometimes life gets really heavy. I am glad you are talking about it.",
            "You do not have to pretend everything is fine. I am here to listen.",
            "That kind of situation would wear anyone down. Be gentle with yourself.",
            "I no go lie, that one heavy. But I dey here with you through am. 💙",
            "E go better — this pain you dey feel now no go last forever. Hold on."
        ],
        "escalating": [
            "I can see this has been really weighing on you. Please know that you matter deeply and things can get better. 💙",
            "It sounds like things have been really difficult lately. Have you been able to talk to someone you trust?",
            "I am genuinely concerned and I want you to know you do not have to face this alone. I am here. 🤝",
            "You have been carrying a lot. Please be gentle with yourself — reaching out like this takes real courage.",
            "I hear you and I want you to know that what you are feeling is real and valid. You deserve support.",
            "This sounds like more than you should carry alone. Please consider reaching out to someone close to you.",
            "Na you strong pass this thing wey dey worry you — but you no need face am alone abeg.",
            "I dey worried about you. You matter and your wellbeing matters. Make you talk to somebody wey you trust."
        ],
        "nigerian_context": [
            "E go better, I promise you. This kind situation no go last forever. 💪",
            "Abeg no let it weigh you down too much — you strong pass this thing.",
            "Na so life be sometimes, but you go scale through. I dey here for you.",
            "You don try well well already. Just hold on small — better days dey come.",
            "Even when e dark like this, e no mean say light no dey come. E go bright again.",
            "This one na test wey go make you stronger — you go look back and thank God.",
            "I know say e dey pain you right now, but you too tough to give up. Keep going.",
            "Naija people strong — and you be part of that strength. This one no go finish you.",
            "Oya breathe small. You dey do better than you think, even if e no feel like am.",
            "No be everything wey heavy suppose break you — some things just dey make you grow."
        ],
        "loss_grief": [
            "I am so deeply sorry for what you are going through. Grief is one of the hardest things to carry.",
            "There are no words that can take this pain away, but please know I am here with you.",
            "It is okay to grieve. Take all the time you need — there is no rush to feel better.",
            "Your pain is completely understandable. Please be very gentle with yourself right now. 💙",
            "Loss changes everything. I am truly sorry you are experiencing this.",
            "Make you no rush yourself to feel okay — grief need time, and that is completely fine."
        ]
    },

    "neutral": {
        "default": [
            "I see, thanks for sharing that with me. Feel free to tell me more.",
            "Alright, I understand. How are you feeling about everything overall?",
            "Got it. Is there anything specific on your mind you would like to talk about?",
            "Okay, I am following you. What else would you like to share?",
            "I hear you. Take your time — I am here to listen without judgment.",
            "That is interesting. What has been on your mind lately beyond that?",
            "I appreciate you sharing that with me. How has your day been overall?",
            "I am here and I am listening. Is there something deeper you would like to explore?",
            "Sometimes things just are what they are. How are you sitting with it?",
            "I dey hear you. Anything wey you wan talk about — I dey here for you."
        ],
        "checking_in": [
            "How are you really doing today? Sometimes the honest answer is different from what we say.",
            "I just want to check in — how has life been treating you lately?",
            "Beyond what is happening on the surface, how are you feeling inside?",
            "Sometimes neutral is actually okay. Are you at peace with where things are?",
            "I dey check on you — how you really dey? No need to form strong if you no dey alright."
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
    cls   = "badge-" + sentiment
    return '<span class="badge ' + cls + '">' + emoji + ' ' + sentiment.capitalize() + '</span>'


def score_item_html(label, color_class, color_hex, value):
    return (
        '<div class="score-item">'
        '  <span class="score-dot ' + color_class + '"></span>'
        '  <span style="color:#a0a0c0;font-weight:500;">' + label.capitalize() + '</span>'
        '  <div class="score-mini-bar">'
        '    <div class="score-mini-fill" style="width:' + f"{value:.1f}" + '%;background:' + color_hex + '"></div>'
        '  </div>'
        '  <span class="score-mini-val">' + f"{value:.1f}" + '%</span>'
        '</div>'
    )


def scores_bar_html(scores):
    """Render the horizontal live scores bar like Image 2"""
    items = (
        score_item_html("positive", "score-dot-positive", "#22c55e", scores["positive"] * 100) +
        score_item_html("negative", "score-dot-negative", "#ef4444", scores["negative"] * 100) +
        score_item_html("neutral", "score-dot-neutral", "#818cf8", scores["neutral"] * 100)
    )
    return '<div class="scores-bar"><span class="scores-bar-label">Live Scores</span>' + items + '</div>'


def empty_scores_bar_html():
    """Render empty scores bar before any message is sent"""
    return (
        '<div class="scores-bar">'
        '  <span class="scores-bar-label">Live Scores</span>'
        '  <div class="score-item">'
        '    <span class="score-dot score-dot-positive"></span>'
        '    <span style="color:#5a5a7a;font-size:12px;">Positive</span>'
        '    <div class="score-mini-bar"><div class="score-mini-fill" style="width:0%;background:#22c55e"></div></div>'
        '    <span class="score-mini-val">--</span>'
        '  </div>'
        '  <div class="score-item">'
        '    <span class="score-dot score-dot-negative"></span>'
        '    <span style="color:#5a5a7a;font-size:12px;">Negative</span>'
        '    <div class="score-mini-bar"><div class="score-mini-fill" style="width:0%;background:#ef4444"></div></div>'
        '    <span class="score-mini-val">--</span>'
        '  </div>'
        '  <div class="score-item">'
        '    <span class="score-dot score-dot-neutral"></span>'
        '    <span style="color:#5a5a7a;font-size:12px;">Neutral</span>'
        '    <div class="score-mini-bar"><div class="score-mini-fill" style="width:0%;background:#818cf8"></div></div>'
        '    <span class="score-mini-val">--</span>'
        '  </div>'
        '</div>'
    )


def format_time():
    """Format current time like '06:58 AM'"""
    return datetime.now().strftime("%I:%M %p").lstrip("0")


# ── Session state ─────────────────────────────────────────────────
if "messages"        not in st.session_state: st.session_state.messages        = []
if "sentiment_trail" not in st.session_state: st.session_state.sentiment_trail = []
if "last_scores"     not in st.session_state: st.session_state.last_scores     = None
if "last_sentiment"  not in st.session_state: st.session_state.last_sentiment  = None
if "total_turns"     not in st.session_state: st.session_state.total_turns     = 0
if "quick_msg"       not in st.session_state: st.session_state.quick_msg       = None
if "timestamps"      not in st.session_state: st.session_state.timestamps      = []


# ── Load model ────────────────────────────────────────────────────
with st.spinner("Loading BERT-LSTM model — please wait a moment..."):
    model, tokenizer, device = load_model()


# ══════════════════════════════════════════════════════════════════
# HEADER (like Image 2)
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="header-bar">
    <div class="header-icon">🧠</div>
    <div class="header-text">
        <div class="header-title">Sentiment-Aware Conversational System</div>
        <div class="header-subtitle">Nigerian Context NLP · BERT-LSTM Architecture</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# LIVE SCORES BAR (like Image 2)
# ══════════════════════════════════════════════════════════════════
if st.session_state.last_scores:
    st.markdown(scores_bar_html(st.session_state.last_scores), unsafe_allow_html=True)
else:
    st.markdown(empty_scores_bar_html(), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# CHAT AREA (like Image 2)
# ══════════════════════════════════════════════════════════════════
chat_area = st.container()

with chat_area:
    if not st.session_state.messages:
        # Welcome screen
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

        # Quick message chips in welcome screen
        cols = st.columns(3)
        for i, qm in enumerate(QUICK_MESSAGES[:3]):
            with cols[i]:
                if st.button(qm, key="welcome_q_" + str(i), use_container_width=True):
                    st.session_state.quick_msg = qm
                    st.rerun()

        cols2 = st.columns(3)
        for i, qm in enumerate(QUICK_MESSAGES[3:]):
            with cols2[i]:
                if st.button(qm, key="welcome_q2_" + str(i), use_container_width=True):
                    st.session_state.quick_msg = qm
                    st.rerun()
    else:
        # Chat messages - render like Image 2
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                # Get timestamp if available
                ts = ""
                if i < len(st.session_state.timestamps):
                    ts = st.session_state.timestamps[i]

                st.markdown(
                    '<div class="message-wrapper">'
                    '  <div class="message-meta message-meta-user">You · ' + ts + '</div>'
                    '  <div class="user-bubble-wrap">'
                    '    <div class="user-text">' + msg["content"] + '</div>'
                    '  </div>'
                    '  <div class="badge-wrap-user">'
                    + badge_html(msg["sentiment"], msg["confidence"]) +
                    '  </div>'
                    '  <div class="confidence-text">' + f"{msg['confidence']:.1f}" + '% confidence</div>'
                    '</div>',
                    unsafe_allow_html=True
                )
            else:
                ts = ""
                if i < len(st.session_state.timestamps):
                    ts = st.session_state.timestamps[i]

                st.markdown(
                    '<div class="message-wrapper">'
                    '  <div class="message-meta message-meta-bot">System · ' + ts + '</div>'
                    '  <div class="bot-bubble-wrap">'
                    '    <div class="bot-text">' + msg["content"] + '</div>'
                    '  </div>'
                    '</div>',
                    unsafe_allow_html=True
                )


# ══════════════════════════════════════════════════════════════════
# INPUT AREA
# ══════════════════════════════════════════════════════════════════
if "input_key" not in st.session_state:
    st.session_state.input_key = 0

st.markdown('<div class="input-area">', unsafe_allow_html=True)

# Input row
col1, col2 = st.columns([6, 1])
with col1:
    user_input = st.text_area(
        label="message",
        placeholder="Type your message in English or Nigerian Pidgin...",
        height=70,
        label_visibility="collapsed",
        key="chat_input_" + str(st.session_state.input_key)
    )
with col2:
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    send = st.button("Send 💬", use_container_width=True, type="primary")

# Reset button (small, below)
cols_reset = st.columns([6, 1])
with cols_reset[1]:
    if st.button("🔄 Reset", key="reset_btn", use_container_width=True):
        st.session_state.messages        = []
        st.session_state.sentiment_trail = []
        st.session_state.last_scores     = None
        st.session_state.last_sentiment  = None
        st.session_state.total_turns     = 0
        st.session_state.timestamps      = []
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)


# ── Handle quick message ──────────────────────────────────────────
if st.session_state.quick_msg:
    user_input = st.session_state.quick_msg
    st.session_state.quick_msg = None
    send = True

# ── Process message ───────────────────────────────────────────────
if send and user_input and user_input.strip():
    # Increment key to force input reset
    st.session_state.input_key += 1
    text = user_input.strip()
    current_time = format_time()

    sentiment, confidence, scores = predict_sentiment(text, model, tokenizer, device)
    st.session_state.sentiment_trail.append(sentiment)

    response = get_response(
        sentiment,
        len(st.session_state.messages) // 2,
        st.session_state.sentiment_trail
    )

    # Store user message with timestamp
    st.session_state.messages.append({
        "role":       "user",
        "content":    text,
        "sentiment":  sentiment,
        "confidence": confidence * 100
    })
    st.session_state.timestamps.append(current_time)

    # Store bot response with timestamp
    st.session_state.messages.append({
        "role":    "bot",
        "content": response
    })
    st.session_state.timestamps.append(format_time())

    st.session_state.last_scores    = scores
    st.session_state.last_sentiment = sentiment
    st.session_state.total_turns   += 1
    st.rerun()
