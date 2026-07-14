import torch
import random
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Model, tokenizer and device are loaded in wsgi.py before gunicorn starts
from wsgi import sentiment_model, tokenizer, device

# ── Sentiment prediction ──────────────────────────────────────────────────────
def predict_sentiment(text):
    encoding = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = sentiment_model(input_ids, attention_mask)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred   = torch.argmax(logits, dim=1).item()

    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    return {
        "sentiment":  label_map[pred],
        "confidence": float(probs[pred]),
        "scores": {
            "negative": round(float(probs[0]), 4),
            "neutral":  round(float(probs[1]), 4),
            "positive": round(float(probs[2]), 4)
        }
    }

# ── Response templates ────────────────────────────────────────────────────────
RESPONSES = {
    "positive": {
        "default": [
            "That is wonderful to hear! I am really glad things are going well for you. 😊",
            "Fantastic! Keep that positive energy going — you truly deserve it!",
            "That is great news! What else has been making your day good?",
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

# ── Dialogue state ────────────────────────────────────────────────────────────
conversation_history = []
sentiment_trail      = []

def get_response(sentiment, turn_count):
    escalating = (
        len(sentiment_trail) >= 2 and
        all(s == "negative" for s in sentiment_trail[-2:])
    )
    if sentiment == "negative" and escalating:
        pool = RESPONSES["negative"]["escalating"]
    elif sentiment == "negative":
        pool = (
            RESPONSES["negative"]["default"] +
            RESPONSES["negative"]["nigerian_context"]
        )
    elif sentiment == "positive" and turn_count > 2:
        pool = (
            RESPONSES["positive"]["follow_up"] +
            RESPONSES["positive"]["achievement"]
        )
    elif sentiment == "positive":
        pool = RESPONSES["positive"]["default"]
    elif sentiment == "neutral" and turn_count > 1:
        pool = (
            RESPONSES["neutral"]["default"] +
            RESPONSES["neutral"]["checking_in"]
        )
    else:
        pool = RESPONSES["neutral"]["default"]
    return random.choice(pool)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data       = request.json
    user_input = data.get("message", "").strip()
    if not user_input:
        return jsonify({"error": "Empty message"}), 400

    result     = predict_sentiment(user_input)
    sentiment  = result["sentiment"]
    confidence = result["confidence"]

    sentiment_trail.append(sentiment)
    response = get_response(sentiment, len(conversation_history))

    conversation_history.append({
        "user":      user_input,
        "sentiment": sentiment,
        "bot":       response
    })

    return jsonify({
        "response":   response,
        "sentiment":  sentiment,
        "confidence": round(confidence * 100, 1),
        "scores":     result["scores"],
        "turn":       len(conversation_history)
    })

@app.route("/reset", methods=["POST"])
def reset():
    conversation_history.clear()
    sentiment_trail.clear()
    return jsonify({"status": "Conversation reset successfully"})

@app.route("/history", methods=["GET"])
def history():
    return jsonify({"history": conversation_history})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "model":  "BERT-LSTM v2",
        "device": str(device)
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
