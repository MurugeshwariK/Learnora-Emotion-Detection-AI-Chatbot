from flask import Flask, render_template, request, jsonify
from deepface import DeepFace
from gtts import gTTS
import os
import uuid
import requests

app = Flask(__name__)

# --- Configs ---
CONFUSION_EMOTIONS = ['neutral','sad', 'fear']
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3"

# Ensure audio folder exists
os.makedirs("static/audio", exist_ok=True)

# --- Home Route ---
@app.route('/')
def home():
    return render_template("index.html")

# --- Emotion Detection Page ---
@app.route('/detect')
def detect():
    return render_template("detect.html")

# --- Chatbot Page ---
@app.route('/chat', methods=['GET', 'POST'])
def chat_page():
    if request.method == 'POST':
        question = request.form.get('question', '')
        try:
            response = requests.post(OLLAMA_API_URL, json={
                "model": MODEL_NAME,
                "prompt": question,
                "stream": False
            })
            response.raise_for_status()
            reply = response.json().get("response", "No response.")
        except Exception as e:
            reply = f"Error: {str(e)}"
        return render_template("chatbot.html", question=question, answer=reply)
    return render_template("chatbot.html")


# --- API: Emotion Detection ---
@app.route('/api/emotion', methods=['POST'])
def detect_emotion():
    try:
        file = request.files['frame']
        img_path = f"temp_{uuid.uuid4()}.jpg"
        file.save(img_path)

        result = DeepFace.analyze(img_path, actions=['emotion'], enforce_detection=False)
        os.remove(img_path)

        dominant_emotion = result[0]['dominant_emotion']
        is_confused = dominant_emotion.lower() in CONFUSION_EMOTIONS

        return jsonify({
            "emotion": dominant_emotion,
            "confused": is_confused
        })
    except Exception as e:
        return jsonify({"error": str(e)})

# --- API: Chatbot Answer ---
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        qn = request.json.get("question", "")
        response = requests.post(OLLAMA_API_URL, json={
            "model": MODEL_NAME,
            "prompt": qn,
            "stream": False
        })
        response.raise_for_status()
        reply = response.json().get("response", "No response.")
        return jsonify({"response": reply})
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"})

# --- API: TTS ---
@app.route('/api/speak', methods=['POST'])
def speak():
    try:
        text = request.json.get("text", "")
        filename = f"static/audio/{uuid.uuid4()}.mp3"
        tts = gTTS(text=text, lang='en')
        tts.save(filename)
        return jsonify({"audio_url": "/" + filename})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
