import streamlit as st
import whisper
import requests
import tempfile
from textblob import TextBlob

st.title("🏥 AI Clinical Note Generator")

st.write("Upload a doctor–patient consultation audio file")

audio_file = st.file_uploader("Upload Audio", type=["mp3", "wav", "m4a"])

if audio_file:

    st.write("Saving audio...")

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(audio_file.read())
        audio_path = tmp.name

    st.write("Loading Whisper model (tiny for speed)...")
    model = whisper.load_model("tiny")  # Tiny is faster, good for demos

    st.write("Transcribing audio...")
    result = model.transcribe(audio_path)

    # ------------------------------
    # Auto-correct transcript
    # ------------------------------
    raw_transcript = result["text"]
    transcript = str(TextBlob(raw_transcript).correct())

    # Medical corrections for common misheard words/drugs
    medical_corrections = {
        "para-systemer": "paracetamol",
        "foolish": "fever",
        "float": "fever"
    }

    for wrong, correct in medical_corrections.items():
        transcript = transcript.replace(wrong, correct)

    st.subheader("Transcript")
    st.write(transcript)

    # ------------------------------
    # Generate SOAP note using Ollama
    # ------------------------------
    st.write("Generating SOAP note using Ollama...")

    prompt = f"""
You are a medical documentation assistant.

From this doctor-patient conversation create:

1. Chief Complaint
2. Symptoms
3. Duration
4. SOAP Note (Subjective, Objective, Assessment, Plan)
5. Prescription (drug name, dosage, frequency, duration)

Conversation:
{transcript}
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "phi",
            "prompt": prompt,
            "stream": False
        }
    )

    ai_output = response.json()["response"]

    st.subheader("AI Clinical Note")
    st.write(ai_output)

    # ------------------------------
    # Generate Patient-Friendly Summary
    # ------------------------------
    st.subheader("Patient Friendly Summary")

    summary_prompt = f"""
Explain this consultation in simple language a patient can understand.

Conversation:
{transcript}

Use very simple English.
"""

    summary = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "phi",
            "prompt": summary_prompt,
            "stream": False
        }
    )

    st.write(summary.json()["response"])