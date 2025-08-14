import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

# Carga el modelo pequeño en CPU (más rápido para probar)
model = WhisperModel("small", device="cpu", compute_type="int8")

SR = 16000      # Frecuencia de muestreo
DURATION = 6    # Duración de la grabación en segundos

def listen_once():
    print("🎤 Habla ahora...")
    audio = sd.rec(int(DURATION * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    print("⏳ Transcribiendo...")
    audio = audio[:, 0].astype(np.float32)
    segments, info = model.transcribe(audio, language="es")
    text = "".join(s.text for s in segments).strip()
    print(f"📝 Texto detectado: {text}")
    return text

if __name__ == "__main__":
    listen_once()
