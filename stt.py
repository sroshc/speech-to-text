import sounddevice as sd
from whisper import transcribe

def is_speech(audio_buffer):

  rms = np.sqrt(np.mean(audio_buffer**2))
  threshold = 0.01
  return rms > threshold

def callback(indata, frames, time, status):

  audio_buffer = indata.copy().astype(np.float32) 


  if is_speech(audio_buffer):
    result = transcribe(audio_buffer, task="transcribe", device="cuda" if torch.cuda.is_available() else "cpu", vad=True)
    print(result["text"])

try:
  # Infinite loop for continuous recording
  with sd.InputStream(callback=callback, samplerate=16000, channels=1):
    print("Speak!")
    sd.blockwait()  # Block the main thread until the recording is stopped
except KeyboardInterrupt:  # Handle Ctrl+C to stop recording gracefully
  print("Recording stopped.")
