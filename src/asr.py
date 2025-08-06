import whisper

class WhisperASR:
    def __init__(self, model_size="small", device="cuda"):
        self.model = whisper.load_model(model_size, device)

    def transcribe(self, audio_path: str) -> dict:
        result = self.model.transcribe(audio_path)
        return result 
