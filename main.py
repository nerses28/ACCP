import os
from src.pipeline import Pipeline

if __name__ == "__main__":
    audio_path = "data/audio/ES2016a.Mix-Headset.wav"

    save_path = "outputs"
    os.makedirs(save_path, exist_ok=True)

    pipeline = Pipeline(hf_token="",
                        openai_api_key="",
                        save_path=save_path)
    _ = pipeline.run(audio_path)

