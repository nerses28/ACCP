from pyannote.audio import Pipeline
import torch

class SpeakerDiarizer:
    def __init__(self, hf_token: str, device="cuda"):
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization@2.1",
            use_auth_token=hf_token
        )
        self.pipeline.to(torch.device(device))

    def diarize(self, audio_path: str):
        diarization = self.pipeline(audio_path)

        #'''
        first_segment = next(diarization.itertracks(yield_label=False))[0]
        offset = first_segment.start

        if offset > 0:
            print(f"[Diarizer] Normalizing speaker timeline by offset: {offset:.2f}s")

            from pyannote.core import Annotation, Segment
            normalized = Annotation()
            for segment, track, label in diarization.itertracks(yield_label=True):
                new_seg = Segment(start=segment.start - offset, end=segment.end - offset)
                normalized[new_seg, track] = label
            return normalized
        #'''
        return diarization

