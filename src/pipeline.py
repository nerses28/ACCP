import json
from src.asr import WhisperASR
from src.diarizer import SpeakerDiarizer
from src.phrase_merger import PhraseMerger
from src.topic_segmenter import TopicSegmenter
from src.summary import MeetingSummarizer
from src.llm_client import LLMClient
from src.intent_detection import IntentDetector
from src.speaker_info import SpeakerInfoExtractor

class Pipeline:
    def __init__(self, hf_token: str, openai_api_key: str, model_size="small", device="cuda", save_path="outputs"):
        self.asr = WhisperASR(model_size=model_size, device=device)
        self.diarizer = SpeakerDiarizer(hf_token=hf_token, device=device)
        self.llm = LLMClient(api_key=openai_api_key)
        self.topic_segmenter = TopicSegmenter(llm_client=self.llm)
        self.summarizer = MeetingSummarizer(llm_client=self.llm)
        self.intent_detector = IntentDetector(llm_client=self.llm)
        self.speaker_info_extractor = SpeakerInfoExtractor(llm_client=self.llm)
        self.save_path = save_path

    def run(self, audio_path: str) -> dict:
        print("[Pipeline] Step 1: Transcribing...")
        asr_result = self.asr.transcribe(audio_path)

        print("[Pipeline] Step 2: Diarizing...")
        diarization = self.diarizer.diarize(audio_path)

        print("[Pipeline] Step 3: Merging speaker labels...")
        for segment in asr_result["segments"]:
            start, end = segment["start"], segment["end"]
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if turn.start <= start < turn.end or turn.start < end <= turn.end:
                    segment["speaker"] = speaker
                    break
            else:
                segment["speaker"] = "unknown"

        with open(self.save_path + "/asr_output.json", "w", encoding="utf-8") as f:
            json.dump(asr_result, f, ensure_ascii=False, indent=2)

        with open(self.save_path + "/asr_output2.json", "w", encoding="utf-8") as f:
            json.dump(asr_result["segments"], f, ensure_ascii=False, indent=2)

        print("[Pipeline] Step 4: Merging phrases...")
        utterances = PhraseMerger.merge_segments(asr_result["segments"])
        with open(self.save_path + "/utterances.json", "w", encoding="utf-8") as f:
            json.dump(utterances, f, ensure_ascii=False, indent=2)

        print("[Pipeline] Step 5: Topic segmentation...")
        phases = self.topic_segmenter.segment(utterances)
        with open(self.save_path + "/phases.json", "w", encoding="utf-8") as f:
            json.dump(phases, f, ensure_ascii=False, indent=2)

        print("[Pipeline] Step 6: Generating summary...")
        summary = self.summarizer.summarize(utterances)
        with open(self.save_path + "/summary.txt", "w", encoding="utf-8") as f:
            json.dump(summary.strip(), f, ensure_ascii=False, indent=2)

        print("[Pipeline] Step 7: Detecting intents...")
        intents = self.intent_detector.detect(utterances)
        with open(self.save_path + "/intents.json", "w", encoding="utf-8") as f:
            json.dump(intents, f, ensure_ascii=False, indent=2)
        print("[Pipeline] Step 8: Extracting speaker information...")
        speaker_info = self.speaker_info_extractor.extract(utterances)
        with open(self.save_path + "/speaker_info.json", "w", encoding="utf-8") as f:
            json.dump(speaker_info, f, ensure_ascii=False, indent=2)

        print("[Pipeline] Step 9: Final Output Assembly...")
        final_output = self.assemble_final_output(
            summary_path=self.save_path + "/summary.txt",
            phases_path=self.save_path + "/phases.json",
            asr_path=self.save_path + "/asr_output.json",
            speaker_info_path=self.save_path + "/speaker_info.json",
            intents_path=self.save_path + "/intents.json",
            utterences_path=self.save_path + "/utterances.json"
        )

        with open(self.save_path + "/final_output.json", "w") as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        return final_output

    def assemble_final_output(
        self,
        summary_path: str,
        phases_path: str,
        asr_path: str,
        speaker_info_path: str,
        intents_path: str,
        utterences_path: str
    ) -> dict:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f) if summary_path.endswith(".json") else f.read().strip()

        with open(phases_path, "r", encoding="utf-8") as f:
            phases = json.load(f)

        with open(utterences_path, "r", encoding="utf-8") as f:
            segments = json.load(f)

        with open(asr_path, "r", encoding="utf-8") as f:
            full_text = json.load(f)["text"]

        with open(speaker_info_path, "r", encoding="utf-8") as f:
            raw_info = json.load(f)
            if isinstance(raw_info, dict) and "raw_response" in raw_info:
                raw = raw_info["raw_response"].strip("```json\n").strip("```")
                speaker_info = json.loads(raw)
            else:
                speaker_info = raw_info

        with open(intents_path, "r", encoding="utf-8") as f:
            intent_data = json.load(f)
            intent_map = {item["id"]: item["intent"] for item in intent_data}

        utterances = []
        for idx, seg in enumerate(segments):
            uid = seg.get("id", idx)
            utterances.append({
                "id": uid,
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "speaker": seg.get("speaker", "unknown"),
                "segment_ids": [uid],
                "intent": intent_map.get(uid, "unknown")
            })

        result = {
            "summary": summary,
            "conversation_phases": phases,
            "text": full_text,
            "speakers_info": speaker_info,
            "utterances": utterances
        }
        return result
