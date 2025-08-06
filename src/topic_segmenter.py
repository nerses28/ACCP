from typing import List, Dict
from src.llm_client import LLMClient

class TopicSegmenter:
    def __init__(self, llm_client: LLMClient, prompt_path: str = "src/prompts/phases_segmentation.txt"):
        self.llm = llm_client
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt = f.read()

    def prepare_input(self, phrases: List[Dict]) -> str:
        lines = []
        for phrase in phrases:
            pid = phrase.get("id")
            speaker = phrase.get("speaker", "unknown")
            text = phrase.get("text", "").strip()
            lines.append(f"[{pid}] {speaker}: {text}")
        return "\n".join(lines)

    def segment(self, phrases: List[Dict]) -> List[Dict]:
        input_text = self.prepare_input(phrases)
        raw_output = self.llm.call(self.prompt, input_text)

        segments = []
        for line in raw_output.strip().splitlines():
            if "|" not in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) != 3:
                continue
            topic, start_id, end_id = parts
            try:
                segments.append({
                    "topic": topic,
                    "start_id": int(start_id),
                    "end_id": int(end_id)
                })
            except ValueError:
                continue

        return segments
