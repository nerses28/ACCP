import json
import re
from src.llm_client import LLMClient


class SpeakerInfoExtractor:
    def __init__(self, llm_client: LLMClient, prompt_path: str = "src/prompts/speaker_info.txt"):
        self.llm = llm_client
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt = f.read()

    def extract(self, utterances: list[dict]) -> dict:
        input_text = "\n".join(f"{utt['speaker']}: {utt['text'].strip()}" for utt in utterances)
        response = self.llm.call(self.prompt, input_text)

        cleaned = re.search(r"{.*}", response, re.DOTALL)
        if cleaned:
            try:
                return json.loads(cleaned.group())
            except Exception:
                print("[WARNING] Failed to parse cleaned JSON block.")
                return {"raw_response": response}

        print("[WARNING] LLM output is not valid JSON or missing JSON block.")
        return {"raw_response": response}
