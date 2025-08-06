from src.llm_client import LLMClient


class MeetingSummarizer:
    def __init__(self, llm_client: LLMClient, prompt_path: str = "src/prompts/meeting_summary.txt"):
        self.llm = llm_client
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt = f.read()


    def summarize(self, utterances: list[dict]) -> str:
        input_text = "\n".join(f"{utt['speaker']}: {utt['text'].strip()}" for utt in utterances)
        return self.llm.call(self.prompt, input_text)
