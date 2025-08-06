from src.llm_client import LLMClient


class IntentDetector:
    def __init__(self, llm_client: LLMClient, prompt_path: str = "src/prompts/intent_detection.txt"):
        self.llm = llm_client
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

    def detect(self, utterances: list[dict]) -> list[dict]:
        results = []
        for i, utt in enumerate(utterances):
            context_utts = utterances[max(0, i - 5): i]
            context = "\n".join(f"{u['speaker']}: {u['text'].strip()}" for u in context_utts)
            target = f"{utt['speaker']}: {utt['text'].strip()}"

            prompt = self.prompt_template.replace("{context}", context).replace("{target}", target)
            intent = self.llm.call(prompt, "")
            results.append({
                "id": utt["id"],
                "speaker": utt["speaker"],
                "text": utt["text"],
                "intent": intent.strip()
            })
            print(i, end="\r")
        print()
        return results

