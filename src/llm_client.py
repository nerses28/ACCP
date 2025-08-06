import openai

class LLMClient:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def call(self, prompt: str, input_text: str) -> str:
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": input_text}
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3
        )

        return response.choices[0].message.content.strip()
