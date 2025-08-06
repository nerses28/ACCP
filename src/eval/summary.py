import evaluate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score

class SummaryEvaluator:
    def __init__(self, ref_path: str, pred_path: str):
        with open(ref_path, "r") as f:
            self.reference = f.read().strip()
        with open(pred_path, "r") as f:
            self.prediction = f.read().strip()

    def compute_bleu(self, reference: str, prediction: str) -> float:
        smoothie = SmoothingFunction().method4
        return sentence_bleu(
            [reference.split()],
            prediction.split(),
            smoothing_function=smoothie
        )

    def evaluate(self):
        ref = self.reference
        pred = self.prediction

        # ROUGE
        rouge = evaluate.load("rouge")
        rouge_scores = rouge.compute(predictions=[pred], references=[ref])

        # BERTScore
        P, R, F1 = bert_score([pred], [ref], lang="en", verbose=False)

        # BLEU
        bleu_score = self.compute_bleu(ref, pred)

        print("\n[SUMMARY EVALUATION]")
        print(f"ROUGE-1: {rouge_scores['rouge1']:.3f}")
        print(f"ROUGE-2: {rouge_scores['rouge2']:.3f}")
        print(f"ROUGE-L: {rouge_scores['rougeL']:.3f}")
        print(f"BERTScore F1: {F1[0].item():.3f}")
        print(f"BLEU: {bleu_score:.3f}")
