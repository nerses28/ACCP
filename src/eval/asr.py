import os
import json
import xml.etree.ElementTree as ET
from jiwer import wer, cer


class ASREvaluator:
    def __init__(self, ref_dir: str, hyp_path: str):
        self.ref_dir = ref_dir
        self.hyp_path = hyp_path

    def load_reference_text(self):
        all_text = []
        for fname in sorted(os.listdir(self.ref_dir)):
            if fname.endswith(".words.xml"):
                path = os.path.join(self.ref_dir, fname)
                tree = ET.parse(path)
                root = tree.getroot()

                for word in root.findall(".//w"):
                    if word.text:
                        all_text.append(word.text.lower())

        return " ".join(all_text)

    def load_hypothesis_text(self):
        with open(self.hyp_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        texts = [seg["text"] for seg in data.get("segments", []) if seg.get("text")]
        return " ".join(texts).lower()

    def evaluate(self):
        print("[ASREval] Loading reference...")
        ref = self.load_reference_text()

        print("[ASREval] Loading hypothesis...")
        hyp = self.load_hypothesis_text()

        print("[ASREval] Calculating metrics...")
        wer_score = wer(ref, hyp)
        cer_score = cer(ref, hyp)

        print(f"WER: {wer_score:.3f}")
        print(f"CER: {cer_score:.3f}")

        return {
            "wer": wer_score,
            "cer": cer_score
        }

