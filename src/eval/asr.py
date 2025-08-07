import os
import json
import xml.etree.ElementTree as ET
from jiwer import wer, cer


class ASREvaluator:
    def __init__(self, ref_dir: str, hyp_path: str):
        self.ref_dir = ref_dir
        self.hyp_path = hyp_path

    def load_reference_text(self):
        all_words = []

        for fname in os.listdir(self.ref_dir):
            if not fname.endswith(".words.xml"):
                continue

            path = os.path.join(self.ref_dir, fname)
            tree = ET.parse(path)
            root = tree.getroot()

            for word in root.findall(".//w"):
                text = word.text
                start = word.attrib.get("starttime")
                if text and start:
                    all_words.append((float(start), text.lower()))

        all_words.sort(key=lambda x: x[0])

        return " ".join(word for _, word in all_words)

    def load_hypothesis_text(self):
        with open(self.hyp_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data["text"].lower()

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

