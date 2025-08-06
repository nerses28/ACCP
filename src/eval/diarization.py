import os
import json
import xml.etree.ElementTree as ET
from pyannote.core import Segment, Annotation
from pyannote.metrics.diarization import DiarizationErrorRate


class DiarizationEvaluator:
    def __init__(self, ref_dir: str, hyp_path: str):
        self.ref_dir = ref_dir
        self.hyp_path = hyp_path

    def load_reference(self) -> Annotation:
        annotation = Annotation()

        for fname in sorted(os.listdir(self.ref_dir)):
            if fname.endswith(".words.xml"):
                speaker = fname.split(".")[1]
                speaker_label = f"SPEAKER_{speaker}"

                path = os.path.join(self.ref_dir, fname)
                tree = ET.parse(path)
                root = tree.getroot()

                for word in root.findall(".//w"):
                    start = float(word.attrib["starttime"])
                    end = float(word.attrib["endtime"])
                    annotation[Segment(start, end)] = speaker_label

        return annotation

    def load_hypothesis(self) -> Annotation:
        with open(self.hyp_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        annotation = Annotation()
        for seg in data.get("segments", []):
            start = seg.get("start")
            end = seg.get("end")
            speaker = seg.get("speaker", "unknown")
            annotation[Segment(start, end)] = speaker

        return annotation

    def evaluate(self):
        print("[DiarizationEval] Loading reference...")
        reference = self.load_reference()

        print("[DiarizationEval] Loading hypothesis...")
        hypothesis = self.load_hypothesis()

        print("[DiarizationEval] Calculating DER...")
        metric = DiarizationErrorRate()
        der = metric(reference, hypothesis)

        print(f"DER: {der:.3f}")
        return {"der": der}

