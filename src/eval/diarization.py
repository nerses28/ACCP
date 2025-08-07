import os
import json
from typing import Dict
import xml.etree.ElementTree as ET
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
import numpy as np
from scipy.optimize import linear_sum_assignment


class DiarizationEvaluator:
    def __init__(self, ref_dir: str, hyp_path: str):
        self.ref_dir = ref_dir
        self.hyp_path = hyp_path

    def load_reference(self) -> Annotation:
        reference = Annotation()
        all_segments = []

        for fname in sorted(os.listdir(self.ref_dir)):
            if not fname.endswith(".words.xml"):
                continue
            path = os.path.join(self.ref_dir, fname)
            tree = ET.parse(path)
            root = tree.getroot()

            speaker = root.attrib.get("{http://nite.sourceforge.net/}id", "")
            speaker_id = speaker.split(".")[1] if "." in speaker else speaker
            speaker_id = f"SPEAKER_{speaker_id[-1]}"

            for w in root.findall(".//w"):
                start = float(w.attrib.get("starttime", -1))
                end = float(w.attrib.get("endtime", -1))
                if start >= 0 and end >= 0:
                    all_segments.append((start, end, speaker_id))

        if not all_segments:
            raise ValueError("No reference segments found.")

        # Normalize timestamps
        min_start = min(start for start, _, _ in all_segments)
        for start, end, speaker in all_segments:
            reference[Segment(start - min_start, end - min_start)] = speaker

        return reference

    def greedy_mapping(self, reference, hypothesis) -> Dict[str, str]:
        ref_speakers = sorted(set(label for _, _, label in reference.itertracks(yield_label=True)))
        hyp_speakers = sorted(
            set(label for _, _, label in hypothesis.itertracks(yield_label=True) if label != "unknown"))

        ref_map = {label: idx for idx, label in enumerate(ref_speakers)}
        hyp_map = {label: idx for idx, label in enumerate(hyp_speakers)}

        cmatrix = np.zeros((len(ref_speakers), len(hyp_speakers)))

        for ref_seg, _, ref_label in reference.itertracks(yield_label=True):
            for hyp_seg, _, hyp_label in hypothesis.itertracks(yield_label=True):
                if hyp_label == "unknown":
                    continue
                overlap = ref_seg & hyp_seg
                if overlap:
                    cmatrix[ref_map[ref_label], hyp_map[hyp_label]] += overlap.duration

        if np.all(cmatrix == 0):
            return {}  # no overlap found

        row_ind, col_ind = linear_sum_assignment(-cmatrix)  # maximize overlap
        return {hyp_speakers[j]: ref_speakers[i] for i, j in zip(row_ind, col_ind)}

    def load_hypothesis(self) -> Annotation:
        hypothesis = Annotation()
        with open(self.hyp_path, "r") as f:
            segments = json.load(f)['segments']

        for seg in segments:
            start = seg["start"]
            end = seg["end"]
            speaker = seg["speaker"]
            hypothesis[Segment(start, end)] = speaker

        return hypothesis

    def evaluate(self):
        print("[DiarizationEval] Loading reference...")
        reference = self.load_reference()

        print("[DiarizationEval] Loading hypothesis...")
        hypothesis = self.load_hypothesis()

        print("[DiarizationEval] Mapping speakers...")
        mapping = self.greedy_mapping(reference, hypothesis)
        mapped_hypothesis = hypothesis.rename_labels(mapping)

        print("[DiarizationEval] Calculating DER...")
        metric = DiarizationErrorRate()
        der = metric(reference, mapped_hypothesis)
        print(f"DER: {der:.3f}")


