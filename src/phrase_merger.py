import re
from typing import List, Dict

class PhraseMerger:
    @staticmethod
    def should_merge(prev_seg: Dict, curr_seg: Dict) -> bool:
        if prev_seg["speaker"] == "unknown" or curr_seg["speaker"] == "unknown":
            return False
        if prev_seg["speaker"] != curr_seg["speaker"]:
            return False
        if not prev_seg["text"].strip().endswith(('.', '!', '?')):
            return True
        return False

    @staticmethod
    def clean_text(text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def merge_segments(segments: List[Dict]) -> List[Dict]:
        if not segments:
            return []

        merged = []
        first = segments[0]
        current = {
            "id": 0,
            "start": first["start"],
            "end": first["end"],
            "text": PhraseMerger.clean_text(first["text"]),
            "speaker": first.get("speaker", "unknown"),
            "segment_ids": [first.get("id", 0)]
        }
        phrase_id = 1

        for seg in segments[1:]:
            cleaned_text = PhraseMerger.clean_text(seg["text"])
            seg_with_clean_text = {**seg, "text": cleaned_text}

            if PhraseMerger.should_merge(current, seg_with_clean_text):
                current["end"] = seg["end"]
                current["text"] = PhraseMerger.clean_text(current["text"] + " " + cleaned_text)
                current["segment_ids"].append(seg.get("id", -1))
            else:
                merged.append(current)
                current = {
                    "id": phrase_id,
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": cleaned_text,
                    "speaker": seg.get("speaker", "unknown"),
                    "segment_ids": [seg.get("id", -1)]
                }
                phrase_id += 1

        merged.append(current)
        return merged
