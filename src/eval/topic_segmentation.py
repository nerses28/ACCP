import xml.etree.ElementTree as ET
import json
from typing import List, Dict
from sklearn.metrics import f1_score, precision_score, recall_score
from nltk.metrics.distance import edit_distance
import numpy as np
import os
import re


def load_word_times(words_dir: str) -> Dict[str, Dict[str, float]]:
    word_times = {}
    for fname in os.listdir(words_dir):
        if not fname.endswith(".words.xml"):
            continue
        tree = ET.parse(os.path.join(words_dir, fname))
        root = tree.getroot()
        for w in root.findall(".//w"):
            wid = w.attrib.get("{http://nite.sourceforge.net/}id")
            start = float(w.attrib.get("starttime", -1))
            end = float(w.attrib.get("endtime", -1))
            if wid and start >= 0 and end >= 0:
                word_times[wid] = {"start": start, "end": end}
    return word_times


def parse_topic_segments(xml_path: str, word_times: Dict[str, Dict[str, float]]) -> List[Dict]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {"nite": "http://nite.sourceforge.net/"}
    segments = []

    i = 0
    d = 0
    for topic in root.findall(".//topic"):
        pointer = topic.find("nite:pointer", ns)
        topic_ref = "other"
        if pointer is not None:
            href = pointer.attrib.get("href", "")
            match = re.search(r'id\(([^)]+)\)', href)
            if match:
                topic_ref = match.group(1)
        all_ids = []
        for child in topic.findall("nite:child", ns):
            href = child.attrib.get("href", "")
            ids = re.findall(r'id\(([^)]+)\)', href)
            all_ids.extend(ids)

        times = [word_times[i] for i in all_ids if i in word_times]
        if not times:
            continue
        start = min(t["start"] for t in times)
        end = max(t["end"] for t in times)
        i += 1
        segments.append({
            "start": start,
            "end": end,
            "topic_ref": topic_ref
        })
    return segments



def get_boundaries_binary(timeline: List[float], segments: List[Dict]) -> List[int]:
    boundaries = [0] * len(timeline)
    for seg in segments:
        end_time = seg["end"]
        for i, t in enumerate(timeline):
            if abs(t - end_time) < 0.25:
                boundaries[i] = 1
                break
    return boundaries


def pk(true: List[int], pred: List[int], k: int) -> float:
    errors = 0
    total = len(true) - k
    for i in range(total):
        ref_same = sum(true[i:i + k]) == 0
        hyp_same = sum(pred[i:i + k]) == 0
        if ref_same != hyp_same:
            errors += 1
    return errors / total


def window_diff(true: List[int], pred: List[int], k: int) -> float:
    errors = 0
    total = len(true) - k
    for i in range(total):
        if abs(sum(true[i:i + k]) - sum(pred[i:i + k])) > 0:
            errors += 1
    return errors / total


class TopicSegmentationEvaluator:
    def __init__(self, ref_topic_path: str, topic_map_path: str, hyp_phrases_path: str, hyp_phases_path: str, words_dir: str):
        word_times = load_word_times(words_dir)
        self.ref_segments = parse_topic_segments(ref_topic_path, word_times)
        with open(topic_map_path, "r") as f:
            self.topic_map = json.load(f)
        with open(hyp_phrases_path, "r") as f:
            self.hyp_phrases = json.load(f)
        with open(hyp_phases_path, "r") as f:
            self.hyp_phases = json.load(f)

    def match_pred_to_ref(self):
        matched = []
        for pred in self.hyp_phases:
            start_time = self.hyp_phrases[pred["start_id"]]["start"]
            end_time = self.hyp_phrases[pred["end_id"]]["end"]
            best_iou = 0
            matched_label = "other"
            for ref in self.ref_segments:
                intersection = max(0, min(end_time, ref["end"]) - max(start_time, ref["start"]))
                union = max(end_time, ref["end"]) - min(start_time, ref["start"])
                score = intersection / union if union > 0 else 0
                if score > best_iou:
                    best_iou = score
                    matched_label = self.topic_map.get(ref["topic_ref"], "other")
            matched.append({
                "pred": pred["topic"],
                "true": matched_label,
                "levenshtein": edit_distance(pred["topic"], matched_label)
            })
        return matched

    def evaluate(self):
        matched = self.match_pred_to_ref()
        pred_labels = [m["pred"] for m in matched]
        true_labels = [m["true"] for m in matched]
        exact_matches = sum(1 for p, t in zip(pred_labels, true_labels) if p == t)
        acc = exact_matches / len(matched)
        avg_lev = np.mean([m["levenshtein"] for m in matched])

        # Time boundaries setup
        max_time = max(p["end"] for p in self.hyp_phrases)
        timeline = np.arange(0, max_time, 0.5).tolist()

        ref_bounds = get_boundaries_binary(timeline, self.ref_segments)
        hyp_bounds = get_boundaries_binary(timeline, [
            {
                "start": self.hyp_phrases[p["start_id"]]["start"],
                "end": self.hyp_phrases[p["end_id"]]["end"]
            } for p in self.hyp_phases
        ])

        k = max(1, int(np.mean([seg["end"] - seg["start"] for seg in self.ref_segments]) / 0.5))
        pk_score = pk(ref_bounds, hyp_bounds, k)
        wd_score = window_diff(ref_bounds, hyp_bounds, k)
        f1 = f1_score(ref_bounds, hyp_bounds)
        precision = precision_score(ref_bounds, hyp_bounds)
        recall = recall_score(ref_bounds, hyp_bounds)
        coverage = sum([any(hyp_bounds[i:i + k]) for i in range(0, len(ref_bounds) - k)]) / (len(ref_bounds) - k)

        print("[TOPIC SEGMENTATION EVAL]")
        print(f"Label Accuracy: {acc:.3f}")
        print(f"Average Levenshtein Distance: {avg_lev:.2f}")
        print("--- Temporal Segmentation ---")
        print(f"P_k: {pk_score:.3f}")
        print(f"WindowDiff: {wd_score:.3f}")
        print(f"F1-score: {f1:.3f} (P: {precision:.3f}, R: {recall:.3f})")
        print(f"Coverage: {coverage:.3f}")

