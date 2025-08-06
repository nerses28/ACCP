import os
import re
import json
import xml.etree.ElementTree as ET
from typing import List, Dict
from sklearn.metrics import classification_report


class IntentEvaluator:
    def __init__(self, words_dir: str, dialog_act_dir: str, intents_path: str, intents_dict_path: str, hyp_utterances_path: str):
        self.word_times = self.load_word_times(words_dir)
        self.dialog_act_paths = self.collect_dialog_act_files(dialog_act_dir)
        self.intents_dict = self.load_json(intents_dict_path)
        self.intents = self.load_json(intents_path)
        self.hyp_utterances = self.load_json(hyp_utterances_path)
        self.ref_intents = self.parse_intents()

    def collect_dialog_act_files(self, dialog_act_dir: str) -> List[str]:
        return [
            os.path.join(dialog_act_dir, fname)
            for fname in os.listdir(dialog_act_dir)
            if fname.endswith(".dialog-act.xml")
        ]

    def load_json(self, path: str):
        with open(path, "r") as f:
            return json.load(f)

    def load_word_times(self, words_dir: str) -> Dict[str, Dict[str, float]]:
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

    def parse_intents(self) -> List[Dict]:
        results = []
        ns = {"nite": "http://nite.sourceforge.net/"}

        for path in self.dialog_act_paths:
            tree = ET.parse(path)
            root = tree.getroot()

            for dact in root.findall(".//dact"):
                pointer = dact.find("nite:pointer", ns)
                if pointer is None:
                    continue

                href = pointer.attrib.get("href", "")
                match = re.search(r'id\(([^)]+)\)', href)
                if not match:
                    continue

                act_id = match.group(1)
                if act_id not in self.intents_dict:
                    continue

                intent = self.intents_dict[act_id]

                child = dact.find("nite:child", ns)
                if child is None:
                    continue

                href = child.attrib.get("href", "")
                ids = re.findall(r'id\(([^)]+)\)', href)
                times = [self.word_times[i] for i in ids if i in self.word_times]

                if not times:
                    continue

                start = min(t["start"] for t in times)
                end = max(t["end"] for t in times)
                results.append({
                    "intent": intent,
                    "start": start,
                    "end": end
                })
        return results

    def match_utterances_to_labels(self) -> List[Dict]:
        labeled = []
        id_to_time = {u["id"]: (u["start"], u["end"]) for u in self.hyp_utterances}
        id_to_pred = {u["id"]: u["intent"].lower() for u in self.intents}

        for uid, (ustart, uend) in id_to_time.items():
            pred = id_to_pred.get(uid, "other").lower()
            match = None
            for r in self.ref_intents:
                overlap = max(0, min(uend, r["end"]) - max(ustart, r["start"]))
                if overlap > 0.5 * (uend - ustart):
                    match = r["intent"].lower()
                    break
            labeled.append({"id": uid, "pred": pred, "true": match or "other"})
        return labeled

    def evaluate(self):
        labeled = self.match_utterances_to_labels()
        y_true = [r["true"] for r in labeled]
        y_pred = [r["pred"] for r in labeled]
        print("[INTENT DETECTION EVAL]")
        print(classification_report(y_true, y_pred, zero_division=0))

    def debug(self):
        print(self.ref_intents)
        pass
