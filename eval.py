from src.eval.asr import ASREvaluator
from src.eval.diarization import DiarizationEvaluator
from src.eval.topic_segmentation import TopicSegmentationEvaluator
from src.eval.summary import SummaryEvaluator

if __name__ == "__main__":
    ref_dir = "data/annotations/"
    hyp_path = "outputs/asr_output.json"

    # ASR
    asr_eval = ASREvaluator(ref_dir, hyp_path)
    asr_eval.evaluate()

    # DIARIZATION
    diar_eval = DiarizationEvaluator(ref_dir, hyp_path)
    diar_eval.evaluate()

    # TOPIC SEGMENTATION
    ref_topic_path = "data/topics/ES2016a.topic.xml"
    topic_map_path= "data/topics/topic_map.json"
    utterances_path = "outputs/utterances.json"
    hyp_phases_path = "outputs/phases.json"
    words_path = "data/words"

    topic_eval = TopicSegmentationEvaluator(
        ref_topic_path=ref_topic_path,
        topic_map_path=topic_map_path,
        hyp_phrases_path=utterances_path,
        hyp_phases_path=hyp_phases_path,
        words_dir=words_path
    )
    topic_eval.evaluate()

    # SUMMARY
    gt_summary_path = "data/sum/ES2016asumm.txt"
    summary_path = "outputs/summary.txt"

    sum_eval = SummaryEvaluator(pred_path=summary_path, ref_path=gt_summary_path)
    sum_eval.evaluate()

    from src.eval.intent import IntentEvaluator

    # INTENT DETECTION
    ref_da_files = "data/dialogue_acts"
    intents_dict_path = "data/intents/intents_dict.json"
    intents_path = "outputs/intents.json"

    intent_eval = IntentEvaluator(
        words_dir=words_path,
        dialog_act_dir=ref_da_files,
        intents_path=intents_path,
        intents_dict_path=intents_dict_path,
        hyp_utterances_path=utterances_path,
    )
    intent_eval.evaluate()

