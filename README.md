# ACCP: Automatic Conversational Content Processing

This repository contains a modular pipeline for analyzing multi-speaker meeting recordings. It performs several tasks including:

- **Speech-to-text transcription**
- **Speaker diarization**
- **Topic segmentation**
- **Meeting summarization**
- **Intent detection**
- **Speaker information extraction**

## Structure

```
ACCP/
├── data/                     # Input files (AMI corpus XMLs, etc.)
├── outputs/                  # Inference outputs (summary, intents, etc.)
├── src/
│   ├── eval/                 # Evaluation scripts for each task
│   ├── prompts/              # Prompt templates for LLM tasks
│   ├── tasks/                # Individual task modules
│   └── pipeline.py           # Orchestration logic
│   └── 
├── requirements.txt
├── main.py                   # Entry point
├── eval.py                   # Run evaluation
└── README.md
```

## How to Run

Make sure you have Python 3.10+ and virtual environment activated.

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables** for OpenAI or other LLM providers.

3. **Run the pipeline**:
   ```bash
   python main.py
   ```
3. **Run the evaluation**:
   ```bash
   python eval.py
   ```
## Tasks

Evaluation metrics include ROUGE, BLEU, BERTScore for summarization, F1 and classification reports for intent detection, and Pk / WindowDiff for topic segmentation.

## Notes

- Built and tested using the AMI Meeting Corpus.
- LLM prompts are customizable in the `src/prompts/` directory.
