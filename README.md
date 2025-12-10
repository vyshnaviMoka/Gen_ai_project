Financial News NER — README

Project: Financial News Named Entity Recognition (FiNER)
Goal: Extract structured financial entities (companies, money, locations, people, events, misc) from unstructured business news using Transformer-based NER and serve results via an API + demo UI.

Table of contents

Overview

Features

Repository structure

Dataset

Preprocessing

Model & Training

Evaluation

Deployment

API usage examples

Demo frontend

Quick start (Colab)

Troubleshooting & notes

Next steps & extensions

References & license

Overview

This repo provides an end-to-end pipeline to build a Financial NER system:

Reconstruct and clean the FiNER-Ord dataset to sentence-level BIO labels.

Fine-tune a Transformer (BERT/FinBERT) for token classification.

Evaluate with seqeval metrics.

Serve the model using FastAPI (/extract endpoint).

Demo with a minimal HTML + JS frontend.

Designed for reproducibility: scripts for preprocessing, training, evaluation, serving, and a demo UI are included.

Features

Robust dataset reconstruction from token-level inputs → sentence-level BIO labeling.

Tokenization + label alignment for subword tokenizers.

Compatibility wrapper for different transformers versions (Trainer creation).

Option to use pretrained yiyanghkust/finbert-ner or train your own bert-base-uncased/FinBERT.

FastAPI server returning token-level labels and aggregated entity spans.

Demo frontend to visualize extractions.

Exportable trained checkpoint (/content/finbert-finer-ner).

Repository structure
.
├── preprocess.py        # Reconstruct & map FiNER dataset -> sentence-level BIO
├── train.py             # Tokenize, align, train and save model
├── evaluate.py          # Run inference on test split, dump sample predictions
├── app.py               # FastAPI server for inference
├── demo.html            # Minimal frontend to call the API
├── requirements.txt
├── README.md
└── utils/               # helper scripts (optional)

Dataset

FiNER-Ord (gtfintechlab/finer-ord) — token-level annotated financial news.
Key fields: gold_token, gold_label, doc_idx, sent_idx.
We reconstruct sentences using doc_idx + sent_idx and map dataset label encoding (e.g., L-0..L-6) to BIO tags:

L-0 -> O
L-1 -> B-PER
L-2 -> B-ORG
L-3 -> B-LOC
L-4 -> B-MISC
L-5 -> B-MONEY
L-6 -> B-EVENT


(If the dataset stores numeric ClassLabel, preprocess.py reads .features to convert ints → label names.)

Preprocessing

preprocess.py:

Loads gtfintechlab/finer-ord.

Skips token rows with gold_token is None safely.

Groups tokens by (doc_idx, sent_idx) to reconstruct sentences.

Maps L-# → BIO labels.

Saves sentence-level dataset to disk (/content/fin_data).

Run:

python preprocess.py


Output: /content/fin_data (HuggingFace DatasetDict on disk).

Model & Training

Two options:

Train from scratch / fine-tune: bert-base-uncased (or FinBERT encoder) + token classification head.

Use pretrained: yiyanghkust/finbert-ner — ready for inference.

train.py:

Loads dataset from /content/fin_data.

Builds label_list, id2label, label2id.

Tokenizes with tokenizer(..., is_split_into_words=True) and aligns labels to subword tokens (-100 for special tokens).

Creates Trainer with DataCollatorForTokenClassification.

Uses evaluate + seqeval for token/entity metrics.

Saves model to /content/finbert-finer-ner.

Run:

python train.py
# After success:
# trainer.save_model("/content/finbert-finer-ner")
# tokenizer.save_pretrained("/content/finbert-finer-ner")


Hyperparameters (default in script):

LR = 5e-5, batch size = 8, epochs = 3 (adjust for GPU/limits), weight decay = 0.01.

Compatibility note: train.py contains a robust TrainingArguments wrapper that sets attributes even if your transformers version lacks newer kwargs.

Evaluation

evaluate.py:

Loads saved model and tokenizer.

Runs HuggingFace pipeline("token-classification", aggregation_strategy="simple") on test sentences.

Saves sample predictions for qualitative analysis: predictions_sample.json.

Use seqeval for per-entity precision/recall/F1.

Run:

python evaluate.py

Deployment (FastAPI)

app.py:

Loads a token-classification pipeline (local trained model or HF model id).

/extract (POST): accepts JSON {"text": "..."}

Returns tokens (token-level labels + confidence) and entities (aggregated spans: text, label, score, offsets).

/health (GET): returns model status.

Run server locally:

uvicorn app:app --host 0.0.0.0 --port 8000


In Colab: use cloudflared or ngrok to expose port 8000 to the web.

API usage examples

Example curl:

curl -X POST "http://localhost:8000/extract" -H "Content-Type: application/json" \
  -d '{"text":"Acme Corp announced a $120 million acquisition of BetaTech."}'


Example response:

{
  "tokens": [
    {"token":"Acme", "label":"B-ORG", "confidence":0.99, "start":0, "end":4}, ...
  ],
  "entities":[
    {"text":"Acme Corp","label":"ORG","score":0.98,"start":0,"end":9},
    {"text":"$120 million","label":"MONEY","score":0.97,"start":22,"end":34}
  ]
}

Demo frontend

demo.html is a minimal single-file app:

Textarea for input, button to call /extract.

Renders token table and collapsed entity chips.

To use, change API_URL to http(s)://<your-host>/extract (Cloudflared URL or local host).

Quick start (Colab) — copy/paste sequence

Install deps:

!pip install -q transformers datasets evaluate seqeval accelerate fastapi uvicorn torch


Create files (use %%writefile in Colab) — preprocess.py, train.py, app.py, demo.html.

Preprocess:

!python preprocess.py


(Option A) Use pretrained model and start server:

# edit app.py MODEL_DIR to "yiyanghkust/finbert-ner", then:
!uvicorn app:app --host 0.0.0.0 --port 8000
# expose via cloudflared/ ngrok


(Option B) Train your own model (GPU recommended):

!python train.py
# then:
!uvicorn app:app --host 0.0.0.0 --port 8000


Expose port via Cloudflared and open demo.html (update API_URL).

Troubleshooting & notes

No model files after training? Ensure trainer.train() completed and then call trainer.save_model(...) & tokenizer.save_pretrained(...).

TrainingArguments TypeError: script includes a compatibility wrapper to set attributes if your transformers version is old. Upgrading transformers and restarting runtime is recommended:

!pip install -q -U "transformers>=4.35.0" "datasets>=2.16.0" "evaluate>=0.4.0"
# then Runtime -> Restart runtime (Colab)


Labels all O: inspect raw dataset label representation; use the diagnostic snippet in preprocess.py or the notebook to detect ClassLabel names and map ints → labels properly.

Memory / OOM: reduce per_device_train_batch_size, use gradient accumulation, or use smaller model.

Inference speed: use GPU and pipeline(..., device=0). For high throughput add batching.

Next steps & extensions

Entity linking: resolve companies → tickers (fuzzy match + KB).

Event extraction templates: structured M&A/funding extraction.

Temporal normalization: convert relative dates → absolute.

Coreference: link pronouns to entities for richer events.

Dashboard: visual analytics & timeline of extracted events.

References & license

Dataset: gtfintechlab/finer-ord (HuggingFace)

Pretrained model: yiyanghkust/finbert-ner (HuggingFace)

Libraries: HuggingFace Transformers, Datasets, FastAPI, PyTorch

License: MIT (change as needed)
