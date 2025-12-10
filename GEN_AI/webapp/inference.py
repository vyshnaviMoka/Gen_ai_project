import os
import sys
import json
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'finer-main'))
from finer import FINER
from models.transformer import Transformer
from data import EXPERIMENTS_RUNS_DIR


def _latest_run_path():
    runs_dir = EXPERIMENTS_RUNS_DIR
    if not os.path.exists(runs_dir):
        return None
    candidates = []
    for name in os.listdir(runs_dir):
        path = os.path.join(runs_dir, name)
        model_dir = os.path.join(path, "model", "weights.h5")
        if os.path.isdir(path) and os.path.exists(model_dir):
            candidates.append((os.path.getmtime(model_dir), path))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


class FinerInference:
    def __init__(self, run_path=None):
        self.run_path = run_path or _latest_run_path()

        if self.run_path:
            cfg_path = os.path.join(self.run_path, "transformer.json")
        else:
            cfg_path = os.path.join(os.path.dirname(__file__), '..', 'finer-main', 'configurations', 'transformer.json')
        with open(cfg_path) as f:
            cfg = json.load(f)

        self.train_params = cfg["train_parameters"]
        self.hyper_params = cfg["hyper_parameters"]

        tag2idx, idx2tag = FINER.load_dataset_tags()
        self.idx2tag = idx2tag
        self.n_classes = len(tag2idx)

        self.model = Transformer(
            model_name=self.train_params["model_name"],
            n_classes=self.n_classes,
            dropout_rate=self.hyper_params.get("dropout_rate", 0.1),
            crf=self.hyper_params.get("crf", False),
            tokenizer=None,
            subword_pooling=self.train_params.get("subword_pooling", "all")
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.train_params["model_name"],
            use_fast=self.train_params.get("use_fast_tokenizer", True)
        )

        batch_size, sequence_length = 1, 32
        dummy = np.ones((batch_size, sequence_length), dtype=np.int32)
        self.model.predict(dummy)

        if self.run_path:
            weights_path = os.path.join(self.run_path, "model", "weights.h5")
            self.model.load_weights(weights_path)
        else:
            url = os.getenv("FINER_WEIGHTS_URL")
            if not url:
                raise RuntimeError("Set FINER_WEIGHTS_URL to a publicly accessible weights.h5 URL")
            import urllib.request, tempfile
            tmp_path = os.path.join(tempfile.gettempdir(), "finer_weights.h5")
            if not os.path.exists(tmp_path):
                urllib.request.urlretrieve(url, tmp_path)
            self.model.load_weights(tmp_path)

    def predict(self, text):
        enc = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.train_params["max_length"],
            truncation=True,
            padding="max_length",
            return_tensors="np"
        )

        x = enc["input_ids"]
        y_prob = self.model.predict(x)

        if self.model.crf:
            y_pred = y_prob.astype("int32")[0]
        else:
            y_pred = np.argmax(y_prob, axis=-1)[0]

        ids = x[0]
        tokens = self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=False)

        results = []
        for tok, lab in zip(tokens, y_pred):
            if tok in ["[PAD]", "[CLS]", "[SEP]"]:
                continue
            results.append({"token": tok, "label": self.idx2tag[int(lab)]})
        return results
