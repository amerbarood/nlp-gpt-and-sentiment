"""Fast checks; no downloads needed."""

import torch

from nlp_lab import gpt, sentiment

TEXT = "First Citizen:\nBefore we proceed any further, hear me speak.\n" * 20


def test_tokenizer_roundtrip():
    tok = gpt.CharTokenizer.from_text(TEXT)
    assert tok.decode(tok.encode("hear me")) == "hear me"


def test_gpt_train_and_generate(tmp_path):
    cfg = gpt.GPTConfig(block_size=16, n_embd=32, n_head=2, n_layer=1, dropout=0.0)
    tcfg = gpt.TrainConfig(batch_size=4, max_iters=3, eval_interval=2, eval_iters=2)
    ckpt = tmp_path / "gpt.pt"
    model, tok = gpt.train(TEXT, cfg, tcfg, torch.device("cpu"), ckpt)
    loaded, loaded_tok = gpt.load_checkpoint(ckpt, torch.device("cpu"))
    text = gpt.generate_text(loaded, loaded_tok, 20, torch.device("cpu"), prompt="First")
    assert text.startswith("First") and len(text) == 25


def test_sentiment_train_save_load(tmp_path):
    model = sentiment.train(epochs=5)
    path = tmp_path / "sentiment.pt"
    model.save(path)
    loaded = sentiment.SentimentModel.load(path)
    assert loaded.predict(["great phone", "this does not work"]) == model.predict(["great phone", "this does not work"])
