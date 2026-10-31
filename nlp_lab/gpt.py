"""Character-level decoder-only transformer trained on Tiny Shakespeare.

Follows Andrej Karpathy's "Let's build GPT" walkthrough. Defaults match the
final model in the notebook: 6 layers, 6 heads, 256-dim embeddings, a
128-character context, 20% dropout, 5,000 steps at lr 3e-4.
"""

from __future__ import annotations

import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DEFAULT_TEXT_PATH = Path("data/input.txt")
DEFAULT_CHECKPOINT = Path("checkpoints/gpt.pt")


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 128
    n_embd: int = 256
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.2


@dataclass
class TrainConfig:
    batch_size: int = 64
    max_iters: int = 5000
    eval_interval: int = 200
    eval_iters: int = 200
    learning_rate: float = 3e-4


# ---------------------------------------------------------------- data


class CharTokenizer:
    def __init__(self, chars: list[str]):
        self.chars = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = dict(enumerate(chars))

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def load_text(path: Path = DEFAULT_TEXT_PATH) -> str:
    """Read Tiny Shakespeare, downloading it to `path` the first time."""
    path = Path(path)
    if not path.exists():
        print(f"Downloading Tiny Shakespeare to {path} ...")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
    return path.read_text(encoding="utf-8")


def split_data(data: torch.Tensor, train_fraction: float = 0.9) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(train_fraction * len(data))
    return data[:n], data[n:]


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: torch.device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------- model


class Head(nn.Module):
    """One head of causal self-attention."""

    def __init__(self, cfg: GPTConfig, head_size: int):
        super().__init__()
        self.key = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # Scaled by the embedding size C, as in the original notebook.
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = self.dropout(F.softmax(wei, dim=-1))
        return wei @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: GPTConfig, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(cfg, head_size) for _ in range(cfg.n_head)])
        self.proj = nn.Linear(cfg.n_head * head_size, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.ReLU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Transformer block: pre-norm attention and feed-forward with residuals."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.sa = MultiHeadAttention(cfg, cfg.n_embd // cfg.n_head)
        self.ffwd = FeedForward(cfg)
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        return x + self.ffwd(self.ln2(x))


class GPTLanguageModel(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        _, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = self.blocks(tok_emb + pos_emb)
        # Note: ln_f is defined but, as in the original notebook, not applied here,
        # so checkpoints match the trained architecture.
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block_size :])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx


# ---------------------------------------------------------------- training


@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg: GPTConfig, tcfg: TrainConfig, device) -> dict[str, float]:
    out = {}
    model.eval()
    for split, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(tcfg.eval_iters)
        for k in range(tcfg.eval_iters):
            X, Y = get_batch(data, cfg.block_size, tcfg.batch_size, device)
            losses[k] = model(X, Y)[1].item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(
    text: str,
    cfg: GPTConfig,
    tcfg: TrainConfig,
    device: torch.device,
    checkpoint: Path | None = DEFAULT_CHECKPOINT,
) -> tuple[GPTLanguageModel, CharTokenizer]:
    tokenizer = CharTokenizer.from_text(text)
    cfg.vocab_size = tokenizer.vocab_size
    print(f"Dataset: {len(text):,} characters, {tokenizer.vocab_size} unique")
    train_data, val_data = split_data(torch.tensor(tokenizer.encode(text), dtype=torch.long))

    model = GPTLanguageModel(cfg).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.learning_rate)

    for step in range(tcfg.max_iters):
        if step % tcfg.eval_interval == 0 or step == tcfg.max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, cfg, tcfg, device)
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        xb, yb = get_batch(train_data, cfg.block_size, tcfg.batch_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    if checkpoint is not None:
        save_checkpoint(model, tokenizer, checkpoint)
    return model, tokenizer


def save_checkpoint(model: GPTLanguageModel, tokenizer: CharTokenizer, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.cfg), "chars": tokenizer.chars, "state_dict": model.state_dict()}, path)
    print(f"Saved model to {path}")


def load_checkpoint(path: Path, device: torch.device) -> tuple[GPTLanguageModel, CharTokenizer]:
    ckpt = torch.load(path, map_location=device)
    model = GPTLanguageModel(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, CharTokenizer(ckpt["chars"])


def generate_text(model: GPTLanguageModel, tokenizer: CharTokenizer, n_chars: int, device, prompt: str = "") -> str:
    model.eval()
    if prompt:
        context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
    return tokenizer.decode(model.generate(context, max_new_tokens=n_chars)[0].tolist())
