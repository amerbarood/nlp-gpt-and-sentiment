"""Command-line entry point.

Examples:
    python main.py gpt-train                       # train the char-level GPT (downloads Tiny Shakespeare)
    python main.py gpt-train --max-iters 500 --n-layer 2 --n-embd 64   # quick, CPU-friendly run
    python main.py gpt-generate --chars 500 --prompt "ROMEO:"
    python main.py sentiment-train                 # train the classifier and save it
    python main.py chat                            # talk to the sentiment chatbot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from nlp_lab import gpt, sentiment


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Character-level GPT and sentiment chatbot")
    parser.add_argument("--seed", type=int, default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    g, t = gpt.GPTConfig(), gpt.TrainConfig()
    p = sub.add_parser("gpt-train", help="train the character-level GPT on Tiny Shakespeare")
    p.add_argument("--text", type=Path, default=gpt.DEFAULT_TEXT_PATH)
    p.add_argument("--checkpoint", type=Path, default=gpt.DEFAULT_CHECKPOINT)
    p.add_argument("--batch-size", type=int, default=t.batch_size)
    p.add_argument("--max-iters", type=int, default=t.max_iters)
    p.add_argument("--eval-interval", type=int, default=t.eval_interval)
    p.add_argument("--eval-iters", type=int, default=t.eval_iters)
    p.add_argument("--lr", type=float, default=t.learning_rate)
    p.add_argument("--block-size", type=int, default=g.block_size)
    p.add_argument("--n-embd", type=int, default=g.n_embd)
    p.add_argument("--n-head", type=int, default=g.n_head)
    p.add_argument("--n-layer", type=int, default=g.n_layer)
    p.add_argument("--dropout", type=float, default=g.dropout)
    p.add_argument("--sample-chars", type=int, default=500, help="characters to generate after training")

    p = sub.add_parser("gpt-generate", help="generate text from a trained GPT checkpoint")
    p.add_argument("--checkpoint", type=Path, default=gpt.DEFAULT_CHECKPOINT)
    p.add_argument("--chars", type=int, default=500)
    p.add_argument("--prompt", default="")

    p = sub.add_parser("sentiment-train", help="train the TF-IDF sentiment classifier")
    p.add_argument("--data", type=Path, default=sentiment.DEFAULT_DATA)
    p.add_argument("--checkpoint", type=Path, default=sentiment.DEFAULT_CHECKPOINT)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)

    p = sub.add_parser("chat", help="chat with the sentiment bot (trains first if no checkpoint)")
    p.add_argument("--checkpoint", type=Path, default=sentiment.DEFAULT_CHECKPOINT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    if args.command == "gpt-train":
        device = get_device()
        cfg = gpt.GPTConfig(
            block_size=args.block_size, n_embd=args.n_embd, n_head=args.n_head,
            n_layer=args.n_layer, dropout=args.dropout,
        )
        tcfg = gpt.TrainConfig(
            batch_size=args.batch_size, max_iters=args.max_iters, eval_interval=args.eval_interval,
            eval_iters=args.eval_iters, learning_rate=args.lr,
        )
        model, tokenizer = gpt.train(gpt.load_text(args.text), cfg, tcfg, device, args.checkpoint)
        if args.sample_chars:
            print("\n--- sample ---")
            print(gpt.generate_text(model, tokenizer, args.sample_chars, device))

    elif args.command == "gpt-generate":
        device = get_device()
        model, tokenizer = gpt.load_checkpoint(args.checkpoint, device)
        print(gpt.generate_text(model, tokenizer, args.chars, device, args.prompt))

    elif args.command == "sentiment-train":
        sentiment.train(args.data, args.epochs, args.lr).save(args.checkpoint)

    elif args.command == "chat":
        if args.checkpoint.exists():
            model = sentiment.SentimentModel.load(args.checkpoint)
        else:
            print("No saved model found, training one first...")
            model = sentiment.train()
            model.save(args.checkpoint)
        sentiment.chat(model)


if __name__ == "__main__":
    main()
