"""Sentiment classifier on UCI Amazon reviews (TF-IDF + feed-forward net) and a chat loop."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.nn import functional as F

DEFAULT_DATA = Path("data/amazon_cells_labelled.txt")
DEFAULT_CHECKPOINT = Path("checkpoints/sentiment.pt")


def clean_text(text: pd.Series) -> pd.Series:
    text = text.str.lower()
    text = text.replace(r"[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+", "", regex=True)  # emails
    text = text.replace(r"((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}", "", regex=True)  # IP addresses
    text = text.replace(r"[^\w\s]", "", regex=True)  # punctuation
    text = text.replace(r"\d", "", regex=True)  # numbers
    return text


def load_reviews(path: Path = DEFAULT_DATA) -> pd.DataFrame:
    data = pd.read_csv(path, delimiter="\t", header=None, names=["Sentence", "Class"])
    data["Sentence"] = clean_text(data["Sentence"])
    return data


class SimpleNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, output_size: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.fc2(F.relu(self.fc1(x))), dim=1)


def _to_tensor(sparse) -> torch.Tensor:
    return torch.from_numpy(np.asarray(sparse.todense())).float()


class SentimentModel:
    """A fitted TF-IDF vectoriser and classifier, usable for prediction."""

    LABELS = {0: "negative", 1: "positive"}

    def __init__(self, vectorizer: TfidfVectorizer, net: SimpleNN):
        self.vectorizer = vectorizer
        self.net = net

    @torch.no_grad()
    def predict(self, sentences: list[str]) -> list[int]:
        cleaned = clean_text(pd.Series(sentences, dtype=str))
        self.net.eval()
        return self.net(_to_tensor(self.vectorizer.transform(cleaned))).argmax(dim=1).tolist()

    def save(self, path: Path = DEFAULT_CHECKPOINT) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "vectorizer": pickle.dumps(self.vectorizer),
                "input_size": self.net.fc1.in_features,
                "hidden_size": self.net.fc1.out_features,
                "state_dict": self.net.state_dict(),
            },
            path,
        )
        print(f"Saved model to {path}")

    @classmethod
    def load(cls, path: Path = DEFAULT_CHECKPOINT) -> "SentimentModel":
        # The checkpoint contains a pickled vectoriser: only load files you created.
        ckpt = torch.load(path, weights_only=False)
        net = SimpleNN(ckpt["input_size"], ckpt["hidden_size"])
        net.load_state_dict(ckpt["state_dict"])
        return cls(pickle.loads(ckpt["vectorizer"]), net)


def train(data_path: Path = DEFAULT_DATA, epochs: int = 100, lr: float = 1e-3, hidden_size: int = 128) -> SentimentModel:
    data = load_reviews(data_path)
    x_train, x_val, y_train, y_val = train_test_split(
        data["Sentence"].values.astype("U"),
        data["Class"].values.astype("int64"),
        test_size=0.10,
        random_state=0,
        shuffle=True,
    )

    vectorizer = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), max_features=50000, max_df=0.5, use_idf=True, norm="l2"
    )
    train_x = _to_tensor(vectorizer.fit_transform(x_train))
    train_y = torch.from_numpy(y_train)
    val_x = _to_tensor(vectorizer.transform(x_val))

    net = SimpleNN(len(vectorizer.vocabulary_), hidden_size)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.NLLLoss()

    for epoch in range(epochs):
        net.train()
        optimizer.zero_grad()
        loss = criterion(net(train_x), train_y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    net.eval()
    with torch.no_grad():
        predictions = net(val_x).argmax(dim=1).numpy()
    print("\nValidation results:")
    print(classification_report(y_val, predictions, target_names=["negative", "positive"], zero_division=0))
    return SentimentModel(vectorizer, net)


def chat(model: SentimentModel) -> None:
    print("Chatbot: Hello! How can I assist you today? (type 'exit' to quit)")
    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_input.strip().lower() == "exit":
            break
        label = SentimentModel.LABELS[model.predict([user_input])[0]]
        print(f"Chatbot: It seems like you're expressing a {label} sentiment.")
    print("Chatbot: Goodbye!")
