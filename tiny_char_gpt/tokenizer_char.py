# tokenizer_char.py
import json
from typing import Dict, List, Tuple


class CharTokenizer:
    """Very simple character-level tokenizer."""

    def __init__(self, stoi: Dict[str, int], itos: Dict[int, str], model_max_length: int):
        self.stoi = stoi
        self.itos = itos
        self.model_max_length = model_max_length

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def save_vocab(self, vocab_path: str) -> None:
        """Save stoi mapping as vocab.json."""
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.stoi, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_vocab(cls, vocab_path: str, model_max_length: int) -> "CharTokenizer":
        """Load tokenizer from vocab.json."""
        with open(vocab_path, "r", encoding="utf-8") as f:
            stoi = json.load(f)
        # stoi: char -> id
        itos = {idx: ch for ch, idx in stoi.items()}
        return cls(stoi=stoi, itos=itos, model_max_length=model_max_length)


def train_tokenizer_from_file(corpus_path: str, model_max_length: int) -> Tuple[CharTokenizer, str]:
    """
    Build a char-level tokenizer from a text file.
    Returns (tokenizer, full_text).
    """
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    tokenizer = CharTokenizer(stoi=stoi, itos=itos, model_max_length=model_max_length)
    return tokenizer, text


def save_tokenizer_config(path: str, model_max_length: int) -> None:
    """Save a minimal tokenizer_config.json."""
    cfg = {
        "tokenizer_class": "CharTokenizer",
        "model_max_length": model_max_length
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
