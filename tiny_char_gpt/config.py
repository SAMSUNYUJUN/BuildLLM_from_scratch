# config.py
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass
class TinyGPTConfig:
    """Configuration for the tiny character-level GPT model."""
    vocab_size: int
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 4
    block_size: int = 32
    dropout: float = 0.0
    model_type: str = "tiny_char_gpt"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # class method means it can be called on the class itself, not on an instance. 
    # so we can use TinyGPTConfig.from_dict(...) to create an instance from a dictionary.
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TinyGPTConfig":
        return cls(**data)


def save_config(config: TinyGPTConfig, path: str) -> None:
    """Save config as config.json."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, ensure_ascii=False, indent=2)


def load_config(path: str) -> TinyGPTConfig:
    """Load config.json."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return TinyGPTConfig.from_dict(data)
