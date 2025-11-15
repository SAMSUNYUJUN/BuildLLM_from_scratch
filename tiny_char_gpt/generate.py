# generate.py
import json
import torch
from safetensors.torch import load_file

from config import load_config
from tokenizer_char import CharTokenizer
from modeling_tinygpt import TinyGPTModel
from tokenizer_BPE import bpe_tokenizer

device = "mps"

def main():
    # 1. load configs
    config = load_config("config.json")

    with open("generation_config.json", "r", encoding="utf-8") as f:
        gen_cfg = json.load(f)
    max_new_tokens = int(gen_cfg.get("max_new_tokens", 200))

    tokenizer = bpe_tokenizer.from_files("bpe_merges.json", "bpe_vocab.json", "tokenizer_config.json")

    # 2. load model weights
    model = TinyGPTModel(config)
    state_dict = load_file("model-00001-of-00001.safetensors")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 3. get prompt from user
    prompt = input("Enter prompt (empty for random start): ")

    if prompt:
        context_ids = tokenizer.encode(prompt)
        context = torch.tensor([context_ids], dtype=torch.long, device=device)
    else:
        # start from id 0 when no prompt
        context = torch.zeros((1, 1), dtype=torch.long, device=device)

    # 4. generate text
    with torch.no_grad():
        out = model.generate(context, max_new_tokens=max_new_tokens)

    text = tokenizer.decode(out[0].tolist())
    print("=== Generated text ===")
    print(text)


if __name__ == "__main__":
    main()
