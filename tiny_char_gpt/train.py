# train.py
import json
import torch
from safetensors.torch import save_file

from config import TinyGPTConfig, save_config
from modeling_tinygpt import TinyGPTModel
from tokenizer_BPE import bpe_tokenizer

# ----------------- training hyperparameters -----------------
batch_size = 16
block_size = 32
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
eval_iters = 200
device = "mps"
# ------------------------------------------------------------

torch.manual_seed(1337)

tokenizer = bpe_tokenizer.from_files("bpe_merges.json", "bpe_vocab.json", "tokenizer_config.json")
with open("tiny_char_gpt/input.txt", "r", encoding="utf-8") as f:
    text = f.read()
# tokenizer, text = train_tokenizer_from_file("../input.txt", model_max_length=block_size)

# 2. dataset split.

# Note that for self-regressive language modeling, we don't need special tokens.
data = torch.tensor(tokenizer.encode(text, add_special_tokens=False), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split: str):
    """Sample a batch of data."""
    source = train_data if split == "train" else val_data
    ix = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i: i + block_size] for i in ix])
    y = torch.stack([source[i + 1: i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model: TinyGPTModel):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# 3. build config + model
config = TinyGPTConfig(
    vocab_size=tokenizer.vocab_size,
    n_embd=64,
    n_head=4,
    n_layer=4,
    block_size=block_size,
    dropout=0.0,
)
model = TinyGPTModel(config).to(device)

print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 4. training loop
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        print(
            f"step {iter}: train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# 5. save model weights as safetensors
state_dict = model.state_dict()
save_file(state_dict, "model-00001-of-00001.safetensors")

# build a minimal index file
total_size = 0
for tensor in state_dict.values():
    total_size += tensor.numel() * tensor.element_size()

index = {
    "metadata": {"total_size": total_size},
    "weight_map": {
        name: "model-00001-of-00001.safetensors" for name in state_dict.keys()
    },
}
with open("model.safetensors.index.json", "w", encoding="utf-8") as f:
    json.dump(index, f, ensure_ascii=False, indent=2)

# 6. save config.json
save_config(config, "config.json")

# 7. save generation_config.json
generation_config = {
    "max_new_tokens": 200,
    "do_sample": True,
    "temperature": 1.0,
    "top_k": 0,
    "top_p": 1.0,
}
with open("generation_config.json", "w", encoding="utf-8") as f:
    json.dump(generation_config, f, ensure_ascii=False, indent=2)

print("Training finished, model and configs saved.")
