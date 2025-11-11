# Tiny Char-GPT

This is a tiny character-level GPT model trained on `input.txt`
(typically the Tiny Shakespeare corpus).  
Code is split into the following files:

- `config.py` — model configuration (embedding size, layers, etc.)
- `tokenizer_char.py` — simple character-level tokenizer
- `modeling_tinygpt.py` — model definition (Transformer blocks + GPT head)
- `train.py` — training script
- `generate.py` — text generation script
- `requirements.txt` — Python dependencies

You need an `input.txt` file in this folder as training corpus.

---

## 1. Install uv

If you don't have **uv** yet, install it first (pick one way):

```bash
# Using pip (works everywhere)
pip install uv

# or, if you use pipx
pipx install uv

# Check it works:
uv --version

```
## 2. Create virtual environment with uv
In the project root:
```bash
# Create a virtualenv named .venv
uv venv .venv

# Activate it:

# mac
source .venv/bin/activate

# windows
.venv\Scripts\Activate.ps1
```

## 3. Install dependencies with uv
With the virtualenv activated:

```bash
uv pip install -r requirements.txt
```

## 4. Training the model

Make sure input.txt is in the same directory, then run:

```bash
python train.py
```

The script will:

train the tiny GPT on input.txt

save:

model-00001-of-00001.safetensors

model.safetensors.index.json

config.json

vocab.json

tokenizer_config.json

generation_config.json


## 5. Generating text

After training finishes:

```bash
python generate.py
```

You’ll be prompted:

Enter prompt (empty for random start):


Type a short prompt and press Enter to continue the text.

Or just press Enter with an empty prompt to generate from scratch.

The script loads:

config.json — model architecture

vocab.json + tokenizer_config.json — tokenizer

model-00001-of-00001.safetensors — weights

generation_config.json — max tokens & sampling settings

and prints the generated text to the console.