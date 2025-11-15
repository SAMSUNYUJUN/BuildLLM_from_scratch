# -*- coding: utf-8 -*-
import json
import os
import regex as re
from typing import List, Dict, Tuple, Optional

# -------- GPT-2 style byte <-> unicode mapping --------
def bytes_to_unicode() -> Dict[int, str]:
    # Map 0..255 to a stable, printable unicode range (GPT-2 trick)
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}

def unicode_to_bytes_map(b2u: Dict[int, str]) -> Dict[str, int]:
    return {u: b for b, u in b2u.items()}


class bpe_tokenizer:
    """
    A minimal, byte-level BPE tokenizer with:
      - regex pretokenization (GPT-2 style)
      - special tokens (pad/bos/eos/unk) with explicit printable strings
      - train / encode / decode
      - save/load in unicode-token form (human-readable)
      - optional GPT-2-style export: encoder.json + merges.txt
    """
    def __init__(
        self,
        train_text: str,
        vocab_size: int,
        use_regex_pretokenize: bool = True,
        special_tokens: Optional[Dict[str, int]] = None,
        special_strings: Optional[Dict[str, str]] = None,
    ):
        # ----- config -----
        self.train_text_str = train_text
        self.vocab_size = int(vocab_size)
        self.use_regex_pretokenize = use_regex_pretokenize
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

        # byte<->unicode maps (GPT-2)
        self.b2u = bytes_to_unicode()
        self.u2b = unicode_to_bytes_map(self.b2u)

        # special tokens and their explicit printable strings
        if special_tokens is None:
            special_tokens = {"pad": 0, "bos": 1, "eos": 2, "unk": 3}
        if special_strings is None:
            special_strings = {"pad": "<pad>", "bos": "<bos>", "eos": "<eos>", "unk": "<unk>"}

        self.special_tokens = dict(special_tokens)
        self.special_strings = dict(special_strings)
        self.num_special = len(self.special_tokens)
        self.id2special = {i: k for k, i in self.special_tokens.items()}

        # model params
        self.merges: Dict[Tuple[int, int], int] = {}  # {(p0,p1): new_id}
        self.rank: Dict[Tuple[int, int], int] = {}    # {(p0,p1): rank}  smaller = earlier
        self.vocab: Dict[int, bytes] = {}             # {id: bytes}

        # train
        self._train_tokenizer()

    # ---------------- I/O ----------------
    @classmethod
    def from_files(
        cls,
        merges_path: str,
        vocab_path: str,
        config_path: Optional[str] = None
    ) -> "bpe_tokenizer":

        merges_raw = json.load(open(merges_path, "r"))

        def parse_pair(k: str) -> Tuple[int, int]:
            k = k.strip()
            if k.startswith("(") and k.endswith(")"):
                k = k[1:-1]
            a, b = k.split(",")
            return (int(a.strip()), int(b.strip()))

        merges = {parse_pair(k): int(v) for k, v in merges_raw.items()}
        vocab_unicode_raw = json.load(open(vocab_path, "r"))

        # defaults
        special_tokens = {"pad": 0, "bos": 1, "eos": 2, "unk": 3}
        special_strings = {"pad": "<pad>", "bos": "<bos>", "eos": "<eos>", "unk": "<unk>"}
        use_regex = True
        pat_str = None

        if config_path and os.path.exists(config_path):
            cfg = json.load(open(config_path, "r"))
            special_tokens = cfg.get("special_tokens", special_tokens)
            special_strings = cfg.get("special_strings", special_strings)
            use_regex = cfg.get("use_regex_pretokenize", use_regex)
            pat_str = cfg.get("pattern", None)

        tok = cls.__new__(cls)
        tok.train_text_str = ""
        tok.vocab_size = max(map(int, vocab_unicode_raw.keys())) + 1
        tok.special_tokens = special_tokens
        tok.special_strings = special_strings
        tok.num_special = len(special_tokens)
        tok.id2special = {i: k for k, i in special_tokens.items()}
        tok.use_regex_pretokenize = use_regex
        tok.pat = re.compile(pat_str) if pat_str else re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        tok.b2u = bytes_to_unicode()
        tok.u2b = unicode_to_bytes_map(tok.b2u)

        tok.merges = merges
        tok.rank = {pair: ridx for ridx, (pair, _) in enumerate(sorted(merges.items(), key=lambda kv: kv[1]))}

        # rebuild vocab from unicode tokens
        inv_special_str = {v: k for k, v in special_strings.items()}  # "<pad>" -> "pad"
        tok.vocab = {}
        for k_str, u_token in vocab_unicode_raw.items():
            k = int(k_str)
            if u_token in inv_special_str or k in tok.id2special:
                tok.vocab[k] = b""  # special: no bytes used in decoding
            else:
                tok.vocab[k] = bytes([tok.u2b[ch] for ch in u_token])

        return tok

    def save_pretrained(
        self,
        merges_path: str = "bpe_merges.json",
        vocab_path: str = "bpe_vocab.json",
        config_path: str = "tokenizer_config.json"
    ):
        # merges: "(p0, p1)" -> new_id
        merges_to_save = {f"({p0}, {p1})": idx for (p0, p1), idx in self.merges.items()}
        json.dump(merges_to_save, open(merges_path, "w"))

        # vocab: int -> unicode token (printable)
        vocab_unicode = {}
        for i, b in self.vocab.items():
            if i in self.id2special:
                tag = self.id2special[i]
                vocab_unicode[i] = self.special_strings.get(tag, f"<{tag}>")
            else:
                vocab_unicode[i] = "".join(self.b2u[bt] for bt in b)
        json.dump({int(k): v for k, v in vocab_unicode.items()}, open(vocab_path, "w"), ensure_ascii=False)

        cfg = {
            "vocab_size": self.vocab_size,
            "num_special": self.num_special,
            "special_tokens": self.special_tokens,
            "special_strings": self.special_strings,
            "use_regex_pretokenize": self.use_regex_pretokenize,
            "pattern": self.pat.pattern,
        }
        json.dump(cfg, open(config_path, "w"), ensure_ascii=False)

    # ---------------- core utils ----------------
    def _bytes_to_ids(self, b: bytes) -> List[int]:
        base = self.num_special
        return [base + bt for bt in b]

    def _ids_to_bytes(self, ids: List[int]) -> bytes:
        return b"".join(self.vocab[i] for i in ids if i not in self.id2special)

    def get_stats(self, ids: List[int]) -> Dict[Tuple[int, int], int]:
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge_once(self, ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
        new_ids, i = [], 0
        L = len(ids)
        while i < L:
            if i < L - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    # ---------------- train ----------------
    def _train_tokenizer(self):
        base = self.num_special
        # init vocab: special placeholders + 256 bytes
        self.vocab = {i: b"" for i in range(self.num_special)}
        for i in range(256):
            self.vocab[base + i] = bytes([i])

        # build initial training sequence (regex pretokenize -> bytes -> ids)
        if self.use_regex_pretokenize:
            parts = [t.encode("utf-8") for t in re.findall(self.pat, self.train_text_str)]
            current = []
            for p in parts:
                current.extend(self._bytes_to_ids(p))
        else:
            current = self._bytes_to_ids(self.train_text_str.encode("utf-8"))

        target_vocab = max(self.vocab_size, base + 256)
        num_merges = max(0, target_vocab - (base + 256))

        # greedy training; stable tie-break by pair order
        for i in range(num_merges):
            stats = self.get_stats(current)
            if not stats:
                break
            pair, _ = max(stats.items(), key=lambda kv: (kv[1], kv[0]))
            new_id = base + 256 + i
            current = self.merge_once(current, pair, new_id)
            self.merges[pair] = new_id

        # build rank by learning order (ascending new_id)
        self.rank = {pair: r for r, (pair, _) in enumerate(sorted(self.merges.items(), key=lambda kv: kv[1]))}

        # recursively materialize merged tokens (bytes)
        for i in range(base + 256, base + 256 + len(self.merges)):
            self.vocab[i] = b""
        for pair, idx in sorted(self.merges.items(), key=lambda kv: kv[1]):
            p0, p1 = pair
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

        self.vocab_size = max(self.vocab.keys()) + 1

    # ---------------- encode / decode ----------------
    def _encode_ids_of_bytes(self, b: bytes) -> List[int]:
        ids = self._bytes_to_ids(b)
        while True:
            stats = self.get_stats(ids)
            if not stats:
                break
            pair = min(stats.keys(), key=lambda p: self.rank.get(p, float("inf")))
            if pair not in self.rank:
                break
            ids = self.merge_once(ids, pair, self.merges[pair])
        return ids

    def encode(self, input_text: str, add_special_tokens: bool = False) -> List[int]:
        ids: List[int] = []
        if self.use_regex_pretokenize:
            for tok in re.finditer(self.pat, input_text):
                b = tok.group(0).encode("utf-8")
                ids.extend(self._encode_ids_of_bytes(b))
        else:
            ids = self._encode_ids_of_bytes(input_text.encode("utf-8"))
        if add_special_tokens and "bos" in self.special_tokens and "eos" in self.special_tokens:
            return [self.special_tokens["bos"]] + ids + [self.special_tokens["eos"]]
        return ids

    def encode_plus(self, input_text: str, add_special_tokens: bool = True) -> Dict:
        input_ids: List[int] = []
        offsets: List[Tuple[int, int]] = []
        if self.use_regex_pretokenize:
            for m in re.finditer(self.pat, input_text):
                start, end = m.span()
                b = m.group(0).encode("utf-8")
                ids_piece = self._encode_ids_of_bytes(b)
                input_ids.extend(ids_piece)
                offsets.extend([(start, end)] * len(ids_piece))
        else:
            b = input_text.encode("utf-8")
            ids_piece = self._encode_ids_of_bytes(b)
            input_ids.extend(ids_piece)
            offsets.extend([(0, len(input_text))] * len(ids_piece))

        if add_special_tokens and "bos" in self.special_tokens and "eos" in self.special_tokens:
            input_ids = [self.special_tokens["bos"]] + input_ids + [self.special_tokens["eos"]]
            offsets = [(-1, -1)] + offsets + [(-1, -1)]
        return {"input_ids": input_ids, "offset_mapping": offsets}

    def encode_batch(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
        return_attention_mask: bool = True,
        add_special_tokens: bool = True
    ) -> Dict[str, List[List[int]]]:
        encoded = [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]
        lengths = [len(x) for x in encoded]
        if max_length is None:
            max_length = max(lengths) if padding else None

        pad_id = self.special_tokens.get("pad", 0)
        out_ids, attn = [], []

        for seq in encoded:
            if truncation and max_length is not None and len(seq) > max_length:
                seq = seq[:max_length]
            if padding and max_length is not None and len(seq) < max_length:
                pad_len = max_length - len(seq)
                out_ids.append(seq + [pad_id] * pad_len)
                if return_attention_mask:
                    attn.append([1] * len(seq) + [0] * pad_len)
            else:
                out_ids.append(seq)
                if return_attention_mask:
                    attn.append([1] * len(seq))

        out = {"input_ids": out_ids}
        if return_attention_mask:
            out["attention_mask"] = attn
        return out

    def decode(self, ids: List[int], skip_special_tokens: bool = True, render_special_tokens: bool = False) -> str:
        # Optionally skip special ids; or render them as explicit strings (e.g., "<bos>")
        out_parts: List[str] = []
        buf: List[int] = []

        def flush_buf():
            if not buf:
                return ""
            b = b"".join(self.vocab[i] for i in buf)
            s = b.decode("utf-8", errors="replace")
            buf.clear()
            return s

        for i in ids:
            if i in self.id2special:
                if not skip_special_tokens:
                    if render_special_tokens:
                        out_parts.append(flush_buf())
                        out_parts.append(self.special_strings.get(self.id2special[i], f"<{self.id2special[i]}>"))
                    # else: keep nothing (drops specials but not others)
                else:
                    # skip specials entirely
                    pass
            else:
                buf.append(i)

        out_parts.append(flush_buf())
        return "".join(out_parts)


if __name__ == "__main__":
    # with open("tiny_char_gpt/input.txt", "r", encoding="utf-8") as f:
    #     txt = f.read()
    #tok = bpe_tokenizer(txt, vocab_size=1000)

    # ids = tok.encode("Hello, world!", add_special_tokens=True)
    # print("ids:", ids)
    # print("decoded:", tok.decode(ids))

    # pack = tok.encode_plus("Hello, world!")
    # print("encode_plus:", pack)

    # batch = tok.encode_batch(["Hello World!", "This is Sam speaking"], padding=True, truncation=True, max_length=100)
    # print("batch ids:", batch["input_ids"])
    # print("batch mask:", batch["attention_mask"])

    # tok.save_pretrained()
    tok = bpe_tokenizer.from_files("bpe_merges.json", "bpe_vocab.json", "tokenizer_config.json")
    batch = tok.encode_batch(["Hello World!", "This is Sam speaking"], padding=True, truncation=True, max_length=None, add_special_tokens=True)
    print("batch ids:", batch["input_ids"])
    print("batch mask:", batch["attention_mask"])
    print("decoded:", [tok.decode(ids,skip_special_tokens=False, render_special_tokens=True) for ids in batch["input_ids"]])
