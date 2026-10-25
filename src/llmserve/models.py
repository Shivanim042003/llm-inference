"""Model and tokenizer loading, plus a tiny random model for CPU tests and smoke runs."""
from __future__ import annotations

import torch

TINY_ID = "tiny-random"


class ByteTokenizer:
    """Offline tokenizer for tests: one token per byte (vocab 256 + specials)."""

    eos_ids: set = set()

    def encode_chat(self, messages):
        text = "\n".join(m.get("content", "") for m in messages)
        ids = list(text.encode("utf-8", errors="ignore"))
        return ids if ids else [1]

    def decode(self, ids):
        return bytes(int(i) % 256 for i in ids).decode("utf-8", errors="replace")


class HFTokenizer:
    def __init__(self, model_id: str):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_id)
        eos = self.tok.eos_token_id
        self.eos_ids = {eos} if eos is not None else set()
        # Qwen chat models also end turns with <|im_end|>
        for name in ("<|im_end|>", "<|endoftext|>"):
            tid = self.tok.convert_tokens_to_ids(name)
            if isinstance(tid, int) and tid >= 0 and tid != self.tok.unk_token_id:
                self.eos_ids.add(tid)

    def encode_chat(self, messages):
        try:
            prompt = self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = "\n".join(m.get("content", "") for m in messages)
        return self.tok(prompt, add_special_tokens=False)["input_ids"]

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True)


def build_tiny_model(seed: int = 0, dtype=torch.float32):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        tie_word_embeddings=False,
    )
    model = Qwen2ForCausalLM(cfg).to(dtype)
    model.eval()
    return model


def torch_dtype(name: str):
    return {"float32": torch.float32, "fp32": torch.float32, "float16": torch.float16,
            "fp16": torch.float16, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            "float64": torch.float64}[name]


def load_model(model_id: str, dtype: str = "float16", quant: str = "none", device: str | None = None):
    """Return (model, tokenizer, eos_ids, device). quant: none | int8 | nf4."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if model_id == TINY_ID:
        model = build_tiny_model(dtype=torch_dtype(dtype if device == "cuda" else "float32"))
        return model.to(device), ByteTokenizer(), set(), device

    from transformers import AutoModelForCausalLM

    kwargs = {}
    if quant in ("int8", "nf4"):
        from transformers import BitsAndBytesConfig

        if quant == "int8":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype(dtype),
            )
        kwargs["device_map"] = {"": 0} if device == "cuda" else None
    dt = torch_dtype(dtype)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dt, **kwargs)
    except TypeError:  # older transformers use torch_dtype
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dt, **kwargs)
    if quant == "none":
        model = model.to(device)
    model.eval()
    tok = HFTokenizer(model_id)
    return model, tok, tok.eos_ids, device


def model_footprint(model):
    """Bytes of parameters and KV-cache bytes per token (fp16/fp32 as loaded)."""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    cfg = model.config
    heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    elem = next(model.parameters()).element_size()
    if elem == 1:  # quantised weights: KV cache stays in compute dtype (assume 2 bytes)
        elem = 2
    kv_per_token = 2 * cfg.num_hidden_layers * heads * head_dim * elem
    return {"param_bytes": param_bytes, "kv_bytes_per_token": kv_per_token}
