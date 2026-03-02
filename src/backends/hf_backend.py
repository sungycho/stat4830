# src/backends/hf_backend.py
from __future__ import annotations

from dataclasses import dataclass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class HFBackend:
    model_name: str
    device: str = "cpu"
    dtype: str = "float32"
    max_new_tokens: int = 16

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        torch_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[self.dtype]

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch_dtype,
        ).to(self.device)

        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        out = self.model.generate(
            **inputs,
            do_sample=False,          # greedy decoding
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        return self.tokenizer.decode(out[0], skip_special_tokens=True)