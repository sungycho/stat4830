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
    max_new_tokens: int = 8
    do_sample: bool = False
    temperature: float = 0.9
    top_p: float = 0.95

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
        print(f"[backend] loaded model: {self.model_name}")
        print(f"[backend] decoding: do_sample={self.do_sample} (greedy={'on' if not self.do_sample else 'off'})")

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Causal LMs must pad on the left so the real prompt tokens are always
        # at the right end of the sequence where the model generates from.
        self.tokenizer.padding_side = "left"

        self.model.eval()

    @torch.no_grad()
    def generate_batch(self, prompts: list[str]) -> list[str]:
        """Tokenize all prompts together and run a single batched forward pass."""
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            formatted = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in prompts
            ]
        else:
            formatted = prompts

        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        input_len = inputs["input_ids"].shape[1]

        generate_kwargs = {
            "do_sample": self.do_sample,
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.do_sample:
            generate_kwargs["temperature"] = self.temperature
            generate_kwargs["top_p"] = self.top_p

        out = self.model.generate(**inputs, **generate_kwargs)

        return [
            self.tokenizer.decode(out[i][input_len:], skip_special_tokens=True).strip()
            for i in range(len(prompts))
        ]

    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        """Single-prompt generation — delegates to generate_batch for consistency."""
        return self.generate_batch([prompt])[0]

    def sync_weights(self, model) -> None:
        pass  # HFBackend is the parameter owner; already in sync
