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
    def generate_batch(self, prompts: list[str], raw: bool = False) -> list[str]:
        """Tokenize all prompts together and run a single batched forward pass.

        raw=True: skip chat-template formatting and tokenize prompts as-is.
        Use this when prompts are already fully formatted (e.g. countdown context strings).
        """
        if not raw and hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
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

        max_len = getattr(self.tokenizer, "model_max_length", None)
        if max_len is not None and max_len > 1_000_000:
            max_len = None  # sentencepiece models report absurdly large limits
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=max_len is not None,
            max_length=max_len,
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

    @torch.no_grad()
    def score_logprobs_batch(
        self, prompts: list[str], label_words: list[str], raw: bool = False
    ) -> list[dict[str, float]]:
        """Single forward pass (no generation). Returns restricted log-softmax over
        label_words for each prompt: {word: log P(word | prompt, label_set)}.

        Each label word is represented by its first subword token as it would
        appear mid-sequence (encoded with a leading space). Multi-token label
        words are approximated by their first token only.

        raw=True: skip chat-template formatting (mirrors generate_batch behaviour).
        Use for base-model completion prompts (e.g. MeZO-style templates).
        """
        # Get the first token ID for each label word as it appears mid-sequence.
        label_token_ids: dict[str, int] = {}
        for word in label_words:
            ids = self.tokenizer.encode(" " + word, add_special_tokens=False)
            label_token_ids[word] = ids[0]

        # Apply chat template if needed (same as generate_batch).
        if not raw and hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
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

        max_len = getattr(self.tokenizer, "model_max_length", None)
        if max_len is not None and max_len > 1_000_000:
            max_len = None  # sentencepiece models report absurdly large limits
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=max_len is not None,
            max_length=max_len,
        ).to(self.device)

        outputs = self.model(**inputs)
        # With left-padding, position -1 is always the last real token.
        last_logits = outputs.logits[:, -1, :]  # [batch, vocab]

        # Restricted log-softmax: softmax over label token logits only.
        token_ids = [label_token_ids[w] for w in label_words]
        label_logits = last_logits[:, token_ids]                          # [batch, n_labels]
        restricted_log_probs = torch.nn.functional.log_softmax(           # [batch, n_labels]
            label_logits, dim=-1
        )

        results = []
        for i in range(len(prompts)):
            results.append({
                word: restricted_log_probs[i, j].item()
                for j, word in enumerate(label_words)
            })
        return results

    @property
    def is_instruct(self) -> bool:
        """True if the tokenizer has a chat template (instruction-tuned model)."""
        return bool(getattr(self.tokenizer, "chat_template", None))

    def sync_weights(self, model) -> None:
        pass  # HFBackend is the parameter owner; already in sync
