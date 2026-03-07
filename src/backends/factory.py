from __future__ import annotations
from src.backends.base import Backend


def create_backend(backend: str, **kwargs) -> Backend:
    if backend == "hf":
        from src.backends.hf_backend import HFBackend
        return HFBackend(**kwargs)
    elif backend == "vllm":
        from src.backends.vllm_backend import VLLMBackend
        return VLLMBackend(**kwargs)
    raise ValueError(f"Unknown backend: {backend!r}. Choose 'hf' or 'vllm'.")
