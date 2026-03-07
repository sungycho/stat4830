class VLLMBackend:
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        raise NotImplementedError("VLLMBackend not yet implemented")

    def generate_batch(self, prompts: list[str]) -> list[str]:
        raise NotImplementedError("VLLMBackend not yet implemented")

    def sync_weights(self, model) -> None:
        raise NotImplementedError("VLLMBackend not yet implemented")
