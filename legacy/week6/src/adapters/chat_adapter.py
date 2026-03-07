def hf_text_to_completion(text: str):
    """
    Convert raw HF text output into Prime-style completion format.
    """
    return [
        {"role": "assistant", "content": text}
    ]