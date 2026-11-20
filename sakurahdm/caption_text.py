"""Tokenize plain image captions without a chat template."""


def encode_text(tokenizer, text, *, max_text_tokens=None):
    ids = tokenizer.encode(text, add_special_tokens=False) if text.strip() else []
    if max_text_tokens is not None:
        if max_text_tokens < 1:
            raise ValueError("max_length must allow at least one text token")
        ids = ids[:max_text_tokens]
    if ids:
        return ids
    # One active token keeps unconditional conditioning nonempty for XUT.
    for token in (tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id):
        if token is not None:
            return [token]
    raise ValueError("Tokenizer needs a PAD, EOS or BOS token for empty captions")
