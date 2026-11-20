"""Use SakuraMoon's caption plan and Qwen framing in the shared VLM."""

from sakuramoon.data.caption import (
    CaptionDropoutProbabilities,
    NlDropoutProbabilities,
    build_caption_plan,
)
from sakuramoon.data.serialize import (
    MAIN_SUFFIX,
    SYSTEM_PREFIX,
    FramingContract,
    serialize_caption,
)


class SakuraMoonCaptions:
    def __init__(self, config):
        self.condition_mode = config["condition_mode"]
        dropout = config["dropout"]
        self.probabilities = CaptionDropoutProbabilities(
            condition_route=dropout["condition_route"],
            condition_only=dropout["condition_only"],
            tag=dropout["tag"],
            candidate_source=dropout["candidate_source"],
            nl=NlDropoutProbabilities(**dropout["nl"]),
            nl_all=dropout.get("nl_all", 0.0),
        )

    @staticmethod
    def fields(raw_metadata):
        from sakuramoon.data.production import parse_modelscope_caption_fields

        return parse_modelscope_caption_fields(raw_metadata)

    def plan(self, fields, seed):
        return build_caption_plan(
            fields, self.probabilities, condition_mode=self.condition_mode, seed=seed
        )

    @staticmethod
    def serialize(plan, tokenizer, framing):
        return serialize_caption(plan, tokenizer, framing)


def caption_framing(tokenizer):
    return FramingContract(
        prefix_tokens=len(tokenizer.encode(SYSTEM_PREFIX, add_special_tokens=False)),
        suffix_tokens=len(tokenizer.encode(MAIN_SUFFIX, add_special_tokens=False)),
        padding_token_id=tokenizer.pad_token_id,
    )


def serialize_prompt(text, tokenizer, *, condition_text=""):
    """Encode a free inference prompt with the same native segment boundaries."""
    return tuple(
        token
        for segment in (SYSTEM_PREFIX, text, MAIN_SUFFIX, condition_text)
        for token in tokenizer.encode(segment, add_special_tokens=False)
    )
