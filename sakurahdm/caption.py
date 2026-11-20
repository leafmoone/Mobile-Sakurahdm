"""Use SakuraMoon's caption plan and Qwen framing in the shared VLM."""

from dataclasses import replace

from sakuramoon.data.caption import (
    CaptionDropoutProbabilities,
    NlDropoutProbabilities,
    build_caption_plan,
)
from sakuramoon.data.serialize import (
    MAIN_SUFFIX,
    SYSTEM_PREFIX,
    FramingContract,
    render_caption_segments,
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
    def serialize(plan, tokenizer, framing, *, max_text_tokens=None):
        caption = serialize_caption(plan, tokenizer, framing)
        if max_text_tokens is None or len(caption.input_ids) <= max_text_tokens:
            return caption
        budget = _payload_budget(max_text_tokens, framing.prefix_tokens, framing.suffix_tokens)
        bounded = _fit_plan(plan, tokenizer, budget)
        return replace(serialize_caption(bounded, tokenizer, framing), truncated=True)


def caption_framing(tokenizer):
    return FramingContract(
        prefix_tokens=len(tokenizer.encode(SYSTEM_PREFIX, add_special_tokens=False)),
        suffix_tokens=len(tokenizer.encode(MAIN_SUFFIX, add_special_tokens=False)),
        padding_token_id=tokenizer.pad_token_id,
    )


def _canonical_ids(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def _payload_budget(max_text_tokens, prefix_tokens, suffix_tokens):
    budget = max_text_tokens - prefix_tokens - suffix_tokens
    if budget < 0:
        raise ValueError("max_length is shorter than the configured caption framing")
    return budget


def _fit_plan(plan, tokenizer, budget):
    """Apply SakuraMoon's drop order with less payload room, at the same boundaries."""

    condition_only = replace(plan, tags=(), nl_text=None, selected_nl=None)

    def condition_size(request):
        _, text = render_caption_segments(replace(condition_only, condition=request))
        return len(_canonical_ids(tokenizer, text))

    def body_size(candidate):
        text, _ = render_caption_segments(candidate)
        return len(_canonical_ids(tokenizer, text))

    condition = plan.condition
    if condition is not None and condition_size(condition) > budget:
        tags = tuple(
            tag for tag in condition.tags
            if condition_size(replace(condition, tags=(tag,))) <= budget
        )
        while tags and condition_size(replace(condition, tags=tags)) > budget:
            tags = tags[:-1]
        condition = replace(condition, tags=tags) if tags else None
        plan = replace(plan, condition=condition)

    body_budget = budget - condition_size(condition)
    if body_size(plan) > body_budget and plan.nl_text is not None:
        plan = replace(plan, nl_text=None, selected_nl=None)
    while body_size(plan) > body_budget:
        plan = replace(plan, tags=plan.tags[:-1])
    return plan


def serialize_prompt(text, tokenizer, *, condition_text="", max_text_tokens=None):
    """Encode a free prompt without allowing its body to consume the framing."""

    prefix, body, suffix, condition = (
        _canonical_ids(tokenizer, segment)
        for segment in (SYSTEM_PREFIX, text, MAIN_SUFFIX, condition_text)
    )
    if max_text_tokens is not None:
        budget = _payload_budget(max_text_tokens, len(prefix), len(suffix))
        condition = condition[:budget]
        body = body[:budget - len(condition)]
    return tuple(prefix + body + suffix + condition)
