"""Reuse SakuraMoon's caption rules for plain-text image conditioning."""

from dataclasses import replace

from sakuramoon.data.caption import (
    CaptionDropoutProbabilities,
    NlDropoutProbabilities,
    build_caption_plan,
)
from sakuramoon.data.serialize import render_caption_segments

from .caption_text import encode_text


class SakuraMoonCaptions:
    def __init__(self, config):
        dropout = config["dropout"]
        self.probabilities = CaptionDropoutProbabilities(
            # Use the native branch that puts both surviving sources in the body.
            # It cannot emit a routed condition or apply condition-only dropout.
            condition_route=1.0,
            condition_only=0.0,
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
        plan = build_caption_plan(
            fields, self.probabilities, condition_mode="artist", seed=seed
        )
        first = {"artist": 0, "character": 1}
        return replace(plan, tags=tuple(sorted(plan.tags, key=lambda tag: first.get(tag.source, 2))))

    @staticmethod
    def serialize(plan, tokenizer, *, max_text_tokens=None):
        while True:
            text, _ = render_caption_segments(plan)
            ids = tokenizer.encode(text, add_special_tokens=False)
            if max_text_tokens is None or len(ids) <= max_text_tokens:
                return text, tuple(ids or encode_text(tokenizer, text, max_text_tokens=max_text_tokens))
            # Keep the upstream truncation priority and complete tag boundaries.
            if plan.nl_text is not None:
                plan = replace(plan, nl_text=None, selected_nl=None)
            else:
                plan = replace(plan, tags=plan.tags[:-1])
