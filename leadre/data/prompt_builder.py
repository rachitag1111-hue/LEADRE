"""
Intent-Aware Prompt Engineering & Augmentation for LEADRE.
Implements Section 4.2 of the paper:
- Multi-component prompt construction
- Learnable soft prompt token insertion
- Profile reordering augmentation
- Auto-regressive sequence prefix slicing
- Explicit and Implicit alignment prompt formats
- DPO preference pair generation
"""

import random
from typing import List, Tuple, Dict, Optional, Any
from .schemas import AdItem, UserProfile, InteractionEvent, UserHistory


class PromptBuilder:
    """
    Constructs and augments training & inference prompts for LEADRE.
    """
    def __init__(
        self,
        soft_token: str = "<Learnable Token>",
        include_soft_token: bool = True,
        max_prompt_tokens_est: int = 2048,
        seed: Optional[int] = 42
    ):
        self.soft_token = soft_token
        self.include_soft_token = include_soft_token
        self.max_prompt_tokens_est = max_prompt_tokens_est
        if seed is not None:
            random.seed(seed)

    def format_task_instruction(self, context_body: str) -> str:
        """
        Formats Task Instruction (Section 4.2.1):
        'The following is an instruction describing a task. Please give a response to complete this request appropriately.
        [Instruction]:<Learnable Token>Assuming you are an ad recommender system, <Prompt>, what ad will the user be interested in next time?'
        """
        token_prefix = f"{self.soft_token}" if self.include_soft_token else ""
        return (
            "The following is an instruction describing a task. Please give a response to complete this request appropriately.\n"
            f"[Instruction]:{token_prefix}Assuming you are an ad recommender system, {context_body} "
            "what ad will the user be interested in next time?"
        )

    def format_user_profile(self, profile: UserProfile, shuffle: bool = False) -> str:
        """
        Formats demographic attributes.
        If shuffle is True, applies 'User Profile Reordering' (Section 4.2.2).
        """
        parts = [
            f"{profile.age}-year-old {profile.sex}",
            f"resident in {profile.residence}",
            f"with a {profile.education_level} degree",
            f"working in {profile.occupation}",
            f"with a {profile.consumption_level} consumption level"
        ]
        if shuffle:
            random.shuffle(parts)
        return ", ".join(parts) + "."

    def format_event_sequence(
        self,
        events: List[InteractionEvent],
        use_sids_for_ads: bool = True
    ) -> str:
        """
        Formats chronologically merged short-term behavior sequence.
        Section 4.2.1 (4) & (5).
        """
        if not events:
            return ""
        entries = [e.to_sequence_entry(use_sids_for_ads=use_sids_for_ads) for e in events]
        sequence_str = "; ".join(entries)
        return (
            f"The most recent interaction behavior sequence details "
            f"(format: time^behavior type^title) are {sequence_str}."
        )

    def build_main_prompt(
        self,
        history: UserHistory,
        event_slice: Optional[List[InteractionEvent]] = None,
        shuffle_profile: bool = False,
        include_summary: bool = True,
        include_content: bool = True
    ) -> str:
        """
        Builds complete Main Task Prompt (Section 4.2.1).
        Supports ablation flags (w.o. content, w.o. summary).
        """
        profile_text = self.format_user_profile(history.profile, shuffle=shuffle_profile)
        
        components = [profile_text]
        if include_summary and history.category_counts:
            components.append(history.to_interest_summary_text())

        events = event_slice if event_slice is not None else history.events
        if not include_content:
            events = [e for e in events if e.domain == "ad"]

        if events:
            seq_text = self.format_event_sequence(events, use_sids_for_ads=True)
            components.append(seq_text)

        context_body = " ".join(components)
        return self.format_task_instruction(context_body)

    # -------------------------------------------------------------------------
    # Curriculum Alignment Tasks (Section 4.3.1)
    # -------------------------------------------------------------------------

    def build_explicit_alignment_pair(self, ad: AdItem) -> Tuple[str, str]:
        """
        Stage 1 (EX): Ad Text Description -> S-ID.
        """
        prompt = (
            f'Given the ad\'s detailed description "{ad.to_template_text()}", '
            f'what is the corresponding ad?'
        )
        response = ad.s_id_string()
        return prompt, response

    def build_implicit_alignment_prompt(
        self,
        history: UserHistory,
        event_slice: List[InteractionEvent],
        shuffle_profile: bool = False
    ) -> str:
        """
        Stage 2 (IM): User interaction sequence with Ad descriptions instead of S-IDs -> Next Ad S-ID.
        """
        profile_text = self.format_user_profile(history.profile, shuffle=shuffle_profile)
        components = [profile_text]
        if history.category_counts:
            components.append(history.to_interest_summary_text())

        seq_text = self.format_event_sequence(event_slice, use_sids_for_ads=False)
        components.append(seq_text)

        context_body = " ".join(components)
        return self.format_task_instruction(context_body)

    # -------------------------------------------------------------------------
    # Sequence Augmentation (Section 4.2.2)
    # -------------------------------------------------------------------------

    @staticmethod
    def slice_positive_ad_interactions(
        events: List[InteractionEvent]
    ) -> List[Tuple[List[InteractionEvent], InteractionEvent]]:
        """
        Ad Positive Interaction Reuse:
        Given S_u = [c1, a2, a3, c4, a5], returns:
        - ([c1], a2)
        - ([c1, a2], a3)
        - ([c1, a2, a3, c4], a5)
        """
        slices = []
        for i, event in enumerate(events):
            if event.domain == "ad" and event.s_ids:
                prior_events = events[:i]
                if prior_events:
                    slices.append((prior_events, event))
        return slices
