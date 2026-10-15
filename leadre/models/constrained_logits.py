"""
Constrained Beam Search Decoding & LogitsProcessor using Trie-Tree.
Implements Algorithm 2 (Constrained Decoding Using Trie-Tree) from Section 3.4.
Ensures the LLM produces strictly valid S-IDs corresponding to active ads in the inventory.
"""

from typing import List, Dict, Tuple, Optional, Any, Callable
import numpy as np
from .trie import TrieTree


class ConstrainedBeamSearch:
    """
    Implements Algorithm 2 from Section 3.4 of the LEADRE paper.
    Expands S-ID lists layer-by-layer through the Trie-Tree using beam search.
    """
    def __init__(
        self,
        trie: TrieTree,
        beam_size: int = 8,
        sid_length: int = 5
    ):
        self.trie = trie
        self.beam_size = beam_size
        self.sid_length = sid_length

    def decode(
        self,
        token_scorer_fn: Callable[[List[str], List[str]], Dict[str, float]]
    ) -> List[Tuple[List[str], str, float]]:
        """
        Executes Algorithm 2:
        Args:
          token_scorer_fn: A callable taking (current_sids_prefix, valid_candidates)
                           and returning a dict {candidate_token: probability/score}.
                           This is supplied by the LLM forward head.

        Returns:
          List of (s_id_sequence, ad_id, cumulative_score) for top-B ads.
        """
        # Step 1: Initialize beam with root
        beams: List[Tuple[List[str], float]] = [([], 1.0)]

        # Step 2: Layer-by-layer expansion (l = 1 to M + 1)
        for l in range(1, self.sid_length + 1):
            new_candidates: List[Tuple[List[str], float]] = []

            # Step 5: for each candidate list c in B
            for current_prefix, cum_score in beams:
                # Step 6: Retrieve valid next S-IDs from Trie
                valid_next_tokens = self.trie.get_valid_next_tokens(current_prefix)
                if not valid_next_tokens:
                    continue

                # Step 8: Obtain scores from LLM
                token_scores = token_scorer_fn(current_prefix, valid_next_tokens)

                # Steps 9-10: Update candidate lists
                for token in valid_next_tokens:
                    token_prob = token_scores.get(token, 1.0 / len(valid_next_tokens))
                    # Multiplicative probability or cumulative log-prob
                    updated_score = cum_score * token_prob
                    new_candidates.append((current_prefix + [token], updated_score))

            if not new_candidates:
                break

            # Step 13: select_top_B(B', B)
            new_candidates.sort(key=lambda x: x[1], reverse=True)
            beams = new_candidates[:self.beam_size]

        # Final step: Resolve S-IDs to Ad IDs
        results: List[Tuple[List[str], str, float]] = []
        for sids, score in beams:
            ad_id = self.trie.resolve_ad_id(sids) or "unknown_ad"
            results.append((sids, ad_id, score))

        return results


class TrieConstrainedLogitsProcessor:
    """
    Hugging Face compatible LogitsProcessor for Trie constrained generation.
    Masks out all tokens in the vocabulary that do not form a valid prefix path in the Trie-Tree.
    """
    def __init__(
        self,
        trie: TrieTree,
        tokenizer: Any,
        prompt_len: int,
        token_to_sid_map: Dict[int, str],
        sid_to_token_map: Dict[str, int]
    ):
        self.trie = trie
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.token_to_sid_map = token_to_sid_map
        self.sid_to_token_map = sid_to_token_map

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        """
        PyTorch logits processor call.
        Sets logits of invalid S-ID tokens to -float('inf').
        """
        # input_ids: (batch_size, current_len)
        # scores: (batch_size, vocab_size)
        try:
            import torch
            batch_size = input_ids.shape[0]
            for b in range(batch_size):
                gen_ids = input_ids[b, self.prompt_len:].tolist()
                current_sids = [self.token_to_sid_map[tid] for tid in gen_ids if tid in self.token_to_sid_map]
                valid_next = self.trie.get_valid_next_tokens(current_sids)
                valid_token_ids = {self.sid_to_token_map[sid] for sid in valid_next if sid in self.sid_to_token_map}

                # Mask everything except valid next tokens
                if valid_token_ids:
                    mask = torch.ones_like(scores[b], dtype=torch.bool)
                    for vid in valid_token_ids:
                        mask[vid] = False
                    scores[b, mask] = -float("inf")
        except ImportError:
            pass

        return scores
