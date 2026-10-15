"""
Trie-Tree Data Structure & Construction for S-IDs.
Implements Algorithm 1 (Trie-Tree Construction) from Section 3.3 of the paper.
Provides prefix-based ad lookup, valid child token retrieval, and ad ID resolution.
"""

from typing import Dict, List, Optional, Any, Set


class TrieNode:
    """
    Node in the ad prefix Trie-Tree.
    """
    def __init__(self, token: str = ""):
        self.token: str = token
        self.children: Dict[str, "TrieNode"] = {}
        self.is_end_of_ad: bool = False
        self.ad_id: Optional[str] = None

    def add_child(self, token: str) -> "TrieNode":
        if token not in self.children:
            self.children[token] = TrieNode(token)
        return self.children[token]


class TrieTree:
    """
    Prefix Trie-Tree indexing all active advertisement S-ID sequences.
    Ensures that LLM decoding is strictly constrained to valid ad sequences in the catalog.
    """
    def __init__(self):
        self.root = TrieNode(token="__ROOT__")
        self.total_ads: int = 0
        self.sids_to_ad_id: Dict[tuple, str] = {}
        self.ad_id_to_sids: Dict[str, List[str]] = {}

    def insert(self, s_ids: List[str], ad_id: str):
        """
        Algorithm 1: Inserts an ad's S-ID sequence into the Trie-Tree.
        """
        cur = self.root
        for token in s_ids:
            cur = cur.add_child(token)
        cur.is_end_of_ad = True
        cur.ad_id = ad_id
        
        sid_tuple = tuple(s_ids)
        self.sids_to_ad_id[sid_tuple] = ad_id
        self.ad_id_to_sids[ad_id] = s_ids
        self.total_ads += 1

    def get_valid_next_tokens(self, current_prefix: List[str]) -> List[str]:
        """
        Retrieves all valid candidate S-ID tokens following current_prefix.
        Used directly in Algorithm 2 (Constrained Decoding) step 6.
        """
        cur = self.root
        for token in current_prefix:
            if token not in cur.children:
                return []
            cur = cur.children[token]
        return list(cur.children.keys())

    def resolve_ad_id(self, s_ids: List[str]) -> Optional[str]:
        """Resolves an S-ID sequence back to the active catalog Ad ID."""
        return self.sids_to_ad_id.get(tuple(s_ids))

    def is_valid_ad(self, s_ids: List[str]) -> bool:
        """Checks if the S-ID sequence represents a complete active ad."""
        cur = self.root
        for token in s_ids:
            if token not in cur.children:
                return False
            cur = cur.children[token]
        return cur.is_end_of_ad

    def get_tree_stats(self) -> Dict[str, Any]:
        """Calculates Trie statistics (depth, branching factor, total nodes)."""
        node_count = 0
        max_depth = 0
        
        def traverse(node: TrieNode, depth: int):
            nonlocal node_count, max_depth
            node_count += 1
            max_depth = max(max_depth, depth)
            for child in node.children.values():
                traverse(child, depth + 1)

        traverse(self.root, 0)
        return {
            "total_ads": self.total_ads,
            "total_nodes": node_count,
            "max_depth": max_depth,
            "root_branches": len(self.root.children)
        }
