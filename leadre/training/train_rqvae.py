"""
Trainer for Ad Indexing via RQ-VAE and Trie Construction.
Implements the complete offline ad indexing pipeline (Section 3.2, 3.3, and 5.1.3).
"""

from typing import List, Dict, Tuple, Any
import numpy as np
from ..data.schemas import AdItem
from ..models.feature_encoder import AdFeatureEncoder
from ..models.rqvae import RQVAE
from ..models.trie import TrieTree


class RQVAETrainer:
    """
    Trains RQ-VAE on ad descriptions, assigns collision-free S-IDs,
    and builds the prefix Trie-Tree for constrained decoding.
    """
    def __init__(
        self,
        d_plm: int = 64,
        d_rq: int = 8,
        num_layers: int = 4,
        codebook_size: int = 1024,
        beta_quant: float = 0.25,
        epochs: int = 20,
        lr: float = 0.01,
        seed: int = 42
    ):
        self.d_plm = d_plm
        self.d_rq = d_rq
        self.num_layers = num_layers
        self.codebook_size = codebook_size
        self.beta_quant = beta_quant
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        
        self.rqvae = RQVAE(
            d_plm=d_plm,
            d_rq=d_rq,
            num_layers=num_layers,
            codebook_size=codebook_size,
            beta_quant=beta_quant,
            seed=seed
        )
        self.encoder = AdFeatureEncoder(embedding_dim=d_plm)

    def fit_and_index(self, ads: List[AdItem]) -> Tuple[TrieTree, Dict[str, Any]]:
        """
        Runs the full ad indexing flow:
        1. Encodes ad descriptions to dense vectors x_a.
        2. Fits RQ-VAE to minimize reconstruction & quantization loss.
        3. Assigns 5-token S-IDs to all ads with collision disambiguation.
        4. Constructs and populates the Trie-Tree.
        """
        # Step 1: Text Embedding
        text_descriptions = [a.to_template_text() for a in ads]
        embeddings = self.encoder.encode_batch(text_descriptions)
        embeddings_by_ad = {a.ad_id: embeddings[i] for i, a in enumerate(ads)}

        # Step 2: Fit RQ-VAE
        loss_history = []
        for epoch in range(self.epochs):
            step_metrics = self.rqvae.fit_step(embeddings, lr=self.lr)
            loss_history.append(step_metrics)

        # Step 3: Assign discrete S-IDs
        sids_map = self.rqvae.assign_sids(embeddings_by_ad)
        for ad in ads:
            ad.s_ids = sids_map[ad.ad_id]

        # Step 4: Build Trie-Tree (Algorithm 1)
        trie = TrieTree()
        for ad in ads:
            trie.insert(ad.s_ids, ad.ad_id)

        # Statistics & Metrics
        collision_stats = self.rqvae.compute_collision_metrics(sids_map)
        usage_stats = self.rqvae.quantizer.get_codebook_usage_stats()
        trie_stats = trie.get_tree_stats()

        summary = {
            "final_loss": loss_history[-1] if loss_history else {},
            "collision_stats": collision_stats,
            "codebook_usage": usage_stats,
            "trie_stats": trie_stats
        }

        return trie, summary
