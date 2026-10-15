"""
Downstream Ranking Feature Enrichment Module.
Implements Section 1 & Section 5.3:
- User-Side Feature: Extends user interest representation via mean/weighted pooling of retrieved ad vectors.
- Item-Side Feature: Computes similarity/match scores between candidate ads and retrieved ads.
- Demonstrated in WeChat Channels to produce an additional +1.43% GMV gain.
"""

from typing import List, Dict, Optional, Any
import numpy as np
from ..data.schemas import AdItem
from ..models.feature_encoder import AdFeatureEncoder


class RankingFeatureExtractor:
    """
    Extracts user-side and item-side ranking features from LEADRE generative retrieval candidates.
    """
    def __init__(self, encoder: AdFeatureEncoder, dim: int = 64):
        self.encoder = encoder
        self.dim = dim

    def compute_user_generative_interest_vector(
        self,
        retrieved_candidates: List[Dict[str, Any]],
        catalog_by_id: Dict[str, AdItem]
    ) -> np.ndarray:
        """
        User-Side Feature:
        Aggregates embeddings of the top retrieved ads into a unified interest representation vector.
        """
        if not retrieved_candidates:
            return np.zeros(self.dim, dtype=np.float32)

        vectors = []
        weights = []
        for cand in retrieved_candidates:
            ad_id = cand["ad_id"]
            if ad_id in catalog_by_id:
                ad_obj = catalog_by_id[ad_id]
                vec = self.encoder.encode_text(ad_obj.to_template_text())
                score = cand.get("retrieval_score", 1.0)
                vectors.append(vec)
                weights.append(score)

        if not vectors:
            return np.zeros(self.dim, dtype=np.float32)

        vectors_np = np.stack(vectors, axis=0)
        weights_np = np.array(weights, dtype=np.float32)
        total_weight = np.sum(weights_np)
        if total_weight > 1e-6:
            weights_np /= total_weight
        else:
            weights_np = np.ones_like(weights_np) / len(weights_np)

        # Weighted combination of retrieved ads
        user_vector = np.sum(vectors_np * weights_np[:, np.newaxis], axis=0)
        norm = np.linalg.norm(user_vector)
        if norm > 1e-6:
            user_vector /= norm

        return user_vector

    def compute_item_match_features(
        self,
        target_ad: AdItem,
        user_interest_vector: np.ndarray,
        retrieved_candidates: List[Dict[str, Any]],
        catalog_by_id: Dict[str, AdItem]
    ) -> Dict[str, float]:
        """
        Item-Side Feature:
        Computes match scores between the target ad and the LEADRE candidate pool:
          1. Cosine similarity with user's generative interest vector.
          2. Maximum similarity with any individual retrieved ad.
          3. Is the target ad present in the retrieved candidate set (boolean flag).
        """
        target_vec = self.encoder.encode_text(target_ad.to_template_text())
        
        # 1. Cosine similarity to user interest vector
        dot_product = float(np.dot(target_vec, user_interest_vector))
        norm_t = np.linalg.norm(target_vec)
        norm_u = np.linalg.norm(user_interest_vector)
        denom = (norm_t * norm_u) if (norm_t * norm_u) > 1e-6 else 1.0
        interest_match_score = float(dot_product / denom)

        # 2. Max similarity to top retrieved ads
        max_sim = 0.0
        is_in_retrieved = False
        for cand in retrieved_candidates:
            if cand["ad_id"] == target_ad.ad_id:
                is_in_retrieved = True
            cand_ad = catalog_by_id.get(cand["ad_id"])
            if cand_ad:
                c_vec = self.encoder.encode_text(cand_ad.to_template_text())
                sim = float(np.dot(target_vec, c_vec) / (np.linalg.norm(target_vec) * np.linalg.norm(c_vec) + 1e-6))
                if sim > max_sim:
                    max_sim = sim

        return {
            "leadre_user_match_score": interest_match_score,
            "leadre_max_candidate_similarity": max_sim,
            "leadre_is_candidate": 1.0 if is_in_retrieved else 0.0
        }
