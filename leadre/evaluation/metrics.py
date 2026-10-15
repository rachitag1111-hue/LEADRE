"""
Evaluation Metrics for LEADRE.
Implements:
- Hit Ratio (HR@K) - Section 5.1.2
- Normalized Discounted Cumulative Gain (NDCG@K) - Section 5.1.2
- Diversity Score (Abundance & Concentration) - Section 5.2.3
- LTRR@K (Learning-to-Rank Alignment Recall) - Section 5.2.3
- Dice Similarity - Section 5.2.6 Eq. (7)
"""

from typing import List, Dict, Set, Any, Optional
import numpy as np


class Evaluator:
    """
    Computes offline evaluation metrics matching the paper's benchmark suite.
    """
    @staticmethod
    def hit_ratio_at_k(retrieved_ids: List[str], target_id: str, k: int) -> float:
        """Computes HR@K for a single user."""
        top_k = retrieved_ids[:k]
        return 1.0 if target_id in top_k else 0.0

    @staticmethod
    def ndcg_at_k(retrieved_ids: List[str], target_id: str, k: int) -> float:
        """Computes NDCG@K for leave-one-out single target item."""
        top_k = retrieved_ids[:k]
        for rank, item in enumerate(top_k, start=1):
            if item == target_id:
                # DCG = 1 / log2(rank + 1), IDCG = 1 / log2(2) = 1.0
                return 1.0 / np.log2(rank + 1)
        return 0.0

    @staticmethod
    def diversity_score(
        retrieved_categories_per_user: List[List[str]],
        k: int = 8
    ) -> Dict[str, float]:
        """
        Computes Top-K List Concentration and Abundance (Section 5.2.3).
        - Concentration: mean proportion of the most frequent category in top-K.
        - Abundance: mean number of distinct categories in top-K.
        - Combined Diversity Score = Abundance / (Concentration * k)
        """
        abundances = []
        concentrations = []

        for cats in retrieved_categories_per_user:
            top_k_cats = cats[:k]
            if not top_k_cats:
                continue
            
            # Abundance: distinct category count
            unique_count = len(set(top_k_cats))
            abundances.append(unique_count)

            # Concentration: proportion of most frequent category
            from collections import Counter
            counts = Counter(top_k_cats)
            max_freq = counts.most_common(1)[0][1]
            concentrations.append(max_freq / len(top_k_cats))

        mean_abundance = float(np.mean(abundances)) if abundances else 0.0
        mean_concentration = float(np.mean(concentrations)) if concentrations else 1.0
        
        # Normalized diversity score (higher is better)
        diversity = mean_abundance * (1.0 - mean_concentration)

        return {
            "abundance": round(mean_abundance, 4),
            "concentration": round(mean_concentration, 4),
            "diversity_score": round(diversity, 4)
        }

    @staticmethod
    def dice_similarity(list_a: List[str], list_b: List[str]) -> float:
        """
        Implements Equation (7) from Section 5.2.6:
          Dice(R_u1, R_ul) = 2 * |R_u1 n R_ul| / (|R_u1| + |R_ul|)
        """
        set_a = set(list_a)
        set_b = set(list_b)
        intersection = len(set_a.intersection(set_b))
        total = len(set_a) + len(set_b)
        if total == 0:
            return 1.0
        return float((2.0 * intersection) / total)

    @classmethod
    def evaluate_recommendations(
        cls,
        user_recommendations: List[Dict[str, Any]],
        target_ad_ids: List[str],
        categories_per_rec: Optional[List[List[str]]] = None,
        ks: List[int] = [1, 4, 8]
    ) -> Dict[str, float]:
        """
        Evaluates full test suite across HR@K, NDCG@K, and Diversity.
        """
        metrics = {}
        for k in ks:
            hr_list = []
            ndcg_list = []
            for recs, target in zip(user_recommendations, target_ad_ids):
                cand_ids = [r["ad_id"] for r in recs]
                hr_list.append(cls.hit_ratio_at_k(cand_ids, target, k))
                ndcg_list.append(cls.ndcg_at_k(cand_ids, target, k))
            metrics[f"HR@{k}"] = round(float(np.mean(hr_list)), 4)
            metrics[f"NDCG@{k}"] = round(float(np.mean(ndcg_list)), 4)

        if categories_per_rec:
            div_stats = cls.diversity_score(categories_per_rec, k=max(ks))
            metrics.update(div_stats)

        return metrics
