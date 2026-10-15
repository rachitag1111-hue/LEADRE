"""
Residual-Quantized Variational AutoEncoder (RQ-VAE) for LEADRE Ads Indexing.
Implements Section 3.2:
- Equations (1)-(6) from the paper
- Multi-layer residual vector quantization
- Reconstruction loss & quantization/commitment loss
- Codebook usage & collision metrics (Table 5)
- Disambiguation code assignment for collision-free S-ID sequences
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np


class ResidualQuantizer:
    """
    Implements Residual Vector Quantization (RVQ) with M codebooks.
    Works natively with NumPy for maximum portability and fast execution.
    Can also be wrapped in PyTorch nn.Module.
    """
    def __init__(
        self,
        num_layers: int = 4,
        codebook_size: int = 1024,
        dim: int = 8,
        beta_quant: float = 0.25,
        seed: int = 42
    ):
        self.num_layers = num_layers
        self.codebook_size = codebook_size
        self.dim = dim
        self.beta_quant = beta_quant
        
        np.random.seed(seed)
        # Initialize codebooks C^l = {e_k^l}
        self.codebooks = [
            np.random.randn(codebook_size, dim).astype(np.float32) * 0.1
            for _ in range(num_layers)
        ]
        self.codebook_usage = [np.zeros(codebook_size, dtype=np.int32) for _ in range(num_layers)]

    def quantize_vector(self, z_hat: np.ndarray) -> Tuple[List[int], np.ndarray, float]:
        """
        Quantizes a single continuous latent vector z_hat of dimension (dim,).
        Follows Eq. (2)-(4):
          c_a^l = argmin_k ||r_a^l - e_k^l||^2
          r_a^{l+1} = r_a^l - e_{c_a^l}^l
          z_a = sum_{l=1}^M e_{c_a^l}^l
        """
        indices: List[int] = []
        quant_loss = 0.0
        r_l = z_hat.copy()
        z_a = np.zeros_like(z_hat)

        for l in range(self.num_layers):
            codebook = self.codebooks[l]  # shape: (K, dim)
            # Compute Euclidean distances
            diffs = codebook - r_l[np.newaxis, :]
            dist_sq = np.sum(diffs ** 2, axis=1)
            best_idx = int(np.argmin(dist_sq))
            indices.append(best_idx)
            self.codebook_usage[l][best_idx] += 1

            e_l = codebook[best_idx]
            z_a += e_l

            # Loss: ||sg(r) - e||^2 + beta * ||r - sg(e)||^2
            quant_loss += float(dist_sq[best_idx] * (1.0 + self.beta_quant))

            # Residual update for next layer
            r_l = r_l - e_l

        return indices, z_a, quant_loss

    def quantize_batch(self, z_hats: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Quantizes batch of vectors of shape (N, dim).
        Returns:
          indices: (N, num_layers)
          quantized: (N, dim)
          avg_quant_loss: float
        """
        n_samples = z_hats.shape[0]
        indices = np.zeros((n_samples, self.num_layers), dtype=np.int32)
        quantized = np.zeros_like(z_hats)
        total_loss = 0.0

        for i in range(n_samples):
            idx_list, z_a, q_loss = self.quantize_vector(z_hats[i])
            indices[i] = idx_list
            quantized[i] = z_a
            total_loss += q_loss

        return indices, quantized, total_loss / max(1, n_samples)

    def get_codebook_usage_stats(self) -> Dict[str, Any]:
        """Computes Average Usage Rate across codebooks (Table 5 in paper)."""
        stats = []
        for l, usage in enumerate(self.codebook_usage):
            active_codes = np.count_nonzero(usage)
            usage_rate = active_codes / self.codebook_size
            stats.append({
                "layer": l + 1,
                "active_codes": active_codes,
                "total_codes": self.codebook_size,
                "usage_rate": usage_rate
            })
        avg_usage = float(np.mean([s["usage_rate"] for s in stats]))
        return {"layers": stats, "average_usage_rate": avg_usage}


class RQVAE:
    """
    Residual-Quantized Variational AutoEncoder for Display Ad Indexing.
    Encodes PLM ad embedding x_a -> latent z_hat -> discrete codes -> reconstructed x_hat.
    """
    def __init__(
        self,
        d_plm: int = 64,
        d_rq: int = 8,
        num_layers: int = 4,
        codebook_size: int = 1024,
        beta_quant: float = 0.25,
        seed: int = 42
    ):
        self.d_plm = d_plm
        self.d_rq = d_rq
        self.num_layers = num_layers
        self.codebook_size = codebook_size
        
        np.random.seed(seed)
        # Encoder weights: W_enc (d_plm, d_rq)
        self.w_enc = np.random.randn(d_plm, d_rq).astype(np.float32) / np.sqrt(d_plm)
        self.b_enc = np.zeros(d_rq, dtype=np.float32)
        
        # RVQ Quantizer
        self.quantizer = ResidualQuantizer(
            num_layers=num_layers,
            codebook_size=codebook_size,
            dim=d_rq,
            beta_quant=beta_quant,
            seed=seed
        )
        
        # Decoder weights: W_dec (d_rq, d_plm)
        self.w_dec = np.random.randn(d_rq, d_plm).astype(np.float32) / np.sqrt(d_rq)
        self.b_dec = np.zeros(d_plm, dtype=np.float32)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Projects dense text embedding into RQ latent space."""
        return np.tanh(np.dot(x, self.w_enc) + self.b_enc)

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Reconstructs text embedding from quantized latent representation."""
        return np.dot(z, self.w_dec) + self.b_dec

    def fit_step(self, x_batch: np.ndarray, lr: float = 0.005) -> Dict[str, float]:
        """
        Executes one forward pass and update step for encoder, decoder, and codebooks.
        Implements Eq. (5) and Eq. (6).
        """
        # Forward
        z_hat = self.encode(x_batch)  # (N, d_rq)
        indices, z_q, quant_loss = self.quantizer.quantize_batch(z_hat)
        x_recon = self.decode(z_q)   # (N, d_plm)

        # Reconstruction loss (Eq. 5)
        recon_loss = float(np.mean((x_batch - x_recon) ** 2))

        # Gradient update for Decoder
        grad_recon = 2.0 * (x_recon - x_batch) / x_batch.shape[0]  # (N, d_plm)
        grad_w_dec = np.dot(z_q.T, grad_recon)
        grad_b_dec = np.sum(grad_recon, axis=0)

        # Gradient update for Encoder
        grad_z_q = np.dot(grad_recon, self.w_dec.T)
        grad_z_hat = grad_z_q * (1.0 - z_hat ** 2)  # derivative of tanh
        grad_w_enc = np.dot(x_batch.T, grad_z_hat)
        grad_b_enc = np.sum(grad_z_hat, axis=0)

        # SGD updates
        self.w_dec -= lr * grad_w_dec
        self.b_dec -= lr * grad_b_dec
        self.w_enc -= lr * grad_w_enc
        self.b_enc -= lr * grad_b_enc

        # Update codebooks toward residual inputs (Vector Quantization EMA / SGD)
        for i in range(x_batch.shape[0]):
            r_l = z_hat[i].copy()
            for l in range(self.num_layers):
                c_idx = indices[i, l]
                e_val = self.quantizer.codebooks[l][c_idx]
                # Pull code vector toward residual
                self.quantizer.codebooks[l][c_idx] += lr * (r_l - e_val)
                r_l -= e_val

        return {
            "recon_loss": recon_loss,
            "quant_loss": quant_loss,
            "total_loss": recon_loss + quant_loss
        }

    def assign_sids(self, embeddings_by_ad: Dict[str, np.ndarray]) -> Dict[str, List[str]]:
        """
        Generates 5-token S-ID lists for all ads:
          - Tokens 1 to M from RQ-VAE: a_i, b_j, c_k, d_m
          - Token M+1 is collision disambiguation code: e_0, e_1...
        """
        level_prefixes = ["a_", "b_", "c_", "d_", "e_", "f_"]
        raw_sids_map: Dict[str, List[str]] = {}
        collision_tracker: Dict[Tuple[str, ...], int] = {}

        # Step 1: Compute M-level S-IDs
        for ad_id, emb in embeddings_by_ad.items():
            z_hat = self.encode(emb[np.newaxis, :])[0]
            indices, _, _ = self.quantizer.quantize_vector(z_hat)
            m_tokens = [f"{level_prefixes[l]}{idx}" for l, idx in enumerate(indices)]
            raw_sids_map[ad_id] = m_tokens

        # Step 2: Disambiguation (Section 3.2):
        # 'To address potential collisions, we introduce an additional code to distinguish them: c_a^(M+1)'
        final_sids_map: Dict[str, List[str]] = {}
        for ad_id, m_tokens in raw_sids_map.items():
            key = tuple(m_tokens)
            disambig_id = collision_tracker.get(key, 0)
            collision_tracker[key] = disambig_id + 1
            
            e_prefix = level_prefixes[self.num_layers] if self.num_layers < len(level_prefixes) else "e_"
            final_tokens = m_tokens + [f"{e_prefix}{disambig_id}"]
            final_sids_map[ad_id] = final_tokens

        return final_sids_map

    @staticmethod
    def compute_collision_metrics(sids_map: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Computes Collision Rate and Max Collision Number (Table 5 in paper).
        Assesses S-ID collisions across the first M levels.
        """
        m_level_counts: Dict[Tuple[str, ...], int] = {}
        for tokens in sids_map.values():
            m_key = tuple(tokens[:-1])
            m_level_counts[m_key] = m_level_counts.get(m_key, 0) + 1

        total_ads = len(sids_map)
        collisions = sum(count - 1 for count in m_level_counts.values() if count > 1)
        collision_rate = collisions / max(1, total_ads)
        max_collision = max(m_level_counts.values()) if m_level_counts else 0

        return {
            "total_ads": total_ads,
            "unique_m_paths": len(m_level_counts),
            "collision_rate": collision_rate,
            "max_collision_number": max_collision
        }
