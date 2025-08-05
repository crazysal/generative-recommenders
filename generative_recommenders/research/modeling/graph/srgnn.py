import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from generative_recommenders.research.modeling.sequential.embedding_modules import (
    EmbeddingModule,
)
from generative_recommenders.research.modeling.sequential.input_features_preprocessors import (
    InputFeaturesPreprocessorModule,
)
from generative_recommenders.research.modeling.sequential.output_postprocessors import (
    OutputPostprocessorModule,
)

from generative_recommenders.research.modeling.similarity_module import (
    SequentialEncoderWithLearnedSimilarityModule,
)
from generative_recommenders.research.rails.similarities.module import SimilarityModule


def get_last_valid_token_embedding(
    lengths: torch.Tensor,
    encoded_embeddings: torch.Tensor,
) -> torch.Tensor:
    """
    Selects the embedding corresponding to the last valid item in each sequence.
    Args:
        lengths: [B] — number of valid tokens in each session
        encoded_embeddings: [B, N, D] — full node embeddings
    Returns:
        [B, D] — last-item node embedding
    """
    B, N, D = encoded_embeddings.shape
    idx = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, D)  # [B, 1, D]
    return encoded_embeddings.gather(dim=1, index=idx).squeeze(1)  # [B, D]


class SRGNNBaseline(SequentialEncoderWithLearnedSimilarityModule):
    """
    SR-GNN baseline for session-based recommendation.
    Uses GRU-based gated graph neural networks for message passing over session graphs.
    """

    def __init__(
        self,
        max_sequence_len: int,
        max_output_len: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout_rate: float,
        embedding_module: EmbeddingModule,
        similarity_module: SimilarityModule,
        input_features_preproc_module: InputFeaturesPreprocessorModule,
        output_postproc_module: OutputPostprocessorModule,
        pooling: str = "last",
        verbose: bool = False,
    ) -> None:
        super().__init__(ndp_module=similarity_module)

        self._max_sequence_length = max_sequence_len + max_output_len
        self._embedding_dim = embedding_dim
        self._hidden_dim = hidden_dim
        self._pooling = pooling.lower()
        self._verbose = verbose

        self._embedding_module = embedding_module
        self._input_features_preproc = input_features_preproc_module
        self._output_postproc = output_postproc_module

        # Linear before GRU to match hidden_dim
        self.linear_in = nn.Linear(embedding_dim, hidden_dim)
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)

        # Linear after GRU to restore original embedding dimension
        self.linear_out = nn.Linear(hidden_dim, embedding_dim)

        self.dropout = nn.Dropout(p=dropout_rate)

    def debug_str(self) -> str:
        return (
            f"SRGNNBaseline("
            f"embedding_dim={self._embedding_dim}, "
            f"hidden_dim={self._hidden_dim}, "
            f"pooling='{self._pooling}'"
            f")"
        )

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self._embedding_module.get_item_embeddings(item_ids)

    def _run_one_layer(
        self,
        x: torch.Tensor,          # [B, N, D] input node features (embedding_dim)
        adj: torch.Tensor,        # [B, N, N] adjacency matrix
        valid_mask: torch.Tensor, # [B, N] float mask for valid tokens
    ) -> torch.Tensor:
        """
        SR-GNN message passing + GRU-based node update.
        """
        # 1. Linear transform input to hidden space
        x_proj = self.linear_in(x)   # [B, N, hidden_dim]

        # 2. Message passing: aggregate neighbor features
        neighbor_agg = torch.bmm(adj, x_proj)  # [B, N, hidden_dim]

        # 3. GRU-style gated update: reshape and apply GRUCell
        B, N, H = neighbor_agg.shape
        x_flat = x_proj.view(B * N, H)
        agg_flat = neighbor_agg.view(B * N, H)
        updated = self.gru_cell(agg_flat, x_flat)  # [B*N, H]
        updated = updated.view(B, N, H)

        # 4. Project back to original embedding dim
        x_out = self.linear_out(updated)  # [B, N, embedding_dim]
        x_out = self.dropout(x_out)

        # 5. Mask padded tokens
        x_out = x_out * valid_mask.unsqueeze(-1)  # [B, N, D]
        return x_out

    def generate_user_embeddings(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Run SR-GNN over the session graph with gated message passing using GRU.
        """
        # 1. Input preprocessing (positional encoding, dropout, masking)
        past_lengths, x, valid_mask = self._input_features_preproc(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
        )  # x: [B, N, D], valid_mask: [B, N, 1]

        adj = past_payloads.get("adj_matrix", None)
        if adj is None:
            raise ValueError("SRGNNBaseline expects 'adj_matrix' in past_payloads")

        # 2. Remove padded tokens from contributing in graph
        valid_mask_flat = valid_mask.squeeze(-1).float()  # [B, N]
        adj = adj * valid_mask_flat.unsqueeze(1) * valid_mask_flat.unsqueeze(2)  # [B, N, N]

        # 3. Run gated graph update
        x_encoded = self._run_one_layer(x, adj, valid_mask_flat)  # [B, N, D]

        return self._output_postproc(x_encoded)  # [B, N, D]

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        batch_id: Optional[int] = None,
    ) -> torch.Tensor:
        return self.generate_user_embeddings(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
        )
    
    def encode(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Returns a pooled embedding representation of the user session graph.

        SR-GNN performs message passing via gated GRU cells and returns an aggregate node embedding,
        typically based on the last item (as in session-based GRNN literature).

        Args:
            past_lengths: [B] — number of valid items per session
            past_ids: [B, N] — item IDs in each session
            past_embeddings: [B, N, D] — raw item embeddings
            past_payloads: dict — must contain "adj_matrix": [B, N, N]

        Returns:
            [B, D] — pooled session representation
        """
        # 1. Run gated message passing over graph
        encoded_seq_embeddings = self.generate_user_embeddings(
            past_lengths,
            past_ids,
            past_embeddings,
            past_payloads,
        )  # [B, N, D]

        # 2. Pooling mechanism (e.g., last, mean, max, etc.)
        return self._pool(past_lengths, encoded_seq_embeddings)  # [B, D]

    def _pool(self, lengths: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Pooling over the sequence dimension [B, N, D] → [B, D]
        """
        B, N, D = x.shape

        if self._pooling == "last":
            return get_last_valid_token_embedding(lengths, x)

        elif self._pooling == "mean":
            mask = torch.arange(N, device=lengths.device)[None, :] < lengths[:, None]
            return (x * mask.unsqueeze(-1).float()).sum(1) / lengths.unsqueeze(1).clamp(min=1)

        elif self._pooling == "sum":
            mask = torch.arange(N, device=lengths.device)[None, :] < lengths[:, None]
            return (x * mask.unsqueeze(-1).float()).sum(1)

        elif self._pooling == "attention":
            # Step 1: get query vector (last valid token per sequence) → [B, D]
            q = get_last_valid_token_embedding(lengths, x)  # [B, D]

            # Step 2: compute attention weights → [B, N]
            attn_scores = (x * q.unsqueeze(1)).sum(dim=-1)  # dot product: q · k_i

            # Mask out invalid tokens
            mask = torch.arange(N, device=lengths.device)[None, :] < lengths[:, None]  # [B, N]
            attn_scores[~mask] = float('-inf')

            attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, N]

            # Step 3: weighted sum → [B, D]
            pooled = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # [B, D]
            return pooled

        else:
            raise ValueError(f"Unknown pooling strategy: {self._pooling}")
        
    def predict(
        self,
        past_ids: torch.Tensor,
        past_ratings: torch.Tensor,
        past_timestamps: torch.Tensor,
        next_timestamps: torch.Tensor,
        target_ids: torch.Tensor,
        batch_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Predict scores for target items based on encoded session representation.

        Args:
            past_ids: [B, N] x int64 — item IDs in the past session
            past_ratings: [B, N] x int64 — optional, may be unused
            past_timestamps: [B, N] x int64 — optional, may be unused
            next_timestamps: [B, N] x int64 — optional, may be unused
            target_ids: [B, X] x int64 — item IDs to score

        Returns:
            [B, X] float — similarity scores between current user embedding and target items
        """
        B, N = past_ids.shape
        past_embeddings = self._embedding_module.get_item_embeddings(past_ids)

        # Simple temporal adjacency: fully connected where past_ids ≠ 0
        mask = (past_ids != 0).float()  # [B, N]
        adj_matrix = mask.unsqueeze(1) * mask.unsqueeze(2)  # [B, N, N]

        user_embeddings = self.encode(
            past_lengths=past_ids.new_full((B,), N),
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads={"adj_matrix": adj_matrix},
        )  # [B, D]

        return self.interaction(user_embeddings, target_ids)  # [B, X]

