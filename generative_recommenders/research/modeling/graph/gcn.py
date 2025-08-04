import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from generative_recommenders.research.modeling.sequential.embedding_modules import EmbeddingModule
from generative_recommenders.research.modeling.sequential.input_features_preprocessors import InputFeaturesPreprocessorModule
from generative_recommenders.research.modeling.sequential.output_postprocessors import OutputPostprocessorModule
from generative_recommenders.research.modeling.similarity_module import SequentialEncoderWithLearnedSimilarityModule
from generative_recommenders.research.rails.similarities.module import SimilarityModule
from generative_recommenders.research.modeling.sequential.utils import get_current_embeddings


def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """
    Symmetric normalization of adjacency matrix: D^{-1/2} A D^{-1/2}
    adj: [B, N, N] unnormalized adjacency matrix
    """
    deg = adj.sum(dim=-1)  # [B, N]
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt = torch.nan_to_num(deg_inv_sqrt, nan=0.0, posinf=0.0, neginf=0.0)
    deg_inv_sqrt = deg_inv_sqrt.unsqueeze(-1)  # [B, N, 1]
    return adj * deg_inv_sqrt * deg_inv_sqrt.transpose(1, 2)  # [B, N, N]



def get_gcn_current_embeddings(
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

class GCNBaseline(SequentialEncoderWithLearnedSimilarityModule):
    """
    Graph Convolutional Network (GCN) baseline for session-based recommendation.
    Applies symmetric normalized adjacency for message passing over item sequences.
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
        self._pooling = pooling.lower()
        self._verbose = verbose

        self._embedding_module = embedding_module
        self._input_features_preproc = input_features_preproc_module
        self._output_postproc = output_postproc_module

        self.gcn_layers = nn.ModuleList([
            nn.Linear(embedding_dim, hidden_dim),
            nn.Linear(hidden_dim, embedding_dim),
        ])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def debug_str(self) -> str:
        return (
            f"GCNBaseline("
            f"embedding_dim={self._embedding_dim}, "
            f"hidden_dim={self.gcn1.out_features}, "
            f"pooling='{self._pooling}'"
            f")"
        )

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self._embedding_module.get_item_embeddings(item_ids)


    def _run_one_layer(
        self,
        i: int,
        x: torch.Tensor,          # [B, N, D]
        adj: torch.Tensor,        # [B, N, N]
        valid_mask: torch.Tensor  # [B, N] float mask
    ) -> torch.Tensor:
        """
        Run one GCN layer: normalize adjacency, message passing, transformation.

        Args:
            i: GCN layer index
            x: [B, N, D] input features
            adj: [B, N, N] adjacency matrix
            valid_mask: [B, N] float mask for valid nodes

        Returns:
            [B, N, D] updated features
        """
        norm_adj = normalize_adjacency(adj)  # [B, N, N]
        x = torch.bmm(norm_adj, x)           # message passing
        x = self.gcn_layers[i](x)            # layer transformation

        if i < len(self.gcn_layers) - 1:
            x = self.relu(x)
            x = self.dropout(x)

        x = x * valid_mask.unsqueeze(-1)     # [B, N, D]
        return x


    def generate_user_embeddings(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Run GCN over the session graph with adjacency-based message passing.

        Returns:
            [B, N, D] final sequence embeddings
        """
        past_lengths, x, valid_mask = self._input_features_preproc(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
        )  # x: [B, N, D], valid_mask: [B, N, 1]

        adj = past_payloads.get("adj_matrix", None)
        if adj is None:
            raise ValueError("GCNBaseline expects 'adj_matrix' in past_payloads")

        # Mask padded rows and cols in adjacency
        valid_mask_flat = valid_mask.squeeze(-1).float()   # [B, N]
        adj = adj * valid_mask_flat.unsqueeze(1) * valid_mask_flat.unsqueeze(2)  # [B, N, N]

        # Run all GCN layers
        for i in range(len(self.gcn_layers)):
            x = self._run_one_layer(i, x, adj, valid_mask_flat)

        return self._output_postproc(x)  # [B, N, D]

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        batch_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass of GCNBaseline.

        Args:
            past_lengths: [B] — actual lengths of each user sequence.
            past_ids: [B, N] — item IDs in user history.
            past_embeddings: [B, N, D] — precomputed item embeddings.
            past_payloads: dict — expected to include:
                - "adj_matrix": [B, N, N] — session-level adjacency matrices per sample.

        Returns:
            Tensor of shape [B, N, D] — GCN-encoded sequence embeddings.
        """
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
        Returns a pooled embedding representation of the user sequence.

        Args:
            past_lengths: [B] x int
            past_ids: [B, N] x int
            past_embeddings: [B, N, D] x float
            past_payloads: dict with keys like "adj_matrix" [B, N, N]

        Returns:
            [B, D] pooled user embeddings
        """
        encoded_seq_embeddings = self.generate_user_embeddings(
            past_lengths,
            past_ids,
            past_embeddings,
            past_payloads,
        )  # [B, N, D]

        return self._pool(past_lengths, encoded_seq_embeddings)

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


    def _pool(self, lengths: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Pooling over the sequence dimension [B, N, D] → [B, D]
        """
        if self._pooling == "last":
            return get_gcn_current_embeddings(lengths, x)
        elif self._pooling == "mean":
            mask = torch.arange(x.size(1), device=lengths.device)[None, :] < lengths[:, None]
            return (x * mask.unsqueeze(-1).float()).sum(1) / lengths.unsqueeze(1).clamp(min=1)
        elif self._pooling == "sum":
            mask = torch.arange(x.size(1), device=lengths.device)[None, :] < lengths[:, None]
            return (x * mask.unsqueeze(-1).float()).sum(1)
        else:
            raise ValueError(f"Unknown pooling strategy: {self._pooling}")

