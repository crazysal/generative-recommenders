'''
GCNBaseline(
  (_ndp_module): DotProductSimilarity()
  (_embedding_module): LocalEmbeddingModule(
    (_item_emb): Embedding(3953, 50, padding_idx=0)
  )
  (_input_features_preproc): LearnablePositionalEmbeddingInputFeaturesPreprocessor(
    (_pos_emb): Embedding(211, 50)
    (_emb_dropout): Dropout(p=0.2, inplace=False)
  )
  (_output_postproc): L2NormEmbeddingPostprocessor()
  (gcn1): Linear(in_features=50, out_features=256, bias=True)
  (gcn2): Linear(in_features=256, out_features=50, bias=True)
  (relu): ReLU()
  (dropout): Dropout(p=0.1, inplace=False)
)
GCNBaseline-d50-h256

🧠 Model Summary:
_embedding_module._item_emb.weight                           | [3953, 50] | 197650 params
_input_features_preproc._pos_emb.weight                      | [211, 50] | 10550 params
gcn1.weight                                                  | [256, 50] | 12800 params
gcn1.bias                                                    | [256] | 256 params
gcn2.weight                                                  | [50, 256] | 12800 params
gcn2.bias                                                    | [50] | 50 params

🔢 Total Trainable Parameters: 234,106

'''
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from generative_recommenders.research.modeling.sequential.embedding_modules import EmbeddingModule
from generative_recommenders.research.modeling.sequential.input_features_preprocessors import (
    InputFeaturesPreprocessorModule,
)
from generative_recommenders.research.modeling.sequential.output_postprocessors import (
    OutputPostprocessorModule,
)
from generative_recommenders.research.modeling.sequential.utils import get_current_embeddings
from generative_recommenders.research.modeling.similarity_module import (
    SequentialEncoderWithLearnedSimilarityModule,
)
from generative_recommenders.research.rails.similarities.module import SimilarityModule

class GCNBaseline(SequentialEncoderWithLearnedSimilarityModule):
    """
    Simple GCN baseline for session-based recommendation.
    Constructs a user graph per sequence and uses GCN to encode item interactions.
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
        verbose: bool = False,
    ) -> None:
        super().__init__(ndp_module=similarity_module)
        self._max_sequence_length: int = max_sequence_len + max_output_len

        self._embedding_module = embedding_module
        self._embedding_dim = embedding_dim
        self._input_features_preproc = input_features_preproc_module
        self._output_postproc = output_postproc_module
        self._verbose = verbose

        self.gcn1 = nn.Linear(self._embedding_dim, hidden_dim)
        self.gcn2 = nn.Linear(hidden_dim, self._embedding_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self._pooling = pooling.lower()

    def debug_str(self) -> str:
        return f"GCNBaseline-d{self._embedding_dim}-h{self.gcn1.out_features}"

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self._embedding_module.get_item_embeddings(item_ids)

    def encode(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        B, N, D = past_embeddings.shape
        _, user_embeddings, valid_mask = self._input_features_preproc(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
        )  # [B, N, D], [B, N, 1]

        masked_embeddings = user_embeddings * valid_mask  # [B, N, D]

        # Identity adjacency for now (placeholder): [B, N, N]
        adj = torch.eye(N, device=masked_embeddings.device).unsqueeze(0).repeat(B, 1, 1)  # [B, N, N]

        x = torch.bmm(adj, masked_embeddings)  # [B, N, D]
        x = self.relu(self.gcn1(x))
        x = self.dropout(x)
        x = torch.bmm(adj, x)
        x = self.gcn2(x)  # [B, N, D]

        return x  # [B, N, D]

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        batch_id: Optional[int] = None,
    ) -> torch.Tensor:
        return self.encode(past_lengths, past_ids, past_embeddings, past_payloads)

    def predict(
        self,
        past_ids: torch.Tensor,
        past_ratings: torch.Tensor,
        past_timestamps: torch.Tensor,
        next_timestamps: torch.Tensor,
        target_ids: torch.Tensor,
        batch_id: Optional[int] = None,
    ) -> torch.Tensor:
        return self.interaction(
            self.encode(
                past_lengths=past_ids.new_full((past_ids.size(0),), past_ids.size(1)),
                past_ids=past_ids,
                past_embeddings=self._embedding_module.get_item_embeddings(past_ids),
                past_payloads={},
            ),
            target_ids,
        )
    def _pool(self, lengths: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self._pooling == "last":
            return get_current_embeddings(lengths, x)
        elif self._pooling == "mean":
            mask = torch.arange(x.size(1), device=lengths.device)[None, :] < lengths[:, None]
            return (x * mask.unsqueeze(-1).float()).sum(1) / lengths.unsqueeze(1).clamp(min=1)
        elif self._pooling == "sum":
            mask = torch.arange(x.size(1), device=lengths.device)[None, :] < lengths[:, None]
            return (x * mask.unsqueeze(-1).float()).sum(1)
        elif self._pooling == "srgnn":
            # Example SR-GNN-style fusion
            last = get_current_embeddings(lengths, x)
            mean = x.mean(1)
            return F.relu(self.w1(last) + self.w2(mean))
        else:
            raise ValueError(f"Unknown pooling strategy: {self._pooling}")
