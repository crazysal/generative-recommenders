'''
MLPBaseline(
  (_ndp_module): DotProductSimilarity()
  (_embedding_module): LocalEmbeddingModule(
    (_item_emb): Embedding(3953, 50, padding_idx=0)
  )
  (_input_features_preproc): LearnablePositionalEmbeddingInputFeaturesPreprocessor(
    (_pos_emb): Embedding(211, 50)
    (_emb_dropout): Dropout(p=0.2, inplace=False)
  )
  (_output_postproc): L2NormEmbeddingPostprocessor()
  (mlp): Sequential(
    (0): Linear(in_features=50, out_features=50, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.2, inplace=False)
    (3): Linear(in_features=50, out_features=50, bias=True)
  )
)
MLPBaseline-d50-h50

🧠 Model Summary:
_embedding_module._item_emb.weight                           | [3953, 50] | 197650 params
_input_features_preproc._pos_emb.weight                      | [211, 50] | 10550 params
mlp.0.weight                                                 | [50, 50] | 2500 params
mlp.0.bias                                                   | [50] | 50 params
mlp.3.weight                                                 | [50, 50] | 2500 params
mlp.3.bias                                                   | [50] | 50 params

🔢 Total Trainable Parameters: 213,300
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


class MLPBaseline(SequentialEncoderWithLearnedSimilarityModule):
    """
    MLP baseline for recommendation.
    This model treats user history as an unordered set, averaging embeddings.
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

        self.mlp = nn.Sequential(
            nn.Linear(self._embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, self._embedding_dim),
        )

    def debug_str(self) -> str:
        return f"MLPBaseline-d{self._embedding_dim}-h{self.mlp[0].out_features}"

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self._embedding_module.get_item_embeddings(item_ids)

    def encode(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,  # [B, N] x int64
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Encode user history as mean pooled embeddings passed through MLP.
        """
        _, user_embeddings, valid_mask = self._input_features_preproc(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
        )

        masked_embeddings = user_embeddings * valid_mask
        sum_embeddings = masked_embeddings.sum(dim=1)  # [B, D]
        valid_lengths = valid_mask.sum(dim=1).clamp(min=1e-6)  # avoid div by 0
        mean_embeddings = self._output_postproc(sum_embeddings / valid_lengths)  # [B, D]

        # return self.mlp(mean_embeddings).unsqueeze(1).repeat(1, self._max_sequence_length, 1)  # shape: [B, N, D]
        return self.mlp(mean_embeddings)  # shape: [B, X]

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        batch_id: Optional[int] = None,
    ) -> torch.Tensor:
        # Repeat the pooled embedding for all time steps → [B, N, D]
        B, N, _ = past_embeddings.shape
        pooled = self.encode(past_lengths, past_ids, past_embeddings, past_payloads)  # [B, D]
        return pooled.unsqueeze(1).repeat(1, N, 1)  # [B, N, D]

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
