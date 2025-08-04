from typing import Optional, Dict
import torch
from torch import nn, Tensor
from generative_recommenders.models.base import (
    SequentialEncoderWithLearnedSimilarityModule,
)
from generative_recommenders.models.embedding.base import EmbeddingModule
from generative_recommenders.models.input_preproc.base import (
    InputFeaturesPreprocessorModule,
)
from generative_recommenders.models.output_postproc.base import (
    OutputPostprocessorModule,
)
from generative_recommenders.models.interaction.base import SimilarityModule
from generative_recommenders.utils.sequence_utils import get_current_embeddings


class MLPBaseline(SequentialEncoderWithLearnedSimilarityModule):
    """
    MLP baseline that encodes a user's sequence history as a mean-pooled embedding
    passed through a simple MLP.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        dropout_rate: float,
        embedding_module: EmbeddingModule,
        similarity_module: SimilarityModule,
        input_features_preproc_module: InputFeaturesPreprocessorModule,
        output_postproc_module: OutputPostprocessorModule,
    ) -> None:
        super().__init__(ndp_module=similarity_module)
        self._embedding_module = embedding_module
        self._input_features_preproc = input_features_preproc_module
        self._output_postproc = output_postproc_module

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def debug_str(self) -> str:
        return f"MLPBaseline-d{self.mlp[0].in_features}-h{self.mlp[0].out_features}"

    def get_item_embeddings(self, item_ids: Tensor) -> Tensor:
        return self._embedding_module.get_item_embeddings(item_ids)

    def generate_user_embeddings(
        self,
        past_ids: Tensor,
        past_embeddings: Tensor,
        past_lengths: Tensor,
        past_payloads: Dict[str, Tensor],
    ) -> Tensor:
        """
        Generates user representation by mean pooling valid item embeddings, followed by an MLP and output post-processing.
        """
        _, seq_embeds, valid_mask = self._input_features_preproc(
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_lengths=past_lengths,
            past_payloads=past_payloads,
        )
        masked = seq_embeds * valid_mask
        summed = masked.sum(dim=1)
        lengths = valid_mask.sum(dim=1).clamp(min=1e-6)
        pooled = summed / lengths  # [B, D]
        return self._output_postproc(self.mlp(pooled))  # [B, D]

    def encode(
        self,
        past_lengths: Tensor,
        past_ids: Tensor,
        past_embeddings: Tensor,
        past_payloads: Dict[str, Tensor],
    ) -> Tensor:
        """
        Returns the current embedding for evaluation — used for similarity ranking.
        """
        x = self.generate_user_embeddings(past_ids, past_embeddings, past_lengths, past_payloads)
        return get_current_embeddings(past_lengths, x)

    def forward(
        self,
        past_lengths: Tensor,
        past_ids: Tensor,
        past_embeddings: Tensor,
        past_payloads: Dict[str, Tensor],
        batch_id: Optional[int] = None,
    ) -> Tensor:
        """
        Returns sequence output [B, N, D] for loss computation. Since the MLP outputs [B, D],
        we repeat across sequence dimension.
        """
        B, N, _ = past_embeddings.shape
        pooled = self.generate_user_embeddings(past_ids, past_embeddings, past_lengths, past_payloads)  # [B, D]
        return pooled.unsqueeze(1).expand(-1, N, -1)  # [B, N, D]

    def predict(
        self,
        past_ids: Tensor,
        past_ratings: Tensor,
        past_timestamps: Tensor,
        next_timestamps: Tensor,
        target_ids: Tensor,
        batch_id: Optional[int] = None,
    ) -> Tensor:
        """
        Optional: For direct dot-product scoring in AR prediction (like SASRec).
        """
        return self.interaction(
            self.encode(
                past_ids=past_ids,
                past_embeddings=self.get_item_embeddings(past_ids),
                past_lengths=past_ids.new_full((past_ids.size(0),), past_ids.size(1)),
                past_payloads={},
            ),
            target_ids,
        )
