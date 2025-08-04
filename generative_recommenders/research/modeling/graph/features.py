from typing import Dict, Tuple, Optional
import torch

class SequentialFeatures(NamedTuple):
    past_lengths: torch.Tensor        # [B]
    past_ids: torch.Tensor            # [B, N]
    past_embeddings: Optional[torch.Tensor]  # optional [B, N, D]
    past_payloads: Dict[str, torch.Tensor]   # e.g. timestamps, ratings, adj_matrix

def seq_features_from_row(
    row: Dict[str, torch.Tensor],
    device: int,
    max_output_length: int = 0,
) -> Tuple[SequentialFeatures, torch.Tensor, torch.Tensor]:
    """
    Converts raw dataloader row to SequentialFeatures + targets, with optional GCN support.

    Args:
        row: Dict from DataLoader containing sequence data and optionally 'adj_matrix'
        device: target device (CPU/GPU)
        max_output_length: used by autoregressive models like HSTU

    Returns:
        SequentialFeatures, target_ids, target_ratings
    """
    historical_lengths = row["history_lengths"].to(device)
    historical_ids = row["historical_ids"].to(device)
    historical_ratings = row["historical_ratings"].to(device)
    historical_timestamps = row["historical_timestamps"].to(device)
    target_ids = row["target_ids"].to(device).unsqueeze(1)
    target_ratings = row["target_ratings"].to(device).unsqueeze(1)
    target_timestamps = row["target_timestamps"].to(device).unsqueeze(1)

    if max_output_length > 0:
        B = historical_lengths.size(0)
        historical_ids = torch.cat([
            historical_ids,
            torch.zeros((B, max_output_length), dtype=historical_ids.dtype, device=device)
        ], dim=1)

        historical_ratings = torch.cat([
            historical_ratings,
            torch.zeros((B, max_output_length), dtype=historical_ratings.dtype, device=device)
        ], dim=1)

        historical_timestamps = torch.cat([
            historical_timestamps,
            torch.zeros((B, max_output_length), dtype=historical_timestamps.dtype, device=device)
        ], dim=1)

        # Append next timestamp at the correct position
        historical_timestamps.scatter_(
            dim=1,
            index=historical_lengths.view(-1, 1),
            src=target_timestamps.view(-1, 1),
        )

    # Construct past_payloads
    past_payloads = {
        "timestamps": historical_timestamps,
        "ratings": historical_ratings,
    }

    # If graph adjacency is available, include it
    if "adj_matrix" in row:
        past_payloads["adj_matrix"] = row["adj_matrix"].to(device)  # [B, N, N]

    features = SequentialFeatures(
        past_lengths=historical_lengths,
        past_ids=historical_ids,
        past_embeddings=None,
        past_payloads=past_payloads,
    )

    return features, target_ids, target_ratings
