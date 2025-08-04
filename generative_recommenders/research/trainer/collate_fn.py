from typing import List, Dict
import torch

def graph_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    batched = {}

    keys = batch[0].keys()
    for key in keys:
        if key == "adj_matrix":
            batched[key] = torch.stack([x[key] for x in batch], dim=0)  # [B, N, N]
        else:
            # Stack scalars and vectors normally
            if isinstance(batch[0][key], torch.Tensor):
                batched[key] = torch.stack([x[key] for x in batch], dim=0)
            else:
                batched[key] = [x[key] for x in batch]  # fallback for non-tensor entries

    return batched
