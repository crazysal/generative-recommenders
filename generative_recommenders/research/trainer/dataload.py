import os
from typing import Optional, Tuple

import gin
import torch

@gin.configurable
def create_data_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    world_size: int,
    rank: int,
    shuffle: bool,
    prefetch_factor: int = 128,
    num_workers: Optional[int] = os.cpu_count(),
    drop_last: bool = False,
    collate_fn=None,  # ✅ Added to support custom collate function
) -> Tuple[
    Optional[torch.utils.data.distributed.DistributedSampler[torch.utils.data.Dataset]],
    torch.utils.data.DataLoader,
]:
    if shuffle:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=0,
            drop_last=drop_last,
        )
    else:
        sampler = None

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers or 0,
        sampler=sampler,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn,  # ✅ Injected custom collate function here
    )

    return sampler, data_loader
