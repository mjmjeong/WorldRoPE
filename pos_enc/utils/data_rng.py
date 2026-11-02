"""Deterministic, model-independent random streams for training data."""

import random

import numpy as np
import torch


_MAX_TRAIN_SAMPLES_PER_RANK = 2**31 - 1


def indexed_sample_seed(data_seed: int, world_rank: int, index: int) -> int:
    """Map a logical sample index to a stable 32-bit RNG seed."""
    mask = 2**64 - 1
    value = (
        int(data_seed)
        + int(world_rank) * 0x9E3779B97F4A7C15
        + int(index)
    ) & mask
    # SplitMix64 finalizer avoids correlations between adjacent logical IDs.
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & 0xFFFFFFFF


class DeterministicSampleDataset(torch.utils.data.Dataset):
    """Make random ``Dataset.__getitem__`` calls stateless and replayable.

    Each logical sample index receives its own Python, NumPy and CPU-Torch
    seed.  The result is independent of model initialization, DataLoader
    worker assignment, prefetch timing and PE/model architecture.
    """

    def __init__(self, dataset, data_seed: int, world_rank: int) -> None:
        if len(dataset) <= 0:
            raise ValueError("Cannot wrap an empty training dataset.")
        self.dataset = dataset
        self.data_seed = int(data_seed)
        self.world_rank = int(world_rank)

    def __len__(self) -> int:
        # Avoid ordinary step-based training ever wrapping to a second epoch.
        return _MAX_TRAIN_SAMPLES_PER_RANK

    def __getitem__(self, index: int):
        index = int(index)
        seed = indexed_sample_seed(self.data_seed, self.world_rank, index)
        random.seed(seed)
        np.random.seed(seed)
        torch.random.default_generator.manual_seed(seed)
        return self.dataset[index % len(self.dataset)]


class ResumableSequentialSampler(torch.utils.data.Sampler):
    """Sequential logical sample IDs with an explicit checkpoint offset."""

    def __init__(self, num_samples: int = _MAX_TRAIN_SAMPLES_PER_RANK) -> None:
        self.num_samples = int(num_samples)
        self.start_index = 0

    def set_start_index(self, start_index: int) -> None:
        start_index = int(start_index)
        if not 0 <= start_index < self.num_samples:
            raise ValueError(
                f"Training-data start index {start_index} is outside "
                f"[0, {self.num_samples})."
            )
        self.start_index = start_index

    def __iter__(self):
        return iter(range(self.start_index, self.num_samples))

    def __len__(self) -> int:
        return self.num_samples - self.start_index


class LegacyScheduleBatchSampler(torch.utils.data.Sampler):
    """Yield the original 16-worker loader's batches, including epoch tails."""

    def __init__(
        self,
        max_steps: int,
        scene_count: int = 66_033,
        batch_size: int = 8,
    ) -> None:
        self.max_steps = int(max_steps)
        self.scene_count = int(scene_count)
        self.batch_size = int(batch_size)
        self.batches_per_epoch = (
            self.scene_count + self.batch_size - 1
        ) // self.batch_size
        self.start_step = 0

    def set_start_step(self, start_step: int) -> None:
        start_step = int(start_step)
        if not 0 <= start_step <= self.max_steps:
            raise ValueError(
                f"Legacy schedule start step {start_step} is outside "
                f"[0, {self.max_steps}]."
            )
        self.start_step = start_step

    def _batch_size_at_step(self, step: int) -> int:
        batch_in_epoch = int(step) % self.batches_per_epoch
        remaining = self.scene_count - batch_in_epoch * self.batch_size
        return min(self.batch_size, remaining)

    def __iter__(self):
        for step in range(self.start_step, self.max_steps + 1):
            logical_start = step * self.batch_size
            yield list(
                range(logical_start, logical_start + self._batch_size_at_step(step))
            )

    def __len__(self) -> int:
        return self.max_steps + 1 - self.start_step


class FixedScheduleBatchSampler(torch.utils.data.Sampler):
    """Yield exact fixed-size batches for an immutable step/slot schedule."""

    def __init__(self, max_steps: int, batch_size: int = 8) -> None:
        self.max_steps = int(max_steps)
        self.batch_size = int(batch_size)
        self.start_step = 0
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

    def set_start_step(self, start_step: int) -> None:
        start_step = int(start_step)
        if not 0 <= start_step <= self.max_steps:
            raise ValueError(
                f"Fixed schedule start step {start_step} is outside "
                f"[0, {self.max_steps}]."
            )
        self.start_step = start_step

    def __iter__(self):
        for step in range(self.start_step, self.max_steps + 1):
            logical_start = step * self.batch_size
            yield list(range(logical_start, logical_start + self.batch_size))

    def __len__(self) -> int:
        return self.max_steps + 1 - self.start_step


def make_dataloader_generator(data_seed: int, world_rank: int) -> torch.Generator:
    """Create a data-only RNG stream independent of model initialization."""
    generator = torch.Generator()
    generator.manual_seed(int(data_seed) + int(world_rank))
    return generator


def seed_dataloader_worker(_: int) -> None:
    """Seed Python and NumPy from the dedicated DataLoader worker seed."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
