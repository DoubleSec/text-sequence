"""Contains tools for initial data preprocessing and then data loading during training."""

import torch
from torch.utils.data import Dataset


def initial_prep(path: str, *args, **kwargs) -> str:
    """Do one-time data preparation.

    This is intended for things that only need to happen once or infrequently:
    - downloading data
    - slow preprocessing
    - splitting
    - etc.

    The return value is intended to be where the prepared file was written."""
    raise NotImplementedError


class TrainingDataset(Dataset):
    """Torch Dataset class for training."""

    def __init__(self, path: str, *args, **kwargs):
        """Load a dataset and do any prep required.

        If at all possible, this should be deterministic, given a specific file.
        """
        super().__init__()
        raise NotImplementedError

    def __len__(self) -> int:
        """Obvious"""
        raise NotImplementedError

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        """Return a single instance of the training dataset, as a dictionary."""
        raise NotImplementedError
