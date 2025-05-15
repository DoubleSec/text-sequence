"""Contains tools for initial data preprocessing and then data loading during training."""

from typing import Any
from itertools import batched, chain

import numpy as np
import torch
from torch.utils.data import Dataset
from pybaseball import statcast
from logzero import logger
import polars as pl
import morphers
from transformers import AutoTokenizer, AutoModel

MORPHER_MAPPING = {
    "categorical": morphers.Integerizer,
    "numeric": morphers.Normalizer,
}


def pad_tensor_dict(tensor_dict, max_length, return_mask: bool = True):
    """
    Pad a tensor dict up to the max length.
    Padded Location = 0
    """

    init_length = next(iter(tensor_dict.values())).shape[0]
    if init_length >= max_length:
        padded_tensor_dict = {k: v[:max_length] for k, v in tensor_dict.items()}
    else:
        padded_tensor_dict = {
            k: torch.nn.functional.pad(v, [0, max_length - init_length], value=0)
            for k, v in tensor_dict.items()
        }
    # FALSE IS NOT PAD, TRUE IS PAD
    if return_mask:
        pad_mask = torch.ones([max_length], dtype=torch.bool)
        pad_mask[: min(init_length, max_length)] = False
        pad_mask = torch.where(pad_mask, float("-inf"), 0.0)
        return padded_tensor_dict, pad_mask
    else:
        return padded_tensor_dict


def initial_prep(
    path: str,
    start_date: str,
    end_date: str,
) -> str:
    """Do one-time data preparation.
    Here we'll just download statcast data and save it.
    # We'll also embed the outcomes with modernbert
    """

    init_data = statcast(start_dt=start_date, end_dt=end_date)
    init_data = (
        pl.DataFrame(init_data)
        .with_columns(
            outcome_text=pl.concat_str(
                pl.col("des"), pl.lit(" ("), pl.col("player_name"), pl.lit(")")
            )
            .alias("outcome_text")
            .tolist()
        )
        .unique()
    )

    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5").to("cuda:0")
    model.eval()

    def make_embedding(text: str):
        encoded_input = tokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        )

        with torch.inference_mode():
            model_output = model(
                **{k: v.to(model.device) for k, v in encoded_input.items()}
            )
            embeddings = model_output[0][:, 0]
            return (
                torch.nn.functional.normalize(embeddings, p=2, dim=1)
                .numpy(force=True)
                .squeeze()
            )

    data = init_data.select(
        "game_pk",
        "at_bat_number",
        pl.concat_str(
            pl.col("des"), pl.lit(" ("), pl.col("player_name"), pl.lit(")")
        ).alias("outcome_text"),
    ).unique()

    embeddings = list(
        chain.from_iterable(
            make_embedding(batch) for batch in batched(data["outcome_text"], n=2048)
        )
    )
    embeddings = np.stack(embeddings, axis=0)
    data = data.with_columns(embeddings=embeddings)
    data.write_parquet("data/embeddings.parquet")

    return path


class TrainingDataset(Dataset):
    """Torch Dataset class for training."""

    def __init__(
        self,
        path: str,
        embedding_path: str,
        sequence_length: int,
        morpher_spec: (
            dict[str, tuple[type[morphers.base.base.Morpher], dict[str, Any]]] | None
        ) = None,
    ):
        """Load a dataset and do any prep required.

        If at all possible, this should be deterministic, given a specific file.
        """
        super().__init__()
        self.sequence_length = sequence_length

        df = pl.read_parquet(path)

        if morpher_spec is not None:

            # Map feature types to morphers.
            morpher_spec = {
                k: (MORPHER_MAPPING[v[0]], v[1]) for k, v in morpher_spec.items()
            }
            self.morphers = {
                column: morpher.from_data(df[column], **morpher_kwargs)
                for column, (morpher, morpher_kwargs) in morpher_spec.items()
            }

        embeddings = pl.read_parquet(embedding_path)

        self.sequences = (
            df.select(
                "game_pk",
                "at_bat_number",
                "pitch_number",
                *[
                    morpher(morpher.fill_missing(pl.col(column))).alias(column)
                    for column, morpher in self.morphers.items()
                ],
            )
            .sort("pitch_number")
            .group_by("game_pk", "at_bat_number")
            .agg(
                *[pl.col(feature) for feature in self.morphers.keys()],
                n_pitches=pl.col("pitch_number").count(),
            )
            .join(embeddings, on=["game_pk", "at_bat_number"], how="inner")
        )

    def __len__(self) -> int:
        """Obvious"""
        return self.sequences.height

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        """Return a single instance of the training dataset, as a dictionary."""

        row = self.sequences.row(idx, named=True)
        inputs, pad_mask = pad_tensor_dict(
            {
                k: torch.tensor(row[k], dtype=morpher.required_dtype)
                for k, morpher in self.morphers.items()
            },
            max_length=self.sequence_length,
        )
        inputs |= {
            "text_embedding": torch.tensor(row["embeddings"]),
            "pad_mask": pad_mask,
        }
        return inputs


if __name__ == "__main__":

    ds = TrainingDataset(
        "./data/raw_data.parquet",
        "./data/embeddings.parquet",
        sequence_length=16,
        morpher_spec={
            "description": ["categorical", {}],
            "pitch_name": ["categorical", {}],
            "release_speed": ["numeric", {}],
        },
    )

    print(ds[500])
    print(len(ds))
