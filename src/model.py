"""Primarily for the LightningModule that contains the model.

Other network modules, etc. can go here or not, depending on how confusing it is.
"""

import math
from typing import Any

import torch
from torch import nn
import lightning.pytorch as pl
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassPrecision
from morphers.base.base import Morpher


class BoringPositionalEncoding(nn.Module):
    """
    Shamelessly "adapted" from a torch tutorial
    """

    def __init__(self, max_length: int, d_model: int):
        super().__init__()

        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_length, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        return x + self.pe[:, : x.shape[1], :]


class WhateverModel(pl.LightningModule):
    """Model for whatever we're doing."""

    def __init__(
        self,
        optimizer_params: dict[str, Any],
        morpher_dict: dict[str, Morpher],
        d_model: int,
        sequence_length: int,
        d_embedding: int,
        tr_params: dict[str, Any],
        batch_size: int,
    ):
        """Initialize the model.

        - optimizer_params: dictionary of parameters to initialize optimizer with.
        - morpher_dict: initialized morphers from data
        - d_model: size of embeddings, etc.
        """

        super().__init__()
        self.save_hyperparameters()
        self.optimizer_params = optimizer_params

        self.embedders = nn.ModuleDict(
            {
                col: morpher.make_embedding(d_model)
                for col, morpher in morpher_dict.items()
            }
        )
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()
        self.position_encoding = BoringPositionalEncoding(sequence_length, d_model)
        self.register_parameter(
            "cls", nn.Parameter(torch.randn([1, 1, d_model]) * 0.02)
        )

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model,
                nhead=tr_params["n_heads"],
                dim_feedforward=d_model * 3,
                activation="relu",
                batch_first=True,
            ),
            num_layers=tr_params["n_layers"],
        )

        self.sequence_projector = nn.Linear(d_model, d_model)
        self.text_projector = nn.Linear(d_embedding, d_model)

        # self.initialize_weights()

        self.metrics = MetricCollection(
            {
                "precision_20": MulticlassPrecision(num_classes=batch_size, top_k=20),
                "precision_10": MulticlassPrecision(num_classes=batch_size, top_k=10),
                "precision_5": MulticlassPrecision(num_classes=batch_size, top_k=5),
                "precision_1": MulticlassPrecision(num_classes=batch_size, top_k=1),
            }
        )

    def initialize_weights(self):

        # lol this doesn't work
        # #yolo
        for name, param in self.named_parameters():
            if "weight" in name and param.data.dim() == 2:
                nn.init.kaiming_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def configure_optimizers(self):
        """Lightning hook for optimizer setup.

        Body here is just an example, although it probably works okay for a lot of things.
        We can't pass arguments to this directly, so they need to go to the init.
        """

        optimizer = torch.optim.AdamW(self.parameters(), **self.optimizer_params)
        return optimizer

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for the model.

        - x is compatible with the output of data.TrainingDataset.__getitem__, literally the output
          from a dataloader.

        Returns whatever the output of the model is.
        """

        # Sequence inputs
        x_sequence = torch.stack(
            [embedder(x[col]) for col, embedder in self.embedders.items()],
            dim=-1,
        )
        x_sequence = x_sequence.sum(dim=-1)
        x_sequence = self.activation(self.norm(x_sequence))

        # Sequence transformer
        x_sequence = torch.cat(
            [self.cls.expand(x_sequence.shape[0], -1, -1), x_sequence], dim=1
        )
        mask = torch.cat(
            [
                torch.zeros([x_sequence.shape[0], 1], device=x["pad_mask"].device),
                x["pad_mask"],
            ],
            dim=1,
        )
        x_sequence = self.transformer(x_sequence, src_key_padding_mask=mask)
        x_output = self.sequence_projector(x_sequence[:, 0, :])

        return x_output

    def step(self, stage: str, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Generic step for training or validation, assuming they're similar.

        - stage: one of "train" or "valid"
        - x: dictionary of torch tensors, input to model, targets, etc.

        This MUST log "valid_loss" during the validation step in order to have the model checkpointing work as written.

        Returns loss as one-element tensor.
        """

        # Both (n, e)
        x_text = self.text_projector(x["text_embedding"])
        batch_size = x_text.shape[0]

        x_sequence = self.forward(x)

        # (n, n)
        logits = x_sequence @ x_text.T
        targets = torch.arange(batch_size, device=logits.device)

        # Scalar output
        loss = nn.functional.cross_entropy(logits, targets, reduction="mean")

        self.log(f"{stage}_loss", loss)
        metric_results = self.metrics(logits, targets)
        self.log_dict({f"{stage}_{k}": v for k, v in metric_results.items()})
        return loss

    def training_step(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Lightning hook for training."""
        return self.step("train", x)

    def validation_step(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Lightning hook for validation."""
        return self.step("valid", x)
