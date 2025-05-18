"""Train a model."""

import yaml
import torch
from lightning import pytorch as pl

from src.data import TrainingDataset
from src.model import WhateverModel

if __name__ == "__main__":

    # Read the config ---------------------------------------
    # (Probably should use hydra eventually.)

    with open("cfg/config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    # Hashtag reproducibility
    torch.manual_seed(config["random_seed"])

    # Generic dataset preparation ---------------------------

    ds = TrainingDataset(
        path=config["preparation_params"]["data_path"],
        embedding_path=config["preparation_params"]["embedding_path"],
        **config["dataset_params"],
    )
    train_ds, valid_ds, test_ds = torch.utils.data.random_split(ds, [0.75, 0.15, 0.1])

    train_dl = torch.utils.data.DataLoader(
        train_ds,
        shuffle=True,
        **config["dataloader_params"],
        drop_last=True,
    )
    valid_dl = torch.utils.data.DataLoader(
        valid_ds,
        **config["dataloader_params"],
        drop_last=True,
    )
    # For the moment we're not doing anything with the test dataset.

    # Set up the trainer ------------------------------------

    trainer = pl.Trainer(
        **config["trainer_params"],
        logger=pl.loggers.TensorBoardLogger("."),
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath="./model",
                save_top_k=1,
                monitor="valid_loss",
                filename="{epoch}-{valid_loss:.4f}",
            ),
            pl.callbacks.LearningRateMonitor(),
        ],
    )

    with trainer.init_module():

        optimizer_params = config["optimizer_params"] | {
            "n_obs": len(train_ds),
            "n_epochs": config["trainer_params"]["max_epochs"],
        }

        net = WhateverModel(
            **config["model_params"],
            morpher_dict=train_ds.dataset.morphers,
            sequence_length=train_ds.dataset.sequence_length,
            optimizer_params=optimizer_params,
            batch_size=config["dataloader_params"]["batch_size"],
        )
        # Maybe
        net.compile()

    torch.set_float32_matmul_precision("high")
    trainer.fit(net, train_dataloaders=train_dl, val_dataloaders=valid_dl)
