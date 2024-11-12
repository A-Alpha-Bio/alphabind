import logging
import os
import pathlib
from typing import Literal

import click
import lightning as L
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from alphabind.models.dataset import (
    LabelEncodingDataset,
    YMDataModule,
    YMDataset,
)
from alphabind.models.model import (
    TransformerLightning,
    TxRegressorForESMEmbedding,
    TxRegressorLabelEncoded,
)


def train(
    dataset_csv_path: os.PathLike,
    tx_model_path: os.PathLike | None = None,
    max_epochs: int = 100,
    learning_rate: float = 1e-5,
    embedding_type: Literal["esm", "label-encoded"] = "esm",
    random_seed: int | None = None,
    log_dir: str | os.PathLike = "logs",
):
    """
    Simple train function to kick off training using TxRegressorForESMEmbedding.

    Parameters:
        dataset_csv_path: Path of the csv file containing sequence_a, sequence_alpha, Kd. Note that this file cannot contain NaNs
        tx_model_path: Path of pre-trained TxRegressorForESMEmbedding model. This will usually be the path for AlphaBind model. If set
                       to `None`, a new model is instantiated with random weights
        max_epochs: Number of epochs to train/finetune the model
        embedding_type: Controls what type of model to instantiate. Ignored if `tx_model_path` is not `None`.
        random_seed: Seed to set for training. If `None`, no seed will be set.
        log_dir: Specifies the output directory for training logs.

    Returns:
        TxRegressorForESMEmbedding model with weights trained/finetuned to input dataset
    """
    if random_seed is not None:
        seed_everything(random_seed, workers=True)

    if tx_model_path is not None:
        # Load pretrained AlphaBind model
        tx_model = torch.load(tx_model_path, map_location=torch.device("cpu"))

        if isinstance(tx_model, TxRegressorForESMEmbedding):
            datamodule = YMDataModule(
                data_csv_path=dataset_csv_path, dataset_class=YMDataset
            )
        elif isinstance(tx_model, TxRegressorLabelEncoded):
            datamodule = YMDataModule(
                data_csv_path=dataset_csv_path, dataset_class=LabelEncodingDataset
            )
        else:
            raise ValueError(f"Unrecognized model type: {type(tx_model)=}")
    else:
        print("Creating new model for cold start")
        if embedding_type == "esm":
            # Instantiate new model for cold start training
            tx_model = TxRegressorForESMEmbedding(4, 7)
            datamodule = YMDataModule(
                data_csv_path=dataset_csv_path, dataset_class=YMDataset
            )
        elif embedding_type == "label-encoded":
            tx_model = TxRegressorLabelEncoded(4, 7, vocab_size=25)
            datamodule = YMDataModule(
                data_csv_path=dataset_csv_path, dataset_class=LabelEncodingDataset
            )
        else:
            raise ValueError(f"Unrecognized {embedding_type=}")

    lmodel = TransformerLightning(tx_model, lr=learning_rate)
    lmodel.save_hyperparameters()

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=2, filename="best_model_{epoch:02d}"
    )

    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(log_dir)

    trainer = L.Trainer(
        max_epochs=max_epochs,
        check_val_every_n_epoch=1,
        precision="16-mixed",
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model=lmodel, datamodule=datamodule)

    print(
        f"Saving best model from checkpoint: {checkpoint_callback.best_model_path}, with score: {checkpoint_callback.best_model_score}"
    )
    checkpoint = torch.load(checkpoint_callback.best_model_path)
    lmodel.load_state_dict(checkpoint["state_dict"])
    return lmodel.tx_model


@click.command()
@click.option("--dataset_csv_path", required=True, type=click.Path(exists=True))
@click.option("--tx_model_path", default=None, type=click.Path())
@click.option("--max_epochs", default=50, type=click.INT)
@click.option("--learning_rate", default=1e-5, type=click.FLOAT)
@click.option("--output_model_path", default="model_trained.pt", type=click.Path())
@click.option("--seed", default=None, type=click.INT)
@click.option("--embedding_type", default=None, type=click.STRING)
@click.option("--log_dir", default="logs", type=click.Path())
def main(
    dataset_csv_path: os.PathLike,
    tx_model_path: os.PathLike | None = None,
    max_epochs: int = 50,
    learning_rate: float = 1e-5,
    output_model_path: str | os.PathLike = "model_trained.pt",
    seed: int | None = None,
    embedding_type: str | None = None,
    log_dir: str | os.PathLike = "logs",
):
    if embedding_type is None:
        # default to alphabind
        embedding_type = "esm"
    elif embedding_type not in ("esm", "label-encoded"):
        raise ValueError("embedding_type must be either 'esm' or 'label-encoded'")

    trained_model = train(
        dataset_csv_path=dataset_csv_path,
        tx_model_path=tx_model_path,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        embedding_type=embedding_type,
        random_seed=seed,
        log_dir=log_dir,
    )

    torch.save(trained_model, output_model_path)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main(auto_envvar_prefix="ALPHABIND")  # pyright: ignore [reportCallIssue]
