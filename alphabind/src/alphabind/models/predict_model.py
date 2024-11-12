import logging
import os

import click
import pandas as pd
import torch
from tqdm import tqdm

from alphabind.features.featurize_using_esm_2 import (
    get_embedding_using_inferer,
    load_bionemo_inferer,
)
from alphabind.models.model import (
    EnsembleTxRegressorLabelEncoded,
    TxRegressorForESMEmbedding,
    TxRegressorLabelEncoded,
)

allowed_labels = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
    "-",
    "^",
    "&",
    "#",
    "$",
]
label_encoding = {amino_acid: index for index, amino_acid in enumerate(allowed_labels)}


def _featurize_using_esm2nv(
    seq_a_batch: list[str],
    seq_alpha_batch: list[str],
    bionemo_inferer,
    input_device=torch.device("cuda:0"),
):
    if len(seq_a_batch) != len(seq_alpha_batch):
        raise ValueError(
            f"lengths of seq_batch and seq_alpha_batch should match. Found seq_a: {len(seq_a_batch)}, seq_alpha: {len(seq_alpha_batch)}"
        )

    seq_alpha_embeddings = None
    seq_a_embeddings = get_embedding_using_inferer(seq_a_batch, bionemo_inferer)
    seq_alpha_embeddings = get_embedding_using_inferer(seq_alpha_batch, bionemo_inferer)

    embeddings = torch.zeros(
        len(seq_a_batch), 601, 2561, dtype=torch.float16, device=input_device
    )
    padding = torch.ones(len(seq_a_batch), 601, dtype=torch.bool, device=input_device)

    for i, (seq_a, seq_alpha) in enumerate(zip(seq_a_batch, seq_alpha_batch)):
        # TODO: This is somewhat duplicated logic from `dataset.py`. Find a good way to refactor this
        embeddings[i, : len(seq_a), :2560] = seq_a_embeddings[seq_a]
        padding[i, : len(seq_a)] = False

        embeddings[i, len(seq_a), 2560] = 1
        padding[i, len(seq_a)] = False

        embeddings[i, len(seq_a) + 1 : len(seq_a) + 1 + len(seq_alpha), :2560] = (
            seq_alpha_embeddings[seq_alpha]
        )
        padding[i, len(seq_a) + 1 : len(seq_a) + 1 + len(seq_alpha)] = False

    return embeddings, padding


def _featurize_using_label_encoding(
    seq_a_batch: list[str],
    seq_alpha_batch: list[str],
    input_device: torch.device = torch.device("cuda:0"),
    embedded_sequence_length: int = 600,
):
    if len(seq_a_batch) != len(seq_alpha_batch):
        raise ValueError(
            f"lengths of seq_batch and seq_alpha_batch should match. Found seq_a: {len(seq_a_batch)}, seq_alpha: {len(seq_alpha_batch)}"
        )

    embeddings = torch.zeros(
        len(seq_a_batch), 601, dtype=torch.int, device=input_device
    )
    padding = torch.ones(len(seq_a_batch), 601, dtype=torch.bool, device=input_device)

    for i, (sequence_a, sequence_alpha) in enumerate(zip(seq_a_batch, seq_alpha_batch)):
        # TODO: refactor to use constants
        sequence = sequence_a + "#" + sequence_alpha

        # +1 because we added 1 character of delimiter above
        padding_mask = torch.ones(
            (embedded_sequence_length + 1), dtype=torch.bool, device=input_device
        )
        padding_mask[: len(sequence)] = 0

        # pad the remaining embedding with padding character '$'
        sequence = sequence + "$" * (embedded_sequence_length + 1 - len(sequence))

        encoded_sequence = torch.tensor(
            [label_encoding[amino_acid] for amino_acid in sequence],
            dtype=torch.int,
            device=input_device,
        )

        embeddings[i, :] = encoded_sequence
        padding[i, :] = padding_mask

    return embeddings, padding


def predict_using_alphabind(
    sequence_a: list[str],
    sequence_alpha: list[str],
    alphabind_model: TxRegressorForESMEmbedding
    | TxRegressorLabelEncoded
    | EnsembleTxRegressorLabelEncoded,
    bionemo_inferer=None,
    batch_size=16,
    input_device=torch.device("cuda:0"),
    output_device=torch.device("cpu"),
):
    if bionemo_inferer is None and isinstance(
        alphabind_model, TxRegressorForESMEmbedding
    ):
        bionemo_inferer = load_bionemo_inferer()
    alphabind_model.to(input_device)
    alphabind_model.eval()

    kds = []

    for i in tqdm(range(0, len(sequence_a), batch_size)):
        seq_a_batch = list(sequence_a[i : i + batch_size])
        seq_alpha_batch = list(sequence_alpha[i : i + batch_size])

        if isinstance(alphabind_model, TxRegressorForESMEmbedding):
            embeddings, padding = _featurize_using_esm2nv(
                seq_a_batch, seq_alpha_batch, bionemo_inferer, input_device=input_device
            )
        elif isinstance(
            alphabind_model, TxRegressorLabelEncoded | EnsembleTxRegressorLabelEncoded
        ):
            embeddings, padding = _featurize_using_label_encoding(
                seq_a_batch, seq_alpha_batch
            )
        else:
            raise ValueError("model type not supported")

        with torch.no_grad():
            with torch.autocast(
                device_type=input_device.type, dtype=None, enabled=True
            ):
                kd_batch = (
                    alphabind_model(
                        sequence_embeddings=embeddings.to(input_device),
                        padding_mask=padding.to(input_device),
                    )
                    .to(output_device)
                    .numpy()
                    .tolist()
                )
                kds = kds + kd_batch

    return kds[: len(sequence_a)]


def predict_using_alphabind_df_wrapper(ppi_df: pd.DataFrame, **kwargs):
    sequence_a = list(ppi_df.sequence_a)
    sequence_alpha = list(ppi_df.sequence_alpha)

    ppi_df["kd_pred"] = predict_using_alphabind(
        sequence_a=sequence_a, sequence_alpha=sequence_alpha, **kwargs
    )
    return ppi_df


@click.command()
@click.option("--ppi_dataset_path", required=True, type=click.Path(exists=True))
@click.option("--output_dataset_path", required=True, type=click.Path())
@click.option("--trained_model_path", required=True, type=click.Path(exists=True))
def predict(
    ppi_dataset_path: str | os.PathLike,
    output_dataset_path: str | os.PathLike,
    trained_model_path: str | os.PathLike,
):
    ppi_dataset = pd.read_csv(ppi_dataset_path)
    model = torch.load(trained_model_path, map_location=torch.device("cpu"))

    if not isinstance(model, TxRegressorForESMEmbedding):
        raise ValueError(
            f"Provided models must deserialize to `TxRegressorForESMEmbedding`, but got {type(model)=}"
        )

    output_df = predict_using_alphabind_df_wrapper(
        ppi_df=ppi_dataset, alphabind_model=model
    )
    output_df.to_csv(output_dataset_path)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    predict(auto_envvar_prefix="ALPHABIND")  # pyright: ignore [reportCallIssue]
