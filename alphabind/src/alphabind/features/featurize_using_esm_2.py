import hashlib
import json
import logging
import os
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from lightning import LightningModule
from torch import Tensor
from tqdm import tqdm


def load_bionemo_inferer(
    model_name: str = "facebook/esm2_t36_3B_UR50D",
    bionemo_checkpoint_name: str = "esm2nv_3B_converted.nemo",
) -> LightningModule:
    """
    Loads esm2nv 3B model and returns the inferer provided by bionemo. Convenience method to instantiate the bionemo inferer once instead of
    multiple times

    Parameters:
        model_name: [Fill]
        bionemo_model: [Fill]
    """

    # Import bionemo dependencies only if load_bionemo_inferer is called. Loading these takes time.
    from bionemo.model.protein.esm1nv.infer import ESM1nvInference
    from bionemo.triton.utils import load_model_for_inference
    from bionemo.utils.hydra import load_model_config

    BIONEMO_HOME: Path = Path(os.environ["BIONEMO_HOME"]).absolute()

    model_checkpoint_path = BIONEMO_HOME / "models" / bionemo_checkpoint_name
    if not model_checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint file not found at {model_checkpoint_path}"
        )

    config_path = BIONEMO_HOME / "examples" / "protein" / "esm2nv" / "conf"
    print(f"Using model configuration at: {config_path}")

    cfg = load_model_config(config_path=config_path, config_name="infer.yaml")
    cfg.model.downstream_task.restore_from_path = model_checkpoint_path
    cfg.model.tokenizer.model_name = model_name

    inferer = load_model_for_inference(cfg, interactive=True)

    assert isinstance(inferer, ESM1nvInference)

    return inferer


def get_embedding_using_inferer(
    sequences: Sequence[str], bionemo_inferer
) -> dict[str, Tensor]:
    """
    Given a batch of sequences and a bionemo inferer (as provided by load_bionemo_inferer), this function runs inference and returns embeddings

    Parameters:
        sequences: Protein sequences in amino acid notation to be passed through the model to generate embeddings
        bionemo_inferer: [Fill]

    Returns:
        dict where the sequences passed above are the keys and their corresponding embeddings are the values
    """

    sequences = list(set(sequences))  # Deduplicate to avoid redundant inference calls

    if bionemo_inferer is None:
        bionemo_inferer = load_bionemo_inferer()

    sequence_to_embeddings = {}
    hidden_states, pad_masks = bionemo_inferer.seq_to_hiddens(sequences)

    for sequence, hidden_state, pad_mask in zip(
        *[sequences, hidden_states, pad_masks], strict=True
    ):
        truncated_hidden_state = hidden_state[pad_mask]
        sequence_to_embeddings[sequence] = truncated_hidden_state

    return sequence_to_embeddings


def featurize_using_esm2nv(
    sequences: Iterable[str],
    embedding_dir_path: os.PathLike,
    batch_size: int = 16,
    upper_limit_len: int = 450,
    bionemo_inferer=None,
):
    """ """

    seqs = [seq for seq in sequences if len(seq) < upper_limit_len]

    sequence_to_flocation = {}
    for i in tqdm(range(0, len(seqs), batch_size)):
        seq_batch = list(seqs[i : i + batch_size])
        seq_batch_embeddings = get_embedding_using_inferer(seq_batch, bionemo_inferer)

        for sequence, embedding in seq_batch_embeddings.items():
            fname = f"{hashlib.md5(sequence.encode('utf-8')).hexdigest()}.npy"
            flocation = os.path.join(embedding_dir_path, fname)
            np.save(flocation, embedding.cpu().numpy())

            sequence_to_flocation[sequence] = {}
            sequence_to_flocation[sequence]["embedding_location"] = flocation

    return sequence_to_flocation


def featurize_ppis_using_esm(
    ppi_df: pd.DataFrame,
    embedding_dir_path: os.PathLike,
    batch_size: int = 16,
    key_file_name: str | os.PathLike = "sequence_key.json",
    bionemo_inferer=None,
):
    """ """

    sequence_a_mapping = featurize_using_esm2nv(
        set(ppi_df.sequence_a),
        embedding_dir_path=embedding_dir_path,
        batch_size=batch_size,
        bionemo_inferer=bionemo_inferer,
    )

    sequence_alpha_mapping = featurize_using_esm2nv(
        set(ppi_df.sequence_alpha),
        embedding_dir_path=embedding_dir_path,
        batch_size=batch_size,
        bionemo_inferer=bionemo_inferer,
    )

    df_a = pd.DataFrame.from_dict(sequence_a_mapping, orient="index")
    df_a.columns = df_a.columns.map(lambda x: str(x) + "_a")
    df_alpha = pd.DataFrame.from_dict(sequence_alpha_mapping, orient="index")
    df_alpha.columns = df_alpha.columns.map(lambda x: str(x) + "_alpha")

    ppi_df = ppi_df.join(df_a, on="sequence_a", how="left")
    ppi_df = ppi_df.join(df_alpha, on="sequence_alpha", how="left")

    # Save a copy of sequence paths in the directory in case we need to refer to it later
    key_path = os.path.join(embedding_dir_path, key_file_name)
    with open(key_path, "w") as f:
        json.dump(sequence_a_mapping | sequence_alpha_mapping, f)

    logging.info(
        f"The dictionary for sequence to key mapping was stored at: {key_path}"
    )

    return ppi_df
