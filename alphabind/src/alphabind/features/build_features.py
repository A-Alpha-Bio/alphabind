import logging
import pathlib

import click
import pandas as pd

from alphabind.features.featurize_using_esm_2 import (
    featurize_ppis_using_esm,
    load_bionemo_inferer,
)


@click.command()
@click.option("--input_filepath", required=True, type=click.Path(exists=True))
@click.option("--output_filepath", required=True, type=click.Path())
@click.option("--embedding_dir_path", required=True, type=click.Path())
@click.option("--batch_size", default=16, type=click.INT)
def main(input_filepath, output_filepath, embedding_dir_path, batch_size):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("Building features using processed data")

    # Make output directories (recursively) if they do not already exist.
    pathlib.Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(embedding_dir_path).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_filepath)  # Requires a sequence_a and sequence_alpha column

    bionemo_inferer = load_bionemo_inferer()
    df_updated = featurize_ppis_using_esm(
        df,
        embedding_dir_path=embedding_dir_path,
        bionemo_inferer=bionemo_inferer,
        batch_size=batch_size,
    )

    df_updated.to_csv(output_filepath)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main(auto_envvar_prefix="ALPHABIND")  # pyright: ignore [reportCallIssue]
