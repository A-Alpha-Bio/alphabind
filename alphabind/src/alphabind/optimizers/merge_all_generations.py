import logging
import os
import pathlib

import click
import pandas as pd


@click.command()
@click.option("--intermediate_steps_path", required=True, type=click.Path(exists=True))
@click.option("--num_generations", required=True, type=click.INT)
@click.option(
    "--output_file_path", default="all_unique_candidates.csv", type=click.Path()
)
def merge_all_generations(
    intermediate_steps_path: os.PathLike,
    num_generations: int,
    output_file_path: str | os.PathLike = "all_unique_candidates.csv",
):
    # Ensure that output dir exists
    pathlib.Path(output_file_path).parent.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    for i in range(num_generations):
        df = pd.read_csv(os.path.join(intermediate_steps_path, f"generation_{i}.csv"))
        all_dfs.append(df.drop_duplicates(subset=["sequence_a"]).reset_index(drop=True))
    df = pd.concat(all_dfs)
    df = df.drop_duplicates(subset=["sequence_a"]).reset_index(drop=True)
    df.to_csv(output_file_path)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    merge_all_generations(auto_envvar_prefix="ALPHABIND")  # pyright: ignore [reportCallIssue]
