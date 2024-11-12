import logging
import os
import pathlib
from typing import Literal

import click
import pandas as pd
import torch

from alphabind.features.featurize_using_esm_2 import load_bionemo_inferer
from alphabind.optimizers.generator import (
    EditSequenceGenerator,
    ESMSequentialGenerator,
    ESMSimultaneousGenerator,
)
from alphabind.optimizers.optimizer import PPISingleProteinOptimizer
from alphabind.optimizers.scoring_wrapper import AlphaBindBasedPPIScoringWapper
from alphabind.optimizers.selector import MCMCSelector


@click.command()
@click.option("--seed_sequence", required=True, type=click.STRING)
@click.option("--target", required=True, type=click.STRING)
@click.option("--mutation_start_idx", required=True, type=click.INT)
@click.option("--mutation_end_idx", required=True, type=click.INT)
@click.option("--num_seeds", required=True, type=click.INT)
@click.option("--generations", required=True, type=click.INT)
@click.option("--trained_model_path", required=True, type=click.Path(exists=True))
@click.option("--output_file_path", required=True, type=click.Path())
@click.option("--seed", default=None, type=click.INT)
@click.option("--save_intermediate_steps", default=None, type=click.Path())
@click.option("--batch_size", default=16, type=click.INT)
@click.option("--mcmc_temperature", default=0, type=click.FLOAT)
@click.option("--mcmc_temperature_decay", default=1.0, type=click.FLOAT)
@click.option("--generator_type", default="random", type=click.STRING)
def main(
    seed_sequence: str,
    target: str,
    mutation_start_idx: int,
    mutation_end_idx: int,
    num_seeds: int,
    generations: int,
    trained_model_path: os.PathLike,
    output_file_path: os.PathLike,
    seed: int | None = None,
    save_intermediate_steps: os.PathLike | None = None,
    batch_size: int = 16,
    mcmc_temperature: float = 0.0,
    mcmc_temperature_decay: float = 1.0,
    generator_type: Literal[
        "random", "esm-random", "esm-simultaneous-random"
    ] = "random",
):
    # Ensure that output directories exist
    pathlib.Path(output_file_path).parent.mkdir(parents=True, exist_ok=True)
    if save_intermediate_steps is not None:
        pathlib.Path(save_intermediate_steps).mkdir(parents=True, exist_ok=True)

    mutation_region = seed_sequence[mutation_start_idx : (mutation_end_idx + 1)]
    print(f"Optimize seed sequence in mutation region: {mutation_region}")
    model = torch.load(trained_model_path, map_location=torch.device("cpu"))

    selector = MCMCSelector(
        temperature=mcmc_temperature, temperature_decay=mcmc_temperature_decay
    )

    bionemo_inferer = load_bionemo_inferer()

    scorer = AlphaBindBasedPPIScoringWapper(
        model, batch_size=batch_size, bionemo_inferer=bionemo_inferer
    )
    match generator_type:
        case "random":
            generator = EditSequenceGenerator()
        case "esm-random":
            generator = ESMSequentialGenerator(bionemo_inferer=bionemo_inferer)
        case "esm-simultaneous-random":
            generator = ESMSimultaneousGenerator(
                bionemo_inferer=bionemo_inferer, batch_size=batch_size
            )
        case _:
            raise ValueError(
                f"`gerator_type` must be one of ['esm', 'esm-random', 'esm-simultaneous-random'], but got {generator_type=}"
            )

    kd = scorer.predict_using_ppi_df(
        ppi_df=pd.DataFrame({"sequence_a": [seed_sequence], "sequence_alpha": [target]})
    ).kd_pred[0]

    mask = (
        mutation_start_idx * [False]
        + len(mutation_region) * [True]
        + (len(seed_sequence) - (mutation_end_idx + 1)) * [False]
    )
    ppi_df = pd.DataFrame(
        {
            "sequence_a": num_seeds * [seed_sequence],
            "sequence_a_mask": num_seeds * [mask],
            "sequence_alpha": num_seeds * [target],
            "Kd": num_seeds * [kd],
        }
    )

    optimizer = PPISingleProteinOptimizer(
        scorer=scorer, selector=selector, generator=generator, random_seed=seed
    )

    optimized_df, acceptance_df = optimizer.optimize_seqs(
        ppi_df=ppi_df,
        num_generations=generations,
        save_intermediate_steps=save_intermediate_steps,
    )
    optimized_df.to_csv(output_file_path)
    acceptance_df.to_csv("acceptance_rates.csv")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main(auto_envvar_prefix="ALPHABIND")  # pyright: ignore [reportCallIssue]
