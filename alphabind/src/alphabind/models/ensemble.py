import logging
import os

import click
import torch

from alphabind.models.model import EnsembleTxRegressorLabelEncoded


@click.command()
@click.argument("model_paths", nargs=-1, type=click.Path(exists=True))
@click.option("--output_file_path", default="ensemble_model.pt", type=click.Path())
def ensemble(model_paths: list[os.PathLike], output_file_path="ensemble_model.pt"):
    """ """
    submodules = []
    for submodule_path in model_paths:
        submodules.append(torch.load(submodule_path, map_location=torch.device("cpu")))
    model = EnsembleTxRegressorLabelEncoded(modules=submodules)
    torch.save(model, output_file_path)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    ensemble()  # pyright: ignore [reportCallIssue]
