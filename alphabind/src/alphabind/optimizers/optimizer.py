import os
import pathlib
import random

import numpy as np
import pandas as pd
import torch

from alphabind.optimizers.generator import ESMSimultaneousGenerator, SequenceGenerator
from alphabind.optimizers.scoring_wrapper import PPIScoringWrapper
from alphabind.optimizers.selector import SequenceSelector


class PPISingleProteinOptimizer:
    """
    Class to optimize a dataframe of protein sequences using a generator (sequence proposer), a scorer (to compute binding affinity for these sequence),
    and a selector (algorithm to select between the current and the proposed sequence)

    Parameters:
        generator: A SequenceGenerator object which implements generate_proposals to create new proposed sequences
        scorer: A PPIScoringWrapper object which implements score_proposals to predict binding affinity (kd) given sequences
        selector: A SequenceSelector object which implements select_candidates to update current sequence_a in input df (ppi_df in optimize_seqs)
        random_seed: Seed to use for reproducibility.
    """

    def __init__(
        self,
        generator: SequenceGenerator,
        scorer: PPIScoringWrapper,
        selector: SequenceSelector,
        random_seed: int | None = None,
    ):
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
            torch.manual_seed(random_seed)

        self.scorer = scorer
        self.generator = generator
        self.selector = selector

    def generate_proposals(self, ppi_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate proposals using current/seed sequences.

        Parameters:
            ppi_df: A pandas dataframe with sequence_a, sequence_a_mask columns which have following constrains
                    sequence_a: full protein sequence in amino acid notation
                    sequence_a_mask: boolean list equal to length of protein sequence where True represents amino acids
                                     that can be mutated and False represents ones that cannot

        Returns:
            ppi_df with two new columns sequence_a_proposal and sequence_a_mask_proposal. These are proposed sequence and mask
        """
        proposals = ppi_df.apply(
            lambda row: self.generator.generate_proposals(
                row.sequence_a, row.sequence_a_mask
            ),
            axis=1,
        )

        sequence_a_proposals, sequence_a_mask_proposals = zip(*proposals)

        if isinstance(self.generator, ESMSimultaneousGenerator):
            sequence_a_proposals = self.generator.unmask_with_esm(sequence_a_proposals)

        ppi_df["sequence_a_proposal"] = sequence_a_proposals
        ppi_df["sequence_a_mask_proposal"] = sequence_a_mask_proposals
        return ppi_df

    def score_proposals(self, ppi_df: pd.DataFrame) -> pd.DataFrame:
        """
        Score proposed sequences based on scoring function

        Parameters:
            ppi_df: A pandas dataframe with sequence_a_proposed, sequence_alpha columns which have following constrains
                    sequence_a_proposed: full protein sequence in amino acid notation
                    sequence_alpha: target sequence used when optimizing

        Returns:
            ppi_df with Kd_proposal column added. This column is the score of each sequence_a_proposal
        """
        df = pd.DataFrame(
            {
                "sequence_a": ppi_df.sequence_a_proposal,
                "sequence_alpha": ppi_df.sequence_alpha,
            }
        )
        df = self.scorer.predict_using_ppi_df(df)
        ppi_df["Kd_proposal"] = df.kd_pred
        return ppi_df

    def select_candidates(self, ppi_df: pd.DataFrame) -> SequenceSelector.StepOutput:
        """
        Updates df using a selection function

        Parameter:
            ppi_df: A pandas dataframe with columns sequence_a, sequence_a_mask, Kd, sequence_a_proposal, sequence_a_mask_proposal, Kd_proposal. In
                    this dataframe the columns correspond to -
                    sequence_a: current/seed amino acid sequence that we want to optimize
                    sequence_a_mask: current/seed mask (list[bool]) of sequence that we want to optimize
                    Kd: Kd computed for sequence_a (usually w.r.t sequence_alpha)
                    sequence_a_proposal: proposed amino acid sequence to replace sequence_a
                    sequence_a_mask_proposal: proposed mask (list[bool]) of amino acid sequence to replace sequence_a
                    Kd_proposal: Kd computed for sequence_a_proposal

        Returns:
            ppi_df with sequence_a, sequence_a_mask, and Kd updated with acceptance criteria.
        """
        return self.selector.step(ppi_df)

    def optimize_seqs(
        self,
        ppi_df: pd.DataFrame,
        num_generations: int = 30,
        save_intermediate_steps: os.PathLike | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Optimizes ppi_df over multiple generations. This will generate proposals, score them, and then select better ones over time.

        Parameters:
            ppi_df: pandas dataframe to optimize with following columns:
                    sequence_a: full protein sequence in amino acid notation
                    sequence_a_mask: boolean list equal to length of protein sequence where True represents amino acids
                                     that can be mutated and False represents ones that cannot
                    sequence_alpha: target sequence used when optimizing
                    Kd: Kd computed for sequence_a (usually w.r.t sequence_alpha)
            num_generations: Number of times to run the generate, score, select loop.
        """
        if save_intermediate_steps is not None:
            pathlib.Path(save_intermediate_steps).mkdir(parents=True, exist_ok=True)

        acceptance_rates = []
        for generation in range(num_generations):
            print(f"Starting generation {generation}")
            ppi_df = self.generate_proposals(ppi_df)
            ppi_df = self.score_proposals(ppi_df)
            selected_candidates = self.select_candidates(ppi_df)
            ppi_df = selected_candidates["df"]
            acceptance_rates.append(selected_candidates["acceptance_rate"])

            if save_intermediate_steps is not None:
                ppi_df.to_csv(
                    os.path.join(
                        save_intermediate_steps, f"generation_{generation}.csv"
                    )
                )

        acceptance_df = pd.DataFrame(
            {"Generation": range(num_generations), "Acceptance Rate": acceptance_rates}
        )
        return ppi_df, acceptance_df
