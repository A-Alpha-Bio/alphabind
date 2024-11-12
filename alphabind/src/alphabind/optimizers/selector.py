from abc import ABC, abstractmethod
from typing import Literal, TypedDict

import numpy as np
import pandas as pd


class SequenceSelector(ABC):
    """
    Base class to allow different kinds of selection methods for proposed sequences
    """

    class StepOutput(TypedDict):
        df: pd.DataFrame
        acceptance_rate: float

    @abstractmethod
    def step(self, ppi_df: pd.DataFrame) -> StepOutput:
        pass


class MCMCSelector(SequenceSelector):
    """
    Class for selection algorithm using MCMC equations

    Parameters:
        optimization_type: selects minimization or maximization objective. "min" will try to minimize kd whereas "max" will try to maximize kd
        temperature: mcmc parameter which sets the entropy of the system. temperature=0 would result in the better sequence always being selected
                     (hill climbing) whereas temperature>0 will allow for a bad sequence to sometimes be selected in search for a better overall
                     sequence. Note: For alphabind optimization purposes, we set this to 0 to perform hill climbing
        temperature_decay: mcmc parameter to reduce entropy of the system over time. temperature_decay=0 results in normal mcmc/hill climbing whereas
                           temperature_decay>0 performs simulated annealing
        epsilon: mcmc parameter to avoid dividing by zero. This is useful for numerical stability. This will need to increase if kd differences are
                 very small and it results in numerical overflows
    """

    def __init__(
        self,
        optimization_type: Literal["min", "max"] = "min",
        temperature: float = 0.0,
        temperature_decay: float = 1.0,
        epsilon: float = 1e-8,
    ):
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.epsilon = epsilon
        self.optimization_type = optimization_type

    def is_accept(self, kd_current: float, kd_proposal: float) -> bool:
        """
        Function to accept or reject proposal based on mcmc equations

        Parameter:
            kd_current: current or seed Kd to compare the proposal against
            kd_proposal: Kd of the proposed sequence

        Returns:
            boolean value where True means accept proposal whereas False means reject proposal
        """

        if self.optimization_type == "min":
            # note: we want to accept proposals when the delta is negative
            delta_kd = kd_proposal - kd_current

        elif self.optimization_type == "max":
            # if objective is to maximize kd, we want can just invert this equation
            delta_kd = kd_current - kd_proposal

        else:
            raise ValueError("Optimization type should be min or max")

        # Note: This is the log version of acceptance formula
        log_proba = min(0.0, -1 * delta_kd / (self.temperature + self.epsilon))
        proba = np.exp(log_proba)

        return np.random.random() < proba

    def step(self, ppi_df: pd.DataFrame) -> SequenceSelector.StepOutput:
        """
        Given a df with columns sequence_a, sequence_a_mask, and Kd, and columns for sequence_a_proposal, sequence_a_mask_proposal, and Kd_proposal
        this function will update the df's sequence_a, sequence_a_mask and Kd to based on acceptance criteria from MCMCSelector.accept(...).

        Parameters:
            ppi_df: A pandas dataframe with columns sequence_a, sequence_a_mask, Kd, sequence_a_proposal, sequence_a_mask_proposal, Kd_proposal. In
                    this dataframe the columns correspond to -
                    sequence_a: current/seed amino acid sequence that we want to optimize
                    sequence_a_mask: current/seed mask (list[bool]) of sequence that we want to optimize
                    Kd: Kd computed for sequence_a (usually w.r.t sequence_alpha)
                    sequence_a_proposal: proposed amino acid sequence to replace sequence_a
                    sequence_a_mask_proposal: proposed mask (list[bool]) of amino acid sequence to replace sequence_a
                    Kd_proposal: Kd computed for sequence_a_proposal

        Returns:
            ppi_df with seqeuence_a, sequence_a_mask, and Kd updated with acceptance criteria.
        """

        new_seeds = ppi_df.apply(
            lambda row: (
                True,
                row.sequence_a_proposal,
                row.sequence_a_mask_proposal,
                row.Kd_proposal,
            )
            if self.is_accept(row.Kd, row.Kd_proposal)
            else (False, row.sequence_a, row.sequence_a_mask, row.Kd),
            axis=1,
        )
        accepted, sequence_a, sequence_a_mask, kds = zip(*new_seeds)
        acceptance_rate = sum(accepted) * 1.0 / len(accepted)
        print(f"Acceptance Rate: {acceptance_rate}")

        ppi_df.sequence_a = sequence_a
        ppi_df.sequence_a_mask = sequence_a_mask
        ppi_df.Kd = kds

        # decay temperature according to temperature decay parameter. This is useful for simulated annealing-type algorithms
        self.temperature = self.temperature * self.temperature_decay

        return {"df": ppi_df, "acceptance_rate": acceptance_rate}
