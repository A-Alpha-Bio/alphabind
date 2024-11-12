from abc import ABC, abstractmethod

import pandas as pd
from lightning import LightningModule

from alphabind.features.featurize_using_esm_2 import load_bionemo_inferer
from alphabind.models.model import TxRegressorForESMEmbedding
from alphabind.models.predict_model import predict_using_alphabind_df_wrapper


class PPIScoringWrapper(ABC):
    """
    Base class to wrap a scoring function for protein sequences
    """

    @abstractmethod
    def predict_using_ppi_df(self, ppi_df: pd.DataFrame) -> pd.DataFrame:
        pass


class AlphaBindBasedPPIScoringWapper(PPIScoringWrapper):
    """
    Scoring wrapper for AlphaBind-type models

    Parameters:
        model: pytorch model of type TxRegressorForESMEmbedding
    """

    def __init__(
        self,
        model: TxRegressorForESMEmbedding,
        batch_size=16,
        bionemo_inferer: LightningModule | None = None,
    ):
        self.model = model
        self.batch_size = batch_size

        # instantiate bionemo inferer once at the start. Bionemo code tries to re-instantiate the model multiple
        # times if load_bionemo_inferer is called multiple times
        if bionemo_inferer is None:
            self.bionemo_inferer = load_bionemo_inferer()
        else:
            self.bionemo_inferer = bionemo_inferer

    def predict_using_ppi_df(self, ppi_df: pd.DataFrame) -> pd.DataFrame:
        """
        Given a dataframe with protein sequences (in column sequence_a), this function will score the sequences. 'Scoring' here
        refers to predicting the Kd

        Parameters:
            ppi_df: pandas dataframe with columns sequence_a and sequence_alpha. sequence_a is usually the candidate and sequence_alpha is
                    usually the target (though this is not a requirement)

        Returns:
            ppi_df with a new column 'kd_pred' filled in which includes predicted kds of the sequence pairs
        """
        ppi_df = predict_using_alphabind_df_wrapper(
            ppi_df,
            alphabind_model=self.model,
            batch_size=self.batch_size,
            bionemo_inferer=self.bionemo_inferer,
        )
        return ppi_df
