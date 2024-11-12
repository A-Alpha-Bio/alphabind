import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class YMDataset(Dataset):
    """
    Pytorch dataset that loads previously computed ESM embeddings corresponding to provided sequences. In this example, sequence_a is of length 129
    and sequence alpha is of length 345. It will return each embedding in the following format:

        dim_1: 0   1   2  .........   2559 2560
    dim_0:
    0         .5  .3  .2  .........     .3    0
    1         .4  .4  .3  .........     .1    0
    2         .6  .2  .4  .........     .2    0
    3         .4  .4  .8  .........     .6    0
    .
    .
    .
    128       .5  .7  .5  .........     .6    0
    129        0   0   0  .........      0    1
    130       .1  .3  .2  .........     .4    0
    131       .4  .3  .9  .........     .9    0
    .
    .
    .
    472       .6  .2  .2  .........     .2    0
    473       .1  .6  .2  .........     .3    0
    474       .3  .3  .5  .........     .6    0
    475       .0  .0  .0  .........     .0    0
    .
    .
    .
    597       .0  .0  .0  .........     .2    0
    598       .0  .0  .0  .........     .3    0
    599       .0  .0  .0  .........     .6    0
    600       .0  .0  .0  .........     .2    0


    sequence_a embedding get appended to the start of the full embedding (from 0 to 128 of dim=0). A delimiter is then added to the next row (129 of dim=0). sequence_alpha
    is then appended to the remaining tensor (from 130 to 474 of dim=0)

    The embedding tensor is of length 601x2561. The ESM embedding vector is of shape 1x2560. The first 2560 columns of dim=1 are filled with the loaded embeddings, and the
    last column is left empty. In the delimiter column, all the columns except 2560 are 0 and a 1 is added to this position (129,2560 in above example)

    Parameters:
        df (pd.DataFrame): pandas dataframe with columns - embedding_a_location, embedding_alpha_location, Kd. The columns here are as follows:
                           embedding_a_location (os.PathLike): path to the embeddings (usually produced by esm) of sequence_a and saved in a npy format
                           embedding_alpha_location (os.PathLike): path to the embeddings (usually produced by esm) of sequence_alpha and saved in a npy format
                           Kd (float): binding affinity between sequence_a and sequence_alpha
        embedded_sequence_length (int): Embedding length produced by this Dataset. The embeddings are filled within this frame. PPIs whose len(sequence_a) + len(sequence_alpha)
                                        exceeds embedded_sequence_length are dropped
        esm_embedding_size (int): The size of embeddings returned by esm. This is dependent on the size of esm model used to featurize the data

    Returns:
        dictionary of arrays with following keys:
            embedding: The embedding array as described above
            padding_mask: Mask array for the ppi sequence. This mask is False for rows where we have filled in embeddings and delimiter. The rest of the mask is True.\
                          len(mask) = embedding.shape(dim=0)
            kd: binding affinity for the data sample obtained from input df.
    """

    # FIXME: The above docstring is currently conflated with what should now be the docstring for `__getitem__()`.

    def __init__(
        self,
        df: pd.DataFrame,
        embedded_sequence_length: int = 600,
        esm_embedding_size: int = 2560,
    ):
        self.df = df  # TODO: Check if this self.df changes with the change in df in the caller code. If it does, make a copy here instead of assigning a reference
        self.esm_embedding_size = esm_embedding_size
        self.embedded_sequence_length = embedded_sequence_length

    def __getitem__(self, index: int):
        element = self.df.iloc[index]
        embedding_a_path = element.embedding_location_a
        embedding_alpha_path = element.embedding_location_alpha

        full_embedding = np.zeros(
            (self.embedded_sequence_length + 1, self.esm_embedding_size + 1),
            dtype=np.float16,
        )
        padding_mask = np.ones(self.embedded_sequence_length + 1, dtype=np.bool_)

        # Fill sequence_a embedding in the full_embedding tensor form position 0 to length of sequence_a
        full_embedding[: len(element.sequence_a), : self.esm_embedding_size] = np.load(
            embedding_a_path
        )
        padding_mask[: len(element.sequence_a)] = False

        # Fill delimiter at the position after sequence_a embedding
        full_embedding[len(element.sequence_a), self.esm_embedding_size] = 1
        padding_mask[len(element.sequence_a)] = False

        # Fill sequence_alpha embedding in the full_embedding tensor after
        full_embedding[
            len(element.sequence_a) + 1 : (
                len(element.sequence_a) + 1 + len(element.sequence_alpha)
            ),
            : self.esm_embedding_size,
        ] = np.load(embedding_alpha_path)
        padding_mask[
            len(element.sequence_a) + 1 : (
                len(element.sequence_a) + 1 + len(element.sequence_alpha)
            )
        ] = False

        return {
            "embedding": full_embedding,
            "padding_mask": padding_mask,
            "kd": torch.tensor(element.Kd, dtype=torch.float16),
        }

    def __len__(self):
        return len(self.df)


class LabelEncodingDataset(Dataset):
    """ """

    def __init__(self, df: pd.DataFrame, embedded_sequence_length: int = 600):
        # TODO: Add warning to let user know what number of sequences were filtered out because of length
        self.df = df[
            (df.sequence_a.str.len() + df.sequence_alpha.str.len())
            <= embedded_sequence_length
        ]
        self.embedded_sequence_length = embedded_sequence_length

        self.padding_character = "$"
        self.delimiter_character = "#"
        self.allowed_labels = [
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
            self.delimiter_character,
            self.padding_character,
        ]
        self.encoding = {
            amino_acid: index for index, amino_acid in enumerate(self.allowed_labels)
        }

    def __getitem__(self, index: int):
        element = self.df.iloc[index]
        sequence = (
            element.sequence_a + self.delimiter_character + element.sequence_alpha
        )

        # +1 because we added 1 character of delimiter above
        padding_mask = np.ones(self.embedded_sequence_length + 1, dtype=np.bool_)
        padding_mask[: len(sequence)] = 0

        # pad the remaining embedding with padding character '$'
        sequence = sequence + self.padding_character * (
            self.embedded_sequence_length + 1 - len(sequence)
        )

        kd = element.Kd
        encoded_sequence = torch.tensor(
            [self.encoding[amino_acid] for amino_acid in sequence], dtype=torch.int
        )

        return {
            "embedding": encoded_sequence,
            "padding_mask": padding_mask,
            "kd": torch.tensor(kd, dtype=torch.float16),
        }

    def __len__(self):
        return len(self.df)


class YMDataModule(L.LightningDataModule):
    """
    DataModule to load yeast mating datasets as Pytorch Lightning Datamodules

    Parameters:
        data_csv_path: df to be used in YMDataset
        batch_size: batch_size to use for dataloaders
    """

    def __init__(
        self, data_csv_path, dataset_class, batch_size: int = 64, seed: int = 42
    ):
        super().__init__()

        self.data_csv_path = data_csv_path
        self.batch_size = batch_size
        self.dataset_class = dataset_class
        self.seed = seed

    def setup(self, stage: str, split_size=0.9):
        df = pd.read_csv(self.data_csv_path)
        dataset = self.dataset_class(df)

        if stage == "fit":
            num_training_samples = int(len(dataset) * split_size)
            num_valid_samples = len(dataset) - num_training_samples
            self.train_dataset, self.valid_dataset = random_split(
                dataset,
                [num_training_samples, num_valid_samples],
                generator=torch.Generator().manual_seed(self.seed),
            )

        if stage == "test":
            self.test_dataset = dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=7, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=7)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=7)
