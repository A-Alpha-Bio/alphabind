import math
from collections.abc import Callable
from typing import Literal, TypeAlias

import lightning as L
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.nn import functional as F

ActivationFunction: TypeAlias = Callable[[torch.Tensor], torch.Tensor] | nn.Module


class PositionalEncoding(nn.Module):
    """
    PositionalEncoding module as described in 'Attention is all you need': https://arxiv.org/pdf/1706.03762.pdf

    Parameters:
        dim (int): dimension (usually last dimension) size of the tensor to be passed into the module
        max_len (int): Maximum length of the input (usually dim_1, but dim_0 if not using batch_first)
    """

    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))

        pe = torch.zeros(1, max_len, dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        # register so that it is a member but not updated by gradient descent:
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return x


class TxEnc(nn.Module):
    """
    Transformer encoder module. TODO: Add description and reference

    Parameters:
        dim: dimension of the expected tensor (usually last dimension)
        n_heads: Number of heads to use in multihead attention of TransformerEncoderLayer
        n_layers: Number of encoder layers in the TransformerEncoder

    """

    def __init__(self, dim: int, n_heads: int, n_layers: int):
        super().__init__()

        self.pos_encoder = PositionalEncoding(dim=dim)
        tx_encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, batch_first=True, nhead=n_heads, activation="gelu"
        )

        self.tx_encoder = nn.TransformerEncoder(tx_encoder_layer, num_layers=n_layers)

    def forward(
        self,
        sequence_embeddings: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sequence_embeddings = self.pos_encoder(sequence_embeddings)
        output = self.tx_encoder(sequence_embeddings, src_key_padding_mask=padding_mask)

        return output


class TxRegressor(nn.Module):
    """
    Transformer model that takes in embedded tensors in form of tensors and runs it through transformer encoder for regression task

    Parameters:
        dim_embeddings (int): embedding size for the TxEnc
        heads (int): Number of heads in the transformer enc
        layers (int): Number of layers in the transformer enc
        dim_out (int): Output size for the full regression model
        func_out (torch.nn.functional): activation function
        dropout (float): value from 0 to 1, which equals the number proportion of TxEnc outputs dropped (usually during training)
        mode (Literal['last', 'average', 'last_vs_all']): output_mode for the model for how to treat the tx_enc results
    """

    def __init__(
        self,
        heads: int,
        layers: int,
        dim_embedding: int = 128,
        dim_out: int = 1,
        func_out: ActivationFunction = torch.relu,
        dropout: float = 0.1,
        mode: Literal["last", "average", "last_vs_all"] = "average",
    ):
        super().__init__()

        modes_valid = ("last", "average", "last_vs_all")
        if mode not in modes_valid:
            raise ValueError(f"{mode} should be one of: {modes_valid}")

        self.mode = mode

        self.func_out = func_out

        self.tx_enc = TxEnc(dim_embedding, heads, layers)

        self.dropout = nn.Dropout(p=dropout)
        self.lin_out = nn.Linear(dim_embedding, dim_out)

    def forward(
        self, sequence_embeddings: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        output = self.tx_enc(sequence_embeddings, padding_mask=padding_mask)

        if padding_mask is not None:
            last_index = (~padding_mask).nonzero()[-1][1]
        else:
            last_index = output.shape[1] - 1

        if self.mode == "average":
            if padding_mask is None:
                final_output = torch.mean(output[:, :last_index, :], dim=1)
            else:
                output[padding_mask] = 0
                output = torch.sum(output, dim=1)
                final_output = output / torch.sum(~padding_mask, dim=1).unsqueeze(-1)
        elif self.mode == "last":
            final_output = output[:, last_index, :]
        elif self.mode == "last_vs_all":
            output = torch.softmax(output, dim=1)
            final_output = output[:, last_index, :]
        else:
            raise ValueError(f"Unrecognized value {self.mode=}")

        res = self.dropout(final_output)
        res = self.lin_out(self.func_out(res))

        return res.view(-1)


class TxRegressorLabelEncoded(nn.Module):
    """
    Transformer model with an embedding layer attached to it

    Parameters:
        heads (int): Number of transformer heads in the TxRegressor
        layers (int): Number of layers of transformer heads
        embeddings_dim (int): number of features to produce from embedder which are fed into the regressor
        vocab_size (int): size of the embedding per amino acid passed into the model. For example, this will be 20 for a 1-hot vector of amino acids
        padding_idx (str): character used to indicate padding token
        dim_out (int): output dimension of the regressor. This is usually 1 when prediction Kds
        mode (Literal['last', 'average', 'last_vs_all']): Indicates what part of the output vector from the transformer encoder should be used
    """

    def __init__(
        self,
        heads: int = 4,
        layers: int = 7,
        embeddings_dim: int = 128,
        vocab_size: int = 25,
        padding_idx: int = 24,
        dim_out: int = 1,
        mode: Literal["last", "average", "last_vs_all"] = "average",
    ):
        super().__init__()
        self.embedder = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embeddings_dim,
            padding_idx=padding_idx,
        )
        self.tx = TxRegressor(
            heads,
            layers,
            dim_embedding=embeddings_dim,
            dim_out=dim_out,
            mode=mode,
            func_out=nn.LeakyReLU(0.1),
            dropout=0.1,
        )

    def forward(
        self, sequence_embeddings: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        x = self.embedder(sequence_embeddings)
        return self.tx(x, padding_mask)


class EnsembleTxRegressorLabelEncoded(nn.Module):
    """ """

    def __init__(self, modules: list[TxRegressorLabelEncoded]):
        super().__init__()
        self._submodules = nn.ModuleList(modules)

    def forward(self, sequence_embeddings: torch.Tensor, padding_mask: torch.Tensor):
        return torch.mean(
            torch.stack(
                [
                    submodule(sequence_embeddings, padding_mask)
                    for submodule in self._submodules
                ]
            ),
            dim=0,
        )


class TxRegressorForESMEmbedding(nn.Module):
    """
    Transformer model used in ESM-based training. This model takes pre-computed ESM embeddings as input and runs them through the TxRegressor. It expects the input the same
    shape as provided by aacore.ml.datasets.YeastMatingSequenceDatasetUsingESMEmbeddings.

    Parameters:
        heads (int): Number of transformer heads in the TxRegressor
        layers (int): Number of layers of transformer heads
        features (int): Number of features in ESM embedding. This is defaulted to 2561, which is the current size returned by YeastMatingSequenceDatasetUsingESMEmbeddings
        dims_reduce_features (tuple[int, int]): Number of features in the dimensionality reducer from esm embedding to tensor passed into the transformer
        dropout (float): value from 0 to 1, which equals the number proportion of TxEnc outputs dropped (usually during training)
        mode (Literal['last', 'average', 'last_vs_all']): output_mode for the model for how to treat the tx_enc results
    """

    def __init__(
        self,
        heads: int,
        layers: int,
        features: int = 2561,
        dims_reduce_features: tuple[int, int] = (2048, 256),
        dropout: float = 0.1,
        mode: Literal["last", "average", "last_vs_all"] = "average",
    ):
        super().__init__()

        n_features_2, n_features_3 = dims_reduce_features
        self.dim_reducer_1 = nn.Linear(features, n_features_2)
        self.dim_reducer_2 = nn.Linear(n_features_2, n_features_3)
        self.dropout = nn.Dropout(p=dropout)

        self.tx = TxRegressor(
            heads,
            layers,
            dim_embedding=n_features_3,
            dim_out=1,
            dropout=dropout,
            mode=mode,
        )

    def forward(
        self, sequence_embeddings: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        x = self.dim_reducer_1(sequence_embeddings)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.dim_reducer_2(x)
        return self.tx(x, padding_mask)


class TransformerLightning(L.LightningModule):
    """
    Pytorch Lightning module to train the TxRegressor model

    Parameters:
        tx_model: TxRegressorForESMEmbedding module
        lr: Learning rate for training
    """

    def __init__(
        self,
        tx_model: TxRegressorForESMEmbedding | TxRegressorLabelEncoded,
        lr: float = 1e-5,
    ):
        super().__init__()
        self.tx_model = tx_model
        self.lr = lr

        self.kds = []
        self.kd_preds = []

    def training_step(self, batch, batch_idx):
        sequence_embeddings = batch["embedding"]
        padding_mask = batch["padding_mask"]
        kds = batch["kd"]
        kd_preds = self.tx_model(sequence_embeddings, padding_mask)
        loss = F.mse_loss(kd_preds, kds, reduction="mean")
        self.log("loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sequence_embeddings = batch["embedding"]
        padding_mask = batch["padding_mask"]
        kds = batch["kd"]
        kd_preds = self.tx_model(sequence_embeddings, padding_mask)
        loss = F.mse_loss(kd_preds, kds, reduction="mean")
        self.log("val_loss", loss, prog_bar=True)

        self.kds.append(kds.cpu())
        self.kd_preds.append(kd_preds.cpu())
        return loss

    def on_validation_epoch_end(self) -> None:
        kds = torch.cat(self.kds, dim=0)
        kd_preds = torch.cat(self.kd_preds, dim=0)

        rho, pval = spearmanr(kds, kd_preds)
        self.log("spearman_rho", rho, prog_bar=True)
        self.log("spearman_pval", pval, prog_bar=False)

        self.kds.clear()
        self.kd_preds.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
