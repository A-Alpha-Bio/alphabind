import random
import warnings
from abc import ABC, abstractmethod

import torch
from lightning import LightningModule
from tqdm import tqdm

from alphabind.features.featurize_using_esm_2 import load_bionemo_inferer


class SequenceGenerator(ABC):
    """
    Base class for different types of sequence generators for optimization
    """

    all_amino_acids = [
        "V",
        "Y",
        "D",
        "L",
        "P",
        "M",
        "Q",
        "G",
        "F",
        "R",
        "H",
        "A",
        "N",
        "T",
        "I",
        "E",
        "C",
        "S",
        "W",
        "K",
    ]

    @abstractmethod
    def __init__(self) -> None:
        # Remove Cystines from allowed mutations because it usually breaks the antibody
        mutable_amino_acids = self.all_amino_acids.copy()
        mutable_amino_acids.remove("C")

        # Pre-create a dictionary of allowed substitutions so we don't need to run the list comprehension every time
        substitution_dict = {}
        substitution_dict["C"] = mutable_amino_acids
        for amino_acid in mutable_amino_acids:
            substitution_dict[amino_acid] = [
                aa for aa in self.all_amino_acids if aa != amino_acid
            ]

        self.substitution_dict = substitution_dict
        self.mutable_amino_acids = mutable_amino_acids

    @abstractmethod
    def generate_proposals(self, sequence, mask):
        pass


class EditSequenceGenerator(SequenceGenerator):
    """
    Class to randomly edit sequences based on probability of Levenshtein distance and edit type (insertion, deletion, substitution)

    Parameters:
        lev_distance_frequency: Contains a dictionary of edit distance and their frequency. For example, a dictionary of {1: 0.8, 2: 0.2}
                                will choose edit distance of 1 with 0.8 probability and 2 with 0.2 probability.
        edit_type_frequency: Contains a dictionary of edit type and their frequency. The keys for edit type are as follows -
                             'ins': insertion
                             'del': deletion
                             'sub': substitution
                             For example, if edit_type_frequency is {'sub': 0.7, 'ins': 0.3}, this will choose substitution with 0.7
                             probability and insertion with 0.3 probability
        min_mask_length: minimum length to allow for proposed mask (which is the mutation region)
        max_mask_length: maximum length to allow for proposed mask (which is the mutation region)
    """

    def __init__(
        self,
        lev_distance_frequency: dict | None = None,
        edit_type_frequency: dict | None = None,
        min_mask_length: int = 60,
        max_mask_length: int = 100,
    ):
        super().__init__()

        if lev_distance_frequency is None:
            lev_distance_frequency = {1: 0.5, 2: 0.3, 3: 0.2}
        if edit_type_frequency is None:
            edit_type_frequency = {"sub": 0.8, "ins": 0.1, "del": 0.1}

        self.lev_distance_frequency = lev_distance_frequency
        self.edit_type_frequency = edit_type_frequency

        self.min_mask_length = min_mask_length
        self.max_mask_length = max_mask_length

    def substitute(
        self, sequence: str, mask: list[bool], position: int
    ) -> tuple[str, list[bool]]:
        """
        Creates a substitution given a sequence and position. Note: mask is a parameter for consistency, but it not used here

        Parameter:
            sequence: amino acid sequence of seed protein sequence
            mask: list of booleans where True represents mutable amino acid and False represents non-mutable ones. Length of mask should match length of sequence
            position: 0-indexed position where to make the substitution (starting at position 0 of full sequence)

        Returns:
            updated sequence and mask
        """
        current_amino_acid = sequence[position]
        new_amino_acid = random.choice(self.substitution_dict[current_amino_acid])

        sequence_l = list(sequence)
        sequence_l[position] = new_amino_acid
        sequence = "".join(sequence_l)
        return sequence, mask

    def insert(self, sequence, mask, position):
        """
        Creates a insertion given a sequence and position

        Parameter:
            sequence: amino acid sequence of seed protein sequence
            mask: list of booleans where True represents mutable amino acid and False represents non-mutable ones. Length of mask should match length of sequence
            position: 0-indexed position where to make the insertion (starting at position 0 of full sequence)

        Returns:
            updated sequence and mask
        """
        amino_acid_to_insert = random.choice(self.mutable_amino_acids)
        sequence = sequence[:position] + amino_acid_to_insert + sequence[position:]
        mask = mask[:position] + [True] + mask[position:]
        return sequence, mask

    def delete(self, sequence, mask, position):
        """
        Creates a deletion given a sequence and position

        Parameter:
            sequence: amino acid sequence of seed protein sequence
            mask: list of booleans where True represents mutable amino acid and False represents non-mutable ones. Length of mask should match length of sequence
            position: 0-indexed position where to make the deletion (starting at position 0 of full sequence)

        Returns:
            updated sequence and mask
        """
        sequence = sequence[:position] + sequence[position + 1 :]
        mask = mask[:position] + mask[position + 1 :]
        return sequence, mask

    def update_edit_type_frequency_according_to_length(
        self, sequence: str, edit_type_frequency: dict
    ) -> dict:
        """
        Given a sequence and edit_type_frequency, this function updates the edit_type_frequency to remove edits that will exceed max_mask_length or reduce below
        min_mask_length. Note: the output frequencies may not sum to 1

        Parameter:
            sequence: amino acid sequence of seed protein sequence
            edit_type_frequency: Contains a dictionary of edit type and their frequency. The keys for edit type are as follows -
                             'ins': insertion
                             'del': deletion
                             'sub': substitution
                             For example, if edit_type_frequency is {'sub': 0.7, 'ins': 0.3}, this will choose substitution with 0.7
                             probability and insertion with 0.3 probability
        """
        if len(sequence) == self.max_mask_length:
            edit_type_frequency.pop("ins", None)
        elif len(sequence) == self.min_mask_length:
            edit_type_frequency.pop("del", None)
        return edit_type_frequency

    def generate_proposals(
        self, sequence: str, mask: list[bool]
    ) -> tuple[str, list[bool]]:
        """
        Given a protein sequence and mask, this function creates a new protein proposal and accordingly updates the mask too

        Parameters:
            sequence: amino acid sequence of seed protein sequence
            mask: list of booleans where True represents mutable amino acid and False represents non-mutable ones. Length of mask should match length of sequence
        """
        number_of_mutations = random.choices(
            population=list(self.lev_distance_frequency.keys()),
            weights=list(self.lev_distance_frequency.values()),
        )[0]

        for _ in range(number_of_mutations):
            edit_type_frequency_for_this_sequence = (
                self.update_edit_type_frequency_according_to_length(
                    sequence, self.edit_type_frequency.copy()
                )
            )
            mutation_type = random.choices(
                population=list(edit_type_frequency_for_this_sequence.keys()),
                weights=list(edit_type_frequency_for_this_sequence.values()),
            )[0]
            mutation_idx = random.choices(range(len(sequence)), mask)[0]

            if mutation_type == "sub":
                sequence, mask = self.substitute(sequence, mask, mutation_idx)
            elif mutation_type == "ins":
                sequence, mask = self.insert(sequence, mask, mutation_idx)
            elif mutation_type == "del":
                sequence, mask = self.delete(sequence, mask, mutation_idx)
            else:
                raise ValueError(
                    "In edit_type_frequency, the keys should be one of 'sub','ins','del'"
                )

        return sequence, mask


class ESMSequentialGenerator(EditSequenceGenerator):
    """Proposes perturbed sequences by sequentially unmasking residues using ESM.

    Masked positions are unmasked in a random order. The unmasked residue is sampled from ESM's posterior predictions of residue identity at a given masked position.
    """

    # `mask_placeholder` is used as a temporary placeholder because the current mutable
    # window masking scheme relies upon fixed-length amino acid token representations
    # (single character).
    mask_placeholder = "?"

    def __init__(
        self,
        lev_distance_frequency: dict | None = None,
        edit_type_frequency: dict | None = None,
        min_mask_length: int = 60,
        max_mask_length: int = 100,
        bionemo_inferer: LightningModule | None = None,
    ):
        """Initialize the object and, if not provided, bionemo_inferer.

        Args:
            lev_distance_frequency: Contains a dictionary of edit distance and their frequency. For example, a dictionary of {1: 0.8, 2: 0.2}
                                    will choose edit distance of 1 with 0.8 probability and 2 with 0.2 probability. Defaults to None.
            edit_type_frequency: Contains a dictionary of edit type and their frequency. The keys for edit type are as follows -
                                'ins': insertion
                                'del': deletion
                                'sub': substitution
                                For example, if edit_type_frequency is {'sub': 0.7, 'ins': 0.3}, this will choose substitution with 0.7
                                probability and insertion with 0.3 probability. Defaults to None.
            min_mask_length: minimum length to allow for proposed mask (which is the mutation region). Defaults to 60.
            max_mask_length: maximum length to allow for proposed mask (which is the mutation region). Defaults to 100.
            bionemo_inferer: BioNeMo inferer as returned by `bionemo.triton.utils.load_model_for_inference()`. Defaults to None.
        """
        super().__init__()

        if bionemo_inferer is None:
            self.bionemo_inferer = load_bionemo_inferer()
        else:
            self.bionemo_inferer = bionemo_inferer
        self.allowed_esm_vocab_token_mask = torch.tensor(
            [
                token in self.all_amino_acids
                for token in self.bionemo_inferer.tokenizer.vocab
            ],
            dtype=torch.bool,
        )  # Mask for use after truncating dummy tokens from predicted logits.

        if lev_distance_frequency is None:
            lev_distance_frequency = {1: 0.5, 2: 0.3, 3: 0.2}
        if edit_type_frequency is None:
            edit_type_frequency = {"sub": 0.8, "ins": 0.1, "del": 0.1}

        self.lev_distance_frequency = lev_distance_frequency
        self.edit_type_frequency = edit_type_frequency

        self.min_mask_length = min_mask_length
        self.max_mask_length = max_mask_length

        self.mask_token = self.bionemo_inferer.tokenizer.mask_token

    def substitute(
        self, sequence: str, mask: list[bool], position: int
    ) -> tuple[str, list[bool]]:
        """
        Prepares a substitution placeholder given a sequence and position. Note: mask is a parameter for consistency, but it is not used here.

        Parameter:
            sequence: amino acid sequence of seed protein sequence
            mask: list of booleans where True represents mutable amino acid and False represents non-mutable ones. Length of mask should match length of sequence
            position: 0-indexed position where to make the substitution (starting at position 0 of full sequence)

        Returns:
            updated sequence and mask
        """
        sequence_l = list(sequence)
        sequence_l[position] = self.mask_placeholder
        sequence = "".join(sequence_l)
        return sequence, mask

    def insert(self, sequence, mask, position):
        """
        Creates a insertion placeholder given a sequence and position.

        Parameter:
            sequence: amino acid sequence of seed protein sequence
            mask: list of booleans where True represents mutable amino acid and False represents non-mutable ones. Length of mask should match length of sequence
            position: 0-indexed position where to make the insertion (starting at position 0 of full sequence)

        Returns:
            updated sequence and mask
        """
        amino_acid_to_insert = self.mask_placeholder
        sequence = sequence[:position] + amino_acid_to_insert + sequence[position:]
        mask = mask[:position] + [True] + mask[position:]
        return sequence, mask

    def generate_proposals(
        self, sequence: str, mask: list[bool]
    ) -> tuple[str, list[bool]]:
        """
        Given a protein sequence and mask, this function creates a new placeholder protein proposal and accordingly updates the mask.

        Sequences prepared by this routine are intended to be further processed via a call to `self._unmask_with_esm()`.

        Parameters:
            sequence: amino acid sequence of seed protein sequence
            mask: list of booleans where True represents mutable amino acid and False represents non-mutable ones. Length of mask should match length of sequence
        """
        number_of_mutations = random.choices(
            population=list(self.lev_distance_frequency.keys()),
            weights=list(self.lev_distance_frequency.values()),
        )[0]

        for _ in range(number_of_mutations):
            edit_type_frequency_for_this_sequence = (
                self.update_edit_type_frequency_according_to_length(
                    sequence, self.edit_type_frequency.copy()
                )
            )
            mutation_type = random.choices(
                population=list(edit_type_frequency_for_this_sequence.keys()),
                weights=list(edit_type_frequency_for_this_sequence.values()),
            )[0]
            mutation_idx = random.choices(range(len(sequence)), mask)[0]

            if mutation_type == "sub":
                sequence, mask = self.substitute(sequence, mask, mutation_idx)
            elif mutation_type == "ins":
                sequence, mask = self.insert(sequence, mask, mutation_idx)
            elif mutation_type == "del":
                sequence, mask = self.delete(sequence, mask, mutation_idx)
            else:
                raise ValueError(
                    "In edit_type_frequency, the keys should be one of 'sub','ins','del'"
                )

        sequence = self._unmask_with_esm(sequence)

        return sequence, mask

    @torch.no_grad()
    def _hidden_states_to_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Take hidden_states from bionemo_inferer.seq_to_hiddens(seqs) and apply the LM head.

        hidden_states.shape: <n_batch, n_max_seq_length_for_batch, 2560 hidden state dimensions>

        The logit shape has a fixed length of 128 tokens, however, ESM2nv only uses a vocabulary of 33.
        So, we pare down to only the relevant vocabulary classes.
        """
        with torch.autocast(
            device_type="cuda", enabled=self.bionemo_inferer.model.enable_autocast
        ):
            lm_out = self.bionemo_inferer.model.model.lm_head(
                hidden_states, self.bionemo_inferer.model.model.word_embeddings_weight()
            )  # `lm_head` is the language model output head.
        logits = (
            lm_out[:, :, : self.bionemo_inferer.tokenizer.vocab_size].detach().cpu()
        )
        return logits

    @torch.no_grad()
    def _logits_to_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """Converts ESM LM output logits for a single sequence into a valid categorical probability distribution, zeroing the probabilities of forbidden tokens.

        We zero the (post-softmax) probabilities of tokens that are not in `self.all_amino_acids`.

        Args:
            logits: An unbatched tensor of shape  (L, V) containing LM output logits (NOT hidden states!) from ESM for a single sequence.

        Returns:
            A tensor of probabilities with the same shape as `logits`.
        """
        logits = torch.where(
            self.allowed_esm_vocab_token_mask, logits, -torch.inf
        )  # -torch.inf zeros the probabilities after softmax
        probabilities = torch.softmax(logits, dim=-1)

        return probabilities

    @torch.no_grad()
    def _unmask_with_esm(self, sequence: str) -> str:
        """Unmasks a partially masked sequence one residue at a time using ESM.

        Args:
            sequence: An amino acid sequence prepared using `self.generate_proposals()`).

        Returns:
            An unmasked amino acid sequence.
        """
        masked_positions = [
            idx for idx, char in enumerate(sequence) if char == self.mask_placeholder
        ]
        random.shuffle(
            masked_positions
        )  # We unmask tokens sequentially in a random order
        if not masked_positions:
            return sequence

        sequence_residues = [
            token.replace(self.mask_placeholder, self.mask_token) for token in sequence
        ]  # Convert the single-character placeholder mask representation into the true '<mask>' token expected by ESM's tokenizer.

        for position in masked_positions:
            # Unmask a single token.
            hidden_states_batch, pad_masks_batch = self.bionemo_inferer.seq_to_hiddens(
                ["".join(sequence_residues)]
            )

            logits_batch = self._hidden_states_to_logits(hidden_states_batch)
            token_probabilities = self._logits_to_probabilities(
                logits_batch[0][pad_masks_batch[0].detach().cpu()]
            )

            # NOTE: In rare cases, we observed `torch.multinomial` sampling indices at which
            # `input` (`masked_token_probabilities`) was `0.`. This appears to be a
            # difficult to reproduce bug that has been previously reported as fixed, but has
            # since either regressed or was never entirely mitigated. See:
            # https://github.com/pytorch/pytorch/issues/13867
            # https://github.com/pytorch/pytorch/pull/16075
            # https://github.com/pytorch/pytorch/issues/48841
            # https://github.com/pytorch/pytorch/issues/50034
            #
            # TODO: Implement a retry mechanism within this routine such that the current
            # ad-hoc handling in `self._get_valid_residue_from_token_id()` is no longer
            # necessary.
            encoded_unmasked_token = torch.multinomial(
                token_probabilities[position], num_samples=1
            )
            decoded_token = self.bionemo_inferer.tokenizer.ids_to_text(
                [encoded_unmasked_token]
            ).replace(" ", "")

            if len(decoded_token) == 1 and decoded_token[0] in self.all_amino_acids:
                unmasked_residue = decoded_token[0]
            else:
                print(
                    f"WARNING: Invalid detokenization encountered: {decoded_token=}. Selecting an unmasking residue at random from `self.all_amino_acids`."
                )
                # This handles a rare edge case where the tokenizer may not detokenize
                # to a valid amino acid. We do not know the original amino acid here, so
                # we default to selecting an allowed one at random.
                unmasked_residue = random.choices(self.all_amino_acids)[0]

            sequence_residues[position] = unmasked_residue

        return "".join(sequence_residues)


class ESMSimultaneousGenerator(EditSequenceGenerator):
    """Proposes perturbed sequences by sequentially unmasking residues using ESM.

    Masked positions are unmasked in a random order. The unmasked residue is sampled from ESM's posterior predictions of residue identity at a given masked position.
    """

    # `insertion_mask_placeholder` is used as a temporary placeholder because the current mutable
    # window masking scheme relies upon fixed-length amino acid token representations
    # (single character).
    insertion_mask_placeholder = "?"

    def __init__(
        self,
        lev_distance_frequency: dict | None = None,
        edit_type_frequency: dict | None = None,
        min_mask_length: int = 60,
        max_mask_length: int = 100,
        bionemo_inferer: LightningModule | None = None,
        batch_size: int = 512,
    ):
        """Initialize the object and, if not provided, bionemo_inferer.

        Args:
            lev_distance_frequency: Contains a dictionary of edit distance and their frequency. For example, a dictionary of {1: 0.8, 2: 0.2}
                                    will choose edit distance of 1 with 0.8 probability and 2 with 0.2 probability. Defaults to None.
            edit_type_frequency: Contains a dictionary of edit type and their frequency. The keys for edit type are as follows -
                                'ins': insertion
                                'del': deletion
                                'sub': substitution
                                For example, if edit_type_frequency is {'sub': 0.7, 'ins': 0.3}, this will choose substitution with 0.7
                                probability and insertion with 0.3 probability. Defaults to None.
            min_mask_length: minimum length to allow for proposed mask (which is the mutation region). Defaults to 60.
            max_mask_length: maximum length to allow for proposed mask (which is the mutation region). Defaults to 100.
            bionemo_inferer: BioNeMo inferer as returned by `bionemo.triton.utils.load_model_for_inference()`. Defaults to None.
            batch_size: Batch size to use for ESM inference. Defaults to 512.
        """
        super().__init__()

        if bionemo_inferer is None:
            self.bionemo_inferer = load_bionemo_inferer()
        else:
            self.bionemo_inferer = bionemo_inferer
        self.allowed_esm_vocab_token_mask = torch.tensor(
            [
                token in self.all_amino_acids
                for token in self.bionemo_inferer.tokenizer.vocab
            ],
            dtype=torch.bool,
        )  # Mask for use after truncating dummy tokens from predicted logits.

        # Pre-generate a map of token indices
        self.tokenizer_vocab_to_index = {
            token: self.bionemo_inferer.tokenizer.vocab.index(token)
            for token in self.bionemo_inferer.tokenizer.vocab
        }

        if lev_distance_frequency is None:
            lev_distance_frequency = {1: 0.5, 2: 0.3, 3: 0.2}
        if edit_type_frequency is None:
            edit_type_frequency = {"sub": 0.8, "ins": 0.1, "del": 0.1}

        self.lev_distance_frequency = lev_distance_frequency
        self.edit_type_frequency = edit_type_frequency

        self.min_mask_length = min_mask_length
        self.max_mask_length = max_mask_length

        self.batch_size = batch_size

        self.mask_token = self.bionemo_inferer.tokenizer.mask_token

    def substitute(
        self, sequence: str, mask: list[bool], position: int
    ) -> tuple[str, list[bool]]:
        """
        Prepares a substitution placeholder given a sequence and position. Note: mask is a parameter for consistency, but it is not used here.

        Parameter:
            sequence: amino acid sequence of seed protein sequence
            mask: list of booleans where True represents mutable amino acid and False represents non-mutable ones. Length of mask should match length of sequence
            position: 0-indexed position where to make the substitution (starting at position 0 of full sequence)

        Returns:
            updated sequence and mask
        """
        sequence_l = list(sequence)
        sequence_l[position] = sequence_l[
            position
        ].lower()  # We use lower-cased residue representations as an indication that the residue should be unmasked while still tracking the original residue identity.
        sequence = "".join(sequence_l)
        return sequence, mask

    def insert(self, sequence, mask, position):
        """
        Creates a insertion placeholder given a sequence and position

        Parameter:
            sequence: amino acid sequence of seed protein sequence
            mask: list of booleans where True represents mutable amino acid and False represents non-mutable ones. Length of mask should match length of sequence
            position: 0-indexed position where to make the insertion (starting at position 0 of full sequence)

        Returns:
            updated sequence and mask
        """
        amino_acid_to_insert = self.insertion_mask_placeholder
        sequence = sequence[:position] + amino_acid_to_insert + sequence[position:]
        mask = mask[:position] + [True] + mask[position:]
        return sequence, mask

    def generate_proposals(
        self, sequence: str, mask: list[bool]
    ) -> tuple[str, list[bool]]:
        """
        Given a protein sequence and mask, this function creates a new placeholder protein proposal and accordingly updates the mask.

        Sequences prepared by this routine are intended to be further processed via a call to `self._unmask_with_esm()` (or in a batch, by calling `self.unmask_with_esm()`).

        Parameters:
            sequence: amino acid sequence of seed protein sequence
            mask: list of booleans where True represents mutable amino acid and False represents non-mutable ones. Length of mask should match length of sequence
        """
        if not sequence.isupper():
            warnings.warn(
                f"Converting `sequence` to uppercase. Encountered lowercase characters in {sequence=}"
            )
            sequence = sequence.upper()

        number_of_mutations = random.choices(
            population=list(self.lev_distance_frequency.keys()),
            weights=list(self.lev_distance_frequency.values()),
        )[0]

        for _ in range(number_of_mutations):
            edit_type_frequency_for_this_sequence = (
                self.update_edit_type_frequency_according_to_length(
                    sequence, self.edit_type_frequency.copy()
                )
            )
            mutation_type = random.choices(
                population=list(edit_type_frequency_for_this_sequence.keys()),
                weights=list(edit_type_frequency_for_this_sequence.values()),
            )[0]
            mutation_idx = random.choices(range(len(sequence)), mask)[0]

            if mutation_type == "sub":
                sequence, mask = self.substitute(sequence, mask, mutation_idx)
            elif mutation_type == "ins":
                sequence, mask = self.insert(sequence, mask, mutation_idx)
            elif mutation_type == "del":
                sequence, mask = self.delete(sequence, mask, mutation_idx)
            else:
                raise ValueError(
                    "In edit_type_frequency, the keys should be one of 'sub','ins','del'"
                )

        return sequence, mask

    @torch.no_grad()
    def _hidden_states_to_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Take hidden_states from bionemo_inferer.seq_to_hiddens(seqs) and apply the LM head.

        hidden_states.shape: <n_batch, n_max_seq_length_for_batch, 2560 hidden state dimensions>

        The logit shape has a fixed length of 128 tokens, however, ESM2nv only uses a vocabulary of 33.
        So, we pare down to only the relevant vocabulary classes.
        """
        with torch.autocast(
            device_type="cuda", enabled=self.bionemo_inferer.model.enable_autocast
        ):
            lm_out = self.bionemo_inferer.model.model.lm_head(
                hidden_states, self.bionemo_inferer.model.model.word_embeddings_weight()
            )  # `lm_head` is the language model output head.
        logits = (
            lm_out[:, :, : self.bionemo_inferer.tokenizer.vocab_size].detach().cpu()
        )
        return logits

    @torch.no_grad()
    def _logits_to_probabilities(
        self,
        logits: torch.Tensor,
        forbidden_original_residue_logit_indices: list[tuple[int, int, int]],
    ) -> torch.Tensor:
        """Converts ESM LM output logits into valid categorical probability distributions, zeroing the probabilities of forbidden tokens.

        We zero the (post-softmax) probabilities of tokens that are not in self.all_amino_acids as well as at any elements specified in `forbidden_original_residue_logit_indices`.

        Args:
            logits: A batch-first tensor of shape (B, L, V) containing LM output logits (NOT hidden states!) from ESM.
            forbidden_original_residue_logit_indices: A list of indices which specifies elements in `logits` that should be zero after softmax.

        Returns:
            A batch-first tensor of probabilities with the same shape as `logits`.
        """
        logits = torch.where(
            self.allowed_esm_vocab_token_mask, logits, -torch.inf
        )  # -torch.inf zeros the probabilities after softmax
        forbidden_original_idx = torch.tensor(
            forbidden_original_residue_logit_indices, dtype=torch.int
        )
        logits[
            forbidden_original_idx[:, 0],
            forbidden_original_idx[:, 1],
            forbidden_original_idx[:, 2],
        ] = -torch.inf  # Zero the post-softmax probabilities for the original residue of any mutated positions.

        probabilities = torch.softmax(logits, dim=-1)

        return probabilities

    @torch.no_grad()
    def unmask_with_esm(self, sequences: list[str]) -> list[str]:
        """Unmasks all mask-placeholder characters in a list of sequences using ESM.

        The unmasking process is done by sampling each masked position from a single ESM inference pass.

        Args:
            sequences: A list of sequences containing masking placeholder characters, prepared by `self.generate_proposals()`.

        Returns:
            A list of unmasked amino acid sequences.
        """
        output = []
        for i in tqdm(range(0, len(sequences), self.batch_size)):
            output.extend(self._unmask_with_esm(sequences[i : i + self.batch_size]))
        return output

    @torch.no_grad()
    def _unmask_with_esm(self, sequences: list[str]) -> list[str]:
        """Helper routine to unmask a single batch of sequences using ESM.

        The unmasking process is done by sampling each masked position from a single ESM inference pass.

        Args:
            sequences: A list of sequences containing masking placeholder characters, prepared by `self.generate_proposals()`.

        Returns:
            A batch of unmasked amino acid sequences.
        """
        (
            masked_logit_indices,
            masked_residue_position_indices,
            forbidden_original_residue_logit_indices,
        ) = self._get_masked_positions_and_original_residues(sequences)

        sequence_residues = [
            [
                token.upper().replace(self.insertion_mask_placeholder, self.mask_token)
                for token in sequence
            ]
            for sequence in sequences
        ]  # Convert the single-character placeholder mask representation into the true '<mask>' token expected by ESM's tokenizer.

        hidden_states_batch, pad_masks_batch = self.bionemo_inferer.seq_to_hiddens(
            ["".join(residues) for residues in sequence_residues]
        )
        logits_batch = self._hidden_states_to_logits(hidden_states_batch)
        token_probabilities_batch = self._logits_to_probabilities(
            logits_batch, forbidden_original_residue_logit_indices
        )

        residue_values = self._sample_residues_from_probabilities(
            token_probabilities_batch, masked_logit_indices
        )

        sequence_residues = self._replace_masked_positions(
            sequence_residues, masked_residue_position_indices, residue_values
        )

        return ["".join(residues) for residues in sequence_residues]

    # TODO: Improve this by excluding sequences that only contain deletions from ESM
    # unmasking. This will involve needing to index into a subset of all sequences and
    # is omitted for now because the vast majority of sequences contain one or more
    # mask token placeholders.
    def _get_masked_positions_and_original_residues(
        self, sequences: list[str], model_has_sos: bool = True
    ) -> tuple[
        list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int, int]]
    ]:
        """Returns lists of indices for indexing masked logits, masked characters, and disallowed logits.

        For substitutions, this algorithm expects a lower-case placeholder of the
        original residue. For insertions, it expects a placeholder of
        `self.insertion_mask_placeholder` and will denote the original residue as None.

        Args:
            sequence: An amino acid sequence produced by `self.generate_proposals()`
            model_has_sos: Whether self.bionemo_inferer.tokenizer prepends an '<sos>' token.
        """
        masked_logit_indices = []
        masked_residue_position_indices = []  # The indices after subsetting where the returned `pad_mask` from ESM is `True`.
        forbidden_original_residue_logit_indices = []  # Used to set these logits to -torch.inf prior to softmax so that ESM cannot unmask to the original residue

        char_position_offset = 1 if model_has_sos else 0

        for batch_idx, sequence in enumerate(sequences):
            for char_idx, char in enumerate(sequence):
                # NOTE: We need to handle the difference between pad-masked positions and
                # positions after stripping the padding. Note that this is exactly a +1
                # shift to the logit position indices (due to the tokenizer for
                # ESM2 prepending a `<sos>` token).
                #
                # Ultimately, this is because we want to index into a dense matrix (for
                # performance) when performing softmax and masking forbidden original
                # residues.
                if char == self.insertion_mask_placeholder:  # Proxy for insertion
                    masked_logit_indices.append(
                        (batch_idx, char_idx + char_position_offset),
                    )
                    masked_residue_position_indices.append((batch_idx, char_idx))
                elif char.islower():  # Proxy for mutation
                    masked_logit_indices.append(
                        (batch_idx, char_idx + char_position_offset)
                    )
                    masked_residue_position_indices.append((batch_idx, char_idx))

                    # If `char.upper()` is not in the tokenizer vocab (unlikely), then
                    # ESM cannot output the original `char`, so there is no need to add
                    # an entry to `forbidden_original_residue_logit_indices` for this
                    # masked `char`.
                    if char.upper() in self.tokenizer_vocab_to_index:
                        forbidden_original_residue_logit_indices.append(
                            (
                                batch_idx,
                                char_idx + char_position_offset,
                                self.tokenizer_vocab_to_index[char.upper()],
                            )
                        )

        return (
            masked_logit_indices,
            masked_residue_position_indices,
            forbidden_original_residue_logit_indices,
        )

    def _sample_residues_from_probabilities(
        self, probabilities: torch.Tensor, masked_logit_indices: list[tuple[int, int]]
    ) -> list[str]:
        """Given a batched tensor of one-hot probabilities and a list of masked positions at which to sample, generates a sampled list of unmasked residues at all masked positions.

        Args:
            probabilities: A batch-first tensor of one-hot residue probabilities of shape (B, L, V).
            masked_logit_indices: A list of indices that index into `probabilities` to return the residue probabilities associated with a masked position.

        Returns:
            A list of unmasked residues corresponding to the masked positions specified in `masked_logit_indices`.
        """
        masked_indices = torch.tensor(masked_logit_indices, dtype=torch.int)
        masked_token_probabilities = probabilities[
            masked_indices[:, 0], masked_indices[:, 1]
        ]
        # NOTE: In rare cases, we observed `torch.multinomial` sampling indices at which
        # `input` (`masked_token_probabilities`) was `0.`. This appears to be a
        # difficult to reproduce bug that has been previously reported as fixed, but has
        # since either regressed or was never entirely mitigated. See:
        # https://github.com/pytorch/pytorch/issues/13867
        # https://github.com/pytorch/pytorch/pull/16075
        # https://github.com/pytorch/pytorch/issues/48841
        # https://github.com/pytorch/pytorch/issues/50034
        #
        # TODO: Implement a retry mechanism within this routine such that the current
        # ad-hoc handling in `self._get_valid_residue_from_token_id()` is no longer
        # necessary.
        residue_ids = torch.multinomial(masked_token_probabilities, num_samples=1)
        residues = [self._get_valid_residue_from_token_id(id) for id in residue_ids]
        return residues

    def _get_valid_residue_from_token_id(self, token_id: int) -> str:
        """Detokenizes a token index into a valid amino acid residue.

        Args:
            token_id: A token index in the half-open range: [0, `len(self.bionemo_inferer.tokenizer.vocab)`).

        Returns:
            A valid detokenized amino acid. If detokenization yielded an invalid amino acid, one will be sampled uniformly from the allowed set.
        """
        decoded_token = self.bionemo_inferer.tokenizer.ids_to_text([token_id]).replace(
            " ", ""
        )

        if len(decoded_token) == 1 and decoded_token[0] in self.all_amino_acids:
            unmasked_residue = decoded_token[0]
        else:
            print(
                f"WARNING: Invalid detokenization encountered: {decoded_token=}. Selecting an unmasking residue at random from `self.all_amino_acids`."
            )
            # This handles a rare edge case where the tokenizer may not detokenize
            # to a valid amino acid. We do not know the original amino acid here, so
            # we default to selecting an allowed one at random.
            unmasked_residue = random.choices(self.all_amino_acids)[0]

        return unmasked_residue

    def _replace_masked_positions(
        self,
        sequence_residues: list[list[str]],
        residue_idxs: list[tuple[int, int]],
        residue_values: list[str],
    ) -> list[list[str]]:
        """Fills `sequence_residues` with `residue_values` at `residue_idxs`.

        Args:
            sequence_residues: List of character arrays representing amino acid sequences.
            residue_idxs: Indices into `sequence_residues` that designate a specific amino acid to fill with the corresponding element from `residue_values`.
            residue_values: Residue characters to write at `residue_idxs` in `sequence_residues`.

        Returns:
            List of character arrays representing amino acid sequences, with formerly masked residues replaced by `residue_values`.
        """
        for (batch_idx, residue_idx), residue in zip(
            residue_idxs, residue_values, strict=True
        ):
            sequence_residues[batch_idx][residue_idx] = residue

        return sequence_residues
