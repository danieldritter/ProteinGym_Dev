"""Wrapper for ESM models."""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
import torch
from tqdm import tqdm

from proteingym.models.model_repos import esm
from proteingym.utils.scoring_utils import get_optimal_window
from proteingym.wrappers.generic_models import ProteinLanguageModel


class ESMModel(ProteinLanguageModel):
    """Wrapper class for ESM models."""

    def __init__(
        self,
        scoring_strategy: str = "wt-marginals",
        scoring_window: str = "overlapping",
        **kwargs,  # noqa: ANN003
    ) -> None:
        """Initialize ESMModel.

        Args:
            scoring_strategy (str, optional): method for computing scores. One of
                ["wt-marginals", "masked-marginals", "pseudo-ppl"] Defaults to
                "wt-marginals".
            scoring_window (str, optional): how to compute scoring window.
                Defaults to "overlapping".
            kwargs: Additional keyword arguments for ProteinLanguageModel

        """
        super().__init__(**kwargs)
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(
            self.model_checkpoint,
        )
        self.batch_converter = self.alphabet.get_batch_converter()
        self.scoring_strategy = scoring_strategy
        self.scoring_window = scoring_window
        self.model_window = 1024
        self.input_device = self.model.embed_tokens.weight.device

    @property
    def base_model(self) -> torch.nn.Module:
        """Underlying ESM model.

        Returns:
            torch.nn.Module: ESM model

        """
        return self.model

    def predict_logprobs(  # noqa: D102
        self,
        sequences: list[str],
        wt_sequence: str | None = None,
        batch_size: int | None = None,
    ) -> list[float]:
        if batch_size is not None:
            warnings.warn(
                "Batch size is currently ignored for ESM models for inference. "
                "Batch size of 1 is used",
                stacklevel=1,
            )
        if wt_sequence is None:
            msg = "wt_sequence must be provided for ESM models"
            raise ValueError(msg)
        if self.scoring_strategy != "pseudo-ppl" and not all(
            len(sequence) == len(wt_sequence) for sequence in sequences
        ):
            msg = "Some mutated sequences are of different length from the wild"
            "type. Only substitutions are allowed for scoring strategies other"
            "than pseudo-ppl"
            raise ValueError(msg)
        data = [
            ("protein1", wt_sequence),
        ]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.base_model.embed_tokens.weight.device)
        if self.scoring_strategy == "wt-marginals":
            token_probs = self.wt_marginals(batch_tokens)
            scores = self.apply_wt_probabilities(sequences, wt_sequence, token_probs)
        elif self.scoring_strategy == "masked-marginals":
            token_probs = self.masked_marginals(batch_tokens, wt_sequence)
            scores = self.apply_wt_probabilities(sequences, wt_sequence, token_probs)
        elif self.scoring_strategy == "pseudo-ppl":
            scores = self.pseudo_ppl(sequences)
        else:
            msg = f"Scoring strategy {self.scoring_strategy} not supported"
            raise ValueError(msg)
        return scores

    def wt_marginals(self, batch_tokens: torch.Tensor) -> torch.Tensor:
        """Compute scores for wildtype sequence, used later to compute mutant scores.

        Args:
            batch_tokens (torch.Tensor): tensor of shape (1, seq_len) representing
                the tokenized wild type sequence

        Returns:
            token_probs (torch.Tensor): tensor of shape (1, seq_len, vocab_size)
                representing the log probabilities for each position of the wild
                type sequence

        """
        # TODO (@DanielR): Adjust all of this to use self.model_window, instead of
        # hard-coded 1024
        overlap_threshold = 511
        with torch.no_grad():
            if (
                batch_tokens.size(1) > self.model_window
                and self.scoring_window == "overlapping"
            ):
                batch_size, seq_len = batch_tokens.shape  # seq_len includes BOS and EOS
                # Note: batch_size = 1 (need to keep batch dimension to score
                # with model though)
                token_probs = torch.zeros(
                    (batch_size, seq_len, len(self.alphabet)),
                ).to(batch_tokens.device)
                token_weights = torch.zeros((batch_size, seq_len)).to(
                    batch_tokens.device,
                )
                # 1 for 256≤i<1022-256
                weights = torch.ones(self.model_window).to(batch_tokens.device)
                for i in range(1, 257):
                    weights[i] = 1 / (1 + math.exp(-(i - 128) / 16))
                for i in range(1022 - 256, 1023):
                    weights[i] = 1 / (1 + math.exp((i - 1022 + 128) / 16))
                start_left_window = 0
                end_left_window = 1023  # First window is indexed [0-1023]
                start_right_window = (
                    (batch_tokens.size(1) - 1) - 1024 + 1
                )  # Last index is len-1
                end_right_window = batch_tokens.size(1) - 1
                while True:
                    # Left window update
                    left_window_probs = torch.log_softmax(
                        self.model(
                            batch_tokens[
                                :,
                                start_left_window : end_left_window + 1,
                            ],
                        )["logits"],
                        dim=-1,
                    )
                    token_probs[
                        :,
                        start_left_window : end_left_window + 1,
                    ] += left_window_probs * weights.view(-1, 1)
                    token_weights[:, start_left_window : end_left_window + 1] += weights
                    # Right window update
                    right_window_probs = torch.log_softmax(
                        self.model(
                            batch_tokens[
                                :,
                                start_right_window : end_right_window + 1,
                            ].to(batch_tokens.device),
                        )["logits"],
                        dim=-1,
                    )
                    token_probs[
                        :,
                        start_right_window : end_right_window + 1,
                    ] += right_window_probs * weights.view(-1, 1)
                    token_weights[
                        :,
                        start_right_window : end_right_window + 1,
                    ] += weights
                    if end_left_window > start_right_window:
                        # overlap between windows in that last scoring so we
                        # break from the loop
                        break
                    start_left_window += 511
                    end_left_window += 511
                    start_right_window -= 511
                    end_right_window -= 511
                # If central overlap not wide enough, we add one more window
                # at the center
                final_overlap = end_left_window - start_right_window + 1
                if final_overlap < overlap_threshold:
                    start_central_window = int(seq_len / 2) - 512
                    end_central_window = start_central_window + 1023
                    central_window_probs = torch.log_softmax(
                        self.model(
                            batch_tokens[
                                :,
                                start_central_window : end_central_window + 1,
                            ].to(batch_tokens.device),
                        )["logits"],
                        dim=-1,
                    )
                    token_probs[
                        :,
                        start_central_window : end_central_window + 1,
                    ] += central_window_probs * weights.view(-1, 1)
                    token_weights[
                        :,
                        start_central_window : end_central_window + 1,
                    ] += weights
                # Weight normalization
                token_probs = token_probs / token_weights.view(
                    -1,
                    1,
                )  # Add 1 to broadcast
            else:
                token_probs = torch.log_softmax(
                    self.model(batch_tokens)["logits"],
                    dim=-1,
                )
            return token_probs

    def masked_marginals(
        self,
        batch_tokens: torch.Tensor,
        wt_sequence: str,
    ) -> torch.Tensor:
        """Compute scores for the wt sequence using the masked marginals approach.

        masks each position individually and predicting the amino acid distribution
            for that position


        Args:
            batch_tokens (torch.Tensor): tensor of shape (1, seq_len) representing
                the tokenized wild type sequence
            wt_sequence (str): the wild type sequence

        Raises:
            NotImplementedError: if an overlapping scoring window is used

        Returns:
            torch.Tensor: tensor of shape (1, seq_len, vocab_size) representing the
                log probabilities for each position of the wild type sequence

        """
        all_token_probs = []
        for i in range(batch_tokens.size(1)):
            batch_tokens_masked = batch_tokens.clone()
            batch_tokens_masked[0, i] = self.alphabet.mask_idx
            if (
                batch_tokens.size(1) > self.model_window
                and self.scoring_window == "optimal"
            ):
                large_batch_tokens_masked = batch_tokens_masked.clone()
                # TODO (@DanielR): confirm that this logic is correct, rewrite to
                # avoid using wt_sequence if possible
                start, end = get_optimal_window(
                    mutation_position_relative=i,
                    seq_len_wo_special=len(wt_sequence) + 2,
                    model_window=self.model_window,
                )
                batch_tokens_masked = large_batch_tokens_masked[:, start:end]
            elif (
                batch_tokens.size(1) > self.model_window
                and self.scoring_window == "overlapping"
            ):
                msg = "Overlapping not yet implemented for masked-marginals"
                raise NotImplementedError(msg)
            else:
                start = 0
            with torch.no_grad():
                token_probs = torch.log_softmax(
                    self.model(batch_tokens_masked)["logits"],
                    dim=-1,
                )
            all_token_probs.append(token_probs[:, i - start])  # vocab size
        return torch.cat(all_token_probs, dim=0).unsqueeze(0)

    def pseudo_ppl(self, sequences: list[str]) -> list[float]:
        """Compute pseudo-likelihoodds for a list of sequences.

        This is the only scoring method that can handle indels

        Args:
            sequences (List[str]): List of sequences

        Returns:
            List[float]: List of pseudo-likelihoods

        """
        # modify the sequence
        all_seq_probs = []
        for sequence in tqdm(sequences):
            # encode the sequence
            data = [
                ("protein1", sequence),
            ]

            batch_converter = self.alphabet.get_batch_converter()

            _, _, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(self.base_model.embed_tokens.weight.device)
            # compute probabilities at each position
            log_probs = []
            for i in range(1, len(sequence) - 1):
                batch_tokens_masked = batch_tokens.clone()
                batch_tokens_masked[0, i] = self.alphabet.mask_idx
                with torch.no_grad():
                    token_probs = torch.log_softmax(
                        self.model(batch_tokens_masked)["logits"],
                        dim=-1,
                    )
                log_probs.append(
                    token_probs[0, i, self.alphabet.get_idx(sequence[i])].item(),
                )  # vocab size
            all_seq_probs.append(sum(log_probs))
        return all_seq_probs

    def apply_wt_probabilities(
        self,
        sequences: list[str],
        wt_sequence: str,
        token_probs: torch.Tensor,
    ) -> list[float]:
        """Apply the wild type probabilities to a list of sequences.

        Returns the log prob ratio between each sequence and the wild type.
        This is applied after the wt-marginals and masked-marginals approaches

        Args:
            sequences (List[str]): List of sequences to compute log prob ratios for
            wt_sequence (str): wild type sequence
            token_probs (torch.Tensor): tensor of shape (1, seq_len, vocab_size)
                representing the log probabilities for each position of the wild
                type sequence

        Returns:
            List[float]: List of log prob ratios for each sequence

        """
        # TODO (@DanielR): Ignoring offset_idx from original script here for now, since
        # it was always zero.
        # may want to add back later for additional flexibility
        # TODO (@DanielR): Could switch this to iterate over only mutant strings, but
        # that would require passing in
        # the mutants as well as the mutated sequences, so iterating over the sequences
        # for now.
        scores = []
        for seq in sequences:
            score = 0
            for i, aa in enumerate(seq):
                # difference is zero in this case
                if aa == wt_sequence[i]:
                    continue
                score += (
                    token_probs[0, 1 + i, self.alphabet.get_idx(aa)]
                    - token_probs[0, 1 + i, self.alphabet.get_idx(wt_sequence[i])]
                ).item()
            scores.append(score)
        return scores

    def get_embeddings(  # noqa: D102
        self,
        sequences: list[str],
        layers: tuple[int] = (-1,),
        batch_size: int | None = None,
    ) -> dict[int, torch.Tensor]:
        output_reps = None
        # converting negative integers to positives since esm codebase only takes
        # positive integers for repr_layers
        pos_layers = [
            val if val >= 0 else len(self.model.layers) + val + 1 for val in layers
        ]
        if batch_size is None:
            batch_size = len(sequences)
        for i in range(len(sequences) // batch_size):
            batch_seqs = sequences[
                i * batch_size : min((i + 1) * batch_size, len(sequences))
            ]
            _, _, batch_tokens = self.batch_converter(
                [("protein{i}", seq) for i, seq in enumerate(batch_seqs)],
            )
            batch_tokens = batch_tokens.to(self.model.embed_tokens.weight.device)

            output = self.model(batch_tokens, repr_layers=pos_layers)
            if output_reps is None:
                output_reps = output["representations"]
            else:
                output_reps = {
                    key: torch.cat((val, output["representations"][key]), dim=0)
                    for key, val in output_reps.items()
                }
        if output_reps is not None:
            # Converting layer indices back to those passed in
            output_reps = {
                layers[i]: output_reps[pos_layers[i]].type(torch.float32)
                for i in range(len(pos_layers))
            }
        else:
            msg = "Could not get embeddings for given layers"
            raise ValueError(msg)
        return output_reps

    def predict_position_logprobs(  # noqa: D102
        self,
        sequences: list[str],
        wt_sequence: str | None = None,
        batch_size: str | None = None,
    ) -> list[np.ndarray]:
        raise NotImplementedError

    def get_embed_dim(self, layers: tuple[int] = (-1,)) -> list[int]:  # noqa: D102
        # embedding dimension is same at every layer so just return that value
        # multiple times
        return [self.model.embed_dim for i in range(len(layers))]

    def set_trainable_layers(self, layers: tuple[int] = (-1,)) -> None:  # noqa: D102
        # set all parameters to requires_grad = False first
        self.model.requires_grad_(requires_grad=False)
        # Layer 0 is embedding matrix
        if 0 in layers:
            self.model.embed_tokens.requires_grad_(requires_grad=True)
        pos_layers = [
            val if val >= 0 else len(self.model.layers) + val + 1 for val in layers
        ]
        for i, layer in enumerate(self.model.layers):
            if i + 1 in pos_layers:
                layer.requires_grad_(requires_grad=True)
            else:
                layer.requires_grad_(requires_grad=False)
            # Including final layer norm if last layer is trainable
            if i == len(self.model.layers) - 1 and i + 1 in pos_layers:
                self.model.emb_layer_norm_after.requires_grad_(require_grad=True)
