import itertools
import os
import random
from typing import List, Tuple, Union, Optional, Dict

import numpy as np
import torch
from Bio import SeqIO

from proteingym.models.model_repos import esm
from proteingym.utils.scoring_utils import get_optimal_window
from proteingym.wrappers.generic_models import (AlignmentModel,
                                                ProteinLanguageModel)


class MSATransformerModel(AlignmentModel, ProteinLanguageModel):

    def __init__(
        self,
        num_msa_samples: int = 400,
        sampling_strategy: str = "random",
        **kwargs
    ):
        AlignmentModel.__init__(self, **kwargs)
        ProteinLanguageModel.__init__(self, **kwargs)
        self.num_msa_samples = num_msa_samples
        self.sampling_strategy = sampling_strategy
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(
            self.model_checkpoint
        )
        self.batch_converter = self.alphabet.get_batch_converter()
        self.processed_msa = self.read_msa(self.num_msa_samples, self.sampling_strategy)
    
    @property
    def base_model(self):
        return self.model

    def read_msa(self, nseqs: int, sampling_strategy: str) -> List[Tuple[str, str]]:
        """
        Args:
            nseqs (int): number of sequences to sample from the MSA
            sampling_strategy (str): How to sample from the MSA. Must be one of ["random", "first-x-rows", "sequence-reweighting"]

        Returns:
            List[Tuple[str, str]]: List of tuples of the form (sequence_name, sequence)
        """
        # TODO: Maybe change this to use a seeded generator rather than a global seed
        # TODO: Can probably optimize this to avoid making copies of lots of sequences from msa

        msa = []
        if sampling_strategy not in ["random", "first_x_rows", "sequence-reweighting"]:
            raise ValueError(
                "sampling_strategy must be one of ['random', 'first_x_rows', 'sequence-reweighting']"
            )
        if sampling_strategy == "first_x_rows":
            msa = [
                (record.description, str(record.seq))
                for record in itertools.islice(
                    SeqIO.parse(self.alignment_file, "fasta"), nseqs
                )
            ]
        elif sampling_strategy == "random":
            msa = [
                (record.description, str(record.seq))
                for record in SeqIO.parse(self.alignment_file, "fasta")
            ]
            nseqs = min(len(msa), nseqs)
            msa = random.sample(msa, nseqs)
        elif sampling_strategy == "sequence-reweighting":
            assert (
                self.alignment.weights_file is not None
            ), "Weights file must be provided for sequence-reweighting"
            all_sequences_msa = []
            weights = []
            msa = []
            for seq_name in self.alignment.seq_name_to_sequences.keys():
                if seq_name == self.alignment.focus_seq_id:
                    msa.append(
                        (seq_name, self.alignment.seq_name_to_sequences[seq_name])
                    )
                    # TODO: Check that deleting file from alignment here is only done for memory purposes
                    del self.alignment.seq_name_to_weights[seq_name]
                else:
                    if seq_name in self.alignment.seq_name_to_weights:
                        all_sequences_msa.append(
                            (seq_name, self.alignment.seq_name_to_sequences[seq_name])
                        )
                        weights.append(self.alignment.seq_name_to_weights[seq_name])
            if len(all_sequences_msa) > 0:
                weights = (
                    np.array(weights)
                    / np.array(list(self.alignment.seq_name_to_weights.values())).sum()
                )
                msa.extend(
                    random.choices(all_sequences_msa, weights=weights, k=nseqs - 1)
                )
        msa = [(desc, seq.upper()) for desc, seq in msa]
        return msa

    def compute_token_probs(self, wt_sequence: str) -> torch.Tensor:
        """Computes the token probabilities from the model and MSA. These probabilities are used in predict_logprobs to get the probabilities
        of whole mutated sequences.

        Returns:
            token_probs (torch.Tensor): Log probabilities for each token in each position. Tensor of shape (1, seq_len, vocab_size)
        """
        _, _, batch_tokens = self.batch_converter([self.processed_msa])
        all_token_probs = []
        for i in range(batch_tokens.size(2)):
            batch_tokens_masked = batch_tokens.clone()
            # mask out first sequence
            batch_tokens_masked[0, 0, i] = self.alphabet.mask_idx
            if batch_tokens.size(-1) > 1024:
                large_batch_tokens_masked = batch_tokens_masked.clone()
                start, end = get_optimal_window(
                    mutation_position_relative=i,
                    seq_len_wo_special=len(wt_sequence) + 2,
                    model_window=1024,
                )
                print("Start index {} - end index {}".format(start, end))
                batch_tokens_masked = large_batch_tokens_masked[:, :, start:end]
            else:
                start = 0
            with torch.no_grad():
                if not self.nogpu and torch.cuda.is_available():
                    batch_tokens_masked = batch_tokens_masked.cuda()
                token_probs = torch.log_softmax(
                    self.model(batch_tokens_masked)["logits"], dim=-1
                )
            if not self.nogpu and torch.cuda.is_available():
                all_token_probs.append(
                    token_probs[:, 0, i - start].detach().cpu()
                )  # vocab size
            else:
                all_token_probs.append(token_probs[:, 0, i - start].detach())
        token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)
        return token_probs

    def apply_wt_probabilities(
        self, sequences: List[str], wt_sequence: str, token_probs: torch.Tensor
    ) -> List[float]:
        """Applies the wild type probabilities to a list of sequences to get the log prob ratio between each sequence and the wild type.
        This is applied after the wt-marginals and masked-marginals approaches

        Args:
            sequences (List[str]): List of sequences to compute log prob ratios for
            wt_sequence (str): wild type sequence
            token_probs (torch.Tensor): tensor of shape (1, seq_len, vocab_size) representing the log probabilities for each position of the wild type sequence

        Returns:
            List[float]: List of log prob ratios for each sequence
        """
        # TODO: Ignoring offset_idx from original script here for now, since it was always zero.
        # may want to add back later for additional flexibility
        # TODO: Could switch this to iterate over only mutant strings, but that would require passing in
        # the mutants as well as the mutated sequences, so iterating over the sequences for now.
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

    def predict_logprobs(
        self, sequences: List[str], wt_sequence: Union[str, None] = None
    ):
        if wt_sequence is None:
            raise ValueError(
                "Wild type sequence must be provided for MSA transformer model"
            )
        token_probs = self.compute_token_probs(wt_sequence)
        return self.apply_wt_probabilities(sequences, wt_sequence, token_probs)

    def predict_position_logprobs(
        self, sequences: List[str], wt_sequence: Union[str, None] = None
    ) -> List[np.ndarray]:
        pass

    def get_embeddings(self, sequences: List[str], layers: List[int] = [-1], batch_size: Optional[int] = None) -> Dict[int,torch.Tensor]:
        output_reps = None
        # converting negative integers to positives since esm codebase only takes positive integers for repr_layers
        pos_layers = [val if val >= 0 else len(self.model.layers) + val for val in layers]   
        if batch_size is None:
            batch_size = len(sequences)
        for i in range(len(sequences)//batch_size):
            batch_seqs = sequences[i*batch_size:min((i+1)*batch_size,len(sequences))]
            batch_seqs_with_msa = [(f"protein{i}", seq) for i,seq in enumerate(batch_seqs)] + self.processed_msa
            _, _, batch_tokens = self.batch_converter([batch_seqs_with_msa])
            batch_tokens = batch_tokens.to(self.model.embed_tokens.weight.device)
            output = self.model(batch_tokens,repr_layers=pos_layers)
            if output_reps is None:
                output_reps = {key: val[0,-batch_size:] for key,val in output["representations"].items()}
            else:
                output_reps = {key: torch.cat((val,output["representations"][key][0,-batch_size:]),dim=0) for key,val in output_reps.items()}
        if output_reps is not None:
            # Converting layer indices back to those passed in
            output_reps = {layers[i]:output_reps[pos_layers[i]].type(torch.float32) for i in range(len(pos_layers))}
        else:
            raise ValueError("Could not get embeddings for given layers")
        return output_reps

    def get_embed_dim(self, layers: List[int] = [-1]) -> List[int]:
        # embedding dimension is same at every layer so just return that value multiple times
        return [self.model.args.embed_dim for i in range(len(layers))]

    def set_trainable_layers(self, layers: List[int] = [-1]):
        # set all parameters to requires_grad = False first 
        self.model.requires_grad_(False)
        # Layer 0 is embedding matrix 
        if 0 in layers:
            self.model.embed_tokens.requires_grad_(True) 
            self.model.embed_positions.requires_grad_(True)
        # an extra layer norm post embedding that we include with the first layer
        if 1 in layers:
            self.model.emb_layer_norm_before.requires_grad_(True)
        for i, layer in enumerate(self.model.layers): 
            if i+1 in layers:
                layer.requires_grad_(True)
            else:
                layer.requires_grad_(False)
            # Including final layer norm if last layer is trainable 
            if i == len(self.model.layers)-1:
                self.model.emb_layer_norm_after.requires_grad_(True)