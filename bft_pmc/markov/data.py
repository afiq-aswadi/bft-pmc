"""Synthetic finite-state Markov-chain data generation."""

from __future__ import annotations

import torch
from jaxtyping import Float, Int
from torch.distributions import Dirichlet


class MarkovChainDataset:
    """Sample sequences from a fixed pool of Markov transition matrices."""

    def __init__(
        self,
        num_states: int,
        seq_len: int,
        num_chains: int,
        device: torch.device | str,
        seed: int | None = None,
    ) -> None:
        if num_states < 2:
            raise ValueError("num_states must be at least 2.")
        if seq_len < 2:
            raise ValueError("seq_len must be at least 2.")
        if num_chains < 1:
            raise ValueError("num_chains must be at least 1.")

        self.k = num_states
        self.seq_len = seq_len
        self.n_chains = num_chains
        self.device = torch.device(device)
        self.bos_token_id = self.k
        self.vocab_size = self.k + 1

        if seed is not None:
            torch.manual_seed(seed)

        dirichlet = Dirichlet(torch.ones(self.k, device=self.device))
        transition_matrices = dirichlet.sample((self.n_chains, self.k))
        assert torch.isfinite(transition_matrices).all()
        assert (transition_matrices > 0).all()
        self.transition_matrices = transition_matrices
        self.stationary_distributions = self._compute_stationary_batch(
            transition_matrices
        )

    def _compute_stationary_batch(
        self,
        transition_matrices: Float[torch.Tensor, "batch k k"],
    ) -> Float[torch.Tensor, "batch k"]:
        """Compute stationary distributions for a batch of transition matrices."""
        if transition_matrices.ndim != 3:
            raise ValueError(
                "transition_matrices must have shape (batch, k, k), "
                f"got {tuple(transition_matrices.shape)}."
            )
        batch_size, rows, columns = transition_matrices.shape
        if rows != self.k or columns != self.k:
            raise ValueError(
                f"transition_matrices must end in ({self.k}, {self.k}), "
                f"got {tuple(transition_matrices.shape)}."
            )
        if not torch.isfinite(transition_matrices).all():
            raise ValueError("transition_matrices must be finite.")
        if (transition_matrices < 0).any():
            raise ValueError("transition_matrices must be non-negative.")
        if not torch.allclose(
            transition_matrices.sum(dim=-1),
            torch.ones((batch_size, self.k), device=transition_matrices.device),
            atol=1e-5,
        ):
            raise ValueError("Each transition-matrix row must sum to one.")

        transposed = transition_matrices.transpose(-2, -1).float()
        system = transposed - torch.eye(
            self.k,
            dtype=transposed.dtype,
            device=transposed.device,
        )
        system[:, -1, :] = 1.0
        target = torch.zeros(
            (batch_size, self.k),
            dtype=transposed.dtype,
            device=transposed.device,
        )
        target[:, -1] = 1.0
        stationary = torch.linalg.solve(system, target).to(transition_matrices.dtype)
        if not torch.isfinite(stationary).all() or (stationary < -1e-6).any():
            raise ValueError("Could not compute valid stationary distributions.")
        stationary = stationary.clamp_min(0)
        return stationary / stationary.sum(dim=-1, keepdim=True)

    def sample_ood_matrix(self) -> Float[torch.Tensor, "k k"]:
        """Draw a fresh transition matrix outside the training pool."""
        dirichlet = Dirichlet(torch.ones(self.k, device=self.device))
        return dirichlet.sample((self.k,))

    def prepend_bos(
        self,
        tokens: Int[torch.Tensor, "... seq"],
    ) -> Int[torch.Tensor, "batch seq_plus_bos"]:
        """Prepend the BOS token to one sequence or a batch of sequences."""
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        elif tokens.ndim != 2:
            raise ValueError(f"tokens must be 1D or 2D, got {tuple(tokens.shape)}.")
        tokens = tokens.to(device=self.device, dtype=torch.long)

        bos = torch.full(
            (tokens.shape[0], 1),
            fill_value=self.bos_token_id,
            dtype=torch.long,
            device=self.device,
        )
        return torch.cat([bos, tokens], dim=1)

    def _generate_chains(
        self,
        transition_matrices: Float[torch.Tensor, "batch k k"],
        stationary: Float[torch.Tensor, "batch k"],
        length: int,
    ) -> Int[torch.Tensor, "batch length"]:
        """Sample a batch of Markov sequences."""
        if length < 1:
            raise ValueError("length must be positive.")
        batch_size = transition_matrices.shape[0]
        if transition_matrices.shape != (batch_size, self.k, self.k):
            raise ValueError(
                "transition_matrices must have shape "
                f"(batch, {self.k}, {self.k}), got {tuple(transition_matrices.shape)}."
            )
        if stationary.shape != (batch_size, self.k):
            raise ValueError(
                f"stationary must have shape (batch, {self.k}), "
                f"got {tuple(stationary.shape)}."
            )

        sequences = torch.empty(
            (batch_size, length),
            dtype=torch.long,
            device=self.device,
        )
        current_state = torch.multinomial(stationary, 1).squeeze(-1)
        sequences[:, 0] = current_state
        batch_indices = torch.arange(batch_size, device=self.device)

        for position in range(1, length):
            probabilities = transition_matrices[batch_indices, current_state]
            current_state = torch.multinomial(probabilities, 1).squeeze(-1)
            sequences[:, position] = current_state

        return sequences

    def sample_batch(self, batch_size: int) -> Int[torch.Tensor, "batch seq_plus_bos"]:
        """Sample a training batch of BOS-prefixed Markov sequences."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        indices = torch.randint(
            0,
            self.n_chains,
            (batch_size,),
            device=self.device,
        )
        chains = self._generate_chains(
            self.transition_matrices[indices],
            self.stationary_distributions[indices],
            self.seq_len,
        )
        return self.prepend_bos(chains)

    def sample_eval_chains(
        self,
        transition_matrix: Float[torch.Tensor, "k k"],
        length: int,
    ) -> Int[torch.Tensor, " length"]:
        """Sample one evaluation sequence from a specified transition matrix."""
        if transition_matrix.shape != (self.k, self.k):
            raise ValueError(
                f"transition_matrix must have shape ({self.k}, {self.k}), "
                f"got {tuple(transition_matrix.shape)}."
            )
        batched_matrix = transition_matrix.to(self.device).unsqueeze(0)
        stationary = self._compute_stationary_batch(batched_matrix)
        return self._generate_chains(batched_matrix, stationary, length).squeeze(0)
