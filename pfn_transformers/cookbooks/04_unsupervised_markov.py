"""
Cookbook 4: Unsupervised Markov Chains

Demonstrates training a PFN to learn discrete Markov chains with random
transition matrices. The PFN learns transition dynamics from sequences.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch

from pfn_transformerlens import (
    train,
    TrainingConfig,
    UnsupervisedConfig,
)


def sample_transition_matrix(num_states: int) -> torch.Tensor:
    """Sample a random transition matrix using Dirichlet distribution"""
    alpha = torch.ones(num_states)
    transitions = torch.stack(
        [torch.distributions.Dirichlet(alpha).sample() for _ in range(num_states)]
    )
    return transitions


def main():
    output_dir = Path(__file__).parent / "outputs" / "04_markov"
    output_dir.mkdir(parents=True, exist_ok=True)

    num_states = 4

    # custom generator for Markov chains (simpler than using ProbabilisticGenerator)
    class MarkovChainGenerator:
        def __init__(self, num_states: int):
            self.num_states = num_states
            self.input_dim = 1  # required by protocol

        def generate(self, seq_len: int) -> torch.Tensor:
            # sample random transition matrix
            transition_matrix = sample_transition_matrix(self.num_states)

            # generate sequence
            states = torch.zeros(seq_len, dtype=torch.long)
            states[0] = torch.randint(0, self.num_states, (1,))
            for i in range(1, seq_len):
                probs = transition_matrix[states[i - 1]]
                states[i] = torch.multinomial(probs, 1)

            return states

    data_gen = MarkovChainGenerator(num_states)

    model_config = UnsupervisedConfig(
        d_model=128,
        n_layers=4,
        n_heads=4,
        d_head=32,
        n_ctx=128,
        d_vocab=num_states,
        input_type="discrete",
        prediction_type="distribution",
        act_fn="gelu",
    )

    training_config = TrainingConfig(
        batch_size=128,
        num_steps=300,
        learning_rate=1e-3,
        warmup_steps=30,
        log_every=100,
        use_wandb=False,
    )

    model = train(data_gen, model_config, training_config)

    # test: compare learned vs true transitions
    device = next(model.parameters()).device
    true_tm = sample_transition_matrix(num_states)

    # generate sequence using true transition matrix
    def generate_markov_sequence(tm: torch.Tensor, length: int) -> torch.Tensor:
        states = torch.zeros(length, dtype=torch.long)
        states[0] = torch.randint(0, num_states, (1,))
        for i in range(1, length):
            probs = tm[states[i - 1]]
            states[i] = torch.multinomial(probs, 1)
        return states

    sequence = generate_markov_sequence(true_tm, 50).to(device)

    # predict next-state probabilities for each state
    learned_tm = torch.zeros(num_states, num_states)

    for state in range(num_states):
        # find positions where this state occurs (not at end)
        positions = (sequence[:-1] == state).nonzero(as_tuple=True)[0]
        if len(positions) > 0:
            # use last occurrence for prediction
            idx = positions[-1].item()
            context = sequence[: idx + 1].unsqueeze(0)
            with torch.no_grad():
                pred = model.predict_on_prompt(context)
            learned_tm[state] = pred.probs[0, -1].cpu()

    # plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # true transition matrix
    sns.heatmap(
        true_tm.numpy(),
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        ax=axes[0],
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Probability"},
    )
    axes[0].set_title("True Transition Matrix")
    axes[0].set_xlabel("Next State")
    axes[0].set_ylabel("Current State")

    # learned transition matrix
    sns.heatmap(
        learned_tm.numpy(),
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        ax=axes[1],
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Probability"},
    )
    axes[1].set_title("Learned Transition Matrix")
    axes[1].set_xlabel("Next State")
    axes[1].set_ylabel("Current State")

    plt.tight_layout()
    plt.savefig(output_dir / "markov_transitions.png", dpi=150)
    plt.close()

    print(f"Saved: {output_dir / 'markov_transitions.png'}")


if __name__ == "__main__":
    main()
