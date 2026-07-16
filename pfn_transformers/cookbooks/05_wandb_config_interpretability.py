"""
Cookbook 5: WANDB, Config Loading, and Interpretability

Demonstrates:
1. Training with wandb logging
2. Saving and loading models by config
3. Accessing internal transformer for interpretability (activation cache)
"""

import torch
import matplotlib.pyplot as plt
from pathlib import Path

from pfn_transformerlens import (
    train,
    TrainingConfig,
    RegressionConfig,
    DeterministicGenerator,
    checkpointing,
)


def linear_function(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (w * x).sum(dim=-1)


def main():
    output_dir = Path(__file__).parent / "outputs" / "05_wandb_config"
    output_dir.mkdir(parents=True, exist_ok=True)

    # PART 1: Training with wandb (set use_wandb=True to enable)
    print("PART 1: Training with checkpointing")

    input_dim = 3
    prior = torch.distributions.Independent(
        torch.distributions.Normal(torch.zeros(input_dim), torch.ones(input_dim)),
        reinterpreted_batch_ndims=1,
    )

    data_gen = DeterministicGenerator(
        prior=prior,
        function=linear_function,
        input_dim=input_dim,
        noise_std=0.1,
    )

    model_config = RegressionConfig(
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_head=32,
        n_ctx=128,
        d_vocab=30,
        input_dim=input_dim,
        prediction_type="distribution",
        bucket_type="uniform",
        y_min=-3.0,
        y_max=3.0,
        mask_type="autoregressive-pfn",
        act_fn="gelu",
    )

    training_config = TrainingConfig(
        batch_size=64,
        num_steps=100,
        learning_rate=1e-3,
        warmup_steps=10,
        log_every=50,
        use_wandb=False,  # set to True to enable wandb
        checkpoint_dir=str(output_dir / "checkpoints"),
        save_every=100,
    )

    model = train(data_gen, model_config, training_config)

    # manually save checkpoint
    checkpoint_path = output_dir / "checkpoints" / "demo_checkpoint.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = checkpointing.CheckpointMetadata(
        timestamp="2024-01-01",
        wandb_run_id=None,
        wandb_run_name=None,
        wandb_run_url=None,
        git_hash=None,
    )

    # create dummy optimizer for checkpoint
    dummy_optimizer = torch.optim.Adam(model.parameters())

    checkpointing.save_checkpoint(
        checkpoint_path=checkpoint_path,
        step=100,
        model_state=model.state_dict(),
        optimizer_state=dummy_optimizer.state_dict(),
        model_config=model_config,
        training_config=training_config,
        metadata=metadata,
    )
    print(f"Model saved to: {checkpoint_path}")

    # PART 2: Load model from checkpoint
    print("\nPART 2: Loading model from checkpoint")

    loaded_model, _, loaded_metadata = checkpointing.load_checkpoint(
        checkpoint_path, device="auto"
    )
    print(f"Loaded model: {type(loaded_model)}")
    print("Loaded from step: 100")

    # PART 3: Access internal transformer for interpretability
    print("\nPART 3: Accessing activation cache")

    device = next(loaded_model.parameters()).device

    # create sample input
    x_test = torch.randn(1, 10, input_dim).to(device)
    y_test = torch.randn(1, 10).to(device)

    # get predictions WITH cache
    result = loaded_model.predict_on_prompt(x_test[0], y_test[0], return_cache=True)

    # when return_cache=True, returns (prediction, cache)
    if isinstance(result, tuple):
        pred, cache = result
    else:
        pred = result
        cache = pred.cache if hasattr(pred, "cache") else None

    print(f"Prediction type: {type(pred)}")
    print(f"Has cache: {cache is not None}")

    if cache is not None:
        # access transformer activations
        print(f"\nCache keys: {list(cache.keys())[:5]}...")  # show first 5 keys

        # example: visualize attention patterns
        # attention pattern shape: [batch, head, query_pos, key_pos]
        attn_pattern = cache["blocks.0.attn.hook_pattern"][
            0, 0
        ].cpu()  # layer 0, head 0

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(attn_pattern, cmap="viridis", aspect="auto")
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")
        ax.set_title("Attention Pattern (Layer 0, Head 0)")
        plt.colorbar(im, ax=ax, label="Attention Weight")
        plt.tight_layout()
        plt.savefig(output_dir / "attention_pattern.png", dpi=150)
        plt.close()
        print(f"Saved: {output_dir / 'attention_pattern.png'}")

        # example: plot residual stream norms across layers
        layer_norms = []
        for layer_idx in range(model_config.n_layers):
            key = f"blocks.{layer_idx}.hook_resid_post"
            if key in cache:
                resid = cache[key][0].cpu()  # [seq, d_model]
                layer_norms.append(resid.norm(dim=-1).mean().item())

        if layer_norms:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(range(len(layer_norms)), layer_norms, "o-")
            ax.set_xlabel("Layer")
            ax.set_ylabel("Mean Residual Stream Norm")
            ax.set_title("Residual Stream Norms Across Layers")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "residual_norms.png", dpi=150)
            plt.close()
            print(f"Saved: {output_dir / 'residual_norms.png'}")

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
