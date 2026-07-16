"""Weights & Biases logging utilities for PFN training."""

from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from pfn_transformerlens.checkpointing import CheckpointMetadata
from pfn_transformerlens.model.configs import BasePFNConfig


class WandbLogger:
    """Lightweight wandb logging wrapper.

    Handles wandb initialization, metric logging, and checkpoint uploads.
    No-op when use_wandb=False. Lazy imports wandb only when enabled.
    """

    def __init__(
        self,
        training_config,  # TrainingConfig - avoid circular import
        model_config: BasePFNConfig,
        data_config: Any | None = None,
    ) -> None:
        self.enabled = training_config.use_wandb
        self.run_id: str | None = None
        self.run_name: str | None = None
        self.run_url: str | None = None

        # Store configs for artifact metadata
        self.model_config = model_config
        self.training_config = training_config
        if data_config is not None and (
            isinstance(data_config, type) or not is_dataclass(data_config)
        ):
            raise TypeError("data_config must be a dataclass instance")
        self.data_config = asdict(data_config) if data_config is not None else None

        if not self.enabled:
            return

        # lazy import wandb only if enabled
        try:
            import wandb

            self.wandb = wandb
        except ImportError:
            raise ImportError(
                "wandb is required for logging. Install with: uv sync --extra wandb"
            )

        # read env vars as fallbacks
        project = training_config.wandb_project or os.getenv("WANDB_PROJECT")
        entity = training_config.wandb_entity or os.getenv("WANDB_ENTITY")

        # build config dict
        wandb_config = {
            "model": model_config.__dict__,
            "training": training_config.__dict__,
        }
        if self.data_config is not None:
            wandb_config["data"] = self.data_config

        # check if wandb is already initialized (e.g., by wandb sweep agent)
        if self.wandb.run is None:
            # initialize wandb run
            self.wandb.init(
                project=project,
                entity=entity,
                name=training_config.wandb_run_name,
                tags=training_config.wandb_tags,
                notes=training_config.wandb_notes,
                config=wandb_config,
            )
        else:
            # wandb already initialized (sweep agent), just update config
            self.wandb.config.update(wandb_config, allow_val_change=True)

        # Store run info for checkpoint metadata
        if self.wandb.run is not None:
            self.run_id = self.wandb.run.id
            self.run_name = self.wandb.run.name
            self.run_url = self.wandb.run.url

        self.log_model = training_config.wandb_log_model

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics to wandb."""
        if not self.enabled:
            return
        self.wandb.log(metrics, step=step)

    def log_checkpoint(
        self,
        checkpoint_path: str | Path,
        step: int,
        metadata: CheckpointMetadata | None = None,
    ) -> None:
        """Upload checkpoint as wandb artifact with run.id-based naming."""
        if not self.enabled or not self.log_model:
            return

        assert self.wandb.run is not None, "wandb run must be initialized"
        artifact_metadata = {
            "step": step,
            "run_name": self.wandb.run.name,
            "run_id": self.wandb.run.id,
            "run_url": self.wandb.run.url,
        }
        if metadata is not None:
            artifact_metadata.update(
                {
                    "timestamp": metadata.timestamp,
                    "git_hash": metadata.git_hash,
                    "wandb_run_id": metadata.wandb_run_id,
                    "wandb_run_name": metadata.wandb_run_name,
                    "wandb_run_url": metadata.wandb_run_url,
                }
            )

        # Add full configs to artifact metadata for config-based loading
        artifact_metadata["model_config"] = self.model_config.__dict__
        artifact_metadata["training_config"] = self.training_config.__dict__
        if self.data_config is not None:
            artifact_metadata["data_config"] = self.data_config

        # Use run.id for unique artifact naming
        artifact = self.wandb.Artifact(
            name=f"checkpoint-{self.wandb.run.id}-step-{step}",
            type="model",
            description=f"Checkpoint at step {step}",
            metadata=artifact_metadata,
        )
        artifact.add_file(str(checkpoint_path))
        self.wandb.log_artifact(artifact)

    def finish(self) -> None:
        """Cleanup wandb run."""
        if not self.enabled:
            return
        self.wandb.finish()


__all__ = ["WandbLogger"]
