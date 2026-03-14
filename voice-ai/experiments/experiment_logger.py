"""
VAJRA Voice AI — Experiment Logger.

Tracks training experiments with metadata, hyperparameters, and metrics.
Stores experiment data as JSON files for reproducibility.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


class ExperimentLogger:
    """
    Experiment tracking system.

    Stores per-epoch metrics, hyperparameters, and system metadata
    as JSON files in the experiments directory.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    output_dir : str
        Directory for experiment logs (default ``experiments/logs``).
    """

    def __init__(
        self,
        experiment_name: str,
        output_dir: str = "experiments/logs",
    ) -> None:
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.start_time = time.time()
        self.epoch_logs: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}

        # Collect system info
        self._collect_system_info()

        self.experiment_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        log.info(
            "Experiment: %s (id=%s)", experiment_name, self.experiment_id
        )

    def _collect_system_info(self) -> None:
        """Collect system information for reproducibility."""
        import torch

        self.metadata["system"] = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "cuda_version": (
                torch.version.cuda if torch.cuda.is_available() else None
            ),
        }

    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Log training hyperparameters."""
        self.metadata["hyperparameters"] = params
        log.info("Hyperparameters: %s", params)

    def log_epoch(self, metrics: Any) -> None:
        """
        Log metrics for a single epoch.

        Parameters
        ----------
        metrics : EpochMetrics or dict
        """
        if hasattr(metrics, "__dataclass_fields__"):
            data = asdict(metrics)
        elif isinstance(metrics, dict):
            data = metrics
        else:
            data = {"value": str(metrics)}

        data["timestamp"] = time.time()
        self.epoch_logs.append(data)

    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a single metric value."""
        entry = {"metric": name, "value": value, "timestamp": time.time()}
        if step is not None:
            entry["step"] = step
        self.epoch_logs.append(entry)

    def log_evaluation(self, results: Dict[str, Any]) -> None:
        """Log evaluation results."""
        self.metadata["evaluation"] = results

    def save(self) -> Path:
        """
        Save the experiment to a JSON file.

        Returns
        -------
        Path to the saved experiment file.
        """
        elapsed = time.time() - self.start_time

        experiment = {
            "experiment_name": self.experiment_name,
            "experiment_id": self.experiment_id,
            "metadata": self.metadata,
            "training_time_seconds": round(elapsed, 1),
            "epochs": self.epoch_logs,
        }

        filename = f"{self.experiment_id}_{self.experiment_name}.json"
        filepath = self.output_dir / filename

        with filepath.open("w", encoding="utf-8") as fh:
            json.dump(experiment, fh, indent=2, default=str)

        log.info("Experiment saved → %s", filepath)
        return filepath

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the experiment."""
        if not self.epoch_logs:
            return {"experiment": self.experiment_name, "epochs": 0}

        # Find best metrics from epoch logs
        train_losses = [
            e.get("train_loss", float("inf"))
            for e in self.epoch_logs
            if "train_loss" in e
        ]
        val_accs = [
            e.get("val_acc", 0.0)
            for e in self.epoch_logs
            if "val_acc" in e
        ]

        return {
            "experiment": self.experiment_name,
            "total_epochs": len(self.epoch_logs),
            "best_val_acc": max(val_accs) if val_accs else 0.0,
            "best_train_loss": min(train_losses) if train_losses else float("inf"),
            "training_time_s": round(time.time() - self.start_time, 1),
        }
