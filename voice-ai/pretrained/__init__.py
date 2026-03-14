"""
VAJRA Voice AI — Pretrained Weight Management.

Provides a centralized registry, download/verification utilities,
and a CLI to set up all pretrained weights for the voice-ai models.

Usage::

    # Download all pretrained weights
    python -m pretrained.setup_weights

    # Programmatic access
    from pretrained import WeightRegistry, download_weights, verify_weights
"""
from pretrained.registry import WeightEntry, WeightRegistry
from pretrained.downloader import download_weights, verify_weights, setup_all

__all__ = [
    "WeightEntry",
    "WeightRegistry",
    "download_weights",
    "verify_weights",
    "setup_all",
]
