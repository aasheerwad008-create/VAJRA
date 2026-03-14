"""
VAJRA Voice AI — Pretrained Weight Registry.

Defines a central catalogue of all pretrained model weights used by VAJRA,
including source URLs, expected file paths, SHA-256 checksums, and metadata.

Each model's pretrained source is recorded so that:
    - ``setup_weights.py`` can download everything in one step.
    - Runtime code can verify that cached weights are intact.
    - Documentation stays in sync with the actual sources.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

MODEL_DIR = Path(os.getenv("MODEL_DIR", "weights"))


@dataclass(frozen=True)
class WeightEntry:
    """
    Describes a single pretrained weight artefact.

    Parameters
    ----------
    name : str
        Human-readable identifier (e.g. ``"efficientnet_b0_imagenet"``).
    model : str
        VAJRA model that uses these weights.
    source : str
        Download source — a URL, HuggingFace Hub ID, or library identifier.
    source_type : str
        One of ``"timm"``, ``"speechbrain"``, ``"url"``, ``"huggingface"``.
    description : str
        What the weights represent.
    filename : str
        Relative path under ``MODEL_DIR`` where the weights are cached.
    sha256 : str
        Expected SHA-256 hex digest (empty string if not yet known).
    file_size_mb : float
        Approximate file size in MB (0.0 if not yet known).
    trainable : bool
        Whether the weights are used as an initialisation that will be
        fine-tuned (True) or as a frozen feature extractor (False).
    tags : list[str]
        Arbitrary tags for filtering (e.g. ``["backbone", "imagenet"]``).
    """

    name: str
    model: str
    source: str
    source_type: str
    description: str
    filename: str
    sha256: str = ""
    file_size_mb: float = 0.0
    trainable: bool = True
    tags: list = field(default_factory=list)


class WeightRegistry:
    """
    Singleton catalogue of all pretrained weights.

    >>> reg = WeightRegistry()
    >>> for entry in reg.list_all():
    ...     print(entry.name, entry.source_type)
    """

    _instance: Optional["WeightRegistry"] = None

    def __new__(cls) -> "WeightRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._entries: Dict[str, WeightEntry] = {}
            cls._instance._populate()
        return cls._instance

    # ── Public API ────────────────────────────────────────────────────────

    def list_all(self) -> List[WeightEntry]:
        """Return every registered weight entry."""
        return list(self._entries.values())

    def get(self, name: str) -> WeightEntry:
        """Look up a weight entry by name.

        Raises ``KeyError`` if not found.
        """
        return self._entries[name]

    def filter_by_model(self, model: str) -> List[WeightEntry]:
        """Return entries whose *model* field matches ``model``."""
        return [e for e in self._entries.values() if e.model == model]

    def filter_by_source_type(self, source_type: str) -> List[WeightEntry]:
        """Return entries whose *source_type* matches."""
        return [e for e in self._entries.values() if e.source_type == source_type]

    def names(self) -> List[str]:
        """Return the names of all registered weights."""
        return list(self._entries.keys())

    @property
    def model_dir(self) -> Path:
        """Root directory for cached weights."""
        return MODEL_DIR

    # ── Internal ──────────────────────────────────────────────────────────

    def _register(self, entry: WeightEntry) -> None:
        self._entries[entry.name] = entry

    def _populate(self) -> None:
        """Register all known pretrained weight sources."""

        # ── 1. EfficientNet-B0 (ImageNet) ─────────────────────────────────
        self._register(WeightEntry(
            name="efficientnet_b0_imagenet",
            model="SpectrogramModel",
            source="efficientnet_b0",
            source_type="timm",
            description=(
                "ImageNet-1K pretrained EfficientNet-B0 backbone. "
                "Used as the spectrogram classifier backbone with a custom "
                "anti-spoofing head. Downloaded automatically by the timm library."
            ),
            filename="timm/efficientnet_b0.pth",
            file_size_mb=20.5,
            trainable=True,
            tags=["backbone", "imagenet", "spectrogram", "cnn"],
        ))

        # ── 2. ECAPA-TDNN (VoxCeleb) ─────────────────────────────────────
        self._register(WeightEntry(
            name="ecapa_tdnn_voxceleb",
            model="ECAPATDNNClassifier",
            source="speechbrain/spkrec-ecapa-voxceleb",
            source_type="speechbrain",
            description=(
                "ECAPA-TDNN speaker encoder pretrained on VoxCeleb1+2 via "
                "SpeechBrain. Produces 192-d speaker embeddings. Used as the "
                "backbone for anti-spoofing classification and speaker "
                "verification. Downloaded by SpeechBrain's from_hparams()."
            ),
            filename="spkrec-ecapa/embedding_model.ckpt",
            file_size_mb=83.0,
            trainable=False,
            tags=["backbone", "speaker", "voxceleb", "ecapa"],
        ))

        # ── 3. ECAPA-TDNN for Speaker Embedder ───────────────────────────
        self._register(WeightEntry(
            name="ecapa_tdnn_speaker_embedder",
            model="SpeakerEmbedder",
            source="speechbrain/spkrec-ecapa-voxceleb",
            source_type="speechbrain",
            description=(
                "Same ECAPA-TDNN weights as above, shared by the "
                "SpeakerEmbedder module for speaker verification embeddings."
            ),
            filename="spkrec-ecapa/embedding_model.ckpt",
            file_size_mb=83.0,
            trainable=False,
            tags=["speaker", "voxceleb", "ecapa", "embedder"],
        ))

        # ── 4. RawNet2 (random init) ─────────────────────────────────────
        self._register(WeightEntry(
            name="rawnet2_init",
            model="RawNet2",
            source="kaiming_init",
            source_type="init",
            description=(
                "RawNet2 is initialised with Kaiming normal weights for Conv1d "
                "layers and mel-scale sinc filters. No external pretrained "
                "checkpoint — the model is trained end-to-end from scratch on "
                "the anti-spoofing dataset."
            ),
            filename="",
            file_size_mb=0.0,
            trainable=True,
            tags=["rawnet2", "scratch", "sinc"],
        ))

        # ── 5. Codec Artifact Detector (random init) ─────────────────────
        self._register(WeightEntry(
            name="codec_detector_init",
            model="CodecDetectorModel",
            source="kaiming_init",
            source_type="init",
            description=(
                "Lightweight 1-D CNN for neural codec artifact detection. "
                "Initialised with Kaiming weights — no external pretrained "
                "checkpoint. Trained from scratch on codec-artifact data."
            ),
            filename="",
            file_size_mb=0.0,
            trainable=True,
            tags=["codec", "scratch", "cnn"],
        ))
