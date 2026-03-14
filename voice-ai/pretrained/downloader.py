"""
VAJRA Voice AI — Pretrained Weight Downloader & Verifier.

Downloads pretrained weights to the local ``MODEL_DIR`` cache, verifies
integrity via SHA-256 checksums, and provides a ``setup_all()`` function
that prepares every registered weight source in one call.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from pretrained.registry import MODEL_DIR, WeightEntry, WeightRegistry

log = logging.getLogger(__name__)


# ── Public helpers ────────────────────────────────────────────────────────


def verify_weights(entry: WeightEntry, model_dir: Optional[Path] = None) -> bool:
    """
    Check whether the cached weight file exists and (optionally) matches
    its SHA-256 checksum.

    Parameters
    ----------
    entry : WeightEntry
        Registry entry to verify.
    model_dir : Path, optional
        Override for ``MODEL_DIR``.

    Returns
    -------
    bool
        ``True`` if the file exists (and checksum matches when available).
    """
    if not entry.filename:
        # Entries that use random init have no file to check.
        return True

    base = model_dir or MODEL_DIR
    path = base / entry.filename

    if not path.exists():
        log.debug("verify_weights: %s not found at %s", entry.name, path)
        return False

    if entry.sha256:
        digest = _sha256(path)
        if digest != entry.sha256:
            log.warning(
                "verify_weights: checksum mismatch for %s "
                "(expected %s, got %s)",
                entry.name,
                entry.sha256[:16] + "…",
                digest[:16] + "…",
            )
            return False

    return True


def download_weights(
    entry: WeightEntry,
    model_dir: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """
    Download or prepare a single pretrained weight artefact.

    Dispatches to the appropriate downloader based on ``entry.source_type``:
        - ``"timm"`` → uses ``timm.create_model(pretrained=True)``
        - ``"speechbrain"`` → uses ``EncoderClassifier.from_hparams()``
        - ``"url"`` → direct HTTP download
        - ``"init"`` → no-op (random initialisation)

    Parameters
    ----------
    entry : WeightEntry
        Registry entry to download.
    model_dir : Path, optional
        Override for ``MODEL_DIR``.
    force : bool
        If True, re-download even if the file already exists.

    Returns
    -------
    Path
        Path to the downloaded/cached weight file (or ``MODEL_DIR`` for
        init-only entries).

    Raises
    ------
    RuntimeError
        If the download fails for a required weight source.
    """
    base = model_dir or MODEL_DIR
    base.mkdir(parents=True, exist_ok=True)

    if entry.source_type == "init":
        log.info("⏩  %s — random initialisation (no download needed)", entry.name)
        return base

    if not force and verify_weights(entry, base):
        log.info("✅  %s — already cached at %s", entry.name, base / entry.filename)
        return base / entry.filename

    if entry.source_type == "timm":
        return _download_timm(entry, base)
    elif entry.source_type == "speechbrain":
        return _download_speechbrain(entry, base)
    elif entry.source_type == "url":
        return _download_url(entry, base)
    else:
        log.warning(
            "⚠️  %s — unknown source_type %r, skipping", entry.name, entry.source_type
        )
        return base


def setup_all(
    model_dir: Optional[Path] = None,
    force: bool = False,
    skip_errors: bool = True,
) -> Dict[str, bool]:
    """
    Download and verify **all** registered pretrained weights.

    Parameters
    ----------
    model_dir : Path, optional
        Override for ``MODEL_DIR``.
    force : bool
        Re-download even if weights are cached.
    skip_errors : bool
        If True, continue on failure and return status per entry.

    Returns
    -------
    dict[str, bool]
        Mapping from weight name to success status.
    """
    registry = WeightRegistry()
    results: Dict[str, bool] = {}

    log.info("🔧  Setting up pretrained weights in %s", model_dir or MODEL_DIR)
    log.info("    Registered weights: %d", len(registry.list_all()))

    for entry in registry.list_all():
        try:
            download_weights(entry, model_dir=model_dir, force=force)
            results[entry.name] = True
        except Exception as exc:
            log.error("❌  %s — %s", entry.name, exc)
            results[entry.name] = False
            if not skip_errors:
                raise

    succeeded = sum(v for v in results.values())
    total = len(results)
    log.info(
        "🏁  Pretrained weight setup complete: %d/%d succeeded", succeeded, total
    )
    return results


# ── Status helper ─────────────────────────────────────────────────────────


def status(model_dir: Optional[Path] = None) -> List[Dict[str, object]]:
    """
    Return the current status of every registered weight.

    Returns
    -------
    list[dict]
        Each dict has keys: ``name``, ``model``, ``source_type``,
        ``cached``, ``path``.
    """
    registry = WeightRegistry()
    base = model_dir or MODEL_DIR
    rows = []
    for entry in registry.list_all():
        cached = verify_weights(entry, base)
        path = str(base / entry.filename) if entry.filename else "(init)"
        rows.append({
            "name": entry.name,
            "model": entry.model,
            "source_type": entry.source_type,
            "cached": cached,
            "path": path,
        })
    return rows


# ── Private downloaders ──────────────────────────────────────────────────


def _download_timm(entry: WeightEntry, base: Path) -> Path:
    """Trigger a timm model download and cache the weights."""
    log.info("⬇️  %s — downloading via timm (%s)", entry.name, entry.source)
    try:
        import timm
        import torch

        model = timm.create_model(entry.source, pretrained=True, num_classes=0)
        dest = base / entry.filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(dest))
        log.info("✅  %s — saved to %s", entry.name, dest)
        return dest
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download timm weights for {entry.name}: {exc}"
        ) from exc


def _download_speechbrain(entry: WeightEntry, base: Path) -> Path:
    """Trigger a SpeechBrain model download."""
    log.info(
        "⬇️  %s — downloading via SpeechBrain (%s)", entry.name, entry.source
    )
    try:
        from speechbrain.inference.speaker import EncoderClassifier

        savedir = base / entry.filename.split("/")[0]
        savedir.mkdir(parents=True, exist_ok=True)
        EncoderClassifier.from_hparams(
            source=entry.source,
            savedir=str(savedir),
            run_opts={"device": "cpu"},
        )
        log.info("✅  %s — saved to %s", entry.name, savedir)
        return base / entry.filename
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download SpeechBrain weights for {entry.name}: {exc}"
        ) from exc


def _download_url(entry: WeightEntry, base: Path) -> Path:
    """Download weights from a direct URL."""
    import urllib.request

    log.info("⬇️  %s — downloading from %s", entry.name, entry.source)
    dest = base / entry.filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(entry.source, str(dest))
        log.info("✅  %s — saved to %s", entry.name, dest)
        return dest
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download URL weights for {entry.name}: {exc}"
        ) from exc


def _sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
