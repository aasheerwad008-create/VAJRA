#!/usr/bin/env python3
"""
VAJRA Voice AI — Pretrained Weight Setup CLI.

Downloads and verifies all pretrained model weights in one step.

Usage::

    # From voice-ai directory:
    python -m pretrained.setup_weights [--model-dir ./weights] [--force] [--status]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the voice-ai package root is importable.
_VOICE_AI_ROOT = str(Path(__file__).resolve().parent.parent)
if _VOICE_AI_ROOT not in sys.path:
    sys.path.insert(0, _VOICE_AI_ROOT)

from pretrained.downloader import setup_all, status
from pretrained.registry import MODEL_DIR, WeightRegistry


def _print_status(model_dir: Path | None = None) -> None:
    """Print a table showing the current state of all registered weights."""
    rows = status(model_dir)
    print()
    print(f"{'Name':<35} {'Model':<25} {'Type':<13} {'Cached':<8} Path")
    print("─" * 120)
    for r in rows:
        cached_str = "✅ Yes" if r["cached"] else "❌ No"
        print(
            f"{r['name']:<35} {r['model']:<25} {r['source_type']:<13} "
            f"{cached_str:<8} {r['path']}"
        )
    print()


def _print_registry_summary() -> None:
    """Print a summary of all registered pretrained weights."""
    registry = WeightRegistry()
    entries = registry.list_all()
    print()
    print("=" * 72)
    print("VAJRA Voice AI — Pretrained Weight Registry")
    print("=" * 72)
    print(f"  Total registered weights : {len(entries)}")
    print(f"  Weight cache directory   : {MODEL_DIR}")
    print()

    for entry in entries:
        print(f"  📦 {entry.name}")
        print(f"     Model       : {entry.model}")
        print(f"     Source      : {entry.source}")
        print(f"     Source type : {entry.source_type}")
        print(f"     Trainable   : {'Yes' if entry.trainable else 'No (frozen)'}")
        if entry.file_size_mb > 0:
            print(f"     Size (approx): {entry.file_size_mb:.1f} MB")
        print(f"     Description : {entry.description[:80]}…")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VAJRA Voice AI — Download and set up pretrained model weights.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help=f"Directory to store pretrained weights (default: {MODEL_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download weights even if they already exist.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show the current status of cached weights and exit.",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print the full pretrained weight registry and exit.",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        default=True,
        help="Continue on download failure (default: True).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    if args.info:
        _print_registry_summary()
        return

    if args.status:
        _print_status(args.model_dir)
        return

    # Download all weights
    results = setup_all(
        model_dir=args.model_dir,
        force=args.force,
        skip_errors=args.skip_errors,
    )

    # Print final status table
    _print_status(args.model_dir)

    # Exit with error code if any downloads failed
    if not all(results.values()):
        failed = [k for k, v in results.items() if not v]
        print(f"⚠️  Failed downloads: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("✅  All pretrained weights ready.")


if __name__ == "__main__":
    main()
