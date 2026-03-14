"""
VAJRA Voice AI — Dataset Download Automation.

Provides automated download stubs for supported speech datasets.
Actual downloads require user authentication and license agreements
with the respective dataset providers.

Supported datasets:
    - ASVspoof 2019 LA
    - ASVspoof 2024 LA
    - LibriSpeech (train-clean-100)
    - Common Voice (English)
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000


@dataclass
class DatasetInfo:
    """Metadata for a downloadable dataset."""

    name: str
    url: str
    description: str
    license: str
    requires_auth: bool = False


SUPPORTED_DATASETS: Dict[str, DatasetInfo] = {
    "asvspoof2019": DatasetInfo(
        name="ASVspoof 2019 LA",
        url="https://datashare.ed.ac.uk/handle/10283/3336",
        description="Logical access anti-spoofing dataset",
        license="CC BY 4.0",
        requires_auth=True,
    ),
    "asvspoof2024": DatasetInfo(
        name="ASVspoof 2024 LA",
        url="https://www.asvspoof.org/",
        description="Latest anti-spoofing challenge dataset",
        license="Research use",
        requires_auth=True,
    ),
    "librispeech": DatasetInfo(
        name="LibriSpeech train-clean-100",
        url="https://www.openslr.org/12",
        description="100 hours of clean read English speech",
        license="CC BY 4.0",
        requires_auth=False,
    ),
    "commonvoice": DatasetInfo(
        name="Common Voice English",
        url="https://commonvoice.mozilla.org/en/datasets",
        description="Mozilla crowd-sourced speech corpus",
        license="CC-0",
        requires_auth=True,
    ),
}


class DatasetDownloader:
    """
    Automated dataset download manager.

    Downloads are placed in ``<output_dir>/<dataset_name>/`` and
    extracted automatically when possible.

    Parameters
    ----------
    output_dir : str
        Root directory for downloaded datasets.
    """

    def __init__(self, output_dir: str = "data/raw") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def list_datasets(self) -> List[DatasetInfo]:
        """Return metadata for all supported datasets."""
        return list(SUPPORTED_DATASETS.values())

    def download(self, dataset_name: str) -> Path:
        """
        Download a dataset by name.

        Parameters
        ----------
        dataset_name : str
            One of: ``asvspoof2019``, ``asvspoof2024``, ``librispeech``,
            ``commonvoice``.

        Returns
        -------
        Path
            Directory containing the downloaded data.
        """
        if dataset_name not in SUPPORTED_DATASETS:
            raise ValueError(
                f"Unknown dataset: {dataset_name!r}. "
                f"Supported: {list(SUPPORTED_DATASETS.keys())}"
            )

        info = SUPPORTED_DATASETS[dataset_name]
        dest = self.output_dir / dataset_name
        dest.mkdir(parents=True, exist_ok=True)

        if dataset_name == "librispeech":
            return self._download_librispeech(dest)
        elif dataset_name in ("asvspoof2019", "asvspoof2024"):
            return self._download_asvspoof(dataset_name, info, dest)
        elif dataset_name == "commonvoice":
            return self._download_commonvoice(dest)

        return dest

    def download_all(self) -> Dict[str, Path]:
        """Download all supported datasets. Returns mapping of name → path."""
        results: Dict[str, Path] = {}
        for name in SUPPORTED_DATASETS:
            try:
                results[name] = self.download(name)
            except Exception as exc:
                log.warning("Failed to download %s: %s", name, exc)
        return results

    # ── Private download methods ──────────────────────────────────────────

    def _download_librispeech(self, dest: Path) -> Path:
        """Download LibriSpeech train-clean-100 via torchaudio."""
        import torchaudio

        log.info("Downloading LibriSpeech train-clean-100 → %s", dest)
        torchaudio.datasets.LIBRISPEECH(
            root=str(dest), url="train-clean-100", download=True
        )
        log.info("LibriSpeech download complete")
        return dest

    def _download_asvspoof(
        self, name: str, info: DatasetInfo, dest: Path
    ) -> Path:
        """
        ASVspoof datasets require manual download due to license agreements.
        This method creates a README with download instructions.
        """
        readme = dest / "DOWNLOAD_INSTRUCTIONS.md"
        if not readme.exists():
            readme.write_text(
                f"# {info.name}\n\n"
                f"This dataset requires manual download.\n\n"
                f"1. Visit: {info.url}\n"
                f"2. Accept the license agreement ({info.license})\n"
                f"3. Download the LA track files\n"
                f"4. Extract into this directory: {dest}\n\n"
                f"Expected structure:\n"
                f"```\n"
                f"{dest}/\n"
                f"  LA/\n"
                f"    ASVspoof*_LA_cm_protocols/\n"
                f"    ASVspoof*_LA_train/flac/\n"
                f"    ASVspoof*_LA_dev/flac/\n"
                f"    ASVspoof*_LA_eval/flac/\n"
                f"```\n",
                encoding="utf-8",
            )
        log.info(
            "%s requires manual download. See %s", info.name, readme
        )
        return dest

    def _download_commonvoice(self, dest: Path) -> Path:
        """
        Common Voice requires Mozilla account authentication.
        Creates download instructions.
        """
        readme = dest / "DOWNLOAD_INSTRUCTIONS.md"
        if not readme.exists():
            readme.write_text(
                "# Common Voice English\n\n"
                "1. Visit: https://commonvoice.mozilla.org/en/datasets\n"
                "2. Create a Mozilla account and accept the license\n"
                "3. Download the English dataset\n"
                "4. Extract into this directory\n",
                encoding="utf-8",
            )
        log.info("Common Voice requires manual download. See %s", readme)
        return dest
