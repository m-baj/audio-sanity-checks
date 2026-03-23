import os
import shutil
import zipfile
from pathlib import Path

import requests
import torchaudio
import typer
from loguru import logger
from tqdm import tqdm

from audio_sanity_checks.config import RAW_DATA_DIR

app = typer.Typer()

def _download_esc50(target_path: Path):
    url = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
    target_dir = target_path / "esc50"
    temp_extract = target_path / "temp_esc50"
    zip_file = target_path / "esc50.zip"

    if target_dir.exists():
        logger.info(f"ESC-50 already exists at {target_dir}. Skipping.")
        return

    target_path.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Downloading ESC-50 from {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total = int(response.headers.get('content-length', 0))
        with open(zip_file, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc="ESC-50") as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

        logger.info("Extracting files...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_extract)

        source_inner = next(temp_extract.glob("ESC-50*"))
        target_dir.mkdir(parents=True, exist_ok=True)
        
        for folder in ["audio", "meta"]:
            logger.info(f"Moving {folder} to {target_dir}")
            shutil.move(str(source_inner / folder), str(target_dir / folder))

    except Exception as e:
        logger.error(f"Error downloading ESC-50: {e}")
    finally:
        if zip_file.exists():
            os.remove(zip_file)
        if temp_extract.exists():
            shutil.rmtree(temp_extract)


def _download_speech_commands(target_path: Path):
    logger.info("Downloading SpeechCommands via torchaudio...")
    target_path.mkdir(parents=True, exist_ok=True)
    dataset = torchaudio.datasets.SPEECHCOMMANDS(root=str(target_path), download=True)

    for archive in target_path.glob("*.tar.gz"):
        logger.info(f"Removing archive {archive}")
        archive.unlink()

    waveform, _, label, _, _ = dataset[0]
    logger.success(f"SpeechCommands ready. Samples: {len(dataset)}, First label: {label}")


@app.command()
def main(
    subset: str = typer.Option("all", help="Which dataset to download: 'esc50', 'speech_commands', or 'all'")
):
    """
    Download and prepare raw audio datasets for Sanity Checks.
    """
    logger.info(f"Starting data download process in {RAW_DATA_DIR}")

    if subset in ["all", "speech_commands"]:
        _download_speech_commands(RAW_DATA_DIR)

    if subset in ["all", "esc50"]:
        _download_esc50(RAW_DATA_DIR)

    logger.success("Data download and organization complete.")


if __name__ == "__main__":
    app()
