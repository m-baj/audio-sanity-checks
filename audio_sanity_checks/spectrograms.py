import csv
from abc import abstractmethod
from pathlib import Path
from typing import override

from loguru import logger
from tqdm import tqdm
import typer
import torchaudio
from torch.utils.data import Dataset
import torch

from audio_sanity_checks.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


class SpectrogramDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 64,
    ):
        self.dataset_path = dataset_path
        self.sample_rate = sample_rate
        self.transform = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
            ),
            torchaudio.transforms.AmplitudeToDB(),
        )
        self.processed = False

    def _process_spectrogram(self, waveform: torch.Tensor, sample_rate: int):
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, self.sample_rate
            )
        return self.transform(waveform)

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx): ...


class SpeechCommandsSpectrogramDataset(SpectrogramDataset):
    def __init__(
        self,
        dataset_path: Path,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 64,
        subset: str = None,
    ):
        super().__init__(dataset_path, sample_rate, n_fft, hop_length, n_mels)
        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root=str(dataset_path), subset=subset
        )

    def process(self):
        self.spectrograms = []
        for i in tqdm(range(len(self.dataset)), desc="Processing SpeechCommands"):
            waveform, sample_rate, _, _, _ = self.dataset[i]
            spectrogram = self._process_spectrogram(waveform, sample_rate)

            self.spectrograms.append(spectrogram)
        self.processed = True

    @override
    def __len__(self):
        return len(self.dataset)

    @override
    def __getitem__(self, idx):
        waveform, sample_rate, label, _, _ = self.dataset[idx]
        spectrogram = (
            self.spectrograms[idx]
            if self.processed
            else self._process_spectrogram(waveform, sample_rate)
        )
        return spectrogram, label


class ESC50SpectrogramDataset(SpectrogramDataset):
    """ESC-50 spectrograms. Optional ``subset`` filters by official fold:

    * ``training`` — folds 1–3
    * ``validation`` — fold 4
    * ``testing`` — fold 5
    * ``None`` — all folds
    """

    def __init__(
        self,
        dataset_path: Path,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 64,
        subset: str | None = None,
    ):
        super().__init__(dataset_path, sample_rate, n_fft, hop_length, n_mels)
        if subset is not None and subset not in (
            "training",
            "validation",
            "testing",
        ):
            raise ValueError(
                "subset must be None, 'training', 'validation', or 'testing'"
            )
        self.subset = subset
        meta_csv = self.dataset_path / "meta" / "esc50.csv"
        audio_dir = self.dataset_path / "audio"
        self.samples = []
        with open(meta_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fold = int(row["fold"])
                if subset == "training" and fold not in (1, 2, 3):
                    continue
                if subset == "validation" and fold != 4:
                    continue
                if subset == "testing" and fold != 5:
                    continue
                self.samples.append((audio_dir / row["filename"], row["category"]))

    def process(self):
        self.spectrograms = []
        for i in tqdm(range(len(self.samples)), desc="Processing ESC-50"):
            path, _ = self.samples[i]
            waveform, sr = torchaudio.load(path)
            self.spectrograms.append(self._process_spectrogram(waveform, sr))
        self.processed = True

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        spectrogram = (
            self.spectrograms[idx]
            if self.processed
            else self._process_spectrogram(*torchaudio.load(path))
        )
        return spectrogram, label


def process_and_save_speech_commands(
    dataset_class,
    dataset_path,
    processed_path,
    sample_rate: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 64,
    subset=None,
    filename=None,
):
    dataset = dataset_class(
        dataset_path=dataset_path,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        subset=subset,
    )
    dataset.process()
    processed_path.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, processed_path / filename)
    logger.success(f"{dataset_class.__name__} spectrograms saved in {processed_path}")


def generate_spectrograms(
    dataset_class,
    raw_path: Path,
    processed_path: Path,
    *,
    split: bool,
    file_prefix: str,
    log_label: str,
    sample_rate: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 64,
) -> None:
    try:
        if split:
            for subset in ["training", "validation", "testing"]:
                msg = f"Generating spectrograms for {log_label} (subset: {subset})..."
                logger.info(msg)
                process_and_save_speech_commands(
                    dataset_class=dataset_class,
                    dataset_path=raw_path,
                    processed_path=processed_path,
                    sample_rate=sample_rate,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_mels=n_mels,
                    subset=subset,
                    filename=f"{file_prefix}_{subset}.pt",
                )
        else:
            msg = f"Generating spectrograms for {log_label} (subset: all)..."
            logger.info(msg)
            process_and_save_speech_commands(
                dataset_class=dataset_class,
                dataset_path=raw_path,
                processed_path=processed_path,
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                subset=None,
                filename=f"{file_prefix}.pt",
            )
    except Exception as exc:
        logger.exception(f"Failed to load {log_label} dataset: {exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def main(
    subset: str = typer.Argument(
        "all",
        help="Which dataset to generate spectrograms for: 'esc50', 'speech_commands', or 'all'",
    ),
    raw_path: Path = typer.Argument(
        RAW_DATA_DIR,
        help="Path to the raw dataset",
    ),
    processed_path: Path = typer.Argument(
        PROCESSED_DATA_DIR,
        help="Path to the processed dataset",
    ),
    split: bool = typer.Option(
        False,
        "--split",
        "-s",
        help="Whether to split the dataset into training, validation, and testing subsets",
    ),
    sample_rate: int = typer.Option(
        16000,
        "--sample-rate",
        "-r",
        help="Sample rate for the spectrograms",
    ),
    n_fft: int = typer.Option(
        1024,
        "--n-fft",
        "-f",
        help="Number of FFT bins",
    ),
    hop_length: int = typer.Option(
        512,
        "--hop-length",
        "-h",
        help="Hop length for the spectrograms",
    ),
    n_mels: int = typer.Option(
        64,
        "--n-mels",
        "-m",
        help="Number of mel bins",
    ),
):
    """
    Generate spectrograms from dataset.
    """
    logger.info(f"Starting spectrograms generation process in {PROCESSED_DATA_DIR}")

    if subset not in ["all", "speech_commands", "esc50"]:
        logger.error(f"Invalid subset: {subset}")
        raise typer.Exit(code=1)

    if subset in ["all", "speech_commands"]:
        generate_spectrograms(
            SpeechCommandsSpectrogramDataset,
            raw_path,
            processed_path / "speech_commands",
            split=split,
            file_prefix="speech_commands",
            log_label="SpeechCommands",
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

    if subset in ["all", "esc50"]:
        generate_spectrograms(
            ESC50SpectrogramDataset,
            raw_path / "esc50",
            processed_path / "esc50",
            split=split,
            file_prefix="esc50",
            log_label="ESC-50",
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

    logger.success("Spectrograms generation complete.")


if __name__ == "__main__":
    app()
