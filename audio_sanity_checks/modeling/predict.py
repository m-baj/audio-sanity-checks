from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import torch
import matplotlib.pyplot as plt

from audio_sanity_checks.config import MODELS_DIR, PROCESSED_DATA_DIR

from audio_sanity_checks.spectrograms import (
    SpeechCommandsSpectrogramDataset,
    ESC50SpectrogramDataset,
)
from torchvision.models import resnet18, ResNet18_Weights

app = typer.Typer()


def plot_spectrogram(mel, *, label) -> None:

    mel = mel.squeeze(0).detach().cpu().numpy()

    im = plt.imshow(
        mel[0], aspect="auto", origin="lower", interpolation="nearest", cmap="inferno"
    )
    plt.colorbar(im, fraction=0.046, pad=0.04, label="dB")
    plt.title(f"Mel spectrogram ({mel.shape[0]}×{mel.shape[1]}), label: {label!r}")
    plt.xlabel("frame")
    plt.ylabel("mel bin")
    plt.show()


@app.command()
def main(
    dataset_path: Path = typer.Argument("speech_commands", help="Path to the dataset"),
    index: int = typer.Argument(0, help="Index of the sample to predict"),
    split: str = typer.Option("training", "--split", "-s", help="Split to use"),
):
    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()
    model = resnet18(weights=weights).eval()
    # print(model)

    dataset = torch.load(
        PROCESSED_DATA_DIR / dataset_path / f"{dataset_path}_{split}.pt",
        weights_only=False,
    )
    spec, label = dataset[index]
    mel = spec.squeeze(0).detach().cpu().numpy()
    print(mel.shape)
    print(label)

    plot_spectrogram(spec, label=label)

    spec = preprocess(spec)
    print(spec.shape)
    spec = spec.unsqueeze(0)
    pred = model(spec)
    pred_class = pred.argmax(dim=1).item()
    print(pred_class)


if __name__ == "__main__":
    app()
