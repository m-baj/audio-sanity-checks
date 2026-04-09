from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import torch
import matplotlib.pyplot as plt

from audio_sanity_checks.config import MODELS_DIR, PROCESSED_DATA_DIR

from audio_sanity_checks.spectrograms import SpeechCommandsSpectrogramDataset
from torchvision.models import resnet18, ResNet18_Weights

app = typer.Typer()


def plot_spectrogram(mel, *, label) -> None:
    """
    Plot the given mel spectrogram and its 224x224 bilinearly-resized version.

    Args:
        mel (Tensor or ndarray): The input mel spectrogram to plot.
        label (str): Label associated with the spectrogram, for display.
    """
    mel_224 = (
        torch.nn.functional.interpolate(
            mel.float().unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze(0)
        .squeeze(0)
        .numpy()
    )

    mel = mel.squeeze(0).detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(
        mel, aspect="auto", origin="lower", interpolation="nearest", cmap="inferno"
    )
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="dB")
    axes[0].set_title(
        f"Mel spectrogram ({mel.shape[0]}×{mel.shape[1]}), label: {label!r}"
    )
    axes[0].set_xlabel("frame")
    axes[0].set_ylabel("mel bin")

    im1 = axes[1].imshow(
        mel_224,
        aspect="equal",
        origin="lower",
        interpolation="bilinear",
        cmap="inferno",
    )
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="dB")
    axes[1].set_title("Same mel spectrogram rescaled to 224×224 (bilinear)")
    axes[1].set_xlabel("pixel")
    axes[1].set_ylabel("pixel")
    plt.tight_layout()
    plt.show()


@app.command()
def main():
    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()
    model = resnet18(weights=weights).eval()
    # print(model)

    dataset = torch.load(
        PROCESSED_DATA_DIR / "speech_commands" / "speech_commands_training.pt",
        weights_only=False,
    )
    spec, label = dataset[0]
    mel = spec.squeeze(0).detach().cpu().numpy()
    print(mel.shape)

    plot_spectrogram(spec, label=label)

    # ImageNet preprocess expects RGB (3 channels); mel spectrograms are single-channel.
    if spec.shape[0] == 1:
        spec = spec.repeat(3, 1, 1)
    spec = preprocess(spec)
    print(spec.shape)
    spec = spec.unsqueeze(0)
    pred = model(spec)
    pred_class = pred.argmax(dim=1).item()
    print(pred_class)


if __name__ == "__main__":
    app()
