from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import torch
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import RichProgressBar

from audio_sanity_checks.config import (
    PROCESSED_DATA_DIR,
    TRAINING_DIR,
)
from audio_sanity_checks.modeling.models import SpectrogramModel
from audio_sanity_checks.spectrograms import (
    SpeechCommandsSpectrogramDataset,
    ESC50SpectrogramDataset,
)

app = typer.Typer()


def load_datasets(dataset_path: Path):
    train_dataset = torch.load(
        PROCESSED_DATA_DIR / dataset_path / f"{dataset_path}_training.pt",
        weights_only=False,
    )
    val_dataset = torch.load(
        PROCESSED_DATA_DIR / dataset_path / f"{dataset_path}_validation.pt",
        weights_only=False,
    )
    test_dataset = torch.load(
        PROCESSED_DATA_DIR / dataset_path / f"{dataset_path}_testing.pt",
        weights_only=False,
    )
    return train_dataset, val_dataset, test_dataset


@app.command()
def main(
    dataset_name: str = typer.Argument("speech_commands", help="Path to the dataset"),
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Training spectrogram model...")
    if dataset_name not in ["speech_commands", "esc50"]:
        logger.error(f"Invalid dataset path: {dataset_name}")
        raise typer.Exit(code=1)
    model = SpectrogramModel(num_classes=35)
    model.to(device)

    wandb_logger = WandbLogger(
        project="audio-sanity-checks",
        name=f"{dataset_name}-resnet18-test-run",
        save_dir=TRAINING_DIR,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        verbose=False,
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=2,
        monitor="val_loss",
        mode="min",
        dirpath=TRAINING_DIR / "resnet18-test-run",
        filename=f"{dataset_name}" + "-{epoch:02d}-{val_loss:.4f}",
    )

    trainer = L.Trainer(
        max_epochs=model.num_epochs,
        logger=wandb_logger,
        callbacks=[early_stopping, RichProgressBar(), checkpoint_callback],
        deterministic=True,
    )

    train_dataset, val_dataset, test_dataset = load_datasets(dataset_name)
    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, num_workers=4, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    trainer.fit(model, train_loader, val_loader)
    trainer.test(ckpt_path="best", dataloaders=test_loader)
    wandb_logger.experiment.finish()
    logger.success("Spectrogram model training complete.")

    del model
    del trainer

    torch.cuda.empty_cache()


if __name__ == "__main__":
    app()
