import torchaudio
from pathlib import Path
import requests
import zipfile
import shutil
import os


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"


def download_speech_commands():
    if (RAW_DATA_PATH / "SpeechCommands").exists():
        print(f"--- SpeechCommands already exists at {RAW_DATA_PATH / 'SpeechCommands'}. Skipping download. ---")
        return

    dataset = torchaudio.datasets.SPEECHCOMMANDS(root=RAW_DATA_PATH, download=True)
    waveform, sample_rate, label = dataset[0]

    print(f"Dataset downloaded to: {RAW_DATA_PATH.resolve()}")
    print(f"Number of samples: {len(dataset)}")
    print(f"Label of first sample: {label}")
    print(f"Audio shape: {waveform.shape} (Channels x Samples)")


def download_esc50():
    url = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
    
    target_dir = RAW_DATA_PATH / "esc50"
    temp_extract = RAW_DATA_PATH / "temp_esc50"
    zip_file = RAW_DATA_PATH / "esc50.zip"

    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)

    if target_dir.exists():
        print(f"--- ESC-50 already exists at {target_dir}. Skipping download. ---")
        return

    print(f"--- Downloading ESC-50 from {url} ---")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(zip_file, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print("--- Extracting files ---")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(temp_extract)

    source_inner = temp_extract / "ESC-50-master"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    for folder in ["audio", "meta"]:
        print(f"--- Moving {folder} to {target_dir} ---")
        shutil.move(str(source_inner / folder), str(target_dir / folder))

    print("--- Cleanup: removing temporary files ---")

    if zip_file.exists():
        os.remove(zip_file)
    if temp_extract.exists():
        shutil.rmtree(temp_extract)
            
    print(f"--- Success! ESC-50 is ready in {target_dir} ---")


if __name__ == "__main__":
    download_speech_commands()
    download_esc50()
