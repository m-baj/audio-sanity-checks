import csv
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
TRAINING_DIR = MODELS_DIR / "training"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass


import torch
import numpy as np

torch.set_float32_matmul_precision("high")
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# GPU operations have a separate seed we also want to set
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Additionally, some operations on a GPU are implemented stochastic for efficiency
    # We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False


SPEECH_COMMANDS_LABELS_DICT = {
    "backward": 0,
    "bed": 1,
    "bird": 2,
    "cat": 3,
    "dog": 4,
    "down": 5,
    "eight": 6,
    "five": 7,
    "follow": 8,
    "forward": 9,
    "four": 10,
    "go": 11,
    "happy": 12,
    "house": 13,
    "learn": 14,
    "left": 15,
    "marvin": 16,
    "nine": 17,
    "no": 18,
    "off": 19,
    "on": 20,
    "one": 21,
    "right": 22,
    "seven": 23,
    "sheila": 24,
    "six": 25,
    "stop": 16,
    "three": 27,
    "tree": 28,
    "two": 29,
    "up": 30,
    "visual": 31,
    "wow": 32,
    "yes": 33,
    "zero": 34,
}


ESC50_LABELS_DICT = {
    "airplane": 0,
    "breathing": 1,
    "brushing_teeth": 2,
    "can_opening": 3,
    "car_horn": 4,
    "cat": 5,
    "chainsaw": 6,
    "chirping_birds": 7,
    "church_bells": 8,
    "clapping": 9,
    "clock_alarm": 10,
    "clock_tick": 11,
    "coughing": 12,
    "cow": 13,
    "crackling_fire": 14,
    "crickets": 15,
    "crow": 16,
    "crying_baby": 17,
    "dog": 18,
    "door_wood_creaks": 19,
    "door_wood_knock": 20,
    "drinking_sipping": 21,
    "engine": 22,
    "fireworks": 23,
    "footsteps": 24,
    "frog": 25,
    "glass_breaking": 26,
    "hand_saw": 27,
    "helicopter": 28,
    "hen": 29,
    "insects": 30,
    "keyboard_typing": 31,
    "laughing": 32,
    "mouse_click": 33,
    "pig": 34,
    "pouring_water": 35,
    "rain": 36,
    "rooster": 37,
    "sea_waves": 38,
    "sheep": 39,
    "siren": 40,
    "sneezing": 41,
    "snoring": 42,
    "thunderstorm": 43,
    "toilet_flush": 44,
    "train": 45,
    "vacuum_cleaner": 46,
    "washing_machine": 47,
    "water_drops": 48,
    "wind": 49,
}
