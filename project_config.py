"""Konfigurasi bersama dan discovery dataset proyek."""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
RESOURCES_DIR = BASE_DIR / "resources"
CACHE_DIR = BASE_DIR / "cache"
CONFIG_FILE = RESOURCES_DIR / "preprocessing_config.json"


def load_json_file(path: Path) -> dict[str, Any]:
    """Membaca JSON object dari file."""
    if not path.exists():
        raise FileNotFoundError(f"Resource tidak ditemukan: {path.name}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Resource JSON harus berupa object: {path.name}")
    return data


def get_config() -> dict[str, Any]:
    """Mengambil konfigurasi preprocessing pusat."""
    return load_json_file(CONFIG_FILE)


CONFIG = get_config()
MODEL_CONFIG = CONFIG["model"]
TEXT_COLUMN = str(CONFIG["columns"]["text"])
SENTIMENT_COLUMN = str(CONFIG["columns"]["sentiment"])
REQUIRED_COLUMNS = (TEXT_COLUMN, SENTIMENT_COLUMN)
TEST_SIZE = float(CONFIG["split"]["test_size"])
RANDOM_STATE = int(CONFIG["split"]["random_state"])
MODEL_OUTPUT = BASE_DIR / str(MODEL_CONFIG["output_path"])
MODEL_METADATA_OUTPUT = BASE_DIR / str(MODEL_CONFIG["metadata_path"])


def validate_config() -> None:
    """Validasi konfigurasi utama proyek."""
    if not 0 < TEST_SIZE < 1:
        raise ValueError("split.test_size wajib berada antara 0 dan 1.")
    thresholds = CONFIG["thresholds"]
    for key in ("typo", "repeat", "leet"):
        value = float(thresholds[key])
        if not 0 <= value <= 100:
            raise ValueError(f"thresholds.{key} wajib berada antara 0 dan 100.")
    if float(thresholds["emoji_positive"]) <= float(thresholds["emoji_negative"]):
        raise ValueError("threshold emoji positif wajib lebih besar dari negatif.")
    tfidf = CONFIG["tfidf"]
    if int(tfidf["ngram_min"]) <= 0 or int(tfidf["ngram_max"]) < int(tfidf["ngram_min"]):
        raise ValueError("Konfigurasi ngram TF-IDF tidak valid.")
    if float(CONFIG["naive_bayes"]["alpha"]) <= 0:
        raise ValueError("naive_bayes.alpha wajib lebih besar dari 0.")
    if not str(CONFIG["dataset"]["filename_regex"]).strip():
        raise ValueError("dataset.filename_regex wajib diisi.")


validate_config()


def natural_sort_key(path: Path) -> list[int | str]:
    """Mengurutkan path dengan urutan natural."""
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def discover_dataset_files() -> list[Path]:
    """Menemukan dataset CSV dengan regex dari konfigurasi."""
    dataset_config = CONFIG["dataset"]
    glob_pattern = str(dataset_config["glob"])
    filename_regex = re.compile(str(dataset_config["filename_regex"]))
    files = [
        path
        for path in BASE_DIR.glob(glob_pattern)
        if path.is_file() and filename_regex.fullmatch(path.name)
    ]
    files = sorted(files, key=natural_sort_key)
    if not files:
        raise FileNotFoundError(
            f"Tidak ada dataset yang cocok dengan regex {filename_regex.pattern}."
        )
    return files


def label_token(name: str) -> str:
    """Mengambil token label sentimen dari konfigurasi."""
    return str(CONFIG["labels"][name])


def display_label(name: str) -> str:
    """Mengambil label tampilan dari konfigurasi."""
    return str(CONFIG["display_labels"][name])


def sentiment_value(name: str) -> int:
    """Mengambil nilai label dataset dari konfigurasi."""
    return int(CONFIG["sentiment_values"][name])
