"""Pemodelan Analisis Sentimen Bahasa Indonesia.

Semua data kebahasaan dan nilai konfigurasi dibaca dari resource eksternal.
File ini hanya berisi algoritma pemuatan, preprocessing, training, dan logging.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import csv
import hashlib
import importlib.metadata
import json
import logging
import platform
from pathlib import Path
import re
import time
from typing import Any

import emoji
import joblib
import pandas as pd
from emot.core import emot as EmotParser
from rapidfuzz import fuzz, process
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from project_config import (
    BASE_DIR,
    CACHE_DIR,
    CONFIG,
    CONFIG_FILE,
    MODEL_OUTPUT,
    MODEL_METADATA_OUTPUT,
    RANDOM_STATE,
    REQUIRED_COLUMNS,
    RESOURCES_DIR,
    SENTIMENT_COLUMN,
    TEST_SIZE,
    TEXT_COLUMN,
    discover_dataset_files,
    label_token,
)


DATASET_FILES = discover_dataset_files()
DICTIONARY_FILE = BASE_DIR / "kamus_indonesia.txt"
SLANG_FILE = RESOURCES_DIR / "slang_indonesia.json"
LEET_RULES_FILE = RESOURCES_DIR / "leet_rules.json"
ADDITIONAL_DICTIONARY_FILE = RESOURCES_DIR / "additional_dictionary.json"
NEGATIONS_FILE = RESOURCES_DIR / "negations_indonesia.txt"
STOPWORD_FILE = RESOURCES_DIR / "stopwords_indonesia.txt"
STEMMING_EXCLUSIONS_FILE = RESOURCES_DIR / "stemming_exclusions.txt"
EMOTICON_SENTIMENT_FILE = RESOURCES_DIR / "emoticon_sentiment.csv"
EMOJI_SENTIMENT_FILE = CACHE_DIR / "emoji_sentiment.csv"

THRESHOLDS = CONFIG["thresholds"]
PREPROCESSING_CONFIG = CONFIG["preprocessing"]
TFIDF_CONFIG = CONFIG["tfidf"]
NB_CONFIG = CONFIG["naive_bayes"]
TOKEN_PATTERN = re.compile(r"[a-z0-9_]+", re.IGNORECASE)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_PATTERN = re.compile(r"<[^>]+>")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")
REPEATED_CHAR_PATTERN = re.compile(r"([a-z])\1{2,}", re.IGNORECASE)
MOJIBAKE_HINT_PATTERN = re.compile(r"[^\x00-\x7f]")
WORD_SAFE_PATTERN = re.compile(r"[^a-z0-9_\s]", re.IGNORECASE)
VARIATION_SELECTOR_PATTERN = re.compile("[\ufe00-\ufe0f]")

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"
SECTION_WIDTH = 55
FIELD_WIDTH = 28
logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT, level=logging.INFO)
LOGGER = logging.getLogger(__name__)


class FastSetDictionary:
    """Dictionary kompatibel Sastrawi dengan lookup berbasis set."""

    def __init__(self, words: Iterable[str] | None = None) -> None:
        self.words: set[str] = set()
        if words:
            self.add_words(words)

    def contains(self, word: str) -> bool:
        """Memeriksa keberadaan kata."""
        return word in self.words

    def count(self) -> int:
        """Mengembalikan jumlah kata."""
        return len(self.words)

    def add_words(self, words: Iterable[str]) -> None:
        """Menambahkan banyak kata."""
        for word in words:
            self.add(word)

    def add(self, word: str) -> None:
        """Menambahkan satu kata."""
        if word and word.strip():
            self.words.add(word)


@dataclass
class PreprocessingStats:
    """Statistik preprocessing."""

    tokens_checked: int = 0
    typo_fixed: int = 0
    typo_not_found: int = 0
    leet_fixed: int = 0
    repeated_fixed: int = 0
    emoji_found: int = 0
    emoticon_found: int = 0
    slang_normalized: int = 0
    cache_hit: int = 0
    cache_miss: int = 0


STATS = PreprocessingStats()
KATA_BAKU: list[str] = []
KATA_BAKU_SET: set[str] = set()
KATA_BAKU_BY_LENGTH: dict[int, list[str]] = {}
KATA_BAKU_BY_FIRST_AND_LENGTH: dict[tuple[str, int], list[str]] = {}
SLANG_MAP: dict[str, str] = {}
LEET_RULES: dict[str, list[str]] = {}
NEGATION_SET: set[str] = set()
STOP_WORDS: set[str] = set()
STEM_EXCLUSIONS: set[str] = set()
EMOTICON_SENTIMENT: dict[str, str] = {}
EMOJI_SENTIMENT: dict[str, str] = {}
EMOTICON_PARSER = EmotParser()


def create_fast_stemmer() -> Any:
    """Membuat stemmer Sastrawi dengan dictionary lookup cepat."""
    stemmer_obj = StemmerFactory().create_stemmer()
    delegated = getattr(stemmer_obj, "delegatedStemmer", None)
    dictionary = getattr(delegated, "dictionary", None)
    words = getattr(dictionary, "words", None)
    if delegated is not None and words is not None:
        delegated.dictionary = FastSetDictionary(words)
    return stemmer_obj


STEMMER = create_fast_stemmer()


def require_file(path: Path, message: str) -> None:
    """Memastikan file resource tersedia."""
    if not path.exists():
        raise FileNotFoundError(message)
    if path.stat().st_size == 0:
        raise ValueError(f"Resource kosong: {path.name}")


def load_json_object(path: Path) -> dict[str, Any]:
    """Membaca resource JSON object."""
    require_file(path, f"Resource {path.name} tidak ditemukan. Jalankan setup_resources.py.")
    data = pd.read_json(path, typ="series").to_dict()
    if not isinstance(data, dict):
        raise ValueError(f"Resource JSON harus berupa object: {path.name}")
    return data


def load_text_set(path: Path) -> set[str]:
    """Membaca daftar kata dari file teks."""
    require_file(path, f"Resource {path.name} tidak ditemukan. Jalankan setup_resources.py.")
    return {
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def load_dictionary() -> list[str]:
    """Memuat kamus utama dan kosakata tambahan."""
    require_file(DICTIONARY_FILE, f"Kamus tidak ditemukan: {DICTIONARY_FILE.name}")
    words = load_text_set(DICTIONARY_FILE)
    additional = load_json_object(ADDITIONAL_DICTIONARY_FILE).get("words", [])
    if not isinstance(additional, list):
        raise ValueError("additional_dictionary.json wajib berisi list words.")
    words.update(str(word).strip().lower() for word in additional if str(word).strip())
    if not words:
        raise ValueError(f"Kamus tidak ditemukan atau kosong: {DICTIONARY_FILE.name}")
    return sorted(words)


def load_slang_map() -> dict[str, str]:
    """Memuat slang dari resource JSON."""
    data = load_json_object(SLANG_FILE)
    mapping = data.get("slang")
    if not isinstance(mapping, dict):
        raise ValueError("slang_indonesia.json wajib memiliki object slang.")
    result: dict[str, str] = {}
    for key, value in mapping.items():
        if not str(key).strip() or not str(value).strip():
            raise ValueError("slang_indonesia.json berisi key/value kosong.")
        result[str(key).lower()] = str(value).lower()
    return result


def load_leet_rules() -> dict[str, list[str]]:
    """Memuat aturan leetspeak dari JSON."""
    data = load_json_object(LEET_RULES_FILE)
    substitutions = data.get("substitutions")
    if not isinstance(substitutions, dict):
        raise ValueError("leet_rules.json wajib memiliki object substitutions.")
    rules: dict[str, list[str]] = {}
    for key, values in substitutions.items():
        if not isinstance(values, list) or not values:
            raise ValueError("Setiap aturan leetspeak wajib berupa list.")
        rules[str(key)] = [str(value).lower() for value in values if str(value).strip()]
    return rules


def load_stop_words() -> set[str]:
    """Memuat stopword dari resource lokal."""
    words = load_text_set(STOPWORD_FILE)
    additional = load_json_object(ADDITIONAL_DICTIONARY_FILE).get("words", [])
    if not isinstance(additional, list):
        raise ValueError("additional_dictionary.json wajib berisi list words.")
    additional_words = {str(word).lower() for word in additional if str(word).strip()}
    return {word.lower() for word in words} - NEGATION_SET - additional_words


def load_emoticon_sentiment() -> dict[str, str]:
    """Memuat sentimen emoticon dari CSV resource."""
    require_file(
        EMOTICON_SENTIMENT_FILE,
        "Resource emoticon_sentiment.csv tidak ditemukan. Jalankan setup_resources.py.",
    )
    result: dict[str, str] = {}
    with EMOTICON_SENTIMENT_FILE.open(newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            emoticon_value = str(row.get("emoticon", "")).strip()
            sentiment = str(row.get("sentiment", "")).strip()
            if emoticon_value and sentiment:
                result[emoticon_value] = sentiment
    if not result:
        raise ValueError("Resource emoticon_sentiment.csv kosong.")
    return result


def sentiment_from_score(score: float) -> str:
    """Mengubah skor emoji menjadi token sentimen dari konfigurasi."""
    if score >= float(THRESHOLDS["emoji_positive"]):
        return label_token("positive")
    if score <= float(THRESHOLDS["emoji_negative"]):
        return label_token("negative")
    return label_token("neutral")


def normalize_emoji_key(value: str) -> str:
    """Menormalkan emoji untuk lookup resource."""
    return VARIATION_SELECTOR_PATTERN.sub("", value)


def repair_mojibake(text: str) -> str:
    """Memilih decoding dengan jumlah emoji Unicode terbanyak."""
    if not MOJIBAKE_HINT_PATTERN.search(text):
        return text
    candidates = [text]
    for source_encoding in ("latin1", "cp1252"):
        try:
            candidates.append(text.encode(source_encoding, errors="ignore").decode("utf-8"))
        except UnicodeError:
            continue
    try:
        bytes_buffer = bytearray()
        for char in text:
            try:
                bytes_buffer.extend(char.encode("cp1252"))
            except UnicodeEncodeError:
                if ord(char) <= 255:
                    bytes_buffer.append(ord(char))
                else:
                    bytes_buffer.extend(char.encode("utf-8", errors="ignore"))
        candidates.append(bytes(bytes_buffer).decode("utf-8"))
    except UnicodeError:
        LOGGER.debug("Perbaikan mojibake tidak menghasilkan kandidat tambahan.")
    return max(candidates, key=lambda value: emoji.emoji_count(value))


def load_emoji_sentiment() -> dict[str, str]:
    """Memuat sentimen emoji dari cache CSV terstruktur."""
    if not EMOJI_SENTIMENT_FILE.exists() or EMOJI_SENTIMENT_FILE.stat().st_size == 0:
        raise FileNotFoundError(
            "Resource emoji_sentiment.csv tidak tersedia. Jalankan setup_resources.py."
        )
    data = pd.read_csv(EMOJI_SENTIMENT_FILE)
    if data.empty:
        raise ValueError("Resource emoji_sentiment.csv kosong.")
    result: dict[str, str] = {}
    for _, row in data.iterrows():
        emoji_char = repair_mojibake(str(row.get("Emoji", row.get("emoji", ""))).strip())
        if not emoji_char:
            continue
        score = None
        for column in ("Sentiment score", "sentiment_score", "score"):
            if column in row and pd.notna(row[column]):
                score = float(row[column])
                break
        if score is None:
            good_count = float(row.get("Positive", 0))
            bad_count = float(row.get("Negative", 0))
            occurrences = max(float(row.get("Occurrences", 1)), 1.0)
            score = (good_count - bad_count) / occurrences
        token = sentiment_from_score(float(score))
        result[emoji_char] = token
        result[normalize_emoji_key(emoji_char)] = token
    if not result:
        raise ValueError("Resource emoji_sentiment.csv tidak menghasilkan mapping.")
    return result


def initialize_resources() -> None:
    """Memuat seluruh resource preprocessing."""
    global KATA_BAKU, KATA_BAKU_SET, KATA_BAKU_BY_LENGTH
    global KATA_BAKU_BY_FIRST_AND_LENGTH, SLANG_MAP, LEET_RULES
    global NEGATION_SET, STOP_WORDS, STEM_EXCLUSIONS
    global EMOTICON_SENTIMENT, EMOJI_SENTIMENT
    NEGATION_SET = load_text_set(NEGATIONS_FILE)
    STEM_EXCLUSIONS = load_text_set(STEMMING_EXCLUSIONS_FILE) | set(CONFIG["special_tokens"])
    KATA_BAKU = load_dictionary()
    KATA_BAKU_SET = set(KATA_BAKU)
    KATA_BAKU_BY_LENGTH = {}
    KATA_BAKU_BY_FIRST_AND_LENGTH = {}
    for word in KATA_BAKU:
        KATA_BAKU_BY_LENGTH.setdefault(len(word), []).append(word)
        KATA_BAKU_BY_FIRST_AND_LENGTH.setdefault((word[0], len(word)), []).append(word)
    SLANG_MAP = load_slang_map()
    LEET_RULES = load_leet_rules()
    STOP_WORDS = load_stop_words()
    EMOTICON_SENTIMENT = load_emoticon_sentiment()
    EMOJI_SENTIMENT = load_emoji_sentiment()


def reset_normalization_stats() -> None:
    """Mereset statistik runtime."""
    global STATS
    STATS = PreprocessingStats()


def remove_double_spaces(text: str) -> str:
    """Merapikan spasi."""
    return re.sub(r"\s+", " ", text).strip()


def safe_text_token(text: str) -> str:
    """Mengubah deskripsi menjadi token aman."""
    safe_text = WORD_SAFE_PATTERN.sub(" ", text.lower().replace("-", "_"))
    return remove_double_spaces(safe_text.replace(" ", "_"))


def remove_web_noise_preserve_symbols(text: str) -> str:
    """Membersihkan noise web sebelum deteksi emoticon."""
    text = HTML_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    text = HASHTAG_PATTERN.sub(r" \1 ", text)
    return remove_double_spaces(text)


def convert_emoticons(text: str) -> str:
    """Mengubah emoticon berdasarkan metadata emot dan CSV resource."""
    parsed = EMOTICON_PARSER.emoticons(text)
    values = parsed.get("value") or []
    locations = parsed.get("location") or []
    replacements: list[tuple[int, int, str]] = []
    for index, value in enumerate(values):
        if index >= len(locations):
            continue
        location = locations[index]
        if len(location) != 2:
            continue
        sentiment = EMOTICON_SENTIMENT.get(str(value))
        if sentiment is None:
            sentiment = label_token("neutral")
        replacements.append((int(location[0]), int(location[1]), sentiment))
    converted = text
    for start, end, token in sorted(replacements, reverse=True):
        STATS.emoticon_found += 1
        converted = f"{converted[:start]} {token} {converted[end:]}"
    return converted


def convert_emojis(text: str) -> str:
    """Mengubah emoji berdasarkan skor resource lokal."""
    if emoji.emoji_count(text) == 0:
        return text

    def replace(match: str, _data: dict[str, Any] | None = None) -> str:
        STATS.emoji_found += 1
        token = EMOJI_SENTIMENT.get(match) or EMOJI_SENTIMENT.get(normalize_emoji_key(match))
        if token is None:
            description = safe_text_token(emoji.demojize(match).strip(":"))
            token = description or str(PREPROCESSING_CONFIG["unknown_description_token"])
        return f" {token} "

    return emoji.replace_emoji(text, replace=replace)


def clean_structure(text: str) -> str:
    """Membersihkan simbol setelah emoji dan emoticon diproses."""
    text = HTML_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    text = HASHTAG_PATTERN.sub(r" \1 ", text)
    text = WORD_SAFE_PATTERN.sub(" ", text.lower())
    return remove_double_spaces(text)


def repeated_candidates(token: str) -> set[str]:
    """Membuat kandidat token dari huruf berulang."""
    candidates = {token, REPEATED_CHAR_PATTERN.sub(r"\1", token)}
    candidates.add(REPEATED_CHAR_PATTERN.sub(r"\1\1", token))
    current = token
    while REPEATED_CHAR_PATTERN.search(current):
        current = REPEATED_CHAR_PATTERN.sub(r"\1", current)
        candidates.add(current)
    return {candidate for candidate in candidates if candidate}


def leet_candidates(token: str) -> set[str]:
    """Membuat kandidat leetspeak dari resource aturan."""
    limit = int(PREPROCESSING_CONFIG["max_leet_candidates"])
    candidates = {""}
    for char in token:
        replacements = LEET_RULES.get(char, [char])
        if char not in replacements:
            replacements = [char, *replacements]
        candidates = {
            prefix + replacement
            for prefix in candidates
            for replacement in replacements
        }
        if len(candidates) > limit:
            candidates = set(list(candidates)[:limit])
    return candidates


def dictionary_words_near_length(word: str, ratio: float, same_first: bool) -> list[str]:
    """Mengambil kandidat kamus dengan panjang relevan."""
    max_gap = max(1, int(len(word) * ratio))
    candidates: list[str] = []
    for length in range(len(word) - max_gap, len(word) + max_gap + 1):
        if same_first and word:
            candidates.extend(KATA_BAKU_BY_FIRST_AND_LENGTH.get((word[0], length), []))
        else:
            candidates.extend(KATA_BAKU_BY_LENGTH.get(length, []))
    return candidates


def best_dictionary_candidate(candidates: Iterable[str], threshold: float) -> str | None:
    """Memilih kandidat terbaik berdasarkan kamus dan RapidFuzz."""
    unique_candidates = sorted(set(candidates), key=lambda item: (len(item), item))
    for candidate in unique_candidates:
        if candidate in KATA_BAKU_SET:
            return candidate
    best_word = None
    best_score = -1.0
    for candidate in unique_candidates:
        if not candidate.isalpha():
            continue
        matches = process.extract(
            candidate,
            dictionary_words_near_length(candidate, ratio=0.4, same_first=True),
            scorer=fuzz.WRatio,
            limit=10,
        )
        for word, score, _index in matches:
            if score > best_score:
                best_word = str(word)
                best_score = float(score)
    if best_word and best_score >= threshold:
        return best_word
    return None


@lru_cache(maxsize=int(PREPROCESSING_CONFIG["cache_size"]))
def normalize_token_cached(token: str) -> tuple[str, str]:
    """Normalisasi token dengan cache."""
    if token in STEM_EXCLUSIONS or token in NEGATION_SET or "_" in token:
        return token, "protected"
    if token in SLANG_MAP:
        return SLANG_MAP[token], "slang"
    if token in KATA_BAKU_SET:
        return token, "dictionary"
    if REPEATED_CHAR_PATTERN.search(token):
        candidate = best_dictionary_candidate(
            repeated_candidates(token),
            float(THRESHOLDS["repeat"]),
        )
        if candidate and candidate != token:
            return candidate, "repeated"
    if any(char.isdigit() for char in token) and any(char.isalpha() for char in token):
        candidate = best_dictionary_candidate(
            {
                repeated
                for leet_candidate in leet_candidates(token)
                for repeated in repeated_candidates(leet_candidate)
            },
            float(THRESHOLDS["leet"]),
        )
        if candidate and candidate != token:
            return candidate, "leet"
    if token.isalpha() and len(token) > 3:
        matches = process.extract(
            token,
            dictionary_words_near_length(token, ratio=0.25, same_first=True),
            scorer=fuzz.WRatio,
            limit=10,
        )
        for word, score, _index in matches:
            if score >= float(THRESHOLDS["typo"]) and abs(len(str(word)) - len(token)) <= 1:
                return str(word), "typo"
    return token, "not_found"


def normalize_token(token: str) -> str:
    """Normalisasi satu token dan update statistik."""
    STATS.tokens_checked += 1
    before = normalize_token_cached.cache_info()
    normalized, source = normalize_token_cached(token)
    after = normalize_token_cached.cache_info()
    if after.hits > before.hits:
        STATS.cache_hit += 1
    else:
        STATS.cache_miss += 1
    if source == "slang":
        STATS.slang_normalized += 1
    elif source == "repeated":
        STATS.repeated_fixed += 1
    elif source == "leet":
        STATS.leet_fixed += 1
    elif source == "typo":
        STATS.typo_fixed += 1
    elif source == "not_found":
        STATS.typo_not_found += 1
    return normalized


def combine_negations(tokens: list[str]) -> list[str]:
    """Menggabungkan kata negasi dengan token berikutnya."""
    combined: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in NEGATION_SET and index + 1 < len(tokens):
            next_token = tokens[index + 1]
            if next_token not in STEM_EXCLUSIONS and next_token not in NEGATION_SET:
                combined.append(f"{token}_{next_token}")
                index += 2
                continue
        combined.append(token)
        index += 1
    return combined


@lru_cache(maxsize=int(PREPROCESSING_CONFIG["cache_size"]))
def stem_token_cached(token: str) -> str:
    """Stemming token dengan cache."""
    if "_" in token or token in STEM_EXCLUSIONS or token in KATA_BAKU_SET:
        return token
    return STEMMER.stem(token)


def stem_tokens(tokens: Iterable[str]) -> list[str]:
    """Stemming token tanpa mengubah token terlindungi."""
    return [stem_token_cached(token) for token in tokens]


def clean_text(text: object) -> str:
    """Membersihkan dan menormalisasi teks mentah."""
    raw_text = repair_mojibake(str(text))
    raw_text = remove_web_noise_preserve_symbols(raw_text)
    converted = convert_emojis(convert_emoticons(raw_text))
    cleaned = clean_structure(converted)
    tokens = TOKEN_PATTERN.findall(cleaned)
    normalized = [normalize_token(token) for token in tokens]
    filtered = [
        token
        for token in normalized
        if token in STEM_EXCLUSIONS or "_" in token
        or (
            len(token) >= int(PREPROCESSING_CONFIG["minimum_token_length"])
            and token not in STOP_WORDS
        )
    ]
    negation_combined = combine_negations(filtered)
    return remove_double_spaces(" ".join(stem_tokens(negation_combined)))


def iter_cleaned_texts(texts: Iterable[object]) -> Iterable[str]:
    """Menghasilkan teks yang sudah dipreprocess."""
    for text in texts:
        yield clean_text(text)


def validate_dataset_columns(data: pd.DataFrame, dataset_name: str) -> None:
    """Validasi kolom wajib."""
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Kolom {missing} tidak ditemukan pada {dataset_name}.")


def read_dataset_files(files: Iterable[Path] | None = None) -> list[pd.DataFrame]:
    """Membaca seluruh dataset."""
    selected_files = list(files or discover_dataset_files())
    dataframes: list[pd.DataFrame] = []
    for path in selected_files:
        dataframe = pd.read_csv(path)
        validate_dataset_columns(dataframe, path.name)
        if dataframe.empty:
            raise ValueError(f"Dataset kosong: {path.name}")
        dataframes.append(dataframe)
    return dataframes


def combine_datasets(dataframes: list[pd.DataFrame]) -> pd.DataFrame:
    """Menggabungkan dataset."""
    if not dataframes:
        raise ValueError("Tidak ada dataset yang dapat digabungkan.")
    combined = pd.concat(dataframes, ignore_index=True)
    if combined.empty:
        raise ValueError("Dataset gabungan kosong.")
    return combined


def remove_empty_required_data(data: pd.DataFrame) -> pd.DataFrame:
    """Menghapus baris dengan text/sentiment kosong."""
    cleaned = data.dropna(subset=list(REQUIRED_COLUMNS)).copy()
    text_ok = cleaned[TEXT_COLUMN].astype(str).str.strip().ne("")
    sentiment_ok = cleaned[SENTIMENT_COLUMN].astype(str).str.strip().ne("")
    cleaned = cleaned[text_ok & sentiment_ok].copy()
    if cleaned.empty:
        raise ValueError("Dataset kosong setelah pembersihan data wajib.")
    return cleaned


def normalize_labels(data: pd.DataFrame) -> pd.DataFrame:
    """Validasi dan normalisasi label dataset."""
    normalized = data.copy()
    allowed_values = set(int(value) for value in CONFIG["sentiment_values"].values())
    try:
        labels = normalized[SENTIMENT_COLUMN].map(lambda value: int(str(value).strip()))
    except (TypeError, ValueError) as exc:
        raise ValueError("Label dataset harus berupa nilai numerik konsisten.") from exc
    invalid = sorted(set(labels.dropna().unique()) - allowed_values)
    if invalid:
        raise ValueError(f"Label dataset tidak diizinkan: {invalid}")
    if labels.nunique() < 2:
        raise ValueError("Training membutuhkan minimal dua kelas label.")
    normalized[SENTIMENT_COLUMN] = labels.astype(int)
    distribution = normalized[SENTIMENT_COLUMN].value_counts().sort_index().to_dict()
    LOGGER.info("Distribusi label: %s", distribution)
    return normalized


def resolve_duplicate_texts(data: pd.DataFrame) -> pd.DataFrame:
    """Menangani duplikasi teks dan konflik label sebelum split."""
    conflicts = (
        data.groupby(TEXT_COLUMN)[SENTIMENT_COLUMN]
        .nunique()
        .loc[lambda series: series > 1]
    )
    if not conflicts.empty:
        raise ValueError(
            f"Ditemukan {len(conflicts)} teks dengan konflik label. "
            "Perbaiki dataset sebelum training."
        )
    if str(CONFIG["dataset"]["duplicate_behavior"]) == "drop_exact":
        before = len(data)
        data = data.drop_duplicates(subset=[TEXT_COLUMN, SENTIMENT_COLUMN]).copy()
        removed = before - len(data)
        if removed:
            LOGGER.info("Duplikasi teks-label dihapus sebelum split: %d", removed)
    return data


def preprocess_text(data: pd.DataFrame) -> pd.DataFrame:
    """Memproses kolom text memakai fungsi preprocessing bersama."""
    validate_dataset_columns(data, "DataFrame")
    processed = data.copy()
    text_col = processed.columns.get_loc(TEXT_COLUMN)
    reset_normalization_stats()
    for index, cleaned_text in enumerate(iter_cleaned_texts(processed[TEXT_COLUMN]), start=1):
        processed.iat[index - 1, text_col] = cleaned_text
        if index % int(PREPROCESSING_CONFIG["progress_interval"]) == 0:
            LOGGER.info("Preprocessing %s/%s", index, len(processed))
    return processed


def count_tokens(texts: Iterable[str]) -> tuple[int, int]:
    """Menghitung jumlah token."""
    total = 0
    unique: set[str] = set()
    for text in texts:
        tokens = str(text).split()
        total += len(tokens)
        unique.update(tokens)
    return total, len(unique)


def print_section(title: str) -> None:
    """Log section."""
    LOGGER.info("%s", "=" * SECTION_WIDTH)
    LOGGER.info("%s", title)
    LOGGER.info("%s", "=" * SECTION_WIDTH)


def print_field(label: str, value: object) -> None:
    """Log field."""
    LOGGER.info("%-*s: %s", FIELD_WIDTH, label, value)


def show_normalization_statistics(data: pd.DataFrame, elapsed: float) -> None:
    """Menampilkan statistik preprocessing."""
    total_tokens, unique_tokens = count_tokens(data[TEXT_COLUMN])
    print_section("STATISTIK PREPROCESSING")
    print_field("Jumlah dokumen", len(data))
    print_field("Jumlah token", total_tokens)
    print_field("Token unik", unique_tokens)
    print_field("Jumlah token diperiksa", STATS.tokens_checked)
    print_field("Typo diperbaiki", STATS.typo_fixed)
    print_field("Typo tidak ditemukan", STATS.typo_not_found)
    print_field("Leetspeak diperbaiki", STATS.leet_fixed)
    print_field("Huruf berulang diperbaiki", STATS.repeated_fixed)
    print_field("Emoji ditemukan", STATS.emoji_found)
    print_field("Emoticon ditemukan", STATS.emoticon_found)
    print_field("Slang dinormalisasi", STATS.slang_normalized)
    print_field("Cache hit", STATS.cache_hit)
    print_field("Cache miss", STATS.cache_miss)
    print_field("Waktu preprocessing", f"{elapsed:.3f} detik")


def create_model() -> Pipeline:
    """Membuat pipeline TF-IDF dan MultinomialNB."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(int(TFIDF_CONFIG["ngram_min"]), int(TFIDF_CONFIG["ngram_max"])),
                    min_df=int(TFIDF_CONFIG["min_df"]),
                    max_df=float(TFIDF_CONFIG["max_df"]),
                ),
            ),
            ("naive_bayes", MultinomialNB(alpha=float(NB_CONFIG["alpha"]))),
        ]
    )


def save_model(model: Pipeline, output_path: Path) -> None:
    """Menyimpan model."""
    try:
        joblib.dump(model, output_path)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Model gagal disimpan: {output_path.name}") from exc


def file_sha256(path: Path) -> str:
    """Menghitung checksum SHA256 file."""
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str:
    """Mengambil versi package untuk metadata."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "tidak tersedia"


def resource_checksums() -> dict[str, str]:
    """Checksum resource utama yang memengaruhi preprocessing."""
    paths = [
        CONFIG_FILE,
        DICTIONARY_FILE,
        SLANG_FILE,
        LEET_RULES_FILE,
        ADDITIONAL_DICTIONARY_FILE,
        NEGATIONS_FILE,
        STOPWORD_FILE,
        STEMMING_EXCLUSIONS_FILE,
        EMOTICON_SENTIMENT_FILE,
        EMOJI_SENTIMENT_FILE,
    ]
    return {path.name: file_sha256(path) for path in paths if path.exists()}


def save_model_metadata(
    data: pd.DataFrame,
    model: Pipeline,
    y_test: pd.Series,
    y_pred: Iterable[int],
) -> None:
    """Menyimpan metadata kompatibilitas model."""
    predictions = pd.Series(list(y_pred))
    metadata = {
        "model_version": str(CONFIG["model"]["version"]),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset_files": [path.name for path in DATASET_FILES],
        "dataset_checksum": {
            path.name: file_sha256(path) for path in DATASET_FILES if path.exists()
        },
        "data_count": int(len(data)),
        "label_distribution": {
            str(key): int(value)
            for key, value in data[SENTIMENT_COLUMN].value_counts().sort_index().items()
        },
        "config_checksum": file_sha256(CONFIG_FILE),
        "resource_checksum": resource_checksums(),
        "library_versions": {
            "python": platform.python_version(),
            "pandas": package_version("pandas"),
            "scikit-learn": package_version("scikit-learn"),
            "joblib": package_version("joblib"),
            "Sastrawi": package_version("Sastrawi"),
            "RapidFuzz": package_version("RapidFuzz"),
            "emoji": package_version("emoji"),
            "emot": package_version("emot"),
        },
        "evaluation_metrics": {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
        },
        "pipeline_steps": list(model.named_steps.keys()),
    }
    MODEL_METADATA_OUTPUT.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Training model end-to-end."""
    start = time.perf_counter()
    print_section("A. MEMBACA DATASET")
    data = combine_datasets(read_dataset_files())
    print_field("Jumlah file", len(DATASET_FILES))
    print_field("Jumlah data", len(data))
    print_section("B. MEMBERSIHKAN DATA")
    data = remove_empty_required_data(data)
    data = normalize_labels(data)
    data = resolve_duplicate_texts(data)
    print_field("Jumlah data bersih", len(data))
    print_section("C. PREPROCESSING")
    preprocessing_start = time.perf_counter()
    data = preprocess_text(data)
    show_normalization_statistics(data, time.perf_counter() - preprocessing_start)
    print_section("D. TRAIN TEST SPLIT")
    x_train, x_test, y_train, y_test = train_test_split(
        data[TEXT_COLUMN],
        data[SENTIMENT_COLUMN],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data[SENTIMENT_COLUMN],
    )
    print_field("Jumlah data training", len(x_train))
    print_field("Jumlah data testing", len(x_test))
    print_section("E. TRAINING MODEL")
    model = create_model()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    LOGGER.info("Training selesai")
    print_section("F. SIMPAN MODEL")
    save_model(model, MODEL_OUTPUT)
    save_model_metadata(data, model, y_test, y_pred)
    print_field("File model", MODEL_OUTPUT.name)
    print_field("Metadata model", MODEL_METADATA_OUTPUT.name)
    print_section("G. WAKTU EKSEKUSI")
    print_field("Lama proses", f"{time.perf_counter() - start:.3f} detik")


initialize_resources()


if __name__ == "__main__":
    main()
