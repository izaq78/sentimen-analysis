"""Exploratory data analysis for the sentiment dataset.

This module performs dataset loading, validation, and text-level EDA for a
set of CSV files. It logs dataset structure, quality, review lengths, label
distribution, and a final summary suitable for NLP project reporting.
"""

import io
import logging
import time
from pathlib import Path
from typing import Sequence

import pandas as pd
from pandas import DataFrame, Series

from project_config import (
    SENTIMENT_COLUMN,
    TEXT_COLUMN,
    discover_dataset_files,
    display_label,
    sentiment_value,
)

__author__ = "Rifqi Khoirul"
__version__ = "1.0.0"
__all__ = ["main"]

BASE_DIR = Path(__file__).resolve().parent
DATASET_FILES = discover_dataset_files()
REQUIRED_COLUMNS = [TEXT_COLUMN, SENTIMENT_COLUMN]
BAD_VALUE = sentiment_value("negative")
GOOD_VALUE = sentiment_value("positive")
BAD_LABEL = display_label("negative")
GOOD_LABEL = display_label("positive")
SECTION_WIDTH = 55
FIELD_WIDTH = 24
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"
logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
    level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def log_section(title: str) -> None:
    """Log a consistent section title separator."""
    separator = "=" * SECTION_WIDTH
    LOGGER.info(separator)
    LOGGER.info(title)
    LOGGER.info(separator)


def log_field(label: str, value: object) -> None:
    """Log a key-value line with consistent alignment."""
    LOGGER.info("%-*s: %s", FIELD_WIDTH, label, value)


def render_dataframe(dataframe: DataFrame, max_rows: int = 10) -> str:
    """Render a DataFrame as a string for logging."""
    with io.StringIO() as buffer:
        buffer.write(dataframe.head(max_rows).to_string(index=False))
        return buffer.getvalue()


def log_dataframe_info(dataframe: DataFrame) -> None:
    """Log pandas DataFrame info output."""
    with io.StringIO() as buffer:
        dataframe.info(buf=buffer)
        LOGGER.info("%s", buffer.getvalue())


def read_dataset_files(files: Sequence[Path]) -> list[DataFrame]:
    """Read all CSV dataset files after validating file availability."""
    dataframes: list[DataFrame] = []

    for dataset_file in files:
        if not dataset_file.exists():
            raise FileNotFoundError(
                f"File dataset tidak ditemukan: {dataset_file.name}"
            )

        dataframe = pd.read_csv(dataset_file)
        validate_dataset_columns(dataframe, dataset_file.name)
        dataframes.append(dataframe)

    if len(dataframes) != len(files):
        raise ValueError(
            f"Jumlah DataFrame tidak sesuai. Ditemukan {len(dataframes)}, "
            f"seharusnya {len(files)}."
        )

    LOGGER.info("Berhasil membaca %d file dataset.", len(files))
    return dataframes


def validate_dataset_columns(data: DataFrame, dataset_name: str) -> None:
    """Validate required columns and basic dataset integrity."""
    if data.empty:
        raise ValueError(f"Dataset kosong: {dataset_name}.")

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Kolom {missing_columns} tidak ditemukan pada {dataset_name}."
        )

    if data[SENTIMENT_COLUMN].isnull().all():
        raise ValueError(
            f"Kolom {SENTIMENT_COLUMN} seluruhnya kosong pada {dataset_name}."
        )


def combine_datasets(dataframes: Sequence[DataFrame]) -> DataFrame:
    """Combine all validated DataFrames into a single DataFrame."""
    combined_data = pd.concat(dataframes, ignore_index=True)
    if combined_data.empty:
        raise ValueError("Dataset kosong setelah digabungkan.")

    return combined_data


def get_memory_usage_kb(data: DataFrame) -> float:
    """Return total DataFrame memory usage in kilobytes."""
    return data.memory_usage(deep=True).sum() / 1024


def get_column_summary(data: DataFrame) -> tuple[int, int]:
    """Return the number of numeric and categorical columns."""
    numeric_columns = data.select_dtypes(include="number").columns
    categorical_columns = data.columns.difference(numeric_columns)
    return len(numeric_columns), len(categorical_columns)


def format_sentiment_label(label: object) -> str:
    """Convert sentiment values into readable label names."""
    if label in [BAD_VALUE, float(BAD_VALUE), str(BAD_VALUE), f"{float(BAD_VALUE):.1f}"]:
        return BAD_LABEL
    if label in [GOOD_VALUE, float(GOOD_VALUE), str(GOOD_VALUE), f"{float(GOOD_VALUE):.1f}"]:
        return GOOD_LABEL
    return str(label)


def get_text_length(data: DataFrame) -> Series:
    """Return the character length of each review."""
    return data[TEXT_COLUMN].astype(str).str.len()


def get_word_count(data: DataFrame) -> Series:
    """Return the word count of each review."""
    return data[TEXT_COLUMN].astype(str).str.split().str.len()


def get_series_statistics(series: Series) -> tuple[int, int, float, float]:
    """Return min, max, mean, and median values for a numeric series."""
    return (
        int(series.min()),
        int(series.max()),
        float(series.mean()),
        float(series.median()),
    )


def log_series_statistics(label_prefix: str, metrics: Series) -> None:
    """Log statistics for a numeric text metric with consistent labels."""
    minimum, maximum, mean_value, median_value = get_series_statistics(metrics)
    label_lower = label_prefix.lower()
    log_field(f"{label_prefix} terpendek", minimum)
    log_field(f"{label_prefix} terpanjang", maximum)
    log_field(f"Rata-rata {label_lower}", f"{mean_value:.2f}")
    log_field(f"Median {label_lower}", f"{median_value:.2f}")


def log_top_reviews(
    data: DataFrame,
    sorting_series: Series,
    top_n: int = 3,
) -> None:
    """Log top longest or shortest reviews based on a numeric metric."""
    top_reviews = data.loc[sorting_series.nlargest(top_n).index]
    for index, review in enumerate(top_reviews[TEXT_COLUMN].tolist(), start=1):
        LOGGER.info("%d. %s", index, review)


def show_dataset_information(data: DataFrame) -> None:
    """Display high-level dataset information."""
    log_section("A. INFORMASI DATASET")
    memory_usage_kb = get_memory_usage_kb(data)
    memory_usage_mb = memory_usage_kb / 1024
    log_field("Jumlah file dataset", f"{len(DATASET_FILES)} File")
    log_field("Jumlah data", data.shape[0])
    log_field("Jumlah kolom", data.shape[1])
    log_field("Shape dataset", data.shape)
    log_field("Memory Usage (KB)", f"{memory_usage_kb:.2f}")
    log_field("Memory Usage (MB)", f"{memory_usage_mb:.2f}")


def show_dataset_structure(data: DataFrame) -> None:
    """Display column names, column groups, and data types."""
    numeric_count, categorical_count = get_column_summary(data)
    log_section("B. STRUKTUR DATASET")
    LOGGER.info("Nama seluruh kolom:")
    for index, column_name in enumerate(data.columns, start=1):
        LOGGER.info("%d. %s", index, column_name)
    log_field("Jumlah kolom numerik", numeric_count)
    log_field("Jumlah kolom kategorikal", categorical_count)
    LOGGER.info("Tipe data:")
    LOGGER.info("%s", data.dtypes.to_string())


def show_data_quality(data: DataFrame) -> tuple[int, int]:
    """Display missing value and duplicate information."""
    missing_values = data.isnull().sum()
    total_missing_values = int(missing_values.sum())
    missing_percentages = (missing_values / len(data)) * 100
    duplicate_count = int(data.duplicated().sum())
    duplicate_percentage = (duplicate_count / len(data)) * 100
    log_section("C. KUALITAS DATA")
    LOGGER.info("Missing Value tiap kolom:")
    LOGGER.info("%s", missing_values.to_string())
    LOGGER.info("Persentase Missing Value:")
    LOGGER.info("%s", missing_percentages.round(2).to_string())
    log_field("Total Missing Value", total_missing_values)
    log_field("Jumlah Duplicate", duplicate_count)
    log_field("Persentase Duplicate", f"{duplicate_percentage:.2f}%")
    if duplicate_count > 0:
        duplicate_rows = data[data.duplicated(keep=False)].head(10)
        LOGGER.info("Contoh data duplicate maksimal 10 baris:")
        LOGGER.info("%s", render_dataframe(duplicate_rows, max_rows=10))
    else:
        LOGGER.info("Dataset tidak memiliki data duplicate.")
    return total_missing_values, duplicate_count


def show_text_length_analysis(data: DataFrame) -> None:
    """Display character and word count statistics for each review."""
    log_section("D. ANALISIS PANJANG REVIEW")
    char_length = get_text_length(data)
    word_count = get_word_count(data)
    log_series_statistics("Karakter", char_length)
    log_series_statistics("Kata", word_count)
    unique_reviews = data[TEXT_COLUMN].nunique(dropna=False)
    duplicate_reviews = int(data[TEXT_COLUMN].duplicated().sum())
    log_field("Unique review", unique_reviews)
    log_field("Duplicate review", duplicate_reviews)
    LOGGER.info("\nContoh review terpanjang:")
    log_top_reviews(data, char_length)
    shortest_indices = char_length.nsmallest(3).index
    LOGGER.info("\nContoh review terpendek:")
    for index, review in enumerate(
        data.loc[shortest_indices, TEXT_COLUMN],
        start=1,
    ):
        LOGGER.info("%d. %s", index, review)


def show_label_distribution(data: DataFrame) -> tuple[int, int]:
    """Display sentiment label counts, percentages, and imbalance analysis."""
    labels = data[SENTIMENT_COLUMN].apply(format_sentiment_label)
    bad_count = int((labels == BAD_LABEL).sum())
    good_count = int((labels == GOOD_LABEL).sum())
    total_labels = bad_count + good_count
    if total_labels == 0:
        bad_percentage = good_percentage = 0.0
    else:
        bad_percentage = (bad_count / total_labels) * 100
        good_percentage = (good_count / total_labels) * 100
    imbalance_ratio = (
        max(bad_count, good_count)
        / min(bad_count, good_count)
        if min(bad_count, good_count) > 0
        else float("inf")
    )
    log_section("E. DISTRIBUSI LABEL")
    log_field(f"Jumlah {BAD_LABEL}", bad_count)
    log_field(f"Jumlah {GOOD_LABEL}", good_count)
    log_field(f"Persentase {BAD_LABEL}", f"{bad_percentage:.2f}%")
    log_field(f"Persentase {GOOD_LABEL}", f"{good_percentage:.2f}%")
    log_field("Imbalance Ratio", f"{imbalance_ratio:.2f}")
    LOGGER.info("Unique label: %s", labels.unique().tolist())
    LOGGER.info(
        "Mapping label: %s -> %s, %s -> %s",
        BAD_VALUE,
        BAD_LABEL,
        GOOD_VALUE,
        GOOD_LABEL,
    )
    return good_count, bad_count


def show_dataset_sample(data: DataFrame) -> None:
    """Display the first 10 rows of the dataset."""
    log_section("F. CONTOH DATA")
    LOGGER.info("%s", render_dataframe(data, max_rows=10))


def show_dataset_info(data: DataFrame) -> None:
    """Log pandas DataFrame information."""
    log_section("G. INFO DATASET")
    log_dataframe_info(data)


def show_dataset_statistics(data: DataFrame) -> None:
    """Display descriptive statistics for all columns."""
    log_section("H. STATISTIK DATASET")
    stats = data.describe(include="all").transpose()
    LOGGER.info("%s", stats.to_string())


def show_dataset_summary(
    data: DataFrame,
    total_missing_values: int,
    duplicate_count: int,
    good_count: int,
    bad_count: int,
) -> None:
    """Display a compact summary of the EDA results."""
    numeric_count, categorical_count = get_column_summary(data)
    memory_usage_kb = get_memory_usage_kb(data)
    memory_usage_mb = memory_usage_kb / 1024
    log_section("I. RINGKASAN DATASET")
    log_field("Jumlah file", f"{len(DATASET_FILES)} File")
    log_field("Jumlah data", data.shape[0])
    log_field("Jumlah kolom", data.shape[1])
    log_field("Memory KB", f"{memory_usage_kb:.2f}")
    log_field("Memory MB", f"{memory_usage_mb:.2f}")
    log_field("Kolom numerik", numeric_count)
    log_field("Kolom kategorikal", categorical_count)
    log_field("Missing value", total_missing_values)
    log_field("Duplicate", duplicate_count)
    log_field(GOOD_LABEL, good_count)
    log_field(BAD_LABEL, bad_count)


def show_conclusion(
    total_missing_values: int,
    duplicate_count: int,
    good_count: int,
    bad_count: int,
    data_size: int,
) -> None:
    """Display an automatic conclusion based on EDA results."""
    log_section("J. KESIMPULAN")
    LOGGER.info("Jumlah data: %d", data_size)
    if total_missing_values == 0:
        LOGGER.info("Tidak ditemukan missing value.")
    else:
        LOGGER.info("Ditemukan %d missing value.", total_missing_values)
    if duplicate_count == 0:
        LOGGER.info("Tidak ditemukan duplicate.")
    else:
        LOGGER.info("Ditemukan %d duplicate.", duplicate_count)
    if good_count == bad_count:
        LOGGER.info("Label relatif seimbang.")
    else:
        LOGGER.info(
            "Label tidak seimbang: %d positif, %d negatif.",
            good_count,
            bad_count,
        )
    LOGGER.info(
        "Dataset siap dianalisis lebih lanjut pada tahap preprocessing dan "
        "pemodelan."
    )


def main() -> None:
    """Run the complete EDA workflow."""
    start_time = time.time()
    dataframes = read_dataset_files(DATASET_FILES)
    data = combine_datasets(dataframes)
    show_dataset_information(data)
    show_dataset_structure(data)
    total_missing_values, duplicate_count = show_data_quality(data)
    good_count, bad_count = show_label_distribution(data)
    show_text_length_analysis(data)
    show_dataset_sample(data)
    show_dataset_info(data)
    show_dataset_statistics(data)
    show_dataset_summary(
        data=data,
        total_missing_values=total_missing_values,
        duplicate_count=duplicate_count,
        good_count=good_count,
        bad_count=bad_count,
    )
    show_conclusion(
        total_missing_values=total_missing_values,
        duplicate_count=duplicate_count,
        good_count=good_count,
        bad_count=bad_count,
        data_size=data.shape[0],
    )
    execution_time = time.time() - start_time
    log_section("K. WAKTU EKSEKUSI")
    log_field("Waktu proses EDA", f"{execution_time:.3f} detik")


if __name__ == "__main__":
    main()
