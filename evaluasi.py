"""Evaluate a sentiment classification model with dataset loading,
preprocessing, prediction, and evaluation metrics logging.

This module reads multiple CSV dataset files, validates and cleans data,
preprocesses the test subset, loads a trained model pipeline, and logs
accuracy, precision, recall, F1-score, classification report, and confusion
matrix.
"""

import logging
import time
from pathlib import Path
from typing import Sequence, Tuple, TypeAlias

import joblib  # type: ignore[import]
import pandas as pd  # type: ignore[import]
from pandas import DataFrame, Series  # type: ignore[import]
from sklearn.metrics import (  # type: ignore[import]
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split  # type: ignore[import]
from sklearn.pipeline import Pipeline  # type: ignore[import]

from project_config import (
    MODEL_OUTPUT,
    RANDOM_STATE,
    SENTIMENT_COLUMN,
    TEST_SIZE,
    TEXT_COLUMN,
    discover_dataset_files,
    display_label,
)

EvaluationMetrics: TypeAlias = Tuple[float, float, float, float]

try:
    from pemodelan import (
        preprocess_text,
        normalize_labels,
        resolve_duplicate_texts,
    )
except ImportError as err:
    raise ImportError(
        "Gagal mengimpor dari pemodelan.py. Pastikan file pemodelan.py berada "
        "di direktori yang sama dengan evaluasi.py."
    ) from err

__author__ = "Senior Python Engineer"
__version__ = "1.0.0"
__all__ = ["run_evaluation", "main"]

# Configuration
BASE_DIR = Path(__file__).resolve().parent
DATASET_FILES = discover_dataset_files()
MODEL_FILE = MODEL_OUTPUT
SECTION_WIDTH = 55
FIELD_WIDTH = 20
LOG_DATE_FORMAT = "%H:%M:%S"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
TARGET_NAMES = [display_label("negative"), display_label("positive")]

logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
    level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)

# Helper functions


def print_section(title: str) -> None:
    """Log a consistent section title separator.

    Args:
        title: Section title.
    """
    separator = "=" * SECTION_WIDTH
    LOGGER.info("%s", separator)
    LOGGER.info("%s", title)
    LOGGER.info("%s", separator)


def print_field(label: str, value: object) -> None:
    """Log a key-value line with consistent alignment.

    Args:
        label: Field label.
        value: Field value.
    """
    LOGGER.info("%-*s: %s", FIELD_WIDTH, label, value)


def load_dataset_files(files: Sequence[Path]) -> DataFrame:
    """Read and combine the dataset files.

    Args:
        files: List of dataset file paths.

    Returns:
        Combined dataset dataframe.

    Raises:
        FileNotFoundError: If any dataset file is missing.
        ValueError: If any dataset file is empty.
    """
    dataframes: list[DataFrame] = []

    for file_path in files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"File dataset tidak ditemukan: {file_path.name}"
            )

        dataframe = pd.read_csv(file_path)
        if dataframe.empty:
            raise ValueError(f"Dataset kosong: {file_path.name}")

        dataframes.append(dataframe)

    return pd.concat(dataframes, ignore_index=True)


def validate_dataset_columns(data: pd.DataFrame) -> None:
    """Validate dataset columns required for evaluation.

    Args:
        data: Combined dataset dataframe.

    Raises:
        ValueError: If required columns are missing.
    """
    missing_columns = []
    if TEXT_COLUMN not in data.columns:
        missing_columns.append(TEXT_COLUMN)
    if SENTIMENT_COLUMN not in data.columns:
        missing_columns.append(SENTIMENT_COLUMN)

    if missing_columns:
        raise ValueError(
            f"Kolom wajib {missing_columns} tidak ditemukan pada dataset."
        )


def remove_empty_required_data(data: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with missing or empty required values.

    Args:
        data: Dataset dataframe.

    Returns:
        Cleaned dataset dataframe.

    Raises:
        ValueError: If the cleaned dataset is empty.
    """
    cleaned_data = data.dropna(subset=[TEXT_COLUMN, SENTIMENT_COLUMN]).copy()

    text_not_empty = (
        cleaned_data[TEXT_COLUMN].astype(str).str.strip() != ""
    )
    sentiment_not_empty = (
        cleaned_data[SENTIMENT_COLUMN].astype(str).str.strip() != ""
    )
    cleaned_data = cleaned_data[text_not_empty & sentiment_not_empty].copy()

    if cleaned_data.empty:
        raise ValueError("Dataset kosong setelah pembersihan data wajib.")

    return cleaned_data


def split_data(data: DataFrame) -> tuple[Series, Series, Series, Series]:
    """Split dataset into training and testing subsets.

    Args:
        data: Clean dataset.

    Returns:
        Tuple containing X_train, X_test, y_train, y_test.
    """
    features = data[TEXT_COLUMN]
    labels = data[SENTIMENT_COLUMN]

    return train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )


def preprocess_test_data(x_test: pd.Series, y_test: pd.Series) -> pd.Series:
    """Preprocess test data using the shared preprocessing function.

    Args:
        x_test: Test features.
        y_test: Test labels.

    Returns:
        Preprocessed test features.
    """
    test_dataframe = pd.DataFrame(
        {TEXT_COLUMN: x_test, SENTIMENT_COLUMN: y_test}
    )
    preprocessed_df = preprocess_text(test_dataframe)
    return preprocessed_df[TEXT_COLUMN]


def evaluate_predictions(y_true: Series, y_pred: Series) -> EvaluationMetrics:
    """Compute evaluation metrics for classification predictions.

    Args:
        y_true: Actual labels.
        y_pred: Predicted labels.

    Returns:
        Tuple containing accuracy, precision, recall, and F1 score.
    """
    return (
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, zero_division=0),
        recall_score(y_true, y_pred, zero_division=0),
        f1_score(y_true, y_pred, zero_division=0),
    )


def log_classification_report(y_true: pd.Series, y_pred: pd.Series) -> None:
    """Log the classification report and confusion matrix.

    Args:
        y_true: Actual labels.
        y_pred: Predicted labels.
    """
    LOGGER.info("")
    LOGGER.info("Classification Report")
    LOGGER.info("")

    report = classification_report(
        y_true,
        y_pred,
        target_names=TARGET_NAMES,
        zero_division=0,
    )
    for line in report.splitlines():
        LOGGER.info(line)

    LOGGER.info("")
    LOGGER.info("Confusion Matrix")
    LOGGER.info("")

    matrix = confusion_matrix(y_true, y_pred)
    for line in str(matrix).splitlines():
        LOGGER.info(line)


def load_model(model_path: Path) -> Pipeline:
    """Load the trained model from disk.

    Args:
        model_path: Path to the model file.

    Returns:
        Trained model pipeline.

    Raises:
        FileNotFoundError: If the model file is missing.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"File model tidak ditemukan: {model_path.name}. "
            "Harap jalankan pemodelan.py terlebih dahulu untuk melatih model."
        )

    return joblib.load(model_path)


def run_evaluation() -> None:
    """Execute the dataset loading, preprocessing, and model evaluation.

    This function orchestrates dataset loading, cleaning, train-test splitting,
    preprocessing of test examples, model loading, prediction, and result
    logging.
    """
    print_section("A. MEMBACA DATASET")
    data = load_dataset_files(DATASET_FILES)
    validate_dataset_columns(data)
    print_field("Jumlah file", len(DATASET_FILES))
    print_field("Jumlah data", len(data))

    print_section("B. MEMBERSIHKAN DATA")
    data = remove_empty_required_data(data)
    data = normalize_labels(data)
    data = resolve_duplicate_texts(data)
    print_field("Jumlah data bersih", len(data))

    x_train, x_test, _, y_test = split_data(data)

    print_section("C. PREPROCESSING")
    LOGGER.info("Preprocessing dimulai")
    preprocessing_start_time = time.perf_counter()
    x_test_preprocessed = preprocess_test_data(x_test, y_test)
    preprocessing_time = time.perf_counter() - preprocessing_start_time
    LOGGER.info("Preprocessing selesai")
    print_field("Waktu preprocessing", f"{preprocessing_time:.3f} detik")

    print_section("D. TRAIN TEST SPLIT")
    print_field("Jumlah data training", len(x_train))
    print_field("Jumlah data testing", len(x_test))

    print_section("E. LOAD MODEL")
    model = load_model(MODEL_FILE)
    LOGGER.info("Model berhasil dimuat")
    print_field("File model", MODEL_FILE.name)

    print_section("F. PREDIKSI")
    LOGGER.info("Prediksi dimulai")
    y_pred = model.predict(x_test_preprocessed)
    LOGGER.info("Prediksi selesai")

    print_section("G. EVALUASI MODEL")
    accuracy, precision, recall, f1 = evaluate_predictions(y_test, y_pred)
    print_field("Accuracy", f"{accuracy * 100:.2f}%")
    print_field("Precision", f"{precision * 100:.2f}%")
    print_field("Recall", f"{recall * 100:.2f}%")
    print_field("F1 Score", f"{f1 * 100:.2f}%")

    log_classification_report(y_test, y_pred)


def main() -> None:
    """Entry point for evaluation workflow."""
    start_time = time.perf_counter()

    try:
        run_evaluation()
        execution_time = time.perf_counter() - start_time
        print_section("H. WAKTU EKSEKUSI")
        print_field("Lama proses", f"{execution_time:.3f} detik")
    except FileNotFoundError as fnf_err:
        LOGGER.error("File tidak ditemukan: %s", fnf_err)
    except ValueError as val_err:
        LOGGER.error("Kesalahan nilai/data: %s", val_err)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.error("Terjadi kesalahan tidak terduga: %s", exc)


if __name__ == "__main__":
    main()
