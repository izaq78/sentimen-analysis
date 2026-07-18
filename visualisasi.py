"""Visualisasi dataset dan hasil evaluasi model untuk proyek Analisis Sentimen.

File ini hanya membaca dataset, model siap-pakai, dan hasil evaluasi (jika perlu),
kemudian menghasilkan beberapa grafik yang disimpan ke folder ``hasil_visualisasi/``.

Standar: PEP8, Google style docstrings, logging, pathlib, type hints.
"""
from __future__ import annotations

import contextlib
import logging
import pickle
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol, TypeAlias, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import load
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split
from sklearn.metrics import (  # type: ignore[import-untyped]
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from project_config import (
    BASE_DIR,
    MODEL_OUTPUT,
    RANDOM_STATE,
    RESOURCES_DIR,
    SENTIMENT_COLUMN,
    TEST_SIZE,
    TEXT_COLUMN,
    discover_dataset_files,
    display_label,
    load_json_file,
    sentiment_value,
)
from pemodelan import (
    iter_cleaned_texts,
    normalize_labels,
    preprocess_text,
    resolve_duplicate_texts,
)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
DatasetFrame: TypeAlias = pd.DataFrame
FigureSize: TypeAlias = tuple[float, float]
LabelSequence: TypeAlias = Iterable[int]
MetricValues: TypeAlias = dict[str, float]
Predictions: TypeAlias = np.ndarray | list[Any]
WordCountSeries: TypeAlias = pd.Series

# ---------------------------------------------------------------------------
# Path and dataset constants
# ---------------------------------------------------------------------------
DATASET_FILES = discover_dataset_files()
REQUIRED_COLUMNS: tuple[str, ...] = (TEXT_COLUMN, SENTIMENT_COLUMN)
VISUALIZATION_CONFIG = load_json_file(RESOURCES_DIR / "visualization_config.json")
OUTPUT_FILES = VISUALIZATION_CONFIG["output_files"]
FIGURE_CONFIG = VISUALIZATION_CONFIG["figure"]
PALETTE_CONFIG = VISUALIZATION_CONFIG["palette"]
METRICS_CONFIG = VISUALIZATION_CONFIG["metrics"]
OUTPUT_DIR = BASE_DIR / str(VISUALIZATION_CONFIG["output_directory"])
MODEL_PATH = MODEL_OUTPUT

# ---------------------------------------------------------------------------
# Output filename constants
# ---------------------------------------------------------------------------
OUTPUT_LABEL_DISTRIBUTION = str(OUTPUT_FILES["label_distribution"])
OUTPUT_REVIEW_LENGTH_HISTOGRAM = str(OUTPUT_FILES["review_length_histogram"])
OUTPUT_REVIEW_LENGTH_BOXPLOT = str(OUTPUT_FILES["review_length_boxplot"])
OUTPUT_TOP_WORDS = str(OUTPUT_FILES["top_words"])
OUTPUT_CONFUSION_MATRIX = str(OUTPUT_FILES["confusion_matrix"])
OUTPUT_MODEL_METRICS = str(OUTPUT_FILES["model_metrics"])

# ---------------------------------------------------------------------------
# Figure and plotting constants
# ---------------------------------------------------------------------------
FIG_SIZE: FigureSize = tuple(FIGURE_CONFIG["default_size"])
BOXPLOT_FIG_SIZE: FigureSize = tuple(FIGURE_CONFIG["boxplot_size"])
TOP_WORDS_FIG_SIZE: FigureSize = tuple(FIGURE_CONFIG["top_words_size"])
FIGURE_DPI = int(FIGURE_CONFIG["dpi"])
BBOX_INCHES = str(FIGURE_CONFIG["bbox_inches"])
HISTOGRAM_BINS = int(VISUALIZATION_CONFIG["histogram"]["bins"])
TOP_N_WORDS = int(VISUALIZATION_CONFIG["top_words"]["count"])
BAR_LABEL_OFFSET_RATIO = float(VISUALIZATION_CONFIG["bar_labels"]["offset_ratio"])
METRIC_X_MIN = float(METRICS_CONFIG["x_min"])
METRIC_X_MAX = float(METRICS_CONFIG["x_max"])
METRIC_TEXT_OFFSET = float(METRICS_CONFIG["text_offset"])
METRIC_DECIMAL_PLACES = int(METRICS_CONFIG["decimal_places"])
SKLEARN_ZERO_DIVISION = int(METRICS_CONFIG["zero_division"])
SEABORN_THEME = str(VISUALIZATION_CONFIG["theme"]["seaborn"])

# ---------------------------------------------------------------------------
# Color and palette constants
# ---------------------------------------------------------------------------
COLOR_HISTOGRAM = str(PALETTE_CONFIG["histogram"])
COLOR_BOXPLOT = str(PALETTE_CONFIG["boxplot"])
PALETTE_LABEL_DISTRIBUTION = list(PALETTE_CONFIG["label_distribution"])
PALETTE_TOP_WORDS = str(PALETTE_CONFIG["top_words"])
PALETTE_MODEL_METRICS = str(PALETTE_CONFIG["model_metrics"])
CONFUSION_MATRIX_CMAP = str(PALETTE_CONFIG["confusion_matrix"])
CONFUSION_MATRIX_VALUES_FORMAT = str(PALETTE_CONFIG["confusion_matrix_values_format"])

# ---------------------------------------------------------------------------
# Label and title constants
# ---------------------------------------------------------------------------
SENTIMENT_LABELS: dict[int, str] = {
    sentiment_value("negative"): display_label("negative"),
    sentiment_value("positive"): display_label("positive"),
}
LABEL_ORDER: tuple[str, ...] = (
    display_label("positive"),
    display_label("negative"),
)
UNKNOWN_LABEL = "Unknown"
CONFUSION_MATRIX_DISPLAY_LABELS: tuple[str, ...] = (
    display_label("negative"),
    display_label("positive"),
)

TITLE_LABEL_DISTRIBUTION = "Distribusi Label"
TITLE_REVIEW_LENGTH_HISTOGRAM = "Histogram Panjang Review (Jumlah Kata)"
TITLE_REVIEW_LENGTH_BOXPLOT = "Boxplot Panjang Review (Jumlah Kata)"
TITLE_TOP_WORDS_TEMPLATE = "Top {count} Kata Paling Sering"
TITLE_CONFUSION_MATRIX = "Confusion Matrix"
TITLE_MODEL_PERFORMANCE = "Performa Model"

AXIS_LABEL_COUNT = "Jumlah"
AXIS_LABEL_WORD_COUNT = "Jumlah kata"
AXIS_LABEL_FREQUENCY = "Frekuensi"
AXIS_LABEL_WORD_FREQUENCY = "Frekuensi"

METRIC_NAME_ACCURACY = "Accuracy"
METRIC_NAME_PRECISION = "Precision"
METRIC_NAME_RECALL = "Recall"
METRIC_NAME_F1 = "F1"
METRIC_NAMES: tuple[str, ...] = (
    METRIC_NAME_ACCURACY,
    METRIC_NAME_PRECISION,
    METRIC_NAME_RECALL,
    METRIC_NAME_F1,
)

# ---------------------------------------------------------------------------
# Logging constants
# ---------------------------------------------------------------------------
SECTION_WIDTH = 50
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"
SECTION_READ_DATASET = "A. MEMBACA DATASET"
SECTION_VISUALIZATION_PREFIX = "B. MEMBUAT VISUALISASI"
SECTION_LABEL_DISTRIBUTION = f"{SECTION_VISUALIZATION_PREFIX} - Distribusi Label"
SECTION_REVIEW_LENGTH_HISTOGRAM = (
    f"{SECTION_VISUALIZATION_PREFIX} - Histogram Panjang Review"
)
SECTION_REVIEW_LENGTH_BOXPLOT = (
    f"{SECTION_VISUALIZATION_PREFIX} - Boxplot Panjang Review"
)
SECTION_TOP_WORDS = f"{SECTION_VISUALIZATION_PREFIX} - Top Words"
SECTION_CONFUSION_MATRIX = f"{SECTION_VISUALIZATION_PREFIX} - Confusion Matrix"
SECTION_MODEL_METRICS = f"{SECTION_VISUALIZATION_PREFIX} - Model Metrics"

LOGGER = logging.getLogger(__name__)


class Predictor(Protocol):
    """Protocol for objects that expose a scikit-learn style predict method."""

    def predict(self, samples: Sequence[str]) -> Predictions:
        """Return predictions for the given samples."""
        ...


def configure_logging(level: int = logging.INFO) -> None:
    """Configure logging for this module.

    Applies ``basicConfig`` only when the root logger has no handlers yet,
    preventing duplicate log lines on repeated imports or calls.

    Args:
        level: Logging level applied to the root and module logger.
    """
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            format=LOG_FORMAT,
            datefmt=LOG_DATE_FORMAT,
            level=level,
        )
    else:
        root_logger.setLevel(level)
    LOGGER.setLevel(level)
    LOGGER.debug("Logging dikonfigurasi dengan level %s", logging.getLevelName(level))


def setup_visualization_style() -> None:
    """Apply the global Seaborn theme used by all visualizations.

    Configures Matplotlib/Seaborn styling once before any figure is created.
    """
    sns.set_theme(style=SEABORN_THEME)
    LOGGER.debug("Seaborn theme diterapkan: %s", SEABORN_THEME)


def log_section(title: str) -> None:
    """Log a section header with consistent separators.

    Args:
        title: Section title to display between separator lines.
    """
    separator = "=" * SECTION_WIDTH
    LOGGER.info("%s", separator)
    LOGGER.info("%s", title)
    LOGGER.info("%s", separator)


def read_dataset(files: Iterable[Path]) -> DatasetFrame:
    """Read and combine dataset CSV files into a single DataFrame.

    Args:
        files: Iterable of Path objects pointing to CSV files.

    Returns:
        Combined pandas DataFrame.

    Raises:
        FileNotFoundError: If any file is missing.
        ValueError: If no files are provided or combined DataFrame is empty.
    """
    log_section(SECTION_READ_DATASET)

    dataframes: list[DatasetFrame] = []
    for file_path in files:
        if not file_path.exists():
            LOGGER.error("File dataset tidak ditemukan: %s", file_path)
            raise FileNotFoundError(f"File dataset tidak ditemukan: {file_path}")
        LOGGER.info("Membaca: %s", file_path.name)
        frame = pd.read_csv(file_path)
        LOGGER.debug(
            "File %s berisi %d baris dan %d kolom",
            file_path.name,
            frame.shape[0],
            frame.shape[1],
        )
        dataframes.append(frame)

    if not dataframes:
        raise ValueError("Tidak ada file dataset untuk dibaca.")

    combined = pd.concat(dataframes, ignore_index=True)
    if combined.empty:
        raise ValueError("Dataset gabungan kosong.")

    LOGGER.info("Total baris dataset gabungan: %d", combined.shape[0])
    LOGGER.debug("Kolom dataset gabungan: %s", list(combined.columns))
    return combined


def _validate_required_columns(data: DatasetFrame) -> None:
    """Ensure the dataset contains all required columns.

    Args:
        data: Dataset to validate.

    Raises:
        ValueError: If required columns are missing.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        LOGGER.error("Kolom yang dibutuhkan tidak ada: %s", missing)
        raise ValueError(f"Kolom yang dibutuhkan tidak ada: {missing}")
    LOGGER.debug("Semua kolom wajib tersedia: %s", REQUIRED_COLUMNS)


def _validate_dataframe_not_empty(data: DatasetFrame) -> None:
    """Ensure the dataset contains at least one row.

    Args:
        data: Dataset to validate.

    Raises:
        ValueError: If the DataFrame has no rows.
    """
    if data.empty:
        LOGGER.error("Dataset kosong setelah digabungkan")
        raise ValueError("Dataset kosong.")
    LOGGER.debug("Dataset memiliki %d baris", data.shape[0])


def _validate_no_nan_values(data: DatasetFrame) -> None:
    """Ensure required columns do not contain NaN values.

    Args:
        data: Dataset to validate.

    Raises:
        ValueError: If NaN values are found in required columns.
    """
    nan_counts = {
        TEXT_COLUMN: int(data[TEXT_COLUMN].isna().sum()),
        SENTIMENT_COLUMN: int(data[SENTIMENT_COLUMN].isna().sum()),
    }
    LOGGER.debug("Jumlah nilai NaN per kolom: %s", nan_counts)

    invalid_columns = [
        column for column, count in nan_counts.items() if count > 0
    ]
    if invalid_columns:
        details = ", ".join(
            f"{column}={nan_counts[column]}" for column in invalid_columns
        )
        LOGGER.error("Ditemukan nilai NaN pada kolom wajib: %s", details)
        raise ValueError(f"Ditemukan nilai NaN pada kolom wajib: {details}")


def _validate_no_empty_strings(data: DatasetFrame, column: str) -> None:
    """Ensure the specified column does not contain blank values.

    Args:
        data: Dataset to validate.
        column: Column name to check.

    Raises:
        ValueError: If blank values are found.
    """
    empty_mask = data[column].astype(str).str.strip().eq("")
    empty_count = int(empty_mask.sum())
    LOGGER.debug("Jumlah %s kosong: %d", column, empty_count)

    if empty_count > 0:
        LOGGER.error("Ditemukan %d baris dengan %s kosong", empty_count, column)
        raise ValueError(f"Ditemukan {empty_count} baris dengan {column} kosong")


def _validate_duplicate_rows(data: DatasetFrame) -> None:
    """Log duplicate rows without altering the dataset.

    Duplicate rows are reported as a warning so the workflow can continue.

    Args:
        data: Dataset to inspect.
    """
    duplicate_count = int(data.duplicated().sum())
    LOGGER.debug("Jumlah baris duplikat: %d", duplicate_count)
    if duplicate_count > 0:
        LOGGER.warning(
            "Ditemukan %d baris duplikat; visualisasi tetap dilanjutkan",
            duplicate_count,
        )


def validate_dataset(data: DatasetFrame) -> None:
    """Validate dataset structure and content quality.

    Performs column, emptiness, NaN, blank-value, and duplicate checks.
    Duplicate rows trigger a warning; all other failures raise ``ValueError``.

    Args:
        data: DataFrame to validate.

    Raises:
        ValueError: If the dataset fails structural or content validation.
    """
    LOGGER.debug("Memulai validasi dataset")
    _validate_required_columns(data)
    _validate_dataframe_not_empty(data)
    _validate_no_nan_values(data)
    _validate_no_empty_strings(data, TEXT_COLUMN)
    _validate_no_empty_strings(data, SENTIMENT_COLUMN)
    _validate_duplicate_rows(data)
    LOGGER.debug("Validasi dataset selesai tanpa error")


def create_output_directory(path: Path) -> None:
    """Create output directory if it does not exist.

    Args:
        path: Path to create.

    Raises:
        OSError: If directory cannot be created.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        LOGGER.debug("Folder output siap: %s", path)
    except OSError:
        LOGGER.exception("Gagal membuat folder output: %s", path)
        raise


def save_figure(fig: Figure, path: Path) -> None:
    """Save a Matplotlib figure to disk with error handling.

    Args:
        fig: Matplotlib Figure object.
        path: Destination Path including filename.

    Raises:
        OSError: If saving the figure fails.
        ValueError: If Matplotlib rejects the save parameters.
        RuntimeError: If the figure backend cannot render the output.
    """
    try:
        fig.tight_layout()
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches=BBOX_INCHES)
        LOGGER.info("Gambar tersimpan: %s", path)
        LOGGER.debug("Figure disimpan dengan dpi=%d", FIGURE_DPI)
    except (OSError, ValueError, RuntimeError):
        LOGGER.exception("Gagal menyimpan gambar: %s", path)
        raise


@contextlib.contextmanager
def managed_figure(figsize: FigureSize):
    """Context manager for safely creating and closing Matplotlib figures.

    Args:
        figsize: Figure dimensions in inches.

    Yields:
        Tuple of figure and axes objects.
    """
    fig, ax = plt.subplots(figsize=figsize)
    LOGGER.debug("Figure dibuat dengan ukuran %s", figsize)
    try:
        yield fig, ax
    finally:
        plt.close(fig)


def compute_word_counts(data: DatasetFrame) -> WordCountSeries:
    """Compute per-review word counts from the text column.

    Args:
        data: Dataset containing ``TEXT_COLUMN``.

    Returns:
        Series of word counts for each review.
    """
    if data is None or data.empty or TEXT_COLUMN not in data.columns:
        return pd.Series(dtype=int)

    word_counts = data[TEXT_COLUMN].astype(str).str.split().str.len()
    word_counts = word_counts.fillna(0).astype(int)

    if word_counts.empty:
        return word_counts

    LOGGER.debug(
        "Statistik panjang teks: min=%d, max=%d, mean=%.2f",
        int(word_counts.min()),
        int(word_counts.max()),
        float(word_counts.mean()),
    )
    return word_counts


def add_bar_value_labels(
    ax: Axes, values: Sequence[float] | np.ndarray | pd.Series | None
) -> None:
    """Add centered count labels above bars in a bar chart.

    Args:
        ax: Matplotlib axes containing the bar plot.
        values: Bar heights used to position labels.
    """
    if values is None or len(values) == 0:
        return

    try:
        vals_list = list(values)
    except TypeError:
        return

    if not vals_list:
        return

    offset = max(vals_list) * BAR_LABEL_OFFSET_RATIO
    for index, value in enumerate(vals_list):
        ax.text(index, value + offset, str(int(value)), ha="center")


def collect_tokens(texts: Iterable[str] | None) -> list[str]:
    """Flatten preprocessed texts into a token list.

    Args:
        texts: Iterable of preprocessed text strings.

    Returns:
        List of tokens extracted from all texts.
    """
    tokens: list[str] = []
    if not texts:
        return tokens

    for text in texts:
        if text:
            tokens.extend(text.split())
    LOGGER.debug("Total token terkumpul: %d", len(tokens))
    return tokens


def compute_metric_values(
    y_true: LabelSequence | None,
    y_pred: LabelSequence | None,
) -> MetricValues:
    """Compute classification metrics for model evaluation plots.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        Mapping of metric names to computed scores.
    """
    if y_true is None or y_pred is None:
        LOGGER.warning("y_true atau y_pred adalah None.")
        return {m: 0.0 for m in METRIC_NAMES}

    try:
        y_true_list = list(y_true)
        y_pred_list = list(y_pred)
    except TypeError:
        LOGGER.warning("y_true atau y_pred tidak dapat diiterasi.")
        return {m: 0.0 for m in METRIC_NAMES}

    if not y_true_list or not y_pred_list or len(y_true_list) != len(y_pred_list):
        LOGGER.warning("Label kosong atau panjang tidak sama.")
        return {m: 0.0 for m in METRIC_NAMES}

    metrics: MetricValues = {
        METRIC_NAME_ACCURACY: accuracy_score(y_true_list, y_pred_list),
        METRIC_NAME_PRECISION: precision_score(
            y_true_list,
            y_pred_list,
            zero_division=SKLEARN_ZERO_DIVISION,
        ),
        METRIC_NAME_RECALL: recall_score(
            y_true_list,
            y_pred_list,
            zero_division=SKLEARN_ZERO_DIVISION,
        ),
        METRIC_NAME_F1: f1_score(
            y_true_list,
            y_pred_list,
            zero_division=SKLEARN_ZERO_DIVISION,
        ),
    }
    LOGGER.debug("Nilai metrik model: %s", metrics)
    return metrics


def add_metric_value_labels(
    ax: Axes, values: Sequence[float] | np.ndarray | pd.Series | None
) -> None:
    """Add formatted score labels beside horizontal metric bars.

    Args:
        ax: Matplotlib axes containing the metric bar plot.
        values: Metric scores displayed on the chart.
    """
    if values is None or len(values) == 0:
        return

    try:
        vals_list = list(values)
    except TypeError:
        return

    for index, value in enumerate(vals_list):
        ax.text(
            value + METRIC_TEXT_OFFSET,
            index,
            f"{float(value):.{METRIC_DECIMAL_PLACES}f}",
        )


def plot_label_distribution(data: DatasetFrame | None, output_dir: Path) -> None:
    """Plot and save label distribution (positif vs negatif) as a bar chart.

    Args:
        data: Dataset containing ``SENTIMENT_COLUMN``.
        output_dir: Directory where the plot will be saved.
    """
    if data is None or data.empty or SENTIMENT_COLUMN not in data.columns:
        LOGGER.warning("Data tidak valid untuk plot_label_distribution.")
        return

    log_section(SECTION_LABEL_DISTRIBUTION)

    labels = (
        data[SENTIMENT_COLUMN]
        .map(SENTIMENT_LABELS)
        .fillna(UNKNOWN_LABEL)
    )
    counts = labels.value_counts().reindex(LABEL_ORDER, fill_value=0)
    LOGGER.debug("Distribusi label: %s", counts.to_dict())

    with managed_figure(FIG_SIZE) as (fig, ax):
        label_plot_data = pd.DataFrame({
            "label": counts.index.tolist(),
            "count": counts.values.tolist(),
        })
        sns.barplot(
            data=label_plot_data,
            x="label",
            y="count",
            hue="label",
            palette=PALETTE_LABEL_DISTRIBUTION,
            legend=False,
            ax=ax,
        )
        ax.set_title(TITLE_LABEL_DISTRIBUTION)
        ax.set_ylabel(AXIS_LABEL_COUNT)
        add_bar_value_labels(ax, counts.values)

        save_figure(fig, output_dir / OUTPUT_LABEL_DISTRIBUTION)


def plot_review_length_histogram(data: DatasetFrame | None, output_dir: Path) -> None:
    """Plot histogram of review lengths (word count).

    Args:
        data: Dataset containing ``TEXT_COLUMN``.
        output_dir: Destination directory.
    """
    if data is None or data.empty or TEXT_COLUMN not in data.columns:
        LOGGER.warning("Data tidak valid untuk plot_review_length_histogram.")
        return

    log_section(SECTION_REVIEW_LENGTH_HISTOGRAM)

    word_counts = compute_word_counts(data)
    with managed_figure(FIG_SIZE) as (fig, ax):
        sns.histplot(
            word_counts,
            bins=HISTOGRAM_BINS,
            kde=False,
            ax=ax,
            color=COLOR_HISTOGRAM,
        )
        ax.set_title(TITLE_REVIEW_LENGTH_HISTOGRAM)
        ax.set_xlabel(AXIS_LABEL_WORD_COUNT)
        ax.set_ylabel(AXIS_LABEL_FREQUENCY)

        save_figure(fig, output_dir / OUTPUT_REVIEW_LENGTH_HISTOGRAM)


def plot_review_length_boxplot(data: DatasetFrame | None, output_dir: Path) -> None:
    """Plot boxplot of review lengths to inspect outliers.

    Args:
        data: Dataset containing ``TEXT_COLUMN``.
        output_dir: Destination directory.
    """
    if data is None or data.empty or TEXT_COLUMN not in data.columns:
        LOGGER.warning("Data tidak valid untuk plot_review_length_boxplot.")
        return

    log_section(SECTION_REVIEW_LENGTH_BOXPLOT)

    word_counts = compute_word_counts(data)
    with managed_figure(BOXPLOT_FIG_SIZE) as (fig, ax):
        sns.boxplot(x=word_counts, ax=ax, color=COLOR_BOXPLOT)
        ax.set_title(TITLE_REVIEW_LENGTH_BOXPLOT)
        ax.set_xlabel(AXIS_LABEL_WORD_COUNT)

        save_figure(fig, output_dir / OUTPUT_REVIEW_LENGTH_BOXPLOT)


def plot_top_words(data: DatasetFrame | None, output_dir: Path) -> None:
    """Plot top N most frequent words based on the dataset texts.

    Uses the same preprocessing pipeline as ``pemodelan.py`` to keep
    tokenization and normalization consistent with model training.

    Args:
        data: Dataset containing ``TEXT_COLUMN``.
        output_dir: Destination directory.
    """
    if data is None or data.empty or TEXT_COLUMN not in data.columns:
        LOGGER.warning("Data tidak valid untuk plot_top_words.")
        return

    log_section(SECTION_TOP_WORDS)

    preprocessed_texts = list(iter_cleaned_texts(data[TEXT_COLUMN]))
    tokens = collect_tokens(preprocessed_texts)
    if not tokens:
        LOGGER.warning("Tidak ada token untuk divisualisasikan.")
        return

    most_common = Counter(tokens).most_common(TOP_N_WORDS)
    words, counts = zip(*most_common) if most_common else ([], [])
    LOGGER.debug("Top words: %s", most_common[:5])

    with managed_figure(TOP_WORDS_FIG_SIZE) as (fig, ax):
        words_plot_data = pd.DataFrame({
            "word": list(words),
            "count": list(counts),
        })
        sns.barplot(
            data=words_plot_data,
            x="count",
            y="word",
            hue="word",
            palette=PALETTE_TOP_WORDS,
            legend=False,
            ax=ax,
        )
        ax.set_title(TITLE_TOP_WORDS_TEMPLATE.format(count=TOP_N_WORDS))
        ax.set_xlabel(AXIS_LABEL_WORD_FREQUENCY)
        ax.set_ylabel("Kata")

        save_figure(fig, output_dir / OUTPUT_TOP_WORDS)


def load_model(path: Path) -> Any:
    """Load a serialized model using joblib.

    Args:
        path: Path to the serialized model file.

    Returns:
        Deserialized model object.

    Raises:
        FileNotFoundError: If model file does not exist.
        pickle.UnpicklingError: If the file cannot be unpickled.
        EOFError: If the file is truncated or empty.
        ValueError: If joblib rejects the file contents.
        TypeError: If the deserialized object has an unexpected structure.
    """
    if not path.exists():
        LOGGER.error("Model tidak ditemukan: %s", path)
        raise FileNotFoundError(f"Model tidak ditemukan: {path}")

    LOGGER.debug("Memuat model dari %s", path)
    try:
        model = load(path)
    except (
        EOFError,
        pickle.UnpicklingError,
        ValueError,
        TypeError,
    ):
        LOGGER.exception("Gagal memuat model dari %s", path)
        raise

    LOGGER.info("Model berhasil dimuat dari %s", path)
    return model


def predict_sentiments(model: Any, data: DatasetFrame | None) -> Predictions | None:
    """Generate sentiment predictions from a trained model.

    Args:
        model: Loaded model exposing a ``predict`` method.
        data: Dataset containing ``TEXT_COLUMN``.

    Returns:
        Model predictions, or ``None`` when prediction is unavailable.
    """
    if data is None or data.empty or TEXT_COLUMN not in data.columns:
        LOGGER.warning("Data tidak valid untuk prediksi.")
        return None

    if not callable(getattr(model, "predict", None)):
        LOGGER.info("Model tidak mendukung metode 'predict'; melewati prediksi.")
        return None

    predictor = cast(Predictor, model)
    LOGGER.info("Mulai preprocessing data untuk prediksi (%d baris)", data.shape[0])
    try:
        processed = preprocess_text(data.copy())
        LOGGER.info("Preprocessing selesai")
    except (ValueError, TypeError, KeyError, AttributeError) as e:
        LOGGER.exception("Preprocessing gagal: %s", e)
        return None

    texts = processed[TEXT_COLUMN].astype(str).tolist()

    LOGGER.info("Mulai prediksi")
    try:
        predictions = predictor.predict(texts)
        LOGGER.info("Prediksi selesai untuk %d sampel", len(texts))
        return predictions
    except (ValueError, TypeError, RuntimeError) as e:
        LOGGER.exception("Prediksi gagal: %s", e)
        return None


def plot_confusion_matrix_from_predictions(
    y_true: LabelSequence | None,
    y_pred: LabelSequence | None,
    output_dir: Path,
) -> None:
    """Plot confusion matrix using sklearn's ConfusionMatrixDisplay.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        output_dir: Destination directory.
    """
    if y_true is None or y_pred is None:
        LOGGER.warning("Data label tidak valid.")
        return

    try:
        y_true_list = list(y_true)
        y_pred_list = list(y_pred)
    except TypeError:
        LOGGER.warning("Label tidak dapat diiterasi.")
        return

    if not y_true_list or not y_pred_list:
        LOGGER.warning("Label kosong.")
        return

    log_section(SECTION_CONFUSION_MATRIX)

    matrix = confusion_matrix(y_true_list, y_pred_list)
    LOGGER.debug("Confusion matrix:\n%s", matrix)
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=list(CONFUSION_MATRIX_DISPLAY_LABELS),
    )
    with managed_figure(FIG_SIZE) as (fig, ax):
        display.plot(
            ax=ax,
            cmap=CONFUSION_MATRIX_CMAP,
            values_format=CONFUSION_MATRIX_VALUES_FORMAT,
        )
        ax.set_title(TITLE_CONFUSION_MATRIX)

        save_figure(fig, output_dir / OUTPUT_CONFUSION_MATRIX)


def plot_model_metrics(
    y_true: LabelSequence,
    y_pred: LabelSequence,
    output_dir: Path,
) -> None:
    """Plot model performance metrics: Accuracy, Precision, Recall, F1.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        output_dir: Destination directory.
    """
    log_section(SECTION_MODEL_METRICS)

    metrics = compute_metric_values(y_true, y_pred)
    names = list(METRIC_NAMES)
    values = [metrics[name] for name in names]

    with managed_figure(FIG_SIZE) as (fig, ax):
        metrics_plot_data = pd.DataFrame({
            "metric": names,
            "value": values,
        })
        sns.barplot(
            data=metrics_plot_data,
            x="value",
            y="metric",
            hue="metric",
            palette=PALETTE_MODEL_METRICS,
            legend=False,
            ax=ax,
        )
        ax.set_xlim(METRIC_X_MIN, METRIC_X_MAX)
        ax.set_title(TITLE_MODEL_PERFORMANCE)
        ax.set_ylabel("")
        add_metric_value_labels(ax, values)

        save_figure(fig, output_dir / OUTPUT_MODEL_METRICS)


def generate_dataset_visualizations(
    data: DatasetFrame | None,
    output_dir: Path,
) -> None:
    """Generate all visualizations that depend only on the dataset.

    Args:
        data: Validated dataset.
        output_dir: Directory for saved figures.

    Raises:
        OSError: If a figure cannot be saved.
        ValueError: If plotting fails due to invalid data.
        RuntimeError: If Matplotlib cannot render a figure.
    """
    if data is None or data.empty:
        LOGGER.warning("Dataset kosong, membatalkan visualisasi berbasis dataset.")
        return

    LOGGER.debug("Membuat visualisasi berbasis dataset")
    plot_label_distribution(data, output_dir)
    plot_review_length_histogram(data, output_dir)
    plot_review_length_boxplot(data, output_dir)
    plot_top_words(data, output_dir)


def generate_prediction_visualizations(
    data: DatasetFrame | None,
    output_dir: Path,
) -> None:
    """Generate confusion matrix and metric plots from model predictions.

    Args:
        data: Validated dataset with ground-truth labels.
        output_dir: Directory for saved figures.
    """
    if data is None or data.empty:
        LOGGER.warning("Dataset kosong, membatalkan visualisasi prediksi.")
        return

    y_pred: Predictions | None = None

    try:
        model = load_model(MODEL_PATH)
        _x_train, x_test, _y_train, y_test = train_test_split(
            data[TEXT_COLUMN],
            data[SENTIMENT_COLUMN],
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=data[SENTIMENT_COLUMN],
        )
        test_data = pd.DataFrame(
            {TEXT_COLUMN: x_test, SENTIMENT_COLUMN: y_test}
        )
        try:
            y_pred = predict_sentiments(model, test_data)
        except (ValueError, TypeError, AttributeError, KeyError):
            LOGGER.exception(
                "Model gagal melakukan prediksi pada dataset; mengabaikan."
            )
    except FileNotFoundError:
        LOGGER.warning(
            "Model tidak tersedia, melewati visualisasi berbasis prediksi."
        )
        return
    except (EOFError, pickle.UnpicklingError, ValueError, TypeError):
        LOGGER.exception("Terjadi kesalahan saat memuat model; melewati prediksi.")
        return

    if y_pred is None:
        LOGGER.info(
            "Tidak ada prediksi model atau kolom label; "
            "melewati confusion/metrics."
        )
        return

    try:
        y_true = y_test.astype(int).tolist()
        plot_confusion_matrix_from_predictions(y_true, y_pred, output_dir)
        plot_model_metrics(y_true, y_pred, output_dir)
    except (OSError, ValueError, RuntimeError):
        LOGGER.exception("Gagal membuat visualisasi berbasis prediksi.")


def main() -> None:
    """Generate all visualizations for the sentiment analysis project.

    Steps:
        1. Read and validate dataset.
        2. Create output directory ``hasil_visualisasi/``.
        3. Create dataset-driven plots.
        4. Load model (if available) and create prediction-driven plots.
    """
    configure_logging()
    setup_visualization_style()
    start = time.time()

    data = read_dataset(DATASET_FILES)
    validate_dataset(data)
    data = normalize_labels(data)
    data = resolve_duplicate_texts(data)

    try:
        create_output_directory(OUTPUT_DIR)
    except OSError:
        LOGGER.error("Tidak dapat membuat folder output, berhenti.")
        return

    try:
        generate_dataset_visualizations(data, OUTPUT_DIR)
    except (OSError, ValueError, RuntimeError):
        LOGGER.exception("Gagal membuat beberapa visualisasi dataset")

    generate_prediction_visualizations(data, OUTPUT_DIR)

    elapsed = time.time() - start
    LOGGER.info("Selesai membuat visualisasi dalam %.3f detik", elapsed)


if __name__ == "__main__":
    main()
