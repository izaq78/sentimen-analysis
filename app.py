"""Dashboard Streamlit profesional untuk Analisis Sentimen Bahasa Indonesia.

File ini berperan sebagai presenter/UI layer. Logika EDA, preprocessing,
evaluasi, visualisasi, dan model tetap dipanggil dari modul proyek yang
sudah ada.
"""
from __future__ import annotations

import logging
import platform
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from joblib import load
from pandas.errors import EmptyDataError, ParserError
from sklearn.metrics import classification_report, confusion_matrix

import eda
import evaluasi
import pemodelan
import visualisasi
from project_config import RESOURCES_DIR, load_json_file

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

VISUALIZATION_DIR = visualisasi.OUTPUT_DIR
MODEL_PATH = pemodelan.MODEL_OUTPUT
MODEL_METADATA_PATH = pemodelan.MODEL_METADATA_OUTPUT
APP_CONFIG = load_json_file(RESOURCES_DIR / "app_config.json")
PAGE_CONFIG = APP_CONFIG["page"]
APP_TITLE = str(APP_CONFIG["title"])
TEXT_INPUT_COLUMN = pemodelan.TEXT_COLUMN
APP_COLUMNS = APP_CONFIG["columns"]
PREDICTION_COLUMN = str(APP_COLUMNS["prediction"])
CONFIDENCE_COLUMN = str(APP_COLUMNS["confidence"])
STATUS_COLUMN = str(APP_COLUMNS["status"])
CONFIDENCE_CONFIG = APP_CONFIG["confidence_thresholds"]
CONFIDENCE_HIGH_THRESHOLD = float(CONFIDENCE_CONFIG["high"])
CONFIDENCE_MEDIUM_THRESHOLD = float(CONFIDENCE_CONFIG["medium"])
DOWNLOAD_CONFIG = APP_CONFIG["downloads"]
MENU_ITEMS = tuple(str(item) for item in APP_CONFIG["menu_items"])


@dataclass(frozen=True)
class EvaluationResult:
    """Container hasil evaluasi aktual model."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    report: pd.DataFrame
    matrix: list[list[int]]
    labels: list[Any]
    label_names: list[str]
    train_size: int
    test_size: int


def configure_page() -> None:
    """Konfigurasi awal halaman Streamlit."""
    st.set_page_config(
        page_title=str(PAGE_CONFIG["page_title"]),
        page_icon=str(PAGE_CONFIG["page_icon"]),
        layout=str(PAGE_CONFIG["layout"]),
        initial_sidebar_state=str(PAGE_CONFIG["initial_sidebar_state"]),
    )


def init_session_state() -> None:
    """Inisialisasi session state aplikasi."""
    st.session_state.setdefault("prediction_history", [])


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Membaca dataset lewat fungsi EDA yang sudah tersedia."""
    dataframes = eda.read_dataset_files(eda.DATASET_FILES)
    return eda.combine_datasets(dataframes)


@st.cache_resource(show_spinner=False)
def load_model() -> Any | None:
    """Memuat model terlatih satu kali setelah metadata divalidasi."""
    if not MODEL_PATH.exists():
        return None
    validate_model_metadata()
    return load(MODEL_PATH)


def load_model_metadata() -> dict[str, Any]:
    """Membaca metadata model untuk validasi kompatibilitas preprocessing."""
    if not MODEL_METADATA_PATH.exists():
        raise FileNotFoundError(
            "Metadata model tidak ditemukan. Latih ulang dengan `python pemodelan.py`."
        )
    data = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Metadata model tidak valid.")
    return data


def validate_model_metadata() -> None:
    """Memastikan model dibuat dengan resource dan konfigurasi saat ini."""
    metadata = load_model_metadata()
    current_config_checksum = pemodelan.file_sha256(pemodelan.CONFIG_FILE)
    current_resource_checksum = pemodelan.resource_checksums()
    if metadata.get("config_checksum") != current_config_checksum:
        raise RuntimeError(
            "Konfigurasi preprocessing berubah. Latih ulang model dengan `python pemodelan.py`."
        )
    if metadata.get("resource_checksum") != current_resource_checksum:
        raise RuntimeError(
            "Resource preprocessing berubah. Latih ulang model dengan `python pemodelan.py`."
        )


@st.cache_data(show_spinner=False)
def get_visualization_files() -> list[Path]:
    """Mengambil daftar gambar visualisasi dari folder output."""
    if not VISUALIZATION_DIR.exists():
        return []
    return sorted(
        path
        for path in VISUALIZATION_DIR.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


@st.cache_data(show_spinner=False)
def compute_dataset_summary(data: pd.DataFrame) -> dict[str, Any]:
    """Menghitung ringkasan dataset dengan helper dari eda.py."""
    if data.empty:
        return {
            "total": 0,
            "positive": 0,
            "negative": 0,
            "missing": 0,
            "duplicate": 0,
            "memory_mb": 0.0,
        }

    labels = data[eda.SENTIMENT_COLUMN].apply(eda.format_sentiment_label)
    return {
        "total": len(data),
        "positive": int((labels == eda.GOOD_LABEL).sum()),
        "negative": int((labels == eda.BAD_LABEL).sum()),
        "missing": int(data.isna().sum().sum()),
        "duplicate": int(data.duplicated().sum()),
        "memory_mb": eda.get_memory_usage_kb(data) / 1024,
    }


def dataset_modified_signature() -> tuple[float | None, ...]:
    """Mengambil timestamp seluruh file dataset untuk invalidasi cache."""
    return tuple(
        path.stat().st_mtime if path.exists() else None
        for path in evaluasi.DATASET_FILES
    )


def evaluation_config_signature() -> tuple[float, int, int]:
    """Mengambil konfigurasi evaluasi yang memengaruhi hasil cache."""
    return (
        evaluasi.TEST_SIZE,
        evaluasi.RANDOM_STATE,
        len(evaluasi.DATASET_FILES),
    )


@st.cache_data(show_spinner=False)
def run_actual_evaluation(
    model_modified_time_value: float | None,
    dataset_signature: tuple[float | None, ...],
    config_signature: tuple[float, int, int],
) -> EvaluationResult | None:
    """Menjalankan evaluasi aktual menggunakan fungsi dari evaluasi.py.

    Args:
        model_modified_time_value: Waktu modifikasi model untuk invalidasi cache.
        dataset_signature: Timestamp semua dataset untuk invalidasi cache.
        config_signature: Konfigurasi evaluasi untuk invalidasi cache.

    Returns:
        Hasil evaluasi aktual, atau None jika model belum tersedia.
    """
    del dataset_signature, config_signature

    if model_modified_time_value is None:
        return None

    data = evaluasi.load_dataset_files(evaluasi.DATASET_FILES)
    evaluasi.validate_dataset_columns(data)
    data = evaluasi.remove_empty_required_data(data)
    data = pemodelan.normalize_labels(data)
    data = pemodelan.resolve_duplicate_texts(data)
    x_train, x_test, _, y_test = evaluasi.split_data(data)
    x_test_processed = evaluasi.preprocess_test_data(x_test, y_test)
    model = evaluasi.load_model(evaluasi.MODEL_FILE)
    y_pred = model.predict(x_test_processed)
    accuracy, precision, recall, f1 = evaluasi.evaluate_predictions(
        y_test,
        pd.Series(y_pred),
    )
    report_dict = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )
    report = pd.DataFrame(report_dict).transpose().round(4)
    labels = list(getattr(model, "classes_", sorted(pd.Series(y_test).unique())))
    label_names = [eda.format_sentiment_label(label) for label in labels]
    matrix = confusion_matrix(y_test, y_pred, labels=labels).tolist()
    return EvaluationResult(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        report=report,
        matrix=matrix,
        labels=labels,
        label_names=label_names,
        train_size=len(x_train),
        test_size=len(x_test),
    )


def model_modified_time() -> float | None:
    """Mengambil timestamp model untuk status dan invalidasi cache."""
    if not MODEL_PATH.exists():
        return None
    return MODEL_PATH.stat().st_mtime


def format_percent(value: float | None) -> str:
    """Format angka metrik ke persen."""
    if value is None:
        return "-"
    return f"{value * 100:.2f}%"


def get_confidence_level(confidence: float) -> tuple[str, str]:
    """Mengubah confidence numerik menjadi kategori Bahasa Indonesia.

    Args:
        confidence: Nilai probabilitas kelas prediksi.

    Returns:
        Tuple berisi label kategori dan penjelasan singkat.
    """
    if confidence >= CONFIDENCE_HIGH_THRESHOLD:
        return "Sangat Yakin", "Probabilitas prediksi sangat kuat."
    if confidence >= CONFIDENCE_MEDIUM_THRESHOLD:
        return "Cukup Yakin", "Probabilitas prediksi cukup kuat."
    return "Kurang Yakin", "Probabilitas prediksi masih rendah."


def model_status() -> tuple[str, str]:
    """Menghasilkan status model dan tanggal training dari file model."""
    if not MODEL_PATH.exists():
        return "Model belum tersedia", "-"
    try:
        trained_timestamp = load_model_metadata().get("timestamp")
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        trained_timestamp = None
    trained_at = datetime.fromtimestamp(MODEL_PATH.stat().st_mtime)
    if trained_timestamp:
        try:
            trained_at = datetime.fromisoformat(str(trained_timestamp))
        except ValueError:
            LOGGER.warning(
                "Timestamp metadata model tidak dapat diparse: %s",
                trained_timestamp,
            )
    return "Model siap digunakan", trained_at.strftime("%d %B %Y %H:%M")


def safe_evaluation() -> EvaluationResult | None:
    """Wrapper evaluasi dengan error handling UI."""
    try:
        with st.spinner("Menghitung evaluasi aktual model..."):
            return run_actual_evaluation(
                model_modified_time(),
                dataset_modified_signature(),
                evaluation_config_signature(),
            )
    except (
        FileNotFoundError,
        ValueError,
        KeyError,
        RuntimeError,
        OSError,
    ) as exc:
        LOGGER.exception("Evaluasi gagal: %s", exc)
        st.warning(f"Evaluasi belum dapat ditampilkan: {exc}")
        return None


def metric_grid(items: list[tuple[str, str, str | None]]) -> None:
    """Menampilkan daftar metric dalam grid responsif."""
    columns = st.columns(len(items))
    for column, (label, value, help_text) in zip(columns, items):
        column.metric(label, value, help=help_text)


def render_home(data: pd.DataFrame, model: Any | None) -> None:
    """Halaman ringkasan utama dashboard."""
    st.title("Beranda Dashboard AI Analisis Sentimen")
    st.caption("Ringkasan dataset, model, dan performa aktual.")

    if data.empty:
        st.warning("Dataset kosong atau belum berhasil dimuat.")

    summary = compute_dataset_summary(data)
    evaluation = safe_evaluation() if MODEL_PATH.exists() else None
    status, training_date = model_status()

    if model is None:
        st.warning("Model belum tersedia. Jalankan `python pemodelan.py`.")
    else:
        st.success(status)

    metric_grid(
        [
            ("Total Dataset", f"{summary['total']:,}", None),
            ("Total Positif", f"{summary['positive']:,}", None),
            ("Total Negatif", f"{summary['negative']:,}", None),
            ("Akurasi", format_percent(evaluation.accuracy if evaluation else None), None),
        ]
    )
    metric_grid(
        [
            ("Presisi", format_percent(evaluation.precision if evaluation else None), None),
            ("Recall", format_percent(evaluation.recall if evaluation else None), None),
            ("F1 Score", format_percent(evaluation.f1 if evaluation else None), None),
            ("Status Model", status, None),
        ]
    )

    st.divider()
    col_model, col_data = st.columns(2)
    with col_model.container(border=True):
        st.subheader("Model")
        st.write("Algoritma: Multinomial Naive Bayes")
        st.write("Ekstraksi Fitur: TF-IDF")
        st.write(f"Tanggal Pelatihan: {training_date}")
        st.write(f"File: `{MODEL_PATH.name}`")
    with col_data.container(border=True):
        st.subheader("Dataset")
        st.write(f"Nilai Hilang: {summary['missing']:,}")
        st.write(f"Duplikasi: {summary['duplicate']:,}")
        st.write(f"Memori: {summary['memory_mb']:.2f} MB")

    st.divider()
    col_chart, col_matrix = st.columns(2)
    with col_chart.container(border=True):
        st.subheader("Distribusi Label Dataset")
        label_data = pd.DataFrame(
            {
                "Sentimen": [eda.GOOD_LABEL, eda.BAD_LABEL],
                "Jumlah": [summary["positive"], summary["negative"]],
            }
        )
        st.bar_chart(label_data.set_index("Sentimen"))
    with col_matrix.container(border=True):
        st.subheader("Ringkasan Matriks Kebingungan")
        if evaluation is None:
            st.info("Evaluasi belum tersedia.")
        else:
            st.dataframe(
                pd.DataFrame(
                    evaluation.matrix,
                    index=evaluation.label_names,
                    columns=evaluation.label_names,
                ),
                use_container_width=True,
            )


def render_eda(data: pd.DataFrame) -> None:
    """Halaman EDA."""
    st.title("Analisis Eksplorasi Data")
    if data.empty:
        st.error("Dataset kosong atau gagal dimuat.")
        return

    tab_preview, tab_quality, tab_stats, tab_visual = st.tabs(
        ["Pratinjau Dataset", "Nilai Hilang", "Statistik Review", "Grafik"]
    )

    with tab_preview:
        st.dataframe(data.head(50), use_container_width=True)
        st.caption(f"Menampilkan 50 dari {len(data):,} baris.")

    with tab_quality:
        missing = data.isna().sum().reset_index()
        missing.columns = ["Kolom", "Nilai Hilang"]
        st.dataframe(missing, use_container_width=True, hide_index=True)
        st.metric("Total Nilai Hilang", int(missing["Nilai Hilang"].sum()))
        st.metric("Total Duplikasi", int(data.duplicated().sum()))

    with tab_stats:
        word_counts = eda.get_word_count(data)
        char_lengths = eda.get_text_length(data)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Kata Minimum", int(word_counts.min()))
        col2.metric("Kata Maksimum", int(word_counts.max()))
        col3.metric("Rata-rata Kata", f"{word_counts.mean():.2f}")
        col4.metric("Median Kata", f"{word_counts.median():.2f}")
        stats_df = pd.DataFrame(
            {
                "Metrik": ["Karakter", "Kata"],
                "Minimum": [int(char_lengths.min()), int(word_counts.min())],
                "Maksimum": [int(char_lengths.max()), int(word_counts.max())],
                "Rata-rata": [char_lengths.mean(), word_counts.mean()],
                "Median": [char_lengths.median(), word_counts.median()],
            }
        )
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    with tab_visual:
        render_visual_image(
            "Distribusi Label",
            "1_label_distribution.png",
            "download_eda_graph_0_1_label_distribution.png",
        )
        render_visual_image(
            "Histogram Panjang Review",
            "2_review_length_histogram.png",
            "download_eda_graph_1_2_review_length_histogram.png",
        )
        render_visual_image(
            "Boxplot Panjang Review",
            "3_review_length_boxplot.png",
            "download_eda_graph_2_3_review_length_boxplot.png",
        )


def render_pipeline_step(index: int, title: str, description: str) -> None:
    """Menampilkan satu langkah pipeline."""
    with st.container(border=True):
        cols = st.columns([1, 5])
        cols[0].metric("Tahap", index)
        cols[1].markdown(f"**{title}**")
        cols[1].caption(description)


def render_modeling() -> None:
    """Halaman pemodelan."""
    st.title("Pemodelan")
    st.caption("Pipeline NLP dari dataset sampai prediksi.")
    steps = [
        ("Dataset", "Data CSV digabung dan divalidasi."),
        ("Cleaning", "URL, simbol, dan noise dibersihkan."),
        ("Normalisasi", "Emoji, emoticon, leetspeak, typo, dan huruf berulang dinormalisasi."),
        ("Tokenisasi", "Teks dipisahkan menjadi token."),
        ("Stopword", "Stopword Bahasa Indonesia dihapus."),
        ("Stemming", "Sastrawi mengubah kata ke bentuk dasar."),
        ("TF-IDF", "Teks diubah menjadi fitur numerik."),
        ("Naive Bayes", "Multinomial Naive Bayes mempelajari distribusi fitur."),
        ("Prediksi", "Model menghasilkan label sentimen."),
    ]
    for index, (title, description) in enumerate(steps, start=1):
        render_pipeline_step(index, title, description)

    with st.expander("Konfigurasi Model", expanded=True):
        st.code(
            "Pipeline([('tfidf', TfidfVectorizer(...)), "
            "('naive_bayes', MultinomialNB())])",
            language="python",
        )


def render_evaluation(model: Any | None) -> None:
    """Halaman evaluasi model aktual."""
    st.title("Evaluasi")
    if model is None:
        st.warning("Model belum tersedia. Evaluasi tidak dapat dijalankan.")
        return

    evaluation = safe_evaluation()
    if evaluation is None:
        return

    metric_grid(
        [
            ("Akurasi", format_percent(evaluation.accuracy), None),
            ("Presisi", format_percent(evaluation.precision), None),
            ("Recall", format_percent(evaluation.recall), None),
            ("F1 Score", format_percent(evaluation.f1), None),
        ]
    )
    st.caption(
        f"Data training: {evaluation.train_size:,} | "
        f"Data testing: {evaluation.test_size:,}"
    )

    tab_report, tab_matrix = st.tabs(["Laporan Klasifikasi", "Matriks Kebingungan"])
    with tab_report:
        st.dataframe(evaluation.report, use_container_width=True)
    with tab_matrix:
        matrix_df = pd.DataFrame(
            evaluation.matrix,
            index=evaluation.label_names,
            columns=evaluation.label_names,
        )
        st.dataframe(matrix_df, use_container_width=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        image = ax.imshow(evaluation.matrix, cmap="Blues")
        ax.set_title("Matriks Kebingungan Aktual Model")
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        ax.set_xticks(range(len(evaluation.label_names)))
        ax.set_yticks(range(len(evaluation.label_names)))
        ax.set_xticklabels(evaluation.label_names)
        ax.set_yticklabels(evaluation.label_names)
        max_value = max(max(row) for row in evaluation.matrix) if evaluation.matrix else 0
        for row_idx, row in enumerate(evaluation.matrix):
            for col_idx, value in enumerate(row):
                text_color = "white" if max_value and value > max_value / 2 else "black"
                ax.text(
                    col_idx,
                    row_idx,
                    value,
                    ha="center",
                    va="center",
                    color=text_color,
                )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)
        plt.close(fig)


def render_visual_image(title: str, filename: str, download_key: str) -> None:
    """Menampilkan gambar visualisasi dengan zoom dan download."""
    image_path = VISUALIZATION_DIR / filename
    with st.container(border=True):
        st.subheader(title)
        if not image_path.exists():
            st.warning(f"Gambar tidak ditemukan: {filename}")
            return
        st.image(str(image_path), use_container_width=True)
        with image_path.open("rb") as image_file:
            image_bytes = image_file.read()
            st.download_button(
                label=f"Download {filename}",
                data=image_bytes,
                file_name=filename,
                mime="image/png",
                key=download_key,
            )


def render_visualization() -> None:
    """Halaman visualisasi dari folder hasil_visualisasi."""
    st.title("Visualisasi")
    files = get_visualization_files()
    if not files:
        st.warning("Belum ada gambar visualisasi di folder hasil_visualisasi.")
        return

    file_names = [path.name for path in files]
    selected = st.selectbox(
        "Pilih gambar untuk zoom",
        file_names,
        key="visualization_selected_image",
    )
    selected_index = file_names.index(selected)
    render_visual_image(
        "Pratinjau / Zoom",
        selected,
        f"download_visual_preview_{selected_index}_{selected}",
    )

    st.divider()
    cols = st.columns(2)
    for index, image_path in enumerate(files):
        with cols[index % 2]:
            render_visual_image(
                image_path.stem.replace("_", " ").title(),
                image_path.name,
                f"download_visual_grid_{index}_{image_path.name}",
            )


def preprocess_for_prediction(texts: list[str]) -> pd.DataFrame:
    """Memanggil preprocessing pemodelan.py untuk kebutuhan prediksi."""
    frame = pd.DataFrame(
        {
            pemodelan.TEXT_COLUMN: texts,
            pemodelan.SENTIMENT_COLUMN: [0] * len(texts),
        }
    )
    return pemodelan.preprocess_text(frame)


def predict_texts(model: Any, texts: list[str]) -> pd.DataFrame:
    """Prediksi banyak teks dengan confidence jika tersedia."""
    processed = preprocess_for_prediction(texts)
    cleaned_texts = processed[pemodelan.TEXT_COLUMN].astype(str)
    predictions = model.predict(cleaned_texts)
    readable_predictions = [eda.format_sentiment_label(value) for value in predictions]
    result = pd.DataFrame(
        {TEXT_INPUT_COLUMN: texts, PREDICTION_COLUMN: readable_predictions}
    )

    if callable(getattr(model, "predict_proba", None)):
        probabilities = model.predict_proba(cleaned_texts)
        model_classes = getattr(model, "classes_", None)
        if model_classes is None:
            result[CONFIDENCE_COLUMN] = pd.NA
        else:
            class_index = {label: index for index, label in enumerate(model_classes)}
            confidence_values: list[float | None] = []
            for row_index, prediction in enumerate(predictions):
                prediction_index = class_index.get(prediction)
                if prediction_index is None:
                    confidence_values.append(None)
                else:
                    confidence_values.append(
                        float(probabilities[row_index][prediction_index])
                    )
            result[CONFIDENCE_COLUMN] = confidence_values
    else:
        result[CONFIDENCE_COLUMN] = pd.NA
    return result


def render_single_prediction(model: Any) -> None:
    """UI prediksi satu review."""
    text = st.text_area("Masukkan review", height=160, key="single_review_text")
    if st.button("Analisis Review", type="primary", key="single_review_analyze_button"):
        if not text.strip():
            st.warning("Teks review belum diisi.")
            return
        try:
            with st.spinner("Memproses prediksi..."):
                result = predict_texts(model, [text])
            prediction = result.iloc[0][PREDICTION_COLUMN]
            confidence = result.iloc[0][CONFIDENCE_COLUMN]
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.prediction_history.append(
                {
                    "timestamp": timestamp,
                    "text": text,
                    "sentimen": prediction,
                    CONFIDENCE_COLUMN: confidence,
                }
            )
            st.success(f"Hasil prediksi: {prediction}")
            if pd.notna(confidence):
                confidence_value = float(confidence)
                level, description = get_confidence_level(confidence_value)
                st.progress(
                    confidence_value,
                    text=f"Tingkat Keyakinan: {confidence_value:.2%}",
                )
                st.info(f"{level}: {description}")
            else:
                st.info("Tingkat keyakinan tidak tersedia untuk model ini.")
            st.toast("Prediksi berhasil disimpan ke riwayat.")
        except (ValueError, KeyError, RuntimeError, AttributeError) as exc:
            LOGGER.exception("Prediksi gagal: %s", exc)
            st.error(f"Prediksi gagal: {exc}")


def render_batch_prediction(model: Any) -> None:
    """UI prediksi CSV."""
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="batch_csv_upload")
    if uploaded_file is None:
        st.info(f"CSV harus memiliki kolom `{TEXT_INPUT_COLUMN}`.")
        return
    try:
        data = pd.read_csv(uploaded_file)
    except EmptyDataError:
        st.error("CSV kosong.")
        return
    except ParserError as exc:
        st.error(f"Format CSV tidak valid: {exc}")
        return

    if data.empty:
        st.warning("CSV tidak memiliki baris data.")
        return
    if TEXT_INPUT_COLUMN not in data.columns:
        st.error(f"Kolom wajib `{TEXT_INPUT_COLUMN}` tidak ditemukan.")
        return

    text_series = data[TEXT_INPUT_COLUMN]
    invalid_mask = text_series.isna() | text_series.astype(str).str.strip().eq("")
    invalid_count = int(invalid_mask.sum())
    if invalid_count:
        st.warning(
            f"{invalid_count:,} baris memiliki teks kosong dan tidak akan diprediksi."
        )

    st.dataframe(data.head(20), use_container_width=True)
    if st.button("Prediksi CSV", key="batch_csv_predict_button"):
        try:
            output = data.copy()
            output[PREDICTION_COLUMN] = pd.NA
            output[CONFIDENCE_COLUMN] = pd.NA
            output[STATUS_COLUMN] = "Berhasil"
            output.loc[invalid_mask, STATUS_COLUMN] = "Teks kosong, tidak diprediksi"
            valid_texts = text_series.loc[~invalid_mask].astype(str).tolist()

            with st.spinner("Memproses seluruh review..."):
                if valid_texts:
                    result = predict_texts(model, valid_texts)
                    valid_index = text_series.index[~invalid_mask]
                    output.loc[valid_index, PREDICTION_COLUMN] = result[
                        PREDICTION_COLUMN
                    ].to_numpy()
                    output.loc[valid_index, CONFIDENCE_COLUMN] = result[
                        CONFIDENCE_COLUMN
                    ].to_numpy()
            st.success("Prediksi CSV selesai.")
            st.dataframe(output.head(50), use_container_width=True)
            st.download_button(
                "Download Hasil Prediksi",
                data=output.to_csv(index=False).encode("utf-8-sig"),
                file_name=str(DOWNLOAD_CONFIG["batch_filename"]),
                mime=str(DOWNLOAD_CONFIG["csv_mime"]),
                key="batch_prediction_download",
            )
        except (ValueError, KeyError, RuntimeError, AttributeError) as exc:
            LOGGER.exception("Prediksi CSV gagal: %s", exc)
            st.error(f"Prediksi CSV gagal: {exc}")


def render_history() -> None:
    """Menampilkan riwayat prediksi."""
    history = st.session_state.prediction_history
    if not history:
        return

    st.divider()
    st.subheader("Riwayat Prediksi")
    history_df = pd.DataFrame(history)
    st.dataframe(history_df, use_container_width=True)

    col_download, col_reset = st.columns(2)
    col_download.download_button(
        "Download Riwayat",
        data=history_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=str(DOWNLOAD_CONFIG["history_filename"]),
        mime=str(DOWNLOAD_CONFIG["csv_mime"]),
        key="prediction_history_download",
    )
    if col_reset.button("Reset Riwayat", key="prediction_history_reset_button"):
        st.session_state.prediction_history = []
        st.rerun()

    counts = history_df["sentimen"].value_counts()
    col_bar, col_pie = st.columns(2)
    with col_bar:
        st.caption("Bar Chart")
        st.bar_chart(counts)
    with col_pie:
        st.caption("Pie Chart")
        fig, ax = plt.subplots()
        ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%")
        st.pyplot(fig)
        plt.close(fig)


def render_prediction(model: Any | None) -> None:
    """Halaman prediksi."""
    st.title("Prediksi")
    if model is None:
        st.error("Model belum dimuat. Jalankan training terlebih dahulu.")
        return

    tab_single, tab_batch = st.tabs(["Satu Review", "Upload CSV"])
    with tab_single:
        render_single_prediction(model)
    with tab_batch:
        render_batch_prediction(model)
    render_history()


def render_about(data: pd.DataFrame) -> None:
    """Halaman tentang aplikasi."""
    st.title("Tentang")
    summary = compute_dataset_summary(data)
    status, training_date = model_status()

    col1, col2 = st.columns(2)
    with col1.container(border=True):
        st.subheader("Proyek")
        st.write("Nama Proyek: Analisis Sentimen Bahasa Indonesia")
        st.write("Universitas: Universitas Mercu Buana")
        st.write("Program Studi: Teknik Informatika")
        st.write("Mata Kuliah: Kecerdasan Buatan")
        st.write(f"Jumlah Dataset: {summary['total']:,}")
        st.write(f"Tanggal Pelatihan: {training_date}")
    with col2.container(border=True):
        st.subheader("Teknologi")
        st.write("Algoritma: Multinomial Naive Bayes")
        st.write("Ekstraksi Fitur: TF-IDF")
        st.write("Normalisasi: RapidFuzz, Sastrawi, emoji, emot")
        st.write(f"Versi Python: {platform.python_version()}")
        st.write(f"Versi Streamlit: {st.__version__}")
        st.write(f"Status Model: {status}")


def render_sidebar() -> str:
    """Menampilkan sidebar navigasi."""
    st.sidebar.title("Navigasi")
    st.sidebar.caption("Dashboard Analisis Sentimen")
    selected = st.sidebar.radio("Menu", list(MENU_ITEMS), key="sidebar_menu")
    st.sidebar.divider()
    st.sidebar.caption(f"Terakhir dibuka: {datetime.now():%d %B %Y %H:%M}")
    return selected


def main() -> None:
    """Entry point Streamlit."""
    configure_page()
    init_session_state()

    try:
        data = load_dataset()
    except (FileNotFoundError, ValueError, PermissionError, OSError) as exc:
        LOGGER.exception("Dataset gagal dimuat: %s", exc)
        st.error(f"Dataset gagal dimuat: {exc}")
        data = pd.DataFrame(columns=[eda.TEXT_COLUMN, eda.SENTIMENT_COLUMN])

    try:
        model = load_model()
    except (FileNotFoundError, ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        LOGGER.exception("Model gagal dimuat: %s", exc)
        st.error(f"Model tidak kompatibel: {exc}")
        model = None
    page = render_sidebar()

    if page == "Beranda":
        render_home(data, model)
    elif page == "EDA":
        render_eda(data)
    elif page == "Pemodelan":
        render_modeling()
    elif page == "Evaluasi":
        render_evaluation(model)
    elif page == "Visualisasi":
        render_visualization()
    elif page == "Prediksi":
        render_prediction(model)
    elif page == "Tentang":
        render_about(data)


if __name__ == "__main__":
    main()

