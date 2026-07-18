from pathlib import Path

import pandas as pd

from project_config import SENTIMENT_COLUMN, TEXT_COLUMN, discover_dataset_files

BASE_DIR = Path(__file__).resolve().parent
DATASET_FILES = discover_dataset_files()
SECTION_WIDTH = 60


def print_section(title: str) -> None:
    """Print a consistent section separator."""
    print(f"\n{'=' * SECTION_WIDTH}")
    print(title)
    print(f"{'=' * SECTION_WIDTH}")


def read_dataset(file_path: Path) -> pd.DataFrame | None:
    """Read a CSV dataset and return None when reading fails."""
    try:
        return pd.read_csv(file_path)
    except Exception as error:
        print(f"ERROR: File gagal dibaca. Detail: {error}")
        return None


def get_dataset_issues(data: pd.DataFrame | None, file_path: Path) -> list[str]:
    """Return validation issues found in a dataset."""
    issues: list[str] = []

    if not file_path.exists():
        issues.append("File tidak ditemukan.")
        return issues

    if data is None:
        issues.append("File gagal dibaca.")
        return issues

    if TEXT_COLUMN not in data.columns:
        issues.append('Kolom text tidak ditemukan.')

    if SENTIMENT_COLUMN not in data.columns:
        issues.append('Kolom sentiment tidak ditemukan.')

    return issues


def show_dataset_overview(file_path: Path, data: pd.DataFrame | None) -> None:
    """Display basic information for one dataset."""
    print_section(f"VALIDASI DATASET: {file_path.name}")

    if not file_path.exists():
        print("ERROR: File tidak ditemukan.")
        return

    if data is None:
        return

    print("File berhasil dibaca.")
    print(f"Jumlah baris : {data.shape[0]}")
    print(f"Jumlah kolom : {data.shape[1]}")

    print("\nDaftar nama kolom:")
    for index, column_name in enumerate(data.columns, start=1):
        print(f"{index}. {column_name}")

    print("\n5 baris pertama:")
    print(data.head(5))


def show_column_checks(data: pd.DataFrame | None) -> None:
    """Display checks for text and sentiment columns."""
    if data is None:
        return

    has_text_column = TEXT_COLUMN in data.columns
    has_sentiment_column = SENTIMENT_COLUMN in data.columns

    print("\nPemeriksaan kolom:")
    print(f"Kolom text      : {'Ada' if has_text_column else 'Tidak ada'}")
    print(
        f"Kolom sentiment : {'Ada' if has_sentiment_column else 'Tidak ada'}"
    )

    if has_sentiment_column:
        print("\nValue counts kolom sentiment:")
        print(data[SENTIMENT_COLUMN].value_counts(dropna=False))

        missing_count = data[SENTIMENT_COLUMN].isnull().sum()
        print("\nMissing value kolom sentiment:")
        print(f"Jumlah missing value : {missing_count}")
        print(f"Status missing value : {'Ada' if missing_count > 0 else 'Tidak ada'}")
    else:
        print("\nERROR: Kolom sentiment tidak ditemukan.")


def validate_dataset(file_path: Path) -> tuple[str, list[str]]:
    """Validate one dataset and return its file name with validation issues."""
    data = read_dataset(file_path) if file_path.exists() else None
    show_dataset_overview(file_path, data)
    show_column_checks(data)

    issues = get_dataset_issues(data, file_path)
    return file_path.name, issues


def show_validation_summary(results: list[tuple[str, list[str]]]) -> None:
    """Display valid and problematic dataset summary."""
    valid_datasets = [file_name for file_name, issues in results if not issues]
    problematic_datasets = [
        (file_name, issues) for file_name, issues in results if issues
    ]

    print_section("RINGKASAN VALIDASI DATASET")
    print("Dataset yang valid:")
    if valid_datasets:
        for file_name in valid_datasets:
            print(f"- {file_name}")
    else:
        print("- Tidak ada")

    print("\nDataset yang bermasalah:")
    if problematic_datasets:
        for file_name, issues in problematic_datasets:
            print(f"- {file_name}")
            for issue in issues:
                print(f"  Alasan: {issue}")
    else:
        print("- Tidak ada")


def main() -> None:
    """Run validation for all configured datasets."""
    validation_results = [
        validate_dataset(dataset_file) for dataset_file in DATASET_FILES
    ]
    show_validation_summary(validation_results)


if __name__ == "__main__":
    main()
