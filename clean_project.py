"""Membersihkan file pengembangan sebelum distribusi ZIP."""
from __future__ import annotations

from pathlib import Path
import shutil


BASE_DIR = Path(__file__).resolve().parent
DEVELOPMENT_DIRECTORIES = {
    ".venv",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".vscode",
    ".agents",
    "dataset_asli",
    "backup",
    "backups",
}
DEVELOPMENT_SUFFIXES = {".pyc", ".pyo", ".log", ".tmp", ".bak"}
DEVELOPMENT_CACHE_FILES = {
    "emoji_sentiment_metadata.json",
    "translation_cache.json",
}
PROTECTED_NAMES = {
    "cache",
    "resources",
    "hasil_visualisasi",
}
PROTECTED_SUFFIXES = {".csv", ".pkl", ".json"}


def is_inside_protected_root(path: Path) -> bool:
    """Memeriksa apakah path berada di folder yang tidak boleh disentuh."""
    parts = set(path.relative_to(BASE_DIR).parts)
    return bool(parts.intersection(PROTECTED_NAMES - {"hasil_visualisasi"}))


def remove_directory(path: Path, removed: list[Path]) -> None:
    """Menghapus folder pengembangan secara aman."""
    if path.name in PROTECTED_NAMES:
        return
    shutil.rmtree(path)
    removed.append(path)


def remove_file(path: Path, removed: list[Path]) -> None:
    """Menghapus file cache/temp tanpa menyentuh artefak wajib."""
    if path.suffix.lower() not in DEVELOPMENT_SUFFIXES and is_inside_protected_root(path):
        return
    if path.suffix.lower() in PROTECTED_SUFFIXES and path.suffix.lower() not in DEVELOPMENT_SUFFIXES:
        return
    path.unlink()
    removed.append(path)


def clean_project() -> list[Path]:
    """Membersihkan cache pengembangan dan mengembalikan daftar path terhapus."""
    removed: list[Path] = []
    for directory_name in sorted(DEVELOPMENT_DIRECTORIES):
        directory = BASE_DIR / directory_name
        if directory.exists() and directory.is_dir():
            remove_directory(directory, removed)

    for path in sorted(BASE_DIR.rglob("*")):
        if not path.exists():
            continue
        if path.is_dir() and path.name == "__pycache__":
            remove_directory(path, removed)
        elif path.is_file() and path.suffix.lower() in DEVELOPMENT_SUFFIXES:
            remove_file(path, removed)
        elif is_inside_protected_root(path):
            continue
    cache_dir = BASE_DIR / "cache"
    for filename in sorted(DEVELOPMENT_CACHE_FILES):
        path = cache_dir / filename
        if path.exists() and path.is_file():
            path.unlink()
            removed.append(path)
    return removed


def main() -> None:
    """Entry point pembersihan proyek."""
    removed = clean_project()
    if not removed:
        print("Tidak ada file/folder pengembangan yang perlu dihapus.")
        return
    print("File/folder yang dihapus:")
    for path in removed:
        print(f"- {path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
