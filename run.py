"""Launcher Streamlit dan pemeriksa kesiapan proyek."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys


BASE_DIR = Path(__file__).resolve().parent
REQUIREMENTS_FILE = BASE_DIR / "requirements.txt"
IMPORT_MAP_FILE = BASE_DIR / "resources" / "dependency_import_map.json"


def read_requirements() -> list[str]:
    """Membaca nama package dari requirements.txt."""
    if not REQUIREMENTS_FILE.exists():
        raise FileNotFoundError("requirements.txt tidak ditemukan.")
    packages: list[str] = []
    for line in REQUIREMENTS_FILE.read_text(encoding="utf-8").splitlines():
        cleaned = line.split("#", maxsplit=1)[0].strip()
        if not cleaned:
            continue
        package_name = re.split(r"[<>=!~\[]", cleaned, maxsplit=1)[0].strip()
        if package_name:
            packages.append(package_name)
    return packages


def load_import_map() -> dict[str, str]:
    """Membaca mapping package ke import module dari resource JSON."""
    if not IMPORT_MAP_FILE.exists():
        raise FileNotFoundError(
            "dependency_import_map.json tidak ditemukan. Jalankan setup_resources.py."
        )
    data = json.loads(IMPORT_MAP_FILE.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("dependency_import_map.json harus berupa object.")
    return {str(key): str(value) for key, value in data.items()}


def missing_dependencies() -> list[str]:
    """Mengembalikan daftar dependency yang belum bisa diimport."""
    import_map = load_import_map()
    missing: list[str] = []
    for package_name in read_requirements():
        module_name = import_map.get(package_name, package_name.replace("-", "_"))
        if importlib.util.find_spec(module_name) is None:
            missing.append(package_name)
    return missing


def model_exists() -> bool:
    """Memastikan model hasil training tersedia."""
    return (BASE_DIR / "model_sentimen.pkl").exists()


def resources_manifest_exists() -> bool:
    """Memastikan manifest resource tersedia."""
    return (BASE_DIR / "resources" / "resource_manifest.json").exists()


def run_streamlit_app() -> int:
    """Menjalankan aplikasi Streamlit."""
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(BASE_DIR / "app.py")],
        check=False,
    )
    return 0


def main() -> None:
    """Memastikan dependency tersedia lalu menjalankan Streamlit."""
    missing = missing_dependencies()
    if missing:
        print("Dependency belum tersedia:")
        for package_name in missing:
            print(f"- {package_name}")
        print("Jalankan: python -m pip install -r requirements.txt")
        sys.exit(1)
    if not resources_manifest_exists():
        print("Resource manifest tidak ditemukan. Jalankan: python setup_resources.py")
        sys.exit(1)
    if not model_exists():
        print("Model belum tersedia. Jalankan: python pemodelan.py")
        sys.exit(1)
    sys.exit(run_streamlit_app())


if __name__ == "__main__":
    main()
