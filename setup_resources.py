"""Setup dan validasi resource lokal proyek analisis sentimen."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any
from urllib.error import URLError
from urllib.request import urlretrieve


BASE_DIR = Path(__file__).resolve().parent
RESOURCES_DIR = BASE_DIR / "resources"
CACHE_DIR = BASE_DIR / "cache"
MANIFEST_FILE = RESOURCES_DIR / "resource_manifest.json"

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
LOGGER = logging.getLogger(__name__)

MANIFEST_KEYS = {
    "name",
    "path",
    "type",
    "required",
    "source_url",
    "generator",
    "checksum",
    "checksum_algorithm",
    "license",
    "source_name",
    "validation",
    "version",
    "updated_at",
    "distribution",
}
VALID_DISTRIBUTIONS = {"bundled", "downloadable"}


def project_path(relative_path: str) -> Path:
    """Mengubah path relatif manifest menjadi path absolut proyek."""
    return BASE_DIR / relative_path


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Menulis JSON secara atomik setelah struktur manifest divalidasi."""
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Manifest hasil update bukan JSON object.")
    validate_manifest(parsed)

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=path.parent,
            suffix=".tmp",
        ) as file_obj:
            file_obj.write(text)
            file_obj.flush()
            os.fsync(file_obj.fileno())
            temp_path = Path(file_obj.name)
        temp_data = json.loads(temp_path.read_text(encoding="utf-8"))
        if not isinstance(temp_data, dict):
            raise ValueError("Manifest sementara bukan JSON object.")
        validate_manifest(temp_data)
        temp_path.replace(path)
    except (OSError, ValueError):
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def file_sha256(path: Path) -> str:
    """Menghitung checksum SHA256 file."""
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Membaca file JSON object."""
    if not path.exists():
        raise FileNotFoundError(f"Resource tidak ditemukan: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Resource JSON harus object: {path}")
    return data


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    """Membaca manifest resource proyek."""
    return load_json(path or MANIFEST_FILE)


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Validasi struktur resource manifest."""
    resources = manifest.get("resources")
    if not isinstance(resources, list) or not resources:
        raise ValueError("resource_manifest.json wajib memiliki list resources.")
    names: set[str] = set()
    for entry in resources:
        if not isinstance(entry, dict):
            raise ValueError("Setiap resource manifest wajib berupa object.")
        missing = sorted(MANIFEST_KEYS - set(entry))
        if missing:
            raise ValueError(f"Entry manifest kurang key {missing}: {entry.get('name')}")
        unexpected = sorted(set(entry) - MANIFEST_KEYS)
        if unexpected:
            raise ValueError(f"Entry manifest memiliki key tidak dikenal {unexpected}: {entry.get('name')}")
        name = str(entry["name"])
        if name in names:
            raise ValueError(f"Nama resource duplikat pada manifest: {name}")
        names.add(name)
        distribution = str(entry["distribution"])
        if distribution not in VALID_DISTRIBUTIONS:
            raise ValueError(f"Distribusi resource tidak didukung: {name}")
        if distribution == "downloadable" and not str(entry["source_url"]).strip():
            raise ValueError(f"Resource downloadable wajib memiliki source_url: {name}")
        if distribution == "bundled" and str(entry["type"]) != "directory":
            if not project_path(str(entry["path"])).exists():
                raise FileNotFoundError(f"Resource bundled wajib tidak ditemukan: {name}")
        if str(entry["checksum_algorithm"]).lower() != "sha256":
            raise ValueError(f"Checksum algorithm wajib sha256: {name}")


def validate_checksum(entry: dict[str, Any], path: Path, strict: bool) -> None:
    """Validasi checksum SHA256 resource."""
    checksum = str(entry.get("checksum") or "")
    if not checksum:
        if strict:
            raise ValueError(f"Checksum wajib diisi untuk resource wajib: {entry['name']}")
        return
    actual = file_sha256(path)
    if actual != checksum:
        raise ValueError(
            "Checksum tidak cocok untuk "
            f"{entry['name']}: expected={checksum}, actual={actual}"
        )


def download_resource(
    entry: dict[str, Any],
    path: Path,
    *,
    validate_checksum_strict: bool,
) -> None:
    """Mengunduh resource ke file sementara lalu rename atomik."""
    source_url = str(entry["source_url"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=path.parent, suffix=".download") as tmp:
        temp_path = Path(tmp.name)
    try:
        urlretrieve(source_url, temp_path)
        validate_resource(entry, temp_path)
        validate_checksum(entry, temp_path, strict=validate_checksum_strict)
        path.unlink(missing_ok=True)
        temp_path.replace(path)
    except (OSError, URLError, ValueError) as exc:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Gagal menyiapkan resource {entry['name']}: {exc}") from exc


def ensure_resource_available(
    entry: dict[str, Any],
    path: Path,
    *,
    validate_checksum_strict: bool,
) -> None:
    """Memastikan resource tersedia sesuai mode distribusi."""
    distribution = str(entry["distribution"])
    if str(entry["type"]) == "directory":
        path.mkdir(parents=True, exist_ok=True)
        return
    if path.exists():
        return
    if distribution == "downloadable":
        download_resource(
            entry,
            path,
            validate_checksum_strict=validate_checksum_strict,
        )
        return
    if bool(entry["required"]):
        raise FileNotFoundError(
            f"Resource bundled wajib tidak ditemukan: {path}. "
            "Pastikan file ikut disertakan dalam distribusi proyek."
        )


def validate_json_object(path: Path, validation: dict[str, Any]) -> None:
    """Validasi resource JSON object."""
    data = load_json(path)
    for key in validation.get("required_keys", []):
        if key not in data:
            raise ValueError(f"{path.name} tidak memiliki key wajib: {key}")
    mapping_key = validation.get("mapping_key")
    list_key = validation.get("list_key")
    if mapping_key is not None:
        mapping = data.get(str(mapping_key))
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"{path.name} wajib memiliki object {mapping_key}.")
        for key, value in mapping.items():
            if not str(key).strip() or not str(value).strip():
                raise ValueError(f"Mapping kosong pada {path.name}: {key}")
    if list_key is not None:
        items = data.get(str(list_key))
        if not isinstance(items, list) or not items:
            raise ValueError(f"{path.name} wajib memiliki list {list_key}.")
        if any(not str(item).strip() for item in items):
            raise ValueError(f"Item kosong pada {path.name}: {list_key}")
    if validation.get("non_empty") and not data:
        raise ValueError(f"Resource JSON kosong: {path.name}")


def validate_text(path: Path) -> None:
    """Validasi file teks non-kosong."""
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Resource teks kosong: {path}")
    blocked_markers = ("TO" + "DO", "FIX" + "ME", "PLACE" + "HOLDER")
    if any(marker in text for marker in blocked_markers):
        raise ValueError(f"Resource teks mengandung marker tidak valid: {path}")


def validate_csv(path: Path, validation: dict[str, Any]) -> None:
    """Validasi CSV berdasarkan schema manifest."""
    columns = [str(column) for column in validation.get("columns", [])]
    with path.open(newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        fieldnames = reader.fieldnames or []
        missing = [column for column in columns if column not in fieldnames]
        if missing:
            raise ValueError(f"Kolom CSV hilang pada {path.name}: {missing}")
        rows = list(reader)
    if validation.get("non_empty") and not rows:
        raise ValueError(f"Resource CSV kosong: {path}")
    unique_key = validation.get("unique_key")
    if unique_key:
        values = [str(row.get(str(unique_key), "")).strip() for row in rows]
        if any(not value for value in values):
            raise ValueError(f"Kolom unik kosong pada {path.name}: {unique_key}")
        if len(values) != len(set(values)):
            raise ValueError(f"Duplikasi {unique_key} ditemukan pada {path.name}")
    allowed_values = validation.get("allowed_values") or {}
    for column, allowed in allowed_values.items():
        invalid = sorted({row.get(column, "") for row in rows} - set(allowed))
        if invalid:
            raise ValueError(f"Nilai tidak valid pada {path.name}.{column}: {invalid}")
    if validation.get("not_all_value"):
        column = str(validation["not_all_value"]["column"])
        value = str(validation["not_all_value"]["value"])
        if rows and all(str(row.get(column, "")) == value for row in rows):
            raise ValueError(f"Semua nilai {path.name}.{column} adalah {value}.")


def validate_resource(entry: dict[str, Any], path: Path) -> None:
    """Memvalidasi resource sesuai tipe dan aturan manifest."""
    if not path.exists():
        raise FileNotFoundError(f"Resource tidak ditemukan: {path}")
    if path.is_file() and path.stat().st_size == 0:
        raise ValueError(f"Resource kosong: {path}")
    resource_type = str(entry["type"])
    validation = entry.get("validation") or {}
    if resource_type == "json":
        validate_json_object(path, validation)
    elif resource_type == "text":
        validate_text(path)
    elif resource_type == "csv":
        validate_csv(path, validation)
    elif resource_type == "directory":
        path.mkdir(parents=True, exist_ok=True)
    else:
        raise ValueError(f"Tipe resource tidak dikenal: {resource_type}")


def setup_resource(
    entry: dict[str, Any],
    *,
    validate_checksum_strict: bool = True,
) -> Path:
    """Menyiapkan satu resource dari entry manifest."""
    path = project_path(str(entry["path"]))
    ensure_resource_available(
        entry,
        path,
        validate_checksum_strict=validate_checksum_strict,
    )
    validate_resource(entry, path)
    if path.is_file() and bool(entry["required"]) and validate_checksum_strict:
        validate_checksum(entry, path, strict=True)
    LOGGER.info("Resource valid: %s", path)
    return path


def update_manifest_checksums(manifest: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Memperbarui checksum manifest hanya pada mode eksplisit."""
    changes: list[tuple[str, str, str]] = []
    for entry in manifest["resources"]:
        path = project_path(str(entry["path"]))
        validate_resource(entry, path)
        if path.is_file():
            checksum = file_sha256(path)
            if entry.get("checksum") != checksum:
                changes.append((str(entry["name"]), str(entry.get("checksum") or ""), checksum))
                entry["checksum"] = checksum
    if changes:
        atomic_write_json(MANIFEST_FILE, manifest)
    return changes


def build_parser() -> argparse.ArgumentParser:
    """Membuat parser argumen command line."""
    parser = argparse.ArgumentParser(description="Validasi resource proyek.")
    parser.add_argument(
        "--update-checksums",
        action="store_true",
        help="Perbarui checksum manifest secara eksplisit setelah resource valid.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Menyiapkan seluruh resource lokal berdasarkan manifest."""
    args = build_parser().parse_args(argv)
    RESOURCES_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    manifest = load_manifest()
    validate_manifest(manifest)
    for entry in manifest["resources"]:
        setup_resource(
            entry,
            validate_checksum_strict=not args.update_checksums,
        )
    if args.update_checksums:
        LOGGER.warning(
            "Mode update checksum aktif. Manifest hanya diperbarui setelah semua resource valid."
        )
        changes = update_manifest_checksums(manifest)
        for name, old_checksum, new_checksum in changes:
            LOGGER.warning(
                "Checksum diperbarui untuk %s: %s -> %s",
                name,
                old_checksum,
                new_checksum,
            )
        status = "checksum manifest diperbarui" if changes else "checksum manifest sudah sesuai"
    else:
        status = "manifest tidak diubah"
    LOGGER.info(
        "Resource berhasil divalidasi; %s. Proyek dapat berjalan offline selama "
        "seluruh resource wajib tersedia secara lokal.",
        status,
    )


if __name__ == "__main__":
    main()
