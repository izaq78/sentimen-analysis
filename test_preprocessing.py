"""Pengujian preprocessing dan audit hardcode resource."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path
import py_compile
import pemodelan
import clean_project
import setup_resources
from project_config import BASE_DIR, discover_dataset_files, label_token


RESOURCE_DIR = BASE_DIR / "resources"
MANIFEST_FILE = RESOURCE_DIR / "resource_manifest.json"
BOM_BYTES = b"\xef\xbb\xbf"


def excluded_parts() -> set[str]:
    """Folder yang bukan source/distribusi aktif untuk audit."""
    return {
        ".venv",
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".vscode",
        ".agents",
        "dataset_asli",
        "old",
        "backup",
    }


def project_python_files() -> list[Path]:
    """Mengambil seluruh file Python aktif di luar arsip, cache, dan virtualenv."""
    return sorted(
        path
        for path in BASE_DIR.rglob("*.py")
        if path.is_file()
        and not excluded_parts().intersection(path.relative_to(BASE_DIR).parts)
    )


def active_text_files() -> list[Path]:
    """Mengambil file teks aktif yang wajib bebas BOM."""
    suffixes = {".py", ".json", ".txt", ".md", ".csv"}
    result: list[Path] = []
    for path in BASE_DIR.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(BASE_DIR).parts
        if excluded_parts().intersection(relative_parts):
            continue
        if path.name == "requirements.txt" or path.suffix.lower() in suffixes:
            result.append(path)
    return sorted(result)


def manifest_entries() -> list[dict[str, object]]:
    """Membaca daftar entry manifest."""
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    return list(manifest["resources"])


def required_file_entries() -> list[dict[str, object]]:
    """Entry resource wajib yang berupa file."""
    return [
        entry
        for entry in manifest_entries()
        if entry.get("required") is True and entry.get("type") != "directory"
    ]


def sha256(path: Path) -> str:
    """Menghitung checksum SHA256."""
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cleaned_tokens(text: str) -> set[str]:
    """Mengambil token hasil preprocessing."""
    return set(pemodelan.clean_text(text).split())


def slang_pair() -> tuple[str, str]:
    """Mengambil contoh slang dari resource."""
    slang_map = pemodelan.load_slang_map()
    source, target = next(iter(slang_map.items()))
    return source, target


def emoticon_for(sentiment: str) -> str:
    """Mengambil emoticon dari resource CSV berdasarkan sentimen."""
    with (RESOURCE_DIR / "emoticon_sentiment.csv").open(newline="", encoding="utf-8") as file_obj:
        for row in csv.DictReader(file_obj):
            if row["sentiment"] == sentiment:
                return row["emoticon"]
    raise AssertionError(f"Emoticon dengan sentimen {sentiment} tidak tersedia.")


def test_leetspeak_repeat_dan_emoji_berdasarkan_resource() -> None:
    """Leetspeak dan huruf berulang harus kembali ke kata kamus."""
    tokens = cleaned_tokens("barangnya b4gusss \U0001f60d")
    assert "bagus" in tokens
    assert "b4gusss" not in tokens
    assert label_token("positive") in tokens


def test_slang_dan_negasi_mengikuti_resource() -> None:
    """Slang dan negasi harus berasal dari resource."""
    slang_source, slang_target = slang_pair()
    tokens = cleaned_tokens(f"produk ini {slang_source} bgus \U0001f621")
    assert f"{slang_target}_bagus" in tokens or slang_target in tokens
    assert label_token("negative") in tokens


def test_typo_repeat_dan_emoticon_resource() -> None:
    """Typo, huruf berulang, dan emoticon mengikuti resource."""
    bad_emoticon = emoticon_for(label_token("negative"))
    tokens = cleaned_tokens(f"pengrimannya lamaaaaa {bad_emoticon}")
    assert "pengirimannya" in tokens or "kirim" in tokens
    assert "lama" in tokens
    assert label_token("negative") in tokens


def test_case_punctuation_dan_emoji_hati() -> None:
    """Emoji harus diproses sebelum tanda baca dibersihkan."""
    tokens = cleaned_tokens("BARANGNYA BAGUS BANGET!!! \u2764\ufe0f")
    assert "bagus" in tokens
    assert label_token("positive") in tokens


def test_negasi_tidak_hilang() -> None:
    """Kata negasi dari resource harus digabung."""
    tokens = cleaned_tokens("tidak sesuai deskripsi")
    assert "tidak_sesuai" in tokens


def test_emoticon_diproses_sebelum_simbol_dibersihkan() -> None:
    """Emoticon berlabel harus terbaca sebelum karakter simbol dibersihkan."""
    good_emoticon = emoticon_for(label_token("positive"))
    tokens = cleaned_tokens(f"mantappp bgt {good_emoticon}")
    assert "mantap" in tokens
    assert label_token("positive") in tokens


def test_url_mention_dan_hashtag() -> None:
    """URL dan mention dibuang, isi hashtag dipertahankan."""
    result = pemodelan.clean_text("cek di https://contoh.com @toko #barangbagus")
    tokens = set(result.split())
    assert "contoh" not in tokens
    assert "toko" not in tokens
    assert "barangbagus" in tokens or "barang" in tokens or "bagus" in tokens


def test_dataset_backup_diabaikan_oleh_regex_config() -> None:
    """Dataset discovery harus mengabaikan nama CSV non-dataset."""
    temporary_files = {
        "data_backup.csv",
        "database.csv",
        "dataset.csv",
    }
    created: list[Path] = []
    for name in temporary_files:
        path = BASE_DIR / name
        if not path.exists():
            path.write_text("text,sentiment\ncontoh,1\n", encoding="utf-8")
            created.append(path)
    try:
        discovered = {path.name for path in discover_dataset_files()}
        assert not temporary_files.intersection(discovered)
        assert all(path_name.startswith("data") for path_name in discovered)
        assert all(
            path_name.removeprefix("data").removesuffix(".csv").isdigit()
            for path_name in discovered
        )
    finally:
        for path in created:
            if path.exists():
                path.unlink()


def test_seluruh_file_teks_aktif_tanpa_bom() -> None:
    """File teks aktif tidak boleh diawali BOM UTF-8."""
    for path in active_text_files():
        assert not path.read_bytes().startswith(BOM_BYTES), (
            f"BOM ditemukan pada {path.relative_to(BASE_DIR)}"
        )


def test_compile_seluruh_source_aktif() -> None:
    """Semua source Python aktif harus lolos py_compile."""
    for path in project_python_files():
        py_compile.compile(str(path), doraise=True)


def test_tidak_ada_nama_variabel_terlarang_di_semua_source() -> None:
    """Source aktif tidak boleh memuat nama hardcode terlarang."""
    forbidden = [
        "DE" + "FAULT_",
        "FALL" + "BACK_",
        "POS" + "ITIVE_",
        "NEG" + "ATIVE_",
        "NEGATION" + "_WORDS",
        "STOP" + "WORDS",
        "CUSTOM" + "_STOP" + "WORDS",
        "EMOTICON" + "_FALLBACK",
        "DEFAULT" + "_SLANG",
        "DEFAULT" + "_LEET_RULES",
        "POSITIVE" + "_EMOJI_WORDS",
        "NEGATIVE" + "_EMOJI_WORDS",
        "SPECIAL" + "_TOKENS",
    ]
    for path in project_python_files():
        source = path.read_text(encoding="utf-8")
        for term in forbidden:
            assert term not in source, f"{term} masih ada di {path.name}"


def collection_size(node: ast.AST) -> int:
    """Menghitung ukuran literal collection AST."""
    if isinstance(node, ast.Dict):
        return len(node.keys)
    if isinstance(node, (ast.Set, ast.List, ast.Tuple)):
        return len(node.elts)
    return 0


def assignment_names(node: ast.AST) -> list[str]:
    """Mengambil target assignment yang sederhana."""
    targets: list[ast.AST] = []
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]

    names: list[str] = []
    for target in targets:
        if isinstance(target, ast.Name):
            names.append(target.id.lower())
        elif isinstance(target, ast.Attribute):
            names.append(target.attr.lower())
    return names


def test_tidak_ada_koleksi_kebahasaan_manual_di_semua_source() -> None:
    """Collection besar hanya dilarang untuk target kebahasaan/sentimen."""
    suspicious_name_parts = (
        "slang",
        "leet",
        "stopword",
        "negation",
        "emoticon",
        "emoji",
        "sentiment_keyword",
        "typo",
        "dictionary",
    )
    for path in project_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            names = assignment_names(node)
            value = node.value
            if not isinstance(value, (ast.Dict, ast.Set, ast.List, ast.Tuple)):
                continue
            if any(part in name for name in names for part in suspicious_name_parts):
                assert collection_size(value) == 0, (
                    f"Koleksi kebahasaan manual ditemukan di {path.name}: {names}"
                )


def test_threshold_tidak_disimpan_sebagai_literal_source() -> None:
    """Threshold teknis harus berasal dari konfigurasi resource."""
    for path in project_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            names = assignment_names(node)
            if not any("threshold" in name or "confidence" in name for name in names):
                continue
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int, float)):
                raise AssertionError(
                    f"Threshold literal ditemukan di {path.name}: {names}"
                )


def test_resource_manifest_tanpa_generator_kosong() -> None:
    """Manifest tidak boleh memakai generator kosong atau CSV kosong."""
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    required_keys = {
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
    for entry in manifest["resources"]:
        assert required_keys.issubset(entry), entry.get("name")
        assert entry["checksum_algorithm"] == "sha256", entry["name"]
        assert entry["distribution"] in {"bundled", "downloadable"}, entry["name"]
        assert entry.get("generator") != "empty" + "_csv", entry["name"]
        assert entry.get("generator") != "emoticon" + "_sentiment_neutral", entry["name"]
        assert entry["distribution"] != "generated", entry["name"]
        if entry["type"] == "csv":
            validation = entry.get("validation") or {}
            assert validation.get("non_empty") is True, entry["name"]
    emoji_entry = next(
        entry for entry in manifest["resources"] if entry["name"] == "emoji_sentiment"
    )
    assert emoji_entry["required"] is True
    assert emoji_entry["source_url"]
    assert emoji_entry["checksum"]


def test_resource_wajib_tersedia_atau_punya_builder() -> None:
    """Resource wajib harus tersedia lokal atau punya source/generator."""
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    for entry in manifest["resources"]:
        if not entry.get("required"):
            continue
        path = BASE_DIR / entry["path"]
        assert path.exists() or entry.get("source_url") or entry.get("generator"), entry["name"]


def test_setup_resource_idempotent_dan_checksum_valid() -> None:
    """Setup resource dua kali tidak boleh mengubah manifest setelah sinkron."""
    before_resources = {
        str(entry["path"]): sha256(BASE_DIR / str(entry["path"]))
        for entry in required_file_entries()
    }
    before_manifest = setup_resources.file_sha256(MANIFEST_FILE)
    setup_resources.main([])
    first_checksum = setup_resources.file_sha256(MANIFEST_FILE)
    setup_resources.main([])
    second_checksum = setup_resources.file_sha256(MANIFEST_FILE)
    after_resources = {
        str(entry["path"]): sha256(BASE_DIR / str(entry["path"]))
        for entry in required_file_entries()
    }
    assert before_manifest == first_checksum == second_checksum
    assert before_resources == after_resources


def test_checksum_seluruh_resource_wajib_cocok() -> None:
    """Seluruh resource wajib harus tersedia, tidak kosong, dan checksum cocok."""
    for entry in manifest_entries():
        assert entry["checksum_algorithm"] == "sha256", entry["name"]
        if entry.get("required") is not True:
            continue
        path = BASE_DIR / str(entry["path"])
        assert path.exists(), entry["name"]
        if entry["type"] == "directory":
            assert entry.get("checksum", "") == "", entry["name"]
            continue
        assert path.stat().st_size > 0, entry["name"]
        assert entry.get("checksum"), entry["name"]
        assert sha256(path) == entry["checksum"], entry["name"]


def test_checksum_mismatch_gagal_dan_manifest_tidak_berubah(tmp_path, monkeypatch) -> None:
    """Checksum mismatch pada manifest temporary harus gagal tanpa rewrite manifest."""
    resource = tmp_path / "sample.txt"
    resource.write_text("resource valid\n", encoding="utf-8")
    checksum = sha256(resource)
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "resources": [
            {
                "name": "sample",
                "path": "sample.txt",
                "type": "text",
                "required": True,
                "source_url": "",
                "generator": "",
                "checksum": checksum,
                "checksum_algorithm": "sha256",
                "license": "project",
                "source_name": "test",
                "validation": {"non_empty": True},
                "version": "1.0.0",
                "updated_at": "2026-07-17",
                "distribution": "bundled",
            }
        ]
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    before_manifest = sha256(manifest_path)
    resource.write_text("resource rusak\n", encoding="utf-8")

    monkeypatch.setattr(setup_resources, "BASE_DIR", tmp_path)
    monkeypatch.setattr(setup_resources, "MANIFEST_FILE", manifest_path)
    loaded = setup_resources.load_manifest(manifest_path)
    try:
        setup_resources.validate_manifest(loaded)
        setup_resources.setup_resource(loaded["resources"][0])
    except ValueError as exc:
        assert "Checksum tidak cocok" in str(exc)
    else:
        raise AssertionError("Checksum mismatch tidak menggagalkan setup.")
    assert before_manifest == sha256(manifest_path)


def write_manifest(path: Path, resource_name: str, resource_path: str, resource_type: str, checksum: str, validation: dict[str, object]) -> None:
    """Menulis manifest minimal untuk test resource sementara."""
    manifest = {
        "resources": [
            {
                "name": resource_name,
                "path": resource_path,
                "type": resource_type,
                "required": True,
                "source_url": "",
                "generator": "",
                "checksum": checksum,
                "checksum_algorithm": "sha256",
                "license": "project",
                "source_name": "test",
                "validation": validation,
                "version": "1.0.0",
                "updated_at": "2026-07-18",
                "distribution": "bundled",
            }
        ]
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def test_update_checksums_memperbarui_manifest_valid(tmp_path, monkeypatch) -> None:
    """Mode update checksum harus melewati checksum lama tetapi tetap validasi schema."""
    resource = tmp_path / "sample.txt"
    resource.write_text("resource valid\n", encoding="utf-8")
    manifest_path = tmp_path / "resource_manifest.json"
    write_manifest(
        manifest_path,
        "sample",
        "sample.txt",
        "text",
        sha256(resource),
        {"non_empty": True},
    )
    resource.write_text("resource valid berubah\n", encoding="utf-8")
    new_checksum = sha256(resource)

    monkeypatch.setattr(setup_resources, "BASE_DIR", tmp_path)
    monkeypatch.setattr(setup_resources, "RESOURCES_DIR", tmp_path)
    monkeypatch.setattr(setup_resources, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(setup_resources, "MANIFEST_FILE", manifest_path)

    setup_resources.main(["--update-checksums"])
    updated = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert updated["resources"][0]["checksum"] == new_checksum
    setup_resources.main([])


def test_update_checksums_menolak_resource_rusak(tmp_path, monkeypatch) -> None:
    """Mode update tidak boleh menerima JSON rusak, CSV header-only, TXT kosong, atau file hilang."""
    cases = [
        ("bad.json", "json", "{", {"non_empty": True}),
        ("empty.csv", "csv", "name,value\n", {"columns": ["name", "value"], "non_empty": True}),
        ("empty.txt", "text", "", {"non_empty": True}),
        ("missing.txt", "text", None, {"non_empty": True}),
        ("marker.txt", "text", "PLACE" + "HOLDER\n", {"non_empty": True}),
    ]
    for file_name, resource_type, content, validation in cases:
        case_dir = tmp_path / file_name.replace(".", "_")
        case_dir.mkdir()
        manifest_path = case_dir / "resource_manifest.json"
        resource = case_dir / file_name
        if content is not None:
            resource.write_text(content, encoding="utf-8")
            checksum = sha256(resource)
        else:
            checksum = "0" * 64
        write_manifest(
            manifest_path,
            "sample",
            file_name,
            resource_type,
            checksum,
            validation,
        )
        monkeypatch.setattr(setup_resources, "BASE_DIR", case_dir)
        monkeypatch.setattr(setup_resources, "RESOURCES_DIR", case_dir)
        monkeypatch.setattr(setup_resources, "CACHE_DIR", case_dir)
        monkeypatch.setattr(setup_resources, "MANIFEST_FILE", manifest_path)
        before_manifest = sha256(manifest_path)
        try:
            setup_resources.main(["--update-checksums"])
        except (ValueError, FileNotFoundError, json.JSONDecodeError):
            assert before_manifest == sha256(manifest_path)
        else:
            raise AssertionError(f"Resource rusak diterima pada mode update: {file_name}")


def test_resource_emoji_dan_emoticon_berisi_label_bermakna() -> None:
    """Resource emoji/emoticon wajib non-kosong dan emoticon tidak semuanya netral."""
    emoji_path = BASE_DIR / "cache" / "emoji_sentiment.csv"
    assert emoji_path.exists()
    with emoji_path.open(newline="", encoding="utf-8") as file_obj:
        emoji_rows = list(csv.DictReader(file_obj))
    assert emoji_rows
    assert {"Emoji", "Negative", "Neutral", "Positive"}.issubset(emoji_rows[0])

    with (RESOURCE_DIR / "emoticon_sentiment.csv").open(newline="", encoding="utf-8") as file_obj:
        emoticon_rows = list(csv.DictReader(file_obj))
    sentiments = {row["sentiment"] for row in emoticon_rows}
    assert emoticon_rows
    assert label_token("positive") in sentiments
    assert label_token("negative") in sentiments
    assert sentiments != {label_token("neutral")}


def test_requirements_sesuai_dependency_proyek() -> None:
    """requirements.txt harus sinkron dengan dependency import map proyek."""
    requirements = {
        line.split("#", maxsplit=1)[0].strip().split("<", maxsplit=1)[0].split(">", maxsplit=1)[0]
        for line in (BASE_DIR / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    import_map = json.loads((RESOURCE_DIR / "dependency_import_map.json").read_text(encoding="utf-8"))
    assert "nltk" not in requirements
    assert requirements == set(import_map)
    for line in (BASE_DIR / "requirements.txt").read_text(encoding="utf-8").splitlines():
        cleaned = line.strip()
        if cleaned and not cleaned.startswith("#"):
            assert ">=" in cleaned and "<" in cleaned, cleaned


def test_urutan_emoji_emoticon_sebelum_pembersihan_simbol() -> None:
    """clean_text harus memproses emoticon/emoji sebelum simbol dibersihkan."""
    source = (BASE_DIR / "pemodelan.py").read_text(encoding="utf-8")
    assert source.index("convert_emoticons(raw_text)") < source.index("clean_structure(converted)")
    assert source.index("convert_emojis(convert_emoticons(raw_text))") < source.index("clean_structure(converted)")


def test_label_conflict_ditolak_sebelum_training() -> None:
    """Dataset dengan teks sama tetapi label berbeda harus ditolak."""
    data = pemodelan.pd.DataFrame(
        {
            pemodelan.TEXT_COLUMN: ["bagus", "bagus", "buruk"],
            pemodelan.SENTIMENT_COLUMN: [1, 0, 0],
        }
    )
    normalized = pemodelan.normalize_labels(data)
    try:
        pemodelan.resolve_duplicate_texts(normalized)
    except ValueError as exc:
        assert "konflik label" in str(exc)
    else:
        raise AssertionError("Konflik label tidak ditolak.")


def test_app_evaluasi_visualisasi_tidak_punya_preprocessing_manual() -> None:
    """Modul UI/evaluasi/visualisasi tidak boleh membuat cleaning sendiri."""
    forbidden_calls = ("re.sub", "word_tokenize", "StemmerFactory")
    checked_files = [
        BASE_DIR / "app.py",
        BASE_DIR / "evaluasi.py",
        BASE_DIR / "visualisasi.py",
    ]
    for path in checked_files:
        source = path.read_text(encoding="utf-8")
        assert "preprocess_text(" in source
        for call in forbidden_calls:
            assert call not in source, f"Preprocessing manual {call} ditemukan di {path.name}"


def test_tidak_ada_marker_dan_error_ditelan_di_source_aktif() -> None:
    """Source aktif tidak boleh menyisakan marker kerja atau except pass."""
    forbidden = ("TO" + "DO", "FIX" + "ME", "place" + "holder", "except ValueError:\n            pass")
    for path in project_python_files():
        source = path.read_text(encoding="utf-8")
        for marker in forbidden:
            assert marker not in source, f"Marker terlarang ditemukan di {path.name}: {marker}"


def test_clean_project_mencakup_cache_pengembangan() -> None:
    """clean_project.py harus mencakup folder/file pengembangan distribusi."""
    forbidden_dirs = {".venv", ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".vscode", ".agents"}
    assert forbidden_dirs.issubset(clean_project.DEVELOPMENT_DIRECTORIES)
    assert "dataset_asli" in clean_project.DEVELOPMENT_DIRECTORIES
    assert {".pyc", ".pyo", ".tmp", ".bak", ".log"}.issubset(clean_project.DEVELOPMENT_SUFFIXES)
    assert {".csv", ".pkl", ".json"}.issubset(clean_project.PROTECTED_SUFFIXES)


def test_clean_project_tmp_path_aman_dan_idempotent(tmp_path, monkeypatch) -> None:
    """clean_project.py harus menghapus cache/lama dan menjaga artefak aktif."""
    for dirname in [".venv", ".git", "__pycache__", "dataset_asli"]:
        directory = tmp_path / dirname
        directory.mkdir()
        (directory / "old.pyc").write_bytes(b"cache")
    (tmp_path / "old.bak").write_text("backup", encoding="utf-8")
    (tmp_path / "resources").mkdir()
    (tmp_path / "cache").mkdir()
    (tmp_path / "resources" / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "cache" / "emoji_sentiment.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (tmp_path / "data1.csv").write_text("text,sentiment\nbagus,1\n", encoding="utf-8")
    (tmp_path / "model_sentimen.pkl").write_bytes(b"model")
    (tmp_path / "app.py").write_text("print('ok')\n", encoding="utf-8")

    monkeypatch.setattr(clean_project, "BASE_DIR", tmp_path)
    first = clean_project.clean_project()
    second = clean_project.clean_project()
    assert first
    assert second == []
    for dirname in [".venv", ".git", "__pycache__", "dataset_asli"]:
        assert not (tmp_path / dirname).exists()
    assert not (tmp_path / "old.bak").exists()
    assert (tmp_path / "resources" / "config.json").exists()
    assert (tmp_path / "cache" / "emoji_sentiment.csv").exists()
    assert (tmp_path / "data1.csv").exists()
    assert (tmp_path / "model_sentimen.pkl").exists()
    assert (tmp_path / "app.py").exists()
