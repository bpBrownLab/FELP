"""Release boundary, reproducibility, dependency closure and overwrite checks."""
import json
from pathlib import Path
import tarfile

import pytest

from scripts.build_release import ROOT, DEFAULT_MANIFEST, build_release, check_local_imports, release_files


def test_source_manifest_contains_runtime_and_only_source_directories():
    manifest = json.loads(DEFAULT_MANIFEST.read_text())
    names = release_files(ROOT, manifest)
    check_local_imports(ROOT, names)
    assert "felp/diagnostics/harmonic.py" in names
    assert "scripts/manuscript_uniform_matrix.py" in names
    allowed = {"felp", "ParmEd", "scripts", "tests", "docs", "examples", "release"}
    assert all(len(Path(name).parts) == 1 or Path(name).parts[0] in allowed for name in names)
    assert "felp/sidechain_ensemble/sampling.py" in names
    assert "tests/factories.py" in names


def test_deterministic_archive_and_source_manifest(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("A source fixture.\n")
    manifest = {"schema_version": 1, "package": "fixture", "files": ["README.md"], "scope": "test"}
    first = build_release(source, manifest, tmp_path / "a")
    second = build_release(source, manifest, tmp_path / "b")
    assert first["sha256"] == second["sha256"]
    with tarfile.open(first["archive"]) as archive:
        content = json.load(archive.extractfile("fixture/SOURCE_MANIFEST.json"))
        assert set(content["files_sha256"]) == {"README.md"}
        assert all(m.uid == 0 and m.mtime == 0 for m in archive.getmembers())
    with pytest.raises(FileExistsError):
        build_release(source, manifest, tmp_path / "a")


def test_missing_files_and_missing_local_imports_fail(tmp_path):
    with pytest.raises(FileNotFoundError):
        release_files(tmp_path, {"schema_version": 1, "files": ["missing.py"]})
    (tmp_path / "entry.py").write_text("from scripts.omitted import method\n")
    with pytest.raises(ValueError, match="Unbundled local imports"):
        check_local_imports(tmp_path, ("entry.py",))
