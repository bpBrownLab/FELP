#!/usr/bin/env python3
"""Build a reproducible source archive from the release manifest."""
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "release/jctc2026_manifest.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def release_files(root, manifest):
    """Resolve a closed file selection; fail rather than silently omit inputs."""
    root = Path(root).resolve()
    if manifest.get("schema_version") != 1:
        raise ValueError("Expected release manifest schema_version=1")
    selected = set(manifest["files"])
    for item in manifest.get("trees", []):
        directory = root / item["path"]
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        matches = set()
        for pattern in item["patterns"]:
            matches.update(str(p.relative_to(root)) for p in directory.glob(pattern)
                           if p.is_file() and "__pycache__" not in p.parts)
        if not matches:
            raise ValueError("Empty release tree: " + item["path"])
        selected.update(matches)
    for name in selected:
        relative = Path(name)
        path = root / relative
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Release paths must stay inside the repository: " + name)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError("Missing file or unsupported symlink: " + name)
        path.resolve().relative_to(root)
        if any(part in {".git", "__pycache__", ".pytest_cache"} for part in relative.parts):
            raise ValueError("Build or repository metadata cannot enter the release: " + name)
    return tuple(sorted(selected))


def check_local_imports(root, names):
    """Check static imports of FELP, its selected scripts and test helpers."""
    selected = set(names)
    missing = []
    for name in names:
        if not name.endswith(".py") or name.startswith("ParmEd/"):
            continue
        tree = ast.parse((root / name).read_text(), filename=name)
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    parts = list(Path(name).parent.parts)
                    if node.level > len(parts):
                        continue
                    prefix = parts[:len(parts) - node.level + 1]
                    modules = [".".join(prefix + (node.module.split(".") if node.module else []))]
                elif node.module:
                    modules = [node.module]
            for module in modules:
                if module.split(".")[0] not in {"felp", "scripts", "tests"}:
                    continue
                target = module.replace(".", "/")
                if not (target + ".py" in selected or target + "/__init__.py" in selected
                        or any(p.startswith(target + "/") for p in selected)):
                    missing.append(f"{name}:{node.lineno}: {module}")
    if missing:
        raise ValueError("Unbundled local imports:\n" + "\n".join(sorted(set(missing))))


def build_release(root, manifest, output_dir):
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    names = release_files(root, manifest)
    check_local_imports(root, names)
    hashes = {name: digest(root / name) for name in names}
    package = manifest["package"]
    if not package or Path(package).name != package or package in {".", ".."}:
        raise ValueError("Package name must be a single directory name")
    record = {"schema_version": 1, "package": package, "protocol": "jctc2026",
              "files_sha256": hashes, "scope": manifest["scope"]}
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / (package + ".tar.gz")
    outputs = [archive, output_dir / (package + ".sha256"), output_dir / (package + ".inventory.json")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Release outputs already exist; choose a fresh output directory")

    def add_bytes(tar, name, data, executable=False):
        info = tarfile.TarInfo(package + "/" + name)
        info.size = len(data)
        info.mode = 0o755 if executable else 0o644
        info.mtime = 0
        tar.addfile(info, io.BytesIO(data))

    with archive.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for name in names:
                    path = root / name
                    data = path.read_bytes()
                    if hashlib.sha256(data).hexdigest() != hashes[name]:
                        raise RuntimeError("Source changed during release build: " + name)
                    add_bytes(tar, name, data, bool(path.stat().st_mode & 0o111))
                add_bytes(tar, "SOURCE_MANIFEST.json", (json.dumps(record, indent=2, sort_keys=True) + "\n").encode())
    archive_sha = digest(archive)
    outputs[1].write_text(archive_sha + "  " + archive.name + "\n")
    omitted = {}
    for directory in ("scripts", "docs", "tests"):
        for path in sorted((root / directory).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".md", ".json"}:
                name = str(path.relative_to(root))
                if name not in hashes:
                    omitted[name] = digest(path)
    inventory = {**record, "archive_sha256": archive_sha,
                 "excluded_development_files_sha256": omitted}
    outputs[2].write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    return {"archive": str(archive), "sha256": archive_sha,
            "included_files": len(names), "excluded_development_files": len(omitted),
            "bytes": archive.stat().st_size}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_release(ROOT, json.loads(args.manifest.read_text()), args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
