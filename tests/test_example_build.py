"""Check that disabled examples do not require generated files in the site build."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts import check_examples

ROOT = Path(__file__).resolve().parents[1]
HIDDEN = ("itpa-tae-linear-mhd", "itpa-tae-shear-alfven")


@pytest.fixture
def example_layout(tmp_path):
    docs = tmp_path / "docs"
    for path in ("src/examples", "src/pages/examples", "src/data", "public/examples", "scripts"):
        (docs / path).mkdir(parents=True)
    for slug in ("active", *HIDDEN):
        (docs / "src/examples" / f"{slug}.py").write_text("# Example source\n")
        (docs / "src/pages/examples" / f"{slug}.py.ts").write_text("// Download route\n")
    metadata = {
        "name": "Active", "description": "Published example", "model": "Maxwell",
        "equationsMarkdown": "PDE", "domain": "Cuboid", "steps": 1,
    }
    (docs / "public/examples/active.metadata.json").write_text(json.dumps(metadata))
    (docs / "public/examples/active.plotly.json").write_text("{}")
    (docs / "public/examples/active.html").write_text("<html></html>")
    shutil.copyfile(
        ROOT / "docs/scripts/generate-examples-index.mjs",
        docs / "scripts/generate-examples-index.mjs",
    )
    return docs


def run_checker(monkeypatch, docs, require_figures):
    monkeypatch.setattr(check_examples, "ROOT", docs.parent)
    monkeypatch.setattr(check_examples, "SCRIPTS_DIR", docs / "src/examples")
    monkeypatch.setattr(check_examples, "OUTPUT_DIR", docs / "public/examples")
    monkeypatch.setattr(check_examples, "ROUTES_DIR", docs / "src/pages/examples")
    monkeypatch.setattr("sys.argv", ["check_examples.py"] + (["--require-figures"] if require_figures else []))
    return check_examples.main()


@pytest.mark.parametrize("require_figures", [False, True])
def test_disabled_examples_need_no_generated_files(example_layout, monkeypatch, require_figures):
    assert run_checker(monkeypatch, example_layout, require_figures) == 0


@pytest.mark.parametrize("suffix", ["metadata.json", "plotly.json", "html"])
def test_published_outputs_are_still_required(example_layout, monkeypatch, capsys, suffix):
    (example_layout / "public/examples" / f"active.{suffix}").unlink()
    assert run_checker(monkeypatch, example_layout, True) == 1
    assert f"active.{suffix}" in capsys.readouterr().err


def generate_index(docs):
    return subprocess.run(
        ["node", str(docs / "scripts/generate-examples-index.mjs")],
        capture_output=True, text=True,
    )


def test_index_builds_without_disabled_metadata(example_layout):
    result = generate_index(example_layout)
    assert result.returncode == 0, result.stderr
    entries = json.loads((example_layout / "src/data/examples-index.json").read_text())
    assert [entry["slug"] for entry in entries] == ["active"]


def test_index_requires_published_metadata(example_layout):
    (example_layout / "public/examples/active.metadata.json").unlink()
    result = generate_index(example_layout)
    assert result.returncode == 1
    assert "Missing example metadata for: active" in result.stderr


def test_stale_disabled_metadata_is_ignored(example_layout, monkeypatch):
    for slug in HIDDEN:
        (example_layout / "public/examples" / f"{slug}.metadata.json").write_text("invalid JSON")
    assert run_checker(monkeypatch, example_layout, True) == 0
    result = generate_index(example_layout)
    assert result.returncode == 0, result.stderr
    entries = json.loads((example_layout / "src/data/examples-index.json").read_text())
    assert [entry["slug"] for entry in entries] == ["active"]
