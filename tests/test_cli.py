from __future__ import annotations

from pathlib import Path

import pytest

from sudoku_solver import cli


def test_cli_passes_expected_arguments_to_pipeline(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    def fake_run_pipeline(**kwargs: object) -> None:
        calls.update(kwargs)

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    image_path = tmp_path / "board.jpg"
    image_path.write_bytes(b"image-bytes")
    output_dir = tmp_path / "out"

    exit_code = cli.main(
        [
            "solve",
            str(image_path),
            "--out",
            str(output_dir),
            "--stage",
            "warp",
        ]
    )

    assert exit_code == 0
    assert calls["image_path"] == image_path
    assert calls["output_dir"] == output_dir
    assert calls["stop_after_stage"] == "warp"
    assert calls["save_stages"] is True


def test_cli_returns_nonzero_and_reports_pipeline_errors(monkeypatch, capsys) -> None:
    def fake_run_pipeline(**kwargs: object) -> None:
        raise RuntimeError("ocr failed")

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    exit_code = cli.main(["solve", "missing.jpg", "--out", "out"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "ocr failed" in captured.err
