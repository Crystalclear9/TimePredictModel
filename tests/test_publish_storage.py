"""Injected storage errors must not masquerade as durable publication."""

import errno
import json
from types import SimpleNamespace

import pytest

from timepredict import model, release


def io_failure(*args, **kwargs):
    raise OSError(errno.EIO, "fixture storage failure")


def test_file_sync_failure_is_not_suppressed(tmp_path, monkeypatch):
    path = tmp_path / "model.ubj"
    path.write_bytes(b"fixture")
    monkeypatch.setattr(model.os, "fsync", io_failure)
    with pytest.raises(OSError, match="storage failure"):
        model._fsync_file(path)


def test_posix_directory_sync_failure_is_not_suppressed(monkeypatch):
    closed = []
    monkeypatch.setattr(
        model,
        "os",
        SimpleNamespace(
            name="posix", O_RDONLY=0, open=lambda *args: 7, fsync=io_failure, close=closed.append
        ),
    )
    with pytest.raises(OSError, match="storage failure"):
        model._fsync_directory("fixture-directory")
    assert closed == [7]


@pytest.mark.parametrize("when", ["before_pointer", "after_pointer"])
def test_publication_reports_storage_failure_at_correct_commit_boundary(
    tmp_path, monkeypatch, when
):
    candidate, serving = tmp_path / "candidate", tmp_path / "serving"
    candidate.mkdir()
    serving.mkdir()
    current = serving / "CURRENT"
    current.write_text("old", encoding="utf-8")
    (candidate / "model.ubj").write_bytes(b"fixture")
    (candidate / "report.json").write_text("{}", encoding="utf-8")
    (candidate / "manifest.json").write_text(
        json.dumps({"sha256": {"model.ubj": "fixture", "report.json": "fixture"}}), encoding="utf-8"
    )
    evidence, policy = tmp_path / "evidence.json", tmp_path / "policy.json"
    evidence.write_text("{}", encoding="utf-8")
    policy.write_text("{}", encoding="utf-8")
    # Gate and model validation have full CPU integration tests elsewhere.
    # These fixtures isolate the real copy/fsync/pointer ordering.
    monkeypatch.setattr(
        release, "Model", lambda *args, **kwargs: SimpleNamespace(version="fixture")
    )
    monkeypatch.setattr(release, "evaluate_gate", lambda *args: {"decision": "approved"})
    flushed = []

    def sync_file(path):
        flushed.append(path.name)
        if when == "before_pointer" and path.name == "model.ubj":
            io_failure()

    def sync_directory(path):
        if (
            when == "after_pointer"
            and path == serving
            and current.read_text(encoding="utf-8") != "old"
        ):
            io_failure()

    monkeypatch.setattr(release, "_fsync_file", sync_file, raising=False)
    monkeypatch.setattr(release, "_fsync_directory", sync_directory)
    with pytest.raises(OSError) as error:
        release.promote(candidate, serving, evidence, policy)
    if when == "before_pointer":
        assert current.read_text(encoding="utf-8") == "old"
    else:
        assert type(error.value).__name__ == "PromotionCommitUncertain"
        assert current.read_text(encoding="utf-8") != "old"
        assert {
            "model.ubj",
            "manifest.json",
            "report.json",
            "promotion.json",
            "gate-policy.json",
            "acceptance.json",
        } <= set(flushed)
