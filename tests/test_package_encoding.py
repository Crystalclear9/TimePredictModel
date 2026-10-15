"""UTF-8 packages must load unchanged on hosts with a different locale."""

from pathlib import Path

import numpy as np
import pytest

from timepredict.model import Model
from timepredict.training import train


@pytest.fixture(scope="module")
def multilingual_package(tmp_path_factory):
    call = {"name": "检索📚", "description": "搜索文档", "arguments": {"数量": 3, "query": "缓存"}}
    rows = [
        {
            "generation": f"g{i}",
            "workflow": f"w{i}",
            "workflow_start": float(i),
            "workflow_end": float(i) + 0.1,
            "call": call,
            "lower": 2.0 + i % 3,
            "upper": 2.0 + i % 3,
        }
        for i in range(12)
    ]
    root = tmp_path_factory.mktemp("utf8-package")
    version, _ = train(rows, root, threads=1, rounds=2, sigmas=(0.5,), data_kind="synthetic")
    return root / version, call


@pytest.mark.parametrize("default_encoding", ["cp936", "cp1252"])
def test_multilingual_package_roundtrip_without_utf8_locale(
    multilingual_package, monkeypatch, default_encoding
):
    directory, call = multilingual_package
    read_text = Path.read_text

    def locale_read(path, encoding=None, errors=None):
        return read_text(path, encoding=encoding or default_encoding, errors=errors)

    # Emulate non-UTF-8 Windows defaults, including on UTF-8 CI hosts.
    monkeypatch.setattr(Path, "read_text", locale_read)
    loaded = Model(directory)
    assert loaded.spec.tools == [call["name"]]
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, encoding=None, errors=None: read_text(
            path, encoding=encoding or "utf-8", errors=errors
        ),
    )
    expected = Model(directory)
    np.testing.assert_array_equal(loaded.spec.transform(call), expected.spec.transform(call))
    actual, status = loaded.predict(call)
    assert status == "ready" and actual["unknown_tool"] is False
    assert actual == expected.predict(call)[0]
