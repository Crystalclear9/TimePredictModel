"""Encoder lifecycle/cache concurrency; the backend is explicitly a test substitute."""

from concurrent.futures import ThreadPoolExecutor
import threading
import time

import numpy as np

from timepredict.semantic import FrozenSemanticEncoder, SemanticEncoder


def test_concurrent_initialization_and_identical_call_cache_are_shared(tmp_path, monkeypatch):
    monkeypatch.setenv("TIMEPREDICT_TEST_ENCODER", "1")
    original_init = FrozenSemanticEncoder.__init__
    initialized = []

    def slow_init(self, *args, **kwargs):
        initialized.append(id(self))
        time.sleep(0.02)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(FrozenSemanticEncoder, "__init__", slow_init)
    with ThreadPoolExecutor(max_workers=8) as pool:
        instances = list(pool.map(lambda _: SemanticEncoder(cache_folder=str(tmp_path)), range(24)))
    assert len({id(e) for e in instances}) == 1 and len(initialized) == 1
    encoder = instances[0]
    original_encode = encoder.model.encode
    calls, lock = [], threading.Lock()

    def slow_encode(call):
        with lock:
            calls.append(call)
        time.sleep(0.005)
        return original_encode(call)

    monkeypatch.setattr(encoder.model, "encode", slow_encode)
    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(lambda _: encoder.encode_text("the same text"), range(32)))
    assert len(calls) == 1
    assert all(np.array_equal(values[0], v) for v in values)
    saved = encoder.encode_text("the same text")
    values[0][:] = 0
    assert np.array_equal(encoder.encode_text("the same text"), saved)
