"""Adversarial cases found during the second implementation review."""
import math

from timepredict.client import WindowConsumer


def candidate(generation="old", revision=10):
    return {"key": "k", "generation": generation, "revision": revision, "status": "ready", "in_window": True,
            "age_seconds": 2., "valid_for_seconds": 1.,
            "distribution": {"mu": math.log(3), "sigma": .5, "distribution_calibrated": True}}


def test_replayed_snapshot_cannot_extend_or_revive_a_lease():
    now = [0.]
    consumer = WindowConsumer(lambda: now[0])
    snapshot = {"revision": 10, "snapshot_utc": 100., "H_seconds": 2., "records": [candidate()]}
    consumer.snapshot(snapshot)
    now[0] = .5
    consumer.snapshot(snapshot)
    assert consumer.candidates()[0]["valid_for_seconds"] == .5
    now[0] = 2.
    assert not consumer.candidates()
    consumer.snapshot(snapshot)
    assert not consumer.candidates()


def test_newer_generation_tombstone_fences_delayed_older_generation():
    consumer = WindowConsumer(lambda: 0.)
    consumer.snapshot({"revision": 10, "snapshot_utc": 100., "H_seconds": 2., "records": []})
    consumer.notice({"action": "remove", "key": "k", "generation": "new", "revision": 20})
    consumer.notice({"action": "candidate_update", "key": "k", "generation": "old", "revision": 15,
                     "H_seconds": 2., "candidate": candidate(revision=15)})
    assert not consumer.candidates()


def test_fresh_same_revision_snapshot_can_advance_age_without_extending_expiry():
    now = [0.]
    consumer = WindowConsumer(lambda: now[0])
    consumer.snapshot({"revision": 10, "snapshot_utc": 100., "H_seconds": 2., "records": [candidate()]})
    now[0] = .5
    row = {**candidate(), "age_seconds": 2.5, "valid_for_seconds": .5}
    consumer.snapshot({"revision": 10, "snapshot_utc": 100.5, "H_seconds": 2., "records": [row]})
    assert consumer.candidates()[0]["age_seconds"] == 2.5
    now[0] = 1.1
    assert not consumer.candidates()


def test_new_revision_is_accepted_when_clock_resolution_gives_equal_timestamp():
    consumer = WindowConsumer(lambda: 0.)
    consumer.snapshot({"revision": 10, "snapshot_utc": 100., "H_seconds": 2., "records": [candidate()]})
    consumer.snapshot({"revision": 11, "snapshot_utc": 100., "H_seconds": 2., "records": []})
    assert not consumer.candidates()


def test_authoritative_new_revision_removes_candidate_despite_replica_clock_skew():
    consumer = WindowConsumer(lambda: 0.)
    consumer.snapshot({"revision": 10, "snapshot_utc": 100., "H_seconds": 2., "records": [candidate()]})
    assert consumer.candidates()
    consumer.snapshot({"revision": 11, "snapshot_utc": 99.999, "H_seconds": 2., "records": []})
    assert not consumer.candidates()
