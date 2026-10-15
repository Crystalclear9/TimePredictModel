"""Long-lived notification consumers keep only live timing metadata."""
import math

from timepredict.client import WindowConsumer


def update(generation, revision, action="candidate_update"):
    return {"action": action, "key": "k", "generation": generation, "revision": revision, "H_seconds": 2.,
            "candidate": {"key": "k", "generation": generation, "revision": revision,
                          "status": "ready", "in_window": True, "age_seconds": 2., "valid_for_seconds": 1.,
                          "distribution": {"mu": math.log(3), "sigma": .5, "distribution_calibrated": True}}}


def test_replaced_and_removed_rounds_release_timing_metadata_without_a_read():
    consumer = WindowConsumer(lambda: 0.)
    for revision in range(1, 1001):
        consumer.notice(update(str(revision), revision))
    assert len(consumer.received) == 1
    consumer.notice(update("1000", 1001, "remove"))
    assert not consumer.received and not consumer.cache.records
    # Version fences are per business key, not per historical generation.
    indexes = [v for k, v in vars(consumer.cache).items() if isinstance(v, dict) and k != "records"]
    assert sum(map(len, indexes)) == 1


def test_matching_remove_is_not_hidden_by_a_newer_other_generation_tombstone():
    consumer = WindowConsumer(lambda: 0.)
    consumer.notice(update("current", 10))
    consumer.notice(update("previous", 30, "remove"))
    assert consumer.candidates()[0]["generation"] == "current"
    consumer.notice(update("current", 20, "remove"))
    assert not consumer.candidates()
    consumer.notice(update("current", 25))
    assert not consumer.candidates()


def test_delayed_and_duplicate_notices_do_not_extend_current_lease():
    now = [0.]
    consumer = WindowConsumer(lambda: now[0])
    consumer.notice(update("current", 20))
    now[0] = .75
    consumer.notice(update("old", 15))
    consumer.notice(update("current", 19, "remove"))
    consumer.notice(update("current", 20))
    assert consumer.candidates()[0]["valid_for_seconds"] == .25
    now[0] = 1.1
    assert not consumer.candidates()


def test_internal_update_fences_delayed_candidate_without_creating_timing_state():
    consumer = WindowConsumer(lambda: 0.)
    consumer.notice(update("g", 20, "update"))
    consumer.notice(update("g", 15))
    assert not consumer.candidates() and not consumer.received


def test_snapshot_reclaims_version_fences_for_finished_business_keys():
    consumer = WindowConsumer(lambda: 0.)
    consumer.notice(update("g", 20, "remove"))
    consumer.snapshot({"revision": 20, "snapshot_utc": 100., "H_seconds": 2., "records": []})
    assert not consumer.cache.key_revisions
    consumer.notice(update("g", 19))
    assert not consumer.candidates()
