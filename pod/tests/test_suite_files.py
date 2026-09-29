import importlib
import os
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)


def test_mock_items_are_unique_and_counted():
    mock_server = importlib.import_module("mock_server")
    items = mock_server.make_items(12, "t")
    assert len(items) == 12
    assert len({item["id"] for item in items}) == 12
    assert {item["format"] for item in items} == {"ogg"}


def test_mock_summary_counts_records():
    mock_server = importlib.import_module("mock_server")
    state = mock_server.make_state(4, "t", "/tmp/unused.json")
    state["records"]["mock-t-0"] = {
        "id": "mock-t-0",
        "status": "done",
        "url": "https://cdn.example/tts/english/a.ogg",
        "error": None,
    }
    state["records"]["mock-t-1"] = {"id": "mock-t-1", "status": "failed", "url": None, "error": "silent"}
    summary = mock_server.summarize(state)
    assert summary["done"] == 1
    assert summary["failed"] == 1
    assert summary["failure_reasons"] == {"silent": 1}


def test_mock_tracks_how_results_arrive():
    mock_server = importlib.import_module("mock_server")
    state = mock_server.make_state(4, "t", "/tmp/unused.json")
    state["progress_log"] = [
        {"ts": 10.0, "count": 5, "done": 5},
        {"ts": 12.0, "count": 5, "done": 4},
        {"ts": 15.0, "count": 1, "done": 0},
    ]
    stats = mock_server.incremental_stats(state)
    assert stats["progress_calls"] == 2
    assert stats["first_done_batch"] == 5
    assert stats["done_reported"] == 9
    assert stats["spread_seconds"] == 2.0


def test_suite_env_helpers(monkeypatch):
    run_suite = importlib.import_module("run_suite")
    monkeypatch.delenv("SUITE_TEST_VALUE", raising=False)
    assert run_suite.env_int("SUITE_TEST_VALUE", 5) == 5
    monkeypatch.setenv("SUITE_TEST_VALUE", "9")
    assert run_suite.env_int("SUITE_TEST_VALUE", 5) == 9
    assert run_suite.env_str("SUITE_TEST_VALUE", "x") == "9"


def test_lump_at_the_end_is_flagged_as_not_incremental():
    run_suite = importlib.import_module("run_suite")
    lump = {"progress_calls": 2, "first_done_batch": 100, "done_reported": 200}
    healthy = {"progress_calls": 40, "first_done_batch": 6, "done_reported": 200}
    assert not run_suite.progress_is_incremental(lump, 200)
    assert run_suite.progress_is_incremental(healthy, 200)
    assert run_suite.progress_is_incremental({}, 10)  # too small to judge


def test_cost_projection_adds_startup_once():
    run_suite = importlib.import_module("run_suite")
    stats = {"startup_seconds": 180.0, "marginal_seconds_per_clip": 0.5}
    p = run_suite.project_daily_cost(stats, 0.36, 2000, 2)
    assert p["batch_seconds"] == 1180.0
    assert p["cost_per_batch_usd"] == round(1180 / 3600 * 0.36, 4)
    assert p["cost_per_day_usd"] == round(p["cost_per_batch_usd"] * 2, 4)
    assert run_suite.project_daily_cost({}, 0.36, 2000, 2) is None 