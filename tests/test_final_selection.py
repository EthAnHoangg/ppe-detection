from ppe.evaluate import select_final_runs


def _run(id_, status="completed", smoke=False, has_model=True):
    return {"id": id_, "status": status, "smoke": smoke, "has_model": has_model}


def test_completed_runs_with_models_are_selected():
    todo, skipped = select_final_runs([_run("a"), _run("b")], {}, explicit=False)
    assert todo == ["a", "b"] and skipped == []


def test_unfinished_or_modelless_runs_are_skipped_with_reason():
    todo, skipped = select_final_runs([_run("a", status="in_progress"), _run("b", has_model=False)], {}, explicit=False)
    assert todo == []
    assert dict(skipped) == {"a": "not completed (in_progress)", "b": "no output model"}


def test_smoke_runs_need_explicit_selection():
    assert select_final_runs([_run("s", smoke=True)], {}, explicit=False)[0] == []
    assert select_final_runs([_run("s", smoke=True)], {}, explicit=True)[0] == ["s"]


def test_existing_final_eval_blocks_run():
    todo, skipped = select_final_runs([_run("a")], {"a": ["completed"]}, explicit=True)
    assert todo == [] and skipped == [("a", "test set already used for this run (final-eval completed)")]


def test_failed_final_eval_allows_retry():
    assert select_final_runs([_run("a")], {"a": ["failed", "stopped"]}, explicit=False)[0] == ["a"]
