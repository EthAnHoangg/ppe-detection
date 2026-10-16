import pytest

from ppe.tracking import check_credentials, run_identity


def _cfg(smoke=False):
    return {
        "project": "PPE-Detection",
        "model": {"name": "yolo26s"},
        "dataset": {"version": "v1"},
        "seed": 2,
        "smoke": {"enabled": smoke},
    }


def test_run_identity_names_and_tags_a_normal_run():
    project, name, tags = run_identity(_cfg())
    assert project == "PPE-Detection"
    assert name == "yolo26s · seed2"
    assert tags == ["model:yolo26s", "data:v1", "seed:2"]


def test_smoke_runs_go_to_smoke_subproject_with_smoke_tag():
    project, _, tags = run_identity(_cfg(smoke=True))
    assert project == "PPE-Detection/smoke"
    assert "smoke" in tags


def test_missing_keys_are_named_in_the_error():
    with pytest.raises(RuntimeError) as err:
        check_credentials({"CLEARML_API_ACCESS_KEY": "x"}, offline=False)
    assert "CLEARML_API_SECRET_KEY" in str(err.value)
    assert "CLEARML_API_ACCESS_KEY" not in str(err.value).split("missing:")[1]
    assert ".env.example" in str(err.value)


def test_present_keys_pass():
    check_credentials({"CLEARML_API_ACCESS_KEY": "x", "CLEARML_API_SECRET_KEY": "y"}, offline=False)


def test_offline_mode_skips_key_check():
    check_credentials({}, offline=True)
