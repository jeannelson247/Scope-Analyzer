from pathlib import Path

from scope_agent_turn import default_model_for_role, endpoint_url, model_payload_id, parse_args, SYSTEM_PROMPTS


def test_default_models_are_role_specific():
    assert "Coder-14B" in default_model_for_role("coder")
    assert "Qwen3.5-9B" in default_model_for_role("planner")
    assert "Coder-3B" in default_model_for_role("summarizer")


def test_endpoint_url_uses_openai_compatible_v1():
    assert endpoint_url("127.0.0.1", 8091) == "http://127.0.0.1:8091/v1"


def test_model_payload_id_preserves_local_paths(tmp_path):
    model_dir = tmp_path / "Qwen-local"
    model_dir.mkdir()
    assert model_payload_id(str(model_dir)) == str(model_dir)
    assert model_payload_id("mlx-community/example") == "mlx-community/example"


def test_parse_args_requires_task_or_prompt():
    args = parse_args(["--role", "planner", "--task", "audit"])
    assert args.role == "planner"
    assert args.task == "audit"


def test_role_system_prompts_warn_about_scope():
    assert "Do not claim to have edited files" in SYSTEM_PROMPTS["planner"]
    assert "Respect protected scientific modules" in SYSTEM_PROMPTS["coder"]
