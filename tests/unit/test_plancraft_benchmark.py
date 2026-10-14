"""The Plancraft benchmark class: registration, suites, observation split and the runtime notes."""

from __future__ import annotations

from wikitrace.config import DatasetConfig
from wikitrace.envs import resolve, train_group_for
from wikitrace.envs.textgame.models import TextGameResult
from wikitrace.harness.prompts import render_template

EXCERPT = "envs/textgame/prompts/solver/excerpt.j2"


def test_registered_with_the_record_settings() -> None:
    b = resolve("textgame", "plancraft").benchmark
    assert (b.env_tool, b.end_command, b.step_limit, b.max_turns) == ("take_action", "impossible", 30, 150)
    assert "`take_action` call" in b.raw_metric_vocabulary.step_contents
    assert "`act`" not in b.raw_metric_vocabulary.step_contents
    assert b.solver_intro.startswith("You are crafting in Minecraft.")
    assert train_group_for(b, "plancraft-seen") == train_group_for(b, "plancraft-unseen") == "plancraft"
    assert b.server_env == (("PYTHONHASHSEED", "0"),)
    assert resolve("textgame", "alfworld").benchmark.server_env == resolve("textgame", "scienceworld").benchmark.server_env == ()


def test_suites_share_the_train_group_and_name_the_package_split() -> None:
    b = resolve("textgame", "plancraft").benchmark
    seen = b.load_suite(DatasetConfig(), "plancraft-seen")
    unseen = b.load_suite(DatasetConfig(), "plancraft-unseen")
    assert (len(seen), len(unseen)) == (150, 150)
    assert [s.spec for s in seen[:50]] == [s.spec for s in unseen[:50]]
    assert {s.spec["split"] for s in seen[:50]} == {"train"}
    assert {s.spec["split"] for s in seen[50:]} == {"test"}
    assert [s.split_role for s in seen] == ["train"] * 50 + ["test"] * 100
    assert {s.task_type for s in seen} == {"craft"}
    assert seen[50].task_text.startswith("Craft an item of type: ")
    assert seen[50].meta["complexity_split"] in ("easy", "medium", "hard")


def test_initial_observation_drops_the_objective_line() -> None:
    b = resolve("textgame", "plancraft").benchmark
    obs = "Craft an item of type: andesite\ninventory:\n - diorite [I18] quantity 1"
    info = {"task": "Craft an item of type: andesite"}
    assert b.initial_observation(obs, info) == "inventory:\n - diorite [I18] quantity 1"
    assert b.task_block(obs, info) == obs


def test_gave_up_phrase_only_on_the_end_command() -> None:
    b = resolve("textgame", "plancraft").benchmark
    r = TextGameResult(sample_id="s", db_id="d", question="q", steps=7, ended_by="agent_stopped")
    assert b.steps_text(r) == "7 take_action calls (ended by the agent)"
    r.ended_on_end_command = True
    assert b.steps_text(r) == "7 take_action calls (ended by the agent: it gave up)"


def test_runtime_notes_branch_on_the_tool() -> None:
    act = render_template(EXCERPT, intro="I")
    assert render_template(EXCERPT, intro="I", env_tool="act") == act
    assert "`act` tool" in act and "take_action" not in act
    notes = render_template(EXCERPT, intro="I", env_tool="take_action")
    assert "`take_action` tool" in notes and "`impossible` action" in notes and "`act`" not in notes


def test_only_plancraft_keeps_observation_lines() -> None:
    from wikitrace.config import ToolStep
    from wikitrace.envs.textgame.adapter import render_episode_steps

    obs = "Craft an item of type: stick\ninventory:\n - bamboo [I1] quantity 2"
    step = ToolStep(reasoning="", command="> search: stick", result=obs, is_error=False, state="")
    assert resolve("textgame", "plancraft").benchmark.multiline_observations
    assert not resolve("textgame", "alfworld").benchmark.multiline_observations
    assert render_episode_steps([step], None, "take_action", True) == (
        "[1]\n  take_action │ search: stick\n"
        "  obs         │ Craft an item of type: stick\n"
        "              │ inventory:\n"
        "              │  - bamboo [I1] quantity 2"
    )
    flat = render_episode_steps([step], None, "take_action")
    assert "  obs         │ Craft an item of type: stick inventory: - bamboo [I1] quantity 2" in flat


def test_renders_name_the_take_action_tool() -> None:
    from wikitrace.config import ToolStep
    from wikitrace.envs.textgame.adapter import render_episode_steps

    step = ToolStep(reasoning="r", command="> move: from [I1] to [A1] with quantity 1", result="ok",
                    is_error=False, state="")
    text = render_episode_steps([step], None, "take_action")
    assert text == (
        "[1]\n  reason      │ r\n  take_action │ move: from [I1] to [A1] with quantity 1\n"
        "  obs         │ ok"
    )
    assert "act " not in text
    artifacts = resolve("textgame", "plancraft").raw_layer_shape.metadata_artifacts
    assert "a `take_action` call is an action" in artifacts and "`act`" not in artifacts
    assert "an `act` call is a command" in resolve("textgame", "alfworld").raw_layer_shape.metadata_artifacts
