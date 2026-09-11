from types import SimpleNamespace

import numpy as np
import pytest
import torch
from experiment import TrialRecorder, build_schedule, expand_blocks, prompt_for
from reproducibility import trial_specs
from session import RolloutResult, Session
from teleop import KeyboardReader, PolicyOperator, PolicyOperatorSettings, TeleopChain


def test_build_schedule_orders():
    assert build_schedule([1, 2, 3], 7, "sequential", 0) == [1, 2, 3, 1, 2, 3, 1]
    shuffled = build_schedule([1, 2, 3], 7, "shuffled", 0)
    assert sorted(shuffled[:3]) == [1, 2, 3] and sorted(shuffled[3:6]) == [1, 2, 3] and len(shuffled) == 7
    random_a = build_schedule([1, 2, 3], 20, "random", 5)
    assert random_a == build_schedule([1, 2, 3], 20, "random", 5)
    assert random_a != build_schedule([1, 2, 3], 20, "random", 6)
    assert set(random_a) <= {1, 2, 3}


def test_trial_specs_match_across_task_order_and_use_seeded_state_permutations():
    counts = {1: 50, 2: 50}
    a = trial_specs([1, 2, 1], "libero_goal", seed=7, state_counts=counts)
    b = trial_specs([1, 1, 2], "libero_goal", seed=7, state_counts=counts)
    assert [s.metadata() for s in a] == [b[i].metadata() for i in [0, 2, 1]]
    assert a[0].init_state_id != a[2].init_state_id
    assert len({s.metadata()["pair_id"] for s in a}) == 3
    other_seed = trial_specs([1, 2, 1], "libero_goal", seed=8, state_counts=counts)
    assert [s.init_state_id for s in a] != [s.init_state_id for s in other_seed]
    extended = trial_specs([1] * 100, "libero_goal", seed=7, state_counts=counts)
    assert extended[:2] == [a[0], a[2]]
    assert {s.init_state_id for s in extended[:50]} == set(range(50))
    assert {s.init_state_id for s in extended[50:]} == set(range(50))
    assert len({s.metadata()["pair_id"] for s in extended}) == 100
    assert len({a[0].env_seed, a[0].policy_seed, a[0].operator_seed, a[0].input_noise_seed}) == 4


def test_run_block_reuses_preview_and_records_matching_identity(tmp_path, monkeypatch):
    import experiment
    from config import ExperimentSettings, SessionSettings

    settings = ExperimentSettings("test", SessionSettings(operator="policy", suite="libero_goal"))
    preview = {"observation": "preview"}
    seen = []

    class FakeSession:
        suite, task_id = "libero_goal", 0
        task_description = "open drawer"
        mode, n_guided_steps = "policy", 0
        operator = object()
        chain = TeleopChain(KeyboardReader(clock=lambda: 0.0))
        adapter = SimpleNamespace(label=None)
        view = SimpleNamespace(stream=SimpleNamespace(set_status=lambda **kw: None))

        def show_scene(self, spec):
            seen.append(spec)
            return preview

        def initial_state_hash(self):
            return "verified-state"

        def rollout(self, prompt, **kwargs):
            assert kwargs["initial_observation"] is preview
            assert kwargs["trial_spec"] == seen[-1]
            return RolloutResult(success=True, steps=1, frames=[], metrics={})

        def mode_label(self):
            return self.mode

        def stats_line(self):
            return ""

    monkeypatch.setattr(experiment, "write_video", lambda *args, **kwargs: None)
    monkeypatch.setattr(Session, "initial_state_count", lambda suite, task: 50)
    runs = []
    for block in ("a", "b"):
        directory = tmp_path / block
        directory.mkdir()
        runs.append(experiment.run_block(FakeSession(), settings, [0, 0], directory))
    assert seen[:2] == seen[2:]
    assert [r["pair_id"] for r in runs[0]] == [r["pair_id"] for r in runs[1]]
    with np.load(tmp_path / "a" / "trial_000.npz") as z:
        assert str(z["pair_id"]) == runs[0][0]["pair_id"]
        assert str(z["initial_state_hash"]) == "verified-state"


def test_show_scene_passes_explicit_reset_without_consuming_another_state():
    calls = []
    observation = {"initial": True}

    def reset(**kwargs):
        calls.append(kwargs)
        return observation, {}

    session = Session.__new__(Session)
    session.vec_env = SimpleNamespace(reset=reset, envs=[SimpleNamespace(render=lambda: None)])
    session.view = SimpleNamespace(stream=SimpleNamespace(publish=lambda frame: None))
    spec = trial_specs([0], "libero_goal", seed=7, state_counts={0: 50})[0]
    assert session.show_scene(spec) is observation
    assert calls == [{"seed": spec.env_seed, "options": {"init_state_id": spec.init_state_id}}]


@pytest.mark.parametrize("chunk_steps,update_steps,delay,steps", [(1, None, 0, 2), (10, 20, 4, 31)])
def test_rollout_keeps_preview_state_and_separates_operator_policy_randomness(
    monkeypatch, chunk_steps, update_steps, delay, steps
):
    import session as session_module
    from config import SessionSettings

    preview = {"pixels": {"image": np.zeros((1, 2, 2, 3), dtype=np.uint8)}}
    monkeypatch.setattr(session_module, "preprocess_observation", lambda obs: dict(obs))
    spec = trial_specs([0], "libero_goal", seed=7, state_counts={0: 50})[0]

    def run(operator_draws):
        actions = []
        query_steps, query_seeds, delivered = [], [], []

        def step(action):
            actions.append(action.copy())
            return preview, np.array([0.0]), np.array([False]), np.array([False]), {}

        def unexpected_reset(**kwargs):
            raise AssertionError("The prepared preview must not be reset again")

        session = Session.__new__(Session)
        session.settings = SessionSettings(n_action_steps=chunk_steps, operator="policy")
        session.operator = PolicyOperator(chunk_steps, PolicyOperatorSettings(update_steps, delay))
        session.operator.refresh(np.ones((1, 7)), step=100)  # prior trial must be discarded
        session.chain = TeleopChain(KeyboardReader(clock=lambda: 0.0))
        session.view = SimpleNamespace(
            stream=SimpleNamespace(publish=lambda frame: None, set_status=lambda **kwargs: None)
        )
        session.vec_env = SimpleNamespace(
            reset=unexpected_reset,
            step=step,
            envs=[SimpleNamespace(render=lambda: preview["pixels"]["image"][0])],
        )

        def refresh_operator(obs, step=0):
            query_steps.append(step)
            query_seeds.append(torch.initial_seed())
            torch.rand(operator_draws)
            session.operator.refresh(np.full((1, 7), 0.25 + step / 100), step=step)

        session._refresh_operator = refresh_operator
        session.mode_label = lambda: "test"
        session.metrics = lambda: {}
        session.autocast_ctx = session_module.nullcontext()
        identity = lambda value: value  # noqa: E731
        cached_action = None

        def select_action(obs):
            nonlocal cached_action
            if len(actions) % chunk_steps == 0:
                cached_action = torch.rand(1, 7)
            delivered.append(session.operator.translation)
            return cached_action

        session._rollout_kwargs = lambda: {
            "policy": SimpleNamespace(reset=lambda: None, select_action=select_action),
            "action_hook": None,
            "preprocessor": identity,
            "postprocessor": identity,
            "env_preprocessor": identity,
            "env_postprocessor": identity,
        }
        result = session.rollout(
            "do something", initial_observation=preview, trial_spec=spec, max_steps=steps
        )
        assert result.steps == steps
        assert query_steps == list(range(0, steps, update_steps or chunk_steps))
        assert query_seeds == [(spec.operator_seed + i) % 2**32 for i in range(len(query_steps))]
        for step, command in enumerate(delivered):
            ready = [q for q in query_steps if q + delay <= step]
            np.testing.assert_allclose(command, 0.25 + ready[-1] / 100 if ready else 0)
        return np.stack(actions)

    a, b = run(1), run(100)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a[0], a[chunk_steps])  # consecutive chunks receive different draws


def test_prompt_for():
    assert prompt_for("do something", "pick up the bowl") == "do something"
    assert prompt_for("task", "pick up the bowl") == "pick up the bowl"


def test_trial_recorder_rows_and_per_trial_reads(tmp_path):
    kb = KeyboardReader(clock=lambda: 0.0)
    kb.update({"ArrowUp"}, toggles=0)
    chain = TeleopChain(kb)
    chain.source.translation  # noqa: B018 -- a read before the trial must not count
    recorder = TrialRecorder(chain)
    obs = {"robot_state": {"eef": {"pos": np.zeros((1, 3)), "quat": np.zeros((1, 4))}}}
    chain.source.translation  # noqa: B018
    recorder(
        step=1,
        observation=obs,
        action=np.ones((1, 7)),
        reward=np.array([0.0]),
        terminated=np.array([False]),
        truncated=np.array([False]),
        info={},
    )
    recorder(
        step=2,
        observation={},
        action=np.zeros((1, 7)),
        reward=np.array([1.0]),
        terminated=np.array([True]),
        truncated=np.array([False]),
        info={},
    )
    assert recorder.total_reads == 1
    assert [r["user_reads"] for r in recorder.rows] == [1, 0]
    assert np.isnan(recorder.rows[1]["eef_pos"]).all() and recorder.rows[1]["joint_pos"].shape == (7,)
    recorder.save(tmp_path / "t.npz", success=True, task_id=3)
    z = np.load(tmp_path / "t.npz")
    assert z["action"].shape == (2, 7) and bool(z["success"]) and int(z["task_id"]) == 3
    assert z["user_translation_raw"].shape == (2, 3) and z["terminated"].tolist() == [False, True]


def test_expand_blocks_names_and_sets():
    [single] = expand_blocks("flow_control_rotz20", [], None)
    assert single.name == "flow_control_rotz20" and single.sets == []
    [one] = expand_blocks("flow_control_rotz20", ["control.corruption=null"], None)
    assert one.name == "flow_control_rotz20_corruption-null"
    blocks = expand_blocks("flow_reversal_rotz20", ["experiment.seed=1"], "control.n_reversal_steps=2, 4,10")
    assert [b.name for b in blocks] == [
        "flow_reversal_rotz20_seed-1_n_reversal_steps-2",
        "flow_reversal_rotz20_seed-1_n_reversal_steps-4",
        "flow_reversal_rotz20_seed-1_n_reversal_steps-10",
    ]
    assert blocks[0].sets == ["experiment.seed=1", "control.n_reversal_steps=2"]
    with pytest.raises(ValueError, match="--sweep"):
        expand_blocks("x", [], "control.n_reversal_steps")


def test_summarize_scene_motion_picks_the_largest_change():
    from experiment import summarize_scene_motion

    start = {
        "object_names": ["bowl", "plate"],
        "object_pos": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "articulation_names": ["cabinet_top", "cabinet_middle"],
        "articulation_qpos": np.array([0.0, 0.0]),
    }
    end = {
        "object_names": ["bowl", "plate"],
        "object_pos": np.array([[0.0, 0.0, 0.0], [1.0, 0.03, 0.0]]),
        "articulation_names": ["cabinet_top", "cabinet_middle"],
        "articulation_qpos": np.array([0.0, -0.15]),
    }
    summary = summarize_scene_motion(start, end)
    assert summary["mode"] == "cabinet_middle"
    assert summary["moved"] == pytest.approx({"plate": 0.03, "cabinet_middle": 0.15})
    assert summarize_scene_motion(start, start) == {"mode": None, "moved": {}}
    assert summarize_scene_motion(None, end) == {"mode": None, "moved": {}}


def test_trial_recorder_keeps_scene_rows_and_saves_them(tmp_path):
    from experiment import summarize_scene_motion

    recorder = TrialRecorder(TeleopChain(KeyboardReader(clock=lambda: 0.0)))
    common = {"observation": {}, "action": np.zeros((1, 7)), "reward": np.array([0.0]), "info": {}}

    def scene(x):
        return {
            "object_names": ["bowl"],
            "object_pos": np.array([[x, 0.0, 0.0]], dtype=np.float32),
            "articulation_names": ["drawer"],
            "articulation_qpos": np.array([0.0], dtype=np.float32),
        }

    recorder(step=1, terminated=np.array([False]), truncated=np.array([False]), scene=scene(0.0), **common)
    recorder(step=2, terminated=np.array([False]), truncated=np.array([False]), scene=None, **common)
    recorder(step=3, terminated=np.array([True]), truncated=np.array([False]), scene=scene(0.3), **common)
    assert recorder.scene_first["object_pos"][0, 0] == 0.0 and recorder.scene_last["object_pos"][0, 0] == 0.3
    assert summarize_scene_motion(recorder.scene_first, recorder.scene_last)["mode"] == "bowl"
    recorder.save(tmp_path / "t.npz", success=True)
    z = np.load(tmp_path / "t.npz")
    assert z["object_pos"].shape == (3, 1, 3) and np.isnan(z["object_pos"][1]).all()
    assert list(z["object_names"]) == ["bowl"] and z["articulation_qpos"].shape == (3, 1)
