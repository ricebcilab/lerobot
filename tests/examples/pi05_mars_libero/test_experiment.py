import json
from types import SimpleNamespace

import experiment
import numpy as np
import pytest
import torch
import yaml
from config import ExperimentSettings, SessionSettings, load_experiment_settings
from experiment import TrialRecorder, build_schedule, expand_blocks, prompt_for
from reproducibility import trial_specs
from session import RolloutResult, Session
from teleop import KeyboardReader, PolicyOperator, PolicyOperatorSettings, TeleopChain

from lerobot.policies.pi05.steering import ReversalAdapter


class FakeSession:
    """What run_block needs of a Session: a scene to show, a chain and an adapter to set, and a rollout."""

    suite, task_id = "libero_goal", 0
    task_description = "open drawer"
    view = SimpleNamespace(stream=SimpleNamespace(set_status=lambda **kw: None))
    operator = object()  # a synthetic operator: no Start button to wait for

    def __init__(self, mode="policy", show_scene=lambda spec: {}, rollout=None):
        self.mode, self.tau, self.lam, self.ic_gain = mode, None, 1, "action"
        self.chain = TeleopChain(KeyboardReader(clock=lambda: 0.0))
        self.adapter = ReversalAdapter()
        self._show_scene, self._rollout = show_scene, rollout

    def set_scene(self, suite, task_id):
        self.suite, self.task_id = suite, task_id

    def show_scene(self, spec):
        return self._show_scene(spec)

    def initial_state_hash(self):
        return "verified-state"

    def rollout(self, prompt, **kwargs):
        if self._rollout is not None:
            self._rollout(self, **kwargs)
        return RolloutResult(success=True, steps=1, frames=[], metrics={})

    def mode_label(self):
        return self.mode

    def stats_line(self):
        return ""


@pytest.fixture
def run_block(monkeypatch):
    """run_block without videos or a simulator: `run(session, settings, schedule, directory)` -> records."""
    monkeypatch.setattr(experiment, "write_video", lambda *args, **kwargs: None)
    monkeypatch.setattr(Session, "initial_state_count", lambda suite, task: 50)

    def run(session, settings, schedule, directory):
        directory.mkdir(exist_ok=True)
        specs = experiment.build_trial_specs(settings, schedule)
        return experiment.run_block(session, settings, specs, directory)

    return run


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


def test_run_block_reuses_preview_and_records_matching_identity(tmp_path, run_block):
    settings = ExperimentSettings("test", SessionSettings(operator="policy", suite="libero_goal"))
    preview = {"observation": "preview"}
    seen = []

    def show_scene(spec):
        seen.append(spec)
        return preview

    def rollout(session, **kwargs):
        assert kwargs["initial_observation"] is preview
        assert kwargs["trial_spec"] == seen[-1]

    runs = [
        run_block(FakeSession(show_scene=show_scene, rollout=rollout), settings, [0, 0], tmp_path / block)
        for block in ("a", "b")
    ]
    assert seen[:2] == seen[2:]
    assert [r["pair_id"] for r in runs[0]] == [r["pair_id"] for r in runs[1]]
    with np.load(tmp_path / "a" / "trial_000.npz") as z:
        assert str(z["pair_id"]) == runs[0][0]["pair_id"]
        assert str(z["initial_state_hash"]) == "verified-state"


def test_show_scene_passes_explicit_reset_without_consuming_another_state():
    calls, published = [], []
    observation = {"pixels": {"image": np.arange(12, dtype=np.uint8).reshape(1, 2, 2, 3)}}

    def reset(**kwargs):
        calls.append(kwargs)
        return observation, {}

    session = Session.__new__(Session)
    session.vec_env = SimpleNamespace(reset=reset)
    session.view = SimpleNamespace(stream=SimpleNamespace(publish=published.append))
    spec = trial_specs([0], "libero_goal", seed=7, state_counts={0: 50})[0]
    assert session.show_scene(spec) is observation
    assert calls == [{"seed": spec.env_seed, "options": {"init_state_id": spec.init_state_id}}]
    np.testing.assert_array_equal(published[0], observation["pixels"]["image"][0][::-1, ::-1])


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
        policy_action=np.full((1, 7), 0.25),
        reward=np.array([1.0]),
        terminated=np.array([True]),
        truncated=np.array([False]),
        info={},
    )
    assert recorder.total_reads == 1
    assert [r["user_reads"] for r in recorder.rows] == [1, 0]
    assert np.isnan(recorder.rows[1]["eef_pos"]).all() and recorder.rows[1]["joint_pos"].shape == (7,)
    # The policy's own plan: given explicitly, else the action itself (nothing steered it).
    assert recorder.rows[0]["policy_translation"].tolist() == [1.0, 1.0, 1.0]
    assert recorder.rows[1]["policy_translation"].tolist() == [0.25, 0.25, 0.25]
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


# ---------------------------------------------------------------- per-trial corruption and input gain

RANDOM = "corruption: {random: {rotation_deg: [30, 90], scale: [1, 4]}}, input_gain: auto"
ADAPTED = (
    "mode: shared_flow_reversal_steering, n_reversal_steps: 1, reversal_adapter: {translation: corruption}"
)


def _settings(tmp_path, control: str):
    path = tmp_path / "condition.yaml"
    path.write_text(
        f"scene: {{suite: libero_goal, task_ids: [4]}}\noperator: policy\ncontrol: {{{control}}}\n"
    )
    return load_experiment_settings(path, {})


def _bare_session(control):
    """A Session without a policy or simulator, in the state `apply_control` leaves it."""
    session = Session.__new__(Session)
    session.chain = TeleopChain(KeyboardReader(clock=lambda: 0.0))
    session.adapter = ReversalAdapter()
    session.policy = SimpleNamespace(config=SimpleNamespace(num_inference_steps=10))
    session._frs_policy = None
    session.operator = object()  # a synthetic operator, so no SpaceMouse is looked for
    session.apply_control(control)
    return session


def _run_block_recording_what_is_in_force(tmp_path, run_block, settings, n_trials=3):
    """Run a block against the fake session; returns the (M, F) each rollout saw and the saved records."""
    in_force = []
    session = FakeSession(
        mode=settings.session.control.mode,
        rollout=lambda session, **kwargs: in_force.append(
            (session.chain.corruption.matrix, session.adapter.matrix)
        ),
    )
    run_block(session, settings, [4] * n_trials, tmp_path)
    saved = [json.loads(line) for line in (tmp_path / "trials.jsonl").read_text().splitlines()]
    return in_force, saved


def test_every_method_draws_the_same_corruption_on_a_trial_and_the_adapter_follows_it(tmp_path):
    fc = _settings(tmp_path, f"mode: shared_flow_control, {RANDOM}").session.control
    adapted = _settings(tmp_path, f"{ADAPTED}, {RANDOM}").session.control
    specs = trial_specs([4] * 6, "libero_goal", seed=4, state_counts={4: 50})
    drawn = []
    for spec in specs:
        a, b = experiment.trial_matrices(fc, spec), experiment.trial_matrices(adapted, spec)
        np.testing.assert_array_equal(a.corruption, b.corruption)  # matched across methods
        assert a.adapter is None
        np.testing.assert_array_equal(b.adapter[:3, :3], b.corruption)  # F = M
        np.testing.assert_array_equal(b.adapter[3:, 3:], np.eye(4))
        assert 30 <= a.draw.rotation_deg <= 90
        drawn.append(a.corruption)
    assert len({m.tobytes() for m in drawn}) == len(specs)  # a fresh draw on every trial
    np.testing.assert_array_equal(experiment.trial_matrices(fc, specs[0]).corruption, drawn[0])
    other_seed = trial_specs([4], "libero_goal", seed=5, state_counts={4: 50})[0]
    assert not np.array_equal(experiment.trial_matrices(fc, other_seed).corruption, drawn[0])


def test_a_fixed_corruption_is_the_same_on_every_trial(tmp_path):
    control = _settings(tmp_path, f"{ADAPTED}, corruption: {{rotation_z_deg: 40}}").session.control
    for spec in trial_specs([4] * 2, "libero_goal", seed=4, state_counts={4: 50}):
        matrices = experiment.trial_matrices(control, spec)
        # cos 40 and sin 40 degrees: a push along x comes out turned towards y
        np.testing.assert_allclose(matrices.corruption @ [1, 0, 0], [0.76604444, 0.64278761, 0])
        np.testing.assert_array_equal(matrices.adapter[:3, :3], matrices.corruption)
        assert matrices.draw is None and matrices.record()["corruption_parameters"] is None


def test_run_block_puts_each_trial_s_corruption_in_force_and_records_it(tmp_path, run_block):
    settings = _settings(tmp_path, f"{ADAPTED}, {RANDOM}")
    in_force, saved = _run_block_recording_what_is_in_force(tmp_path, run_block, settings)
    assert len(in_force) == len(saved) == 3
    for trial, ((corruption, adapter), record) in enumerate(zip(in_force, saved, strict=True)):
        np.testing.assert_array_equal(
            adapter[:3, :3], corruption
        )  # the adapter in force followed the corruption
        # the record and the step file hold what was in force while the trial ran
        np.testing.assert_allclose(record["corruption_matrix"], corruption)
        np.testing.assert_allclose(record["reversal_adapter_matrix"], adapter)
        assert set(record["corruption_parameters"]) == {"rotation_axis", "rotation_deg", "scale"}
        assert record["input_gain"] == 0.25
        with np.load(tmp_path / f"trial_{trial:03d}.npz") as z:
            np.testing.assert_allclose(z["corruption_matrix"], corruption)
            np.testing.assert_allclose(z["reversal_adapter_matrix"], adapter)
            assert float(z["input_gain"]) == 0.25
    assert not np.allclose(in_force[0][0], in_force[1][0])  # and it changed from one trial to the next


def test_run_block_records_empty_matrices_when_nothing_is_in_force(tmp_path, run_block):
    settings = _settings(tmp_path, "mode: shared_flow_control")
    in_force, saved = _run_block_recording_what_is_in_force(tmp_path, run_block, settings, n_trials=1)
    assert in_force == [(None, None)]
    record = saved[0]
    assert record["corruption_matrix"] is None and record["reversal_adapter_matrix"] is None
    assert record["corruption_parameters"] is None and record["input_gain"] is None
    with np.load(tmp_path / "trial_000.npz") as z:  # loadable without pickle: NaN stands for "off"
        assert z["corruption_matrix"].shape == (3, 3) and np.isnan(z["corruption_matrix"]).all()
        assert z["reversal_adapter_matrix"].shape == (7, 7) and np.isnan(z["input_gain"])


def test_apply_control_turns_the_input_gain_on_and_off_with_the_block(tmp_path):
    session = _bare_session(_settings(tmp_path, f"mode: shared_flow_control, {RANDOM}").session.control)
    assert session.chain.gain.gain == 0.25
    assert session.chain.corruption.matrix is None  # nothing is in force until a trial draws it
    fixed = _settings(tmp_path, "mode: shared_flow_control, corruption: {rotation_z_deg: 20}")
    session.apply_control(fixed.session.control)
    assert session.chain.gain.gain is None


def test_an_auto_gain_follows_a_live_corruption_change_but_a_number_does_not(tmp_path):
    session = _bare_session(
        _settings(tmp_path, "mode: shared_flow_control, input_gain: auto").session.control
    )
    assert session.chain.gain.gain == 1.0
    session.chain.corruption.matrix = np.diag([4.0, 1.0, 1.0])  # what the REPL's `corruption <file>` does
    assert "0.25" in session.refresh_input_gain() and session.chain.gain.gain == 0.25
    session.chain.corruption.matrix = None  # `corruption off`
    session.refresh_input_gain()
    assert session.chain.gain.gain == 1.0
    session.apply_control(_settings(tmp_path, "mode: shared_flow_control, input_gain: 0.2").session.control)
    session.chain.corruption.matrix = np.diag([4.0, 1.0, 1.0])
    assert "0.2" in session.refresh_input_gain() and session.chain.gain.gain == 0.2


def test_a_random_block_s_config_lists_every_trial_s_corruption_in_plain_yaml(tmp_path):
    settings = _settings(tmp_path, f"mode: shared_flow_control, {RANDOM}")
    specs = trial_specs([4] * 3, "libero_goal", seed=settings.seed, state_counts={4: 50})
    session = SimpleNamespace(
        resolved_matrices=lambda: {"corruption_matrix": None, "reversal_adapter_matrix": None}
    )
    resolved = experiment._provenance(settings, experiment.Block("block", []), [4, 4, 4], specs, session)
    written = yaml.safe_load(yaml.safe_dump(resolved))  # what lands in config.yaml
    assert written["input_gain"] == "auto" and written["resolved_input_gain"] == 0.25
    assert len(written["trial_corruptions"]) == 3
    for spec, entry in zip(specs, written["trial_corruptions"], strict=True):
        in_trial = experiment.trial_matrices(settings.session.control, spec).corruption
        np.testing.assert_allclose(entry["corruption_matrix"], in_trial)
        assert 30 <= entry["corruption_parameters"]["rotation_deg"] <= 90
    fixed = _settings(tmp_path, "mode: shared_flow_control, corruption: {rotation_z_deg: 20}")
    assert "trial_corruptions" not in experiment._provenance(
        fixed, experiment.Block("block", []), [4], specs[:1], session
    )


def test_the_dry_run_description_states_the_family_and_the_gain_in_force(tmp_path):
    text = experiment._describe_block(_settings(tmp_path, f"mode: shared_flow_control, {RANDOM}"))
    assert "random(rotation_deg=30..90, scale=1..4)" in text
    assert "input gain: 0.25" in text
