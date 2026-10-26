import collections
import sys
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[3] / "examples/pi05/feeding_finetune"
if str(EXAMPLE) not in sys.path:
    sys.path.insert(0, str(EXAMPLE))
import relabel_prompts as rp  # noqa: E402

TEMPLATE = "Reach, grasp and bring to mouth the {food}"
FOODS = ["apple", "hotdog", "pear", "bottle of water"]
PROMPTS = ["Pick something up", "Bring an object to mouth", "Grab any food"]


def _episodes(n_per_food=73):
    tasks = {}
    ep = 0
    for food in FOODS:
        for _ in range(n_per_food):
            tasks[ep] = TEMPLATE.format(food=food)
            ep += 1
    return tasks


def test_fraction_is_stratified_per_task():
    tasks = _episodes()
    plan = rp.plan_relabel(tasks, frac=0.4, seed=0, prompts=PROMPTS)
    per_task = collections.Counter(tasks[ep] for ep in plan)
    for task in set(tasks.values()):
        assert per_task[task] == round(0.4 * 73)
    assert len(plan) == 4 * round(0.4 * 73)


def test_prompts_are_balanced_and_differ_from_originals():
    tasks = _episodes()
    plan = rp.plan_relabel(tasks, frac=0.5, seed=1, prompts=PROMPTS)
    counts = collections.Counter(plan.values())
    assert set(counts) == set(PROMPTS)
    assert max(counts.values()) - min(counts.values()) <= 1
    for ep, prompt in plan.items():
        assert prompt != tasks[ep]


def test_deterministic_in_seed():
    tasks = _episodes()
    a = rp.plan_relabel(tasks, frac=0.3, seed=7, prompts=PROMPTS)
    b = rp.plan_relabel(tasks, frac=0.3, seed=7, prompts=PROMPTS)
    c = rp.plan_relabel(tasks, frac=0.3, seed=8, prompts=PROMPTS)
    assert a == b
    assert a != c


def test_edge_fractions():
    tasks = _episodes(n_per_food=10)
    assert rp.plan_relabel(tasks, frac=0.0, seed=0, prompts=PROMPTS) == {}
    full = rp.plan_relabel(tasks, frac=1.0, seed=0, prompts=PROMPTS)
    assert set(full) == set(tasks)


def test_invalid_inputs():
    tasks = _episodes(n_per_food=3)
    with pytest.raises(ValueError):
        rp.plan_relabel(tasks, frac=1.5, seed=0, prompts=PROMPTS)
    with pytest.raises(ValueError):
        rp.plan_relabel(tasks, frac=0.5, seed=0, prompts=[])


def test_converter_default_generic_prompt_matches_deployment_string():
    # The brand-rice free-use graph sends exactly this string; the converter's default list
    # must keep it first so a --generic-prompt-frac build makes it in-distribution.
    pytest.importorskip("av")
    import convert_nwb_to_lerobot as conv  # noqa: E402

    assert conv.FEEDING_GENERIC_PROMPTS[0] == "Reach, grasp and bring to mouth"
    assert "{" not in "".join(conv.FEEDING_GENERIC_PROMPTS)
