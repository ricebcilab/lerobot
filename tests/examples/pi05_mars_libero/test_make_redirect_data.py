import make_redirect_data as mrd
import numpy as np
import torch
from reproducibility import trial_specs


def test_the_plan_redirects_every_task_to_every_other_from_states_the_study_does_not_use():
    plan = mrd.redirect_plan((1, 4, 5), (10, 40), n_starts=3, n_states=50, seed=0, held_out=25)
    assert len(plan) == 3 * 2 * 2 * 3  # ordered pairs x switch steps x start states
    assert {(a.first, a.meant) for a in plan} == {(a, b) for a in (1, 4, 5) for b in (1, 4, 5) if a != b}
    assert [a.meant for a in plan] == sorted(a.meant for a in plan)  # one scene at a time
    # the study's states (its development and test trials) are its trial specs'; the plan keeps off them
    used = mrd.study_states(4, 50, seed=0, n=25)
    specs = trial_specs([4] * 25, "libero_goal", seed=0, state_counts={4: 50})
    assert used == [s.init_state_id for s in specs] and len(set(used)) == 25
    starts = {a.start for a in plan if a.meant == 4}
    assert len(starts) == 3 and not starts & set(used)
    # shards share the plan out without overlap
    parts = [mrd.shard(plan, i, 4) for i in range(4)]
    assert sorted(sum(parts, []), key=plan.index) == plan and all(parts)


def test_a_frame_is_in_the_training_datasets_format():
    processed = {
        "observation.images.image": torch.full((1, 3, 4, 4), 0.5),
        "observation.images.image2": torch.zeros(1, 3, 4, 4),
        "observation.state": torch.arange(8.0)[None],
    }
    frame = mrd.dataset_frame(processed, np.arange(7.0)[None], "put the bowl on the plate", size=4)
    assert (
        frame["observation.images.image"].shape == (4, 4, 3)
        and frame["observation.images.image"].dtype == np.uint8
    )
    assert frame["observation.images.image"][0, 0, 0] == 128 and frame["observation.images.image2"].max() == 0
    np.testing.assert_array_equal(frame["observation.state"], np.arange(8, dtype=np.float32))
    np.testing.assert_array_equal(frame["action"], np.arange(7, dtype=np.float32))
    assert frame["action"].dtype == np.float32 and frame["task"] == "put the bowl on the plate"
    # the simulator renders larger images than the dataset holds: they are resized to the dataset's
    processed["observation.images.image"] = torch.full((1, 3, 6, 6), 0.25)
    small = mrd.dataset_frame(processed, np.zeros((1, 7)), "t", size=4)["observation.images.image"]
    assert small.shape == (4, 4, 3) and np.all(small == 64)
    assert mrd.dataset_frame(processed, np.zeros((1, 7)), "t")["observation.images.image"].shape == (
        256,
        256,
        3,
    )
