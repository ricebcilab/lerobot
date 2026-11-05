# MARS image task module, round 2: teacher demonstrations and relative token geometry

Amends `2026-11-03-mars-image-task-module-design.md` (the round-1 spec): its "Fallback if the
absolute view does not train" paragraph is replaced by this document. Everything else in the
round-1 spec (the modules, the cache, the gate rule, the sweep, the seeds, the report) stands.

## Why

The round-1 gate failed on every row. The diagnosis (ledger, 2026-11-04): the image modules
memorise their 300 training scenes (train-layout hits 89–98%, fresh-layout hits 15–36%,
rotated-scene hits near 0). Two causes, each with evidence:

- **Too few scenes.** 3,000 scripted demonstrations instead of 300 took the `vlm_image` module
  from 27.6% to 69.0% on T2 and from 14.7% to 62.6% on T4 (fresh layouts, seed 0), still under
  the 80% bar, and costing 48–63 min of demonstrating, past the point where fine-tuning wins.
- **Absolute geometry.** A disc token carries the obstacle's offset from the cursor; a VLM
  token carries a learned absolute position (`scene_pos`) and the cursor enters only the query,
  so the module has to discover "patch minus cursor" from 300 scenes, and a per-scene lookup is
  the easier fit.

The user chose (2026-11-04) the relative geometry (change B) as the candidate, and teacher
rollouts (change A) only as a comparison run beside it: on a robot the teacher needs the
obstacle poses at every rollout, so its scene collection is a supervised, semi-automatic
session rather than the cheap part of the recipe, and the paper's claim stays "300 human
demonstrations" with no autonomous collection. Image-space augmentation (round 3; it needs no
obstacle state) is the next step if change B falls short, and a VLM-as-detector pipeline is a
separate branch if it fails entirely.

## Goals, non-goals, anti-goals

Goal: an image task module that passes the gate below on every context with 300 human
demonstrations per context and nothing the demonstrations do not already contain, so the
round-1 grid, pooled arm, seeds and report can run as specified. The teacher arm's result is
a ceiling: what the same module does when scenes are free.

Non-goals this round: the `cnn` source (worse than `vlm_image` at 300 demos; the relative
geometry applies to it and it can be added later), `vlm_prefix`, symmetry augmentation, the
cursor-centred render, a pooled or grid run of the teacher arm, seeds 1–2 (they follow only
after the gate passes).

Anti-goals: a module that passes by seeing the goal (the round-1 flag rule stays); any change
to the parametric modules, the policy, or the `discs` branch's weights or outputs.

## Change A: teacher demonstrations (the comparison arm)

This arm measures the headroom; it is not the candidate and never reaches the grid (see Why).

**The teacher** for a context is its parametric task module, the gate's `oracle_task`
checkpoint (`task_<ctx>_n300_s0_with_<user>`), trained on the 300 human demonstrations. It is
not retrained.

**The rollouts** are the gate's own `oracle_task` rollouts: the `expert_chunk` agent (at every
replan the teacher-adapted policy's chunk with the goal given, played step by step), `teleop`
mode with the readout off (`Readout(None, 0.0, rng)`: nothing corrupted, no noise), the arm
`condition_grid.yaml` overridden with the task's obstacle, `--targets`, `--reach` and
`reversal_adapter: {task: <teacher>}`, the arm's default time limit. Layouts are drawn as the
scripted recipe draws them (`layouts_for`, the same `--p-free` share without the obstacle).
**Only hits are kept**: layouts are drawn 500 at a time until `--n-demos` hits exist (the val
split likewise for `--val-demos`); collisions, wrong targets and timeouts are discarded and
counted. A kept rollout becomes a demonstration **at the policy's step rate**, not at the
arm's control rate: the gate's arm (`gain_mode: time`, `speed_gain: 0.5`, `execute: 2`) plays
each planned step over two control steps, so `path` holds two entries per planned step and
`diff(path)` taken literally would be a half-speed demonstration (a module trained on it would
learn to halve the policy's speed, and the gate, halving again, would time out). So: actions =
`diff(path[::2])`, which equals the planned steps the controller executed (test this equality
on a short rollout, up to clipping at the screen edge); length = `ceil(n / 2)` planned steps,
`n` the control step at which the dwell selected the goal; zero actions after it. The hold
before selection is then about 8 planned steps instead of the scripted 15, which only changes
how many hold states fall inside a demonstration's length; `DemoSampler` already treats steps
after the length as holds. (`closed_loop_splits` takes `diff(path)` literally on the same arm:
leave it alone, it is the user module's recipe and not this round's.) *(Designer's call,
2026-11-04, from the builder check.)*

**Where.** `train_adapter.py --teacher <adapter name>` (task runs only; refused with
`--demo-seconds`). `expert_splits` uses the teacher rollouts in place of `make_demos` for
both splits; everything downstream is unchanged: `ExpertData`, the expert pushes at the
rollout's positions (`expert_pushes`), the field loss through the context's user module, the
validation numbers. `--teacher` with `--pooled` is refused (no pooled teacher run). The run's
meta records the teacher and the counts of rollouts drawn and kept.

**Scene arrays.** `scene_for` runs on the rollouts' layouts as today; nothing is cached to
disk (10,000 scenes are about 10.5 GB of float16 tokens and take minutes to encode); they stay
on the GPU as today (49 GB cards).

**Sizes.** 10,000 training rollouts and 500 validation rollouts per context, 20,000 steps,
batch 512 (the per-task recipe otherwise unchanged).

## Change B: relative token geometry

`LearnedTaskModule(token_geometry="absolute" | "relative")`, default `"absolute"` (today's
learned `scene_pos`; checkpoints without the key load as absolute). `"relative"` is only for
`vlm_image` and `cnn` (`discs` is already relative, `vlm_prefix` has no screen positions: both
raise).

**Relative:** the module keeps a buffer of token centres on the unit screen. For `vlm_image`
the 224 render enters SigLIP unpadded (`image_resolution` is 224), so token k is patch
`(row, col) = divmod(k, 16)` with centre `((col + 0.5) / 16, (row + 0.5) / 16)` in the
render's coordinates (x right, y down, as `render_layouts` draws them); for `cnn` the same with
`S / 16` per side. Each token's geometry is `[rel_x, rel_y, half / scale, |rel|]` with
`rel = (centre - obs["pos"]) / scale` and `half` the token's half-width in screens, through the
same MLP the disc tokens use (`token_in`), and the token is `scene_in(x) + token_in(geo)` (cnn:
`cnn(x) + token_in(geo)`). A relative module has no `scene_pos` parameter. `obs["pos"]` is
required (the cursor always has it). The bind-time token cache is unchanged: `pos` is part of
the observation and fixed within a chunk.

**Tests** (the module's suite): the config round-trips and a checkpoint without the key loads
as absolute; a relative module has no `scene_pos` and its tokens equal `scene_in(x) +
token_in(features)` for hand-computed features at a known `pos`; `discs` and `vlm_prefix` with
`relative` raise; the bound call equals the unbound call for `relative`; the convention test:
with every token's content identical, tokens for a cursor at `p` and at `p + (1/16, 0)` (and
`(0, 1/16)`) are equal after shifting the token grid by one column (one row), which pins the
row/col and x/y convention. The `discs` equivalence tests stay green bitwise.

**Trainer:** `--token-geometry {absolute,relative}` (default absolute), passed to `TaskModule`;
`scene_spec` is unchanged; the checkpoint's meta records it.

## Arms, order, and the rule to continue

Seed 0, `vlm_image` only. **Quick check on T2 and T4 first** (the two contexts with the 3,000
scripted-demo reference, 69.0 / 62.6):

| arm | demos | geometry | checkpoint name (`image/`) |
|---|---|---|---|
| `rel300` | 300 scripted | relative | `task_<ctx>_n300_s0_with_<user>_vlm_image_rel` |
| `teach10k_rel` | 10,000 teacher | relative | `task_<ctx>_n10000_s0_with_<user>_vlm_image_teacher_rel` |

`rel300` is the candidate: change B alone, on the 300 scripted demonstrations round 1 used.
`teach10k_rel` is the comparison: changes A and B, the same module with 10,000 cheap scenes,
the ceiling the candidate is measured against. Names keep `_n<N>_s0` so `run_sweep`'s
`--module-seed` substitution still matches.

A gate sweep `grid_task_check_image2_quick.yaml` (T2 and T4 only, the round-1 gate's arm,
trials and seed) with `oracle_task_<arm>` and the goal-hidden `policy_task_<arm>` for both
arms, against `oracle_task` / `policy_task`, plus one new reference row, `oracle_ft`: the
task's fine-tuned policy (`grid_ft.yaml`'s `model` per task, `flow_goaldrop_<task>_ft_n300_s0`)
under the same oracle agent with no module, the gate-side stand-in for the grid's fine-tuned +
MARS-U arm. Its table goes to the user before anything else runs.

**The rule (the user's, 2026-11-04; it replaces round 1's 80% bar).** Per context, an arm
*passes* when its hits are comparable to or better than the parametric module's (within 10
points below, or above) and its collisions are within +10 points of the parametric module's.
The *floor* is beating the fine-tuned policy's hits (`oracle_ft`): an arm above the floor but
more than 10 points under the parametric module is *marginal*; an arm at or below the floor
*fails*. The goal-hidden flag is unchanged. On T5, where the parametric module hits 74.2%
(collisions 24.4%), comparable therefore means 64.2% or better.

**Then, only if `rel300` passes both quick contexts**: train it on T3 and T5, gate on all
four (`grid_task_check_image2.yaml`), train its pooled version (`task_all_n1200_s0_vlm_image_rel`,
the round-1 pooled recipe), gate the pooled module, then the round-1 grid with its methods
(`mars_vlm_rel`, `mars_vlm_rel_all`) in `grid_image.yaml`, seeds 1–2 and the report as the
round-1 spec says. A marginal `rel300` continues only if the user says so. The round-1 gate
and grid checkpoints are not deleted.

**If `rel300` does not pass, stop and report, whatever `teach10k_rel` did**: the teacher arm
is not a route to the grid. Its number says what the gap is made of: a passing `teach10k_rel`
means the architecture can do it given scenes, and round 3 has to supply scenes from the 300
demonstrations; a failing one means the gap is not only scenes. Round 3 (the user's call) is
image-space augmentation, which needs no obstacle state: mirrors, rotations and shifts of the
rendered scene with the actions transformed to match, and a distance bias `-|rel| / scale` on
the attention logits, which makes far tokens fade and gives most of the cursor-centred view
without re-rendering.

## Costs

Per-task runs: about 45–60 min each alone on one GPU (rollouts for 10,000 hits a few minutes
batched 500 at a time; encoding minutes; 20,000 steps as round 1). Four quick-check jobs on two
GPUs, two per GPU: about one and a half to two hours. Disk: nothing new is cached (723 GB
free). GPU memory: 10.5 GB of scene tokens for a teacher run.

## Left to the builder

The exact signature of the teacher rollout function and how it reuses `run_trials` and
`closed_loop_splits`'s record-to-demo code; the quick gate YAML's exact method names; whether
the rollout batches reuse `run_sweep`'s `method_arm` or build the arm with `config.load_arm`
directly (either, as long as the arm equals the gate's `oracle_task`); the pipeline script's
job list.
