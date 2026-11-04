# MARS task modules that read the scene from an image

Design, 2026-11-03. Status: draft for review.

## Intent

MARS's task module today reads the task context as privileged parameters: one token
per obstacle disc (center, radius), which the cursor environment hands it and which no
robot can hand it. The goal is a task module that reads the context from a picture,
through the vision-language model (VLM) that π0.5 already uses to inform its action
expert, so that the same module design can later be trained on a robot task the policy
was never fine-tuned on, from a few demonstrations, with the policy frozen.

This document covers the first step only: the cursor task, where the parametric module
is a known reference (Fig. 3 of the MARS paper) and every arm can be run on the same
trials. The outcome that sends the work to the robot: image-conditioned task modules
trained on 300 demonstrations per context perform comparably to, or better than, the
parametric ones on the grid, and above the other controls. Which arm to deploy on the
robot is decided from those results, not now.

## What exists (the parts this builds on)

- Policy: `flow_goaldrop`, the frozen goal-free cursor flow policy
  (`lerobot.policies.cursor_flow`). It never saw an obstacle.
- Task module: `LearnedTaskModule` in `lerobot.policies.steering.modeling_steering`.
  Per-step queries from (x_t, v, t, cursor position) cross-attend (2 layers, d=128) to a
  learned null token plus one token per obstacle disc (position relative to the cursor,
  radius, distance, in units of `scale`); the head emits an affine (A, b) on the velocity,
  zero-initialized so a fresh module is the identity. It is bound to an observation with
  `bind(obs)` and acts in both directions of the flow (`ReversalAdapter.field`).
- Contexts: T2 disc, T3 wall (a row of touching discs with a gap), T4 ring around the goal,
  T5 ring around the start; every context is a list of discs (`lerobot.envs.cursor`).
- Trainer: `examples/pi05/mars/cursor/train/train_adapter.py --kind task`. Flow matching
  through the adapted field on the context's demonstrations, plus λ_id toward the identity
  and λ_field through one user's frozen user module; a share `p_free` of layouts has no
  obstacle, where the module must stay the identity; goal dropout as the policy's. A 300-demo
  module trains in about 13 minutes.
- Evaluation: the grid, `configs/sweeps/grid.yaml` on `condition_grid.yaml`: users U1–U4,
  contexts T2–T5 (T1 open field aside), each task module trained with one user's module in
  the loop (T2/U1, T3/U2, T4/U3, T5/U4); the other twelve cells compose modules that never
  met. Tuned per cell on seed 0 (200 trials), tested on 500; later seeds retrain the modules
  and run at seed 0's settings on the same trials. Fig. 3e pools success over all 16 cells.
- Reference numbers on the 16 obstacle cells: MARS-UT (parametric) 73.3%, retrained policy
  72.6%, fine-tuned + MARS-U on 300 demos 46.0%, MARS-U 18.1%, MARS-T 13.2%, teleop 18.8%,
  flow control 11.0%, FRS 14.1%.
- Weights on this machine: `lerobot/pi05_base` (complete, 14 GB) in the Hugging Face cache.
  `google/paligemma-3b-pt-224` holds only its config and tokenizer (gated; no weights).
  Two RTX 6000 Ada (49 GB); GPU 1 idle at the time of writing.

## Goals

1. A task module whose only knowledge of the context comes from a rendered image of the
   layout, through frozen VLM features, trained with the existing trainer and loss.
2. Compare it, on the grid's 16 obstacle cells and trials, against the parametric module
   and the controls below, at 300 demonstrations per context.
3. Test alongside it one module trained on the four contexts pooled, that reads which
   context it is in from the picture.
4. Keep the result transferable: the feature the module consumes must be something π0.5
   exposes on the robot without extra models.

## Non-goals

- Any LIBERO or real-robot run. The transfer path is described at the end, not built.
- Leaving a context out, or designing a new context: "unseen" means, as in Fig. 3, a
  user–task combination the module never trained on, with the policy frozen.
- Touching the user modules, the policy, the baselines, the FRS schedule or the tuning
  protocol. Every existing arm is reused as it stands.
- Language conditioning. The cursor's prompt would carry no information; it belongs to
  the robot step.

## Anti-goals (what would make this a failure even if it ran)

- The module learning the goal from the picture beyond what the parametric module already
  can. Nothing in the render marks the goal; the policy is goal-free and the operator's
  command is what disambiguates. The task geometry itself does point at the goal (T4's ring
  is centred on it, T2's disc and T3's wall sit at mid-reach), but the disc tokens carry that
  too, so the standard is parity with the parametric module's information, measured in the
  gate below (goal-hidden hits), not argued.
- A result that depends on hand-coded perception: a detector that emits disc lists would
  reproduce today's interface and prove nothing about learning context from an image.
- Changing the evaluation so the new arms are not trial-for-trial comparable with Fig. 3.

## Decisions (the user's)

- Demonstration budget: 300 per context, the grid's, for every new arm.
- Arms: per-task image modules (one per context) and one pooled image module, both
  evaluated on the full grid with the diagonal as the trained cells.
- Evaluation standard: Fig. 3c–e's. A task module trains with one user's module in the loop
  and is tested with all four; success pooled over the 16 obstacle cells and per context.
- The VLM is π0.5's own (PaliGemma: SigLIP vision tower + multimodal projector + Gemma),
  taken from the π0.5 checkpoint, frozen.
- Deployment arm for the robot: chosen from the results later.

## Decisions (mine, as the builder's starting point; each may be overridden)

**Scene tokens as a pluggable source.** The task module keeps its cross-attention and
head; what changes is where its keys/values come from. `LearnedTaskModule` grows a
`token_source` in its config:

- `discs` (today's behaviour, the default; existing checkpoints load unchanged),
- `vlm_image`: the 256 tokens `embed_image` returns for a 224×224 render — SigLIP patches
  after the multimodal projector, in Gemma's width (2048). These are the tokens π0.5 feeds
  its language model, and they are identical in kind to what LIBERO would supply.
- `vlm_prefix` (second feature, run after `vlm_image` has a result): the last hidden states
  of PaliGemma's language model over [image tokens, a fixed prompt], i.e. what the action
  expert actually attends to through the prefix. Same module, different keys.
- `cnn`: a small convolutional encoder trained from scratch, jointly with the module, on the
  same render, giving tokens on a coarse grid. This is the control for "does the pretrained
  VLM buy anything at 300 demos".

For any image source the module projects the tokens to d with a trainable linear layer,
adds a learned 2D positional embedding over the token grid, keeps the null token and the
zero-initialized head. The cursor position stays in the queries; the module has to learn
"where is the gap relative to me" from absolute tokens, which the disc tokens gave for free.

**What is rendered.** One image per layout, 224×224 RGB, the unit screen mapped to the
square: obstacle discs and target circles on a plain background, in a high-contrast palette
(dark obstacles, a distinct mid-tone for targets, light background), not the display's pastel
one — the pygame colours span a third of SigLIP's input range for obstacles and a tenth for
targets, and a near-white-on-white render failing would be indistinguishable from the module
failing. Fidelity to the display is not a requirement; the display palette is an optional
ablation, not part of the gate. No goal highlight, no cursor, no HUD, no plan or command
overlay. Targets are drawn, as a camera would show them. Rendered headlessly from the disc
list (not through pygame) so that training and sweeps produce it in batch; anti-aliased, with a
minimum stroke so the wall's 0.015-radius discs (about 3 px at 224) stay visible. The
obstacle-free layouts (`p_free`) render targets only; the module must be the identity there,
as now.

**Per-layout features, precomputed.** Because the render is per layout and the VLM is
frozen, scene tokens are computed once per layout (training: the demo splits; test: once per
trial at its start, batched over a sweep part's trials) and reach the module through the
observation next to `obstacles` — `bind(obs)` and `ReversalAdapter.field` need no change. At
test time the observation is rebuilt from `Layouts` on every chunk
(`SharedController.observation` in `cursor_sa/steering.py`), so per-trial features ride on
`Layouts` (its `__getitem__`, `tile`, `concat`; `lerobot.envs` is mypy-strict) or on the
controller, and `run_trials` / `ExpertChunkAgent` carry them too. The sweeps' layouts are
drawn from fixed seeds, so features can be precomputed per cell and cached on disk, keyed by
layout draw and render spec; sweep parts then never load the VLM. 300 layouts × 256 × 2048 in
bf16 is 0.3 GB; the pooled arm's 1 200 layouts 1.3 GB; the val split adds proportionally.
Keep features on the GPU when they fit, gather per batch otherwise. The image modules train
on the same draw recipe as the parametric ones (same `--seed`, n 300, targets, reach, `p_free`,
obstacle), so each per-task module sees the demonstrations its parametric counterpart saw.

**The vision tower comes from `lerobot/pi05_base`.** Load the π0.5 policy once and use
`PI05Policy.model.paligemma_with_expert.embed_image` (and, for `vlm_prefix`, its language
model), or load only the vision tower and projector weights from that checkpoint into a
bare PaliGemma built from the cached config; either is fine. π0.5 training fine-tunes the
VLM, so these are not Google's PaliGemma weights; they are the right ones, since a LIBERO
fine-tune starts from them and would supply its own at the robot step.

**The pooled arm's data.** The four per-task modules' demonstrations together (300 per
context, each context's demos with that context's user module in the λ_field term: T2/U1,
T3/U2, T4/U3, T5/U4), one module. Its trained cells are the same diagonal; its unseen cells
the same twelve. It gets the same steps per demonstration as a per-task module (so about
four times the steps), tuned on the grid like any MARS variant.

**Controls, all on the grid's trials.** The parametric MARS-UT (existing checkpoints
`task_<ctx>_n300_s0_with_<user>`; not retrained), fine-tuned + MARS-U (existing `grid_ft`),
MARS-U and the three training-free baselines (existing results), and the `cnn` per-task
modules (new). The parametric module is the reference the image module should approach;
fine-tuned + MARS-U is what it has to beat for the cost argument.

**Gate before any sweep.** `grid_task_check`'s test (`configs/sweeps/grid_task_check.yaml`:
the goal-given policy's own chunks executed with each task module, `oracle_task`, against the
policy alone, `oracle`; no results folder is on disk, so the parametric reference is re-run,
about 2 min) on every new module, per context (the pooled module against each context's
parametric module). The rule: `oracle_task` hits ≥ 80% and its collision rate within +10
points of the parametric module's on that context. In the same run, the goal-*hidden* hits of
`policy_task` (the module with no user module): an image module whose goal-hidden hits exceed
the parametric module's by more than 10 points is flagged as seeing more of the goal than the
reference — its grid results are still produced but reported with the flag, for the user to
judge. A module that fails the collision rule goes back to training (render ablation first,
then the fallback below) before it costs a sweep. The fallback is triggered from this gate,
not from the sweep.

**Fallback if the absolute view does not train.** Render per chunk, centered on the cursor
(a window of fixed size in screen units), so the tokens are relative to the agent as the disc
tokens were. Costs one VLM pass per chunk at test time and per demonstration step in training
(precomputable: 300 demos × ~100 steps). Only if the static render fails the gate.

**Seeds.** Seed 0 tuned per cell (200 trials), tested on 500, like the grid; then seeds 1 and
2 retrain the new modules on fresh draws at seed 0's settings (`--module-seed k
--fixed-settings`, the demo-time curve's three seeds), each seed-k task module trained with the
seed-k user modules in the loop as the grid's parametric seeds were; seeds 3–4 if time allows.
The parametric and fine-tuned arms already have five.

**Names.** Checkpoints under `checkpoints/adapters/image/`:
`task_<ctx>_n300_s<k>_with_<user>_<source>` for per-task modules,
`task_all_n1200_s<k>_<source>` for the pooled one (`_n1200_s<k>` so that `run_sweep`'s
`--module-seed` substitution, regex `(_[nt]\d+)_s0`, matches it; `meta` records 300 × 4, the
contexts, the render spec and the feature source). A sweep `configs/sweeps/grid_image.yaml` on
`condition_grid.yaml` with methods `mars_vlm`, `mars_vlm_all`, `mars_cnn` (and `mars_vlm_prefix`
when run), each `{user: cell, task: cell}` with the grid's `tune` block, plus `teleop` (the
trial-identity check, as `grid_ft` does) and the parametric `mars` re-run at its grid settings,
so that the sweep's paired leads and CIs (`lead_over_best`, `beats_all`, `best_other`, its
`focus`/`against`) compare the new arms with the reference inside one run. Parts are merged into
one results folder per seed as the grid's are.

**Deliverable.** A report (markdown plus the sweep's `summary.png`, `grid_summary.md`,
`ranking.csv`): per cell and per context, success of every new arm against the parametric
MARS-UT and fine-tuned + MARS-U (joined from `grid_ft_s<k>` across folders the way
`grid_data` in the figure builder does), the pooled-over-16-cells statistic with its CI over
seeds, and the gate's numbers (collision, goal-hidden hits). Fig. 3 and `build_figures.py` are
not touched; adding the arms to the paper's figure is a separate request once the results say
which arms matter.

**Code placement.** The token-source abstraction and the image token encoders live in
`lerobot.policies.steering` (they are not cursor-specific). The headless layout render lives
next to the layouts, in `lerobot.envs.cursor`. The VLM feature extraction (render → π0.5
`embed_image` / prefix) lives with the steering module or the π0.5 policy, wherever the
import stays lazy: the `mars` extra has no `transformers`, so the VLM steps run under
`--extra pi` (or `all`) and nothing in the `discs`/`cnn` paths may import it. The trainer gains
`--token-source`, `--render` options and the pooled-contexts mode; `convert_checkpoints.py` and
`load_adapter_module` round-trip the new config. One `LearnedTaskModule` class with
`token_source` in its config (not subclasses): `save_adapter_module` checks the exact type.
For `vlm_prefix`, the prompt is a fixed sentence in π0.5's prompt format with no state
tokens (so features stay per layout); whether to mimic the discretized-state prefix is
revisited when that arm is run.

**Tests** follow `tests/policies/steering` and `tests/examples/pi05_mars_cursor`: the render
puts a disc of known center and radius at the right pixels and never draws the goal; each
token source returns tokens of the documented shape and a fresh module is the identity for
every source; a saved module reloads to the same outputs; a `discs` checkpoint from before
this change loads and matches `_reference_steering`; the pooled trainer draws each context's
demos with its own user. The VLM path is tested with `embed_image` stubbed (or a
monkeypatched tiny Gemma config): `get_gemma_config` has no small variant and the existing
small-policy fixture builds a full-size random model gated on CUDA and an HF token, so no
test loads the 14 GB checkpoint.

## Left to the builder

- The exact render routine and anti-aliasing; the minimum stroke width.
- The `cnn` encoder's size and token grid (small; a few hundred thousand parameters).
- Whether `vlm_image` tokens are stored before or after the projector (after is what π0.5
  feeds its LM; before halves the memory; the trainable projection sees either).
- The pooled trainer's batching (one context per batch, or mixed) and its step count.
- Feature caching on disk between runs (keyed by layout and render spec) or recomputation.
- GPU assignment and the order of the sweep parts.

## Risks and how the design meets them

- *300 demos are few for 256 tokens of width 2048.* The module itself stays small (the
  projection is the only wide layer); λ_id and the obstacle-free layouts regularize toward
  the identity; the `cnn` control and the parametric reference bracket the answer. If the
  per-task modules pass the gate but the pooled one does not, that is a finding, not a bug.
- *Thin structures at 224 px.* Anti-aliased render with a minimum stroke; the render test
  checks the wall is visible. If SigLIP still misses the gap, the agent-centered fallback
  effectively zooms in.
- *Absolute tokens vs relative geometry.* Cursor position in the queries plus learned
  positional embeddings; the fallback restores equivariance.
- *Loading the 14 GB checkpoint for a 400M tower.* Load once per process and keep only the
  tower (and the LM for `vlm_prefix`); features are precomputed, so the sweep runners need it
  only at trial start.
- *Sweep cost.* Four new arms × 16 cells at seed 0 with tuning, then two seeds; the parts
  run in parallel per cell as the demo-time sweeps did. Precomputed features keep each part's
  cost at the parametric arm's.

## Transfer path (not built here)

On LIBERO, `LearnedTaskModule` is currently called as (x_t, t, v) with no scene. With
`vlm_prefix`, `bind(obs)` would carry π0.5's own prefix hidden states (or the image tokens
from its `embed_prefix`), which the policy computes anyway for every chunk, so the task
module reads the scene at zero extra inference cost and the trainer is the LIBERO analogue of
`train_adapter.py --kind task` on a task held out from the policy's fine-tuning. That trainer
does not exist yet (`train_user_module.py` is the only LIBERO trainer) and is the next
design.
