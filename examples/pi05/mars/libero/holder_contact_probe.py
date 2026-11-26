"""Test whether camera-conditioned scores rank contact-free prefixes on unseen reset IDs."""

import json
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional
from train_visual_task import AVOID_PROMPT, DemoChunks, save_json


class ContactHead(nn.Module):
    def __init__(self, state_dim, token_dim):
        super().__init__()
        self.query = nn.Linear(state_dim + 30, 32)
        self.keys = nn.Linear(token_dim, 32)
        self.values = nn.Linear(token_dim, 32)
        self.out = nn.Sequential(nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 2))

    def forward(self, views, state, xyz):
        query = self.query(torch.cat((state.flatten(1), xyz.flatten(1)), dim=1)).tanh()
        weights = (self.keys(views) @ query.unsqueeze(-1) / 32**0.5).softmax(1)
        visual = (weights * self.values(views)).sum(1)
        return self.out(torch.cat((query, visual), dim=1))


@torch.no_grad()
def encode(s, path):
    raw, _ = DemoChunks([path], 50, AVOID_PROMPT).batch(np.random.default_rng(0), 1)
    batch = s.preprocessor(raw)
    _, patches = s.policy.camera_field(batch, stage="post")
    tokens, _ = s.adapter.task.tokens({"scene": {"pi05_cameras": patches}}, 1)
    # Retain a 4x4 grid in each of the two camera views.
    grid = tokens[:, 1:].reshape(1, 2, 4, 4, 4, 4, -1).mean((3, 5)).reshape(1, 32, -1)
    return grid[0].float().cpu(), raw["observation.state"][0].float().cpu()


def build_dataset(s, out, files=None):
    from holder_interventions import H, perturb

    def label(start, contact):
        return contact["with"] if contact and start <= contact["step"] < start + H else None

    dataset = []
    for f in sorted(files if files is not None else (out / "pairs").glob("*.npz")):
        row = json.loads(f.with_suffix(".json").read_text())
        views, state = encode(s, f)
        with np.load(f) as data:
            target = data["action"]
            if len(target) != H:
                continue
            candidates = [(target, None)]
            if row["rescue"]:
                start = row["branches"][-1]["start"]
                original = data["original"]
                candidates.insert(0, (original, label(start, row["contact"])))
                dx, dy = original[:, :2].mean(0)
                for branch in row["branches"]:
                    if branch["start"] != start or branch["joint"]:
                        continue
                    if branch["direction"] == "up":
                        direction = np.array([0, 0, 1])
                    elif np.hypot(dx, dy) > 1e-5:
                        direction = np.array([-dy, dx, 0]) / np.hypot(dx, dy)
                        if branch["direction"] == "right":
                            direction = -direction
                    else:
                        continue
                    candidates.append(
                        (perturb(original, direction, branch["magnitude"]), label(start, branch["contact"]))
                    )
        acts = torch.tensor(np.stack([a[:, :3] for a, _ in candidates]), dtype=torch.float32)
        labels = torch.tensor([[float(k == tag) for k in ("robot", "bowl")] for _, tag in candidates])
        dataset.append(
            {
                "file": f.name,
                "group": row["init_state_id"],
                "object": (row["contact"] or {}).get("obstacle") if "contact" in row else None,
                "views": views,
                "state": state,
                "actions": s.steering_policy()._normalized_chunk(acts),
                "labels": labels,
            }
        )
    return dataset


def train_head(rows):
    torch.manual_seed(0)
    head = ContactHead(rows[0]["state"].numel(), rows[0]["views"].shape[-1])
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    sizes = [len(r["actions"]) for r in rows]
    vv = torch.cat([r["views"][None].expand(n, -1, -1) for r, n in zip(rows, sizes, strict=True)])
    ss = torch.cat([r["state"][None].expand(n, -1) for r, n in zip(rows, sizes, strict=True)])
    aa = torch.cat([r["actions"] for r in rows])
    yy = torch.cat([r["labels"] for r in rows])
    weight = ((len(yy) - yy.sum(0)) / yy.sum(0).clamp_min(1)).clamp(1, 10)
    for _ in range(300):
        idx = torch.randint(len(yy), (32,))
        loss = functional.binary_cross_entropy_with_logits(
            head(vv[idx], ss[idx], aa[idx]), yy[idx], pos_weight=weight
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    return head.eval()


def probe(s, out, *, files=None, objects=()):
    manifest = json.loads((out / "manifest.json").read_text())
    s.adapter.load_modules(task_path=manifest["checkpoint"], device=s.policy.config.device)
    rows = build_dataset(s, out, files) if files is not None else build_dataset(s, out)
    groups = sorted({r["group"] for r in rows})
    folds = np.array_split(np.random.default_rng(0).permutation(groups), 4)
    judgments = defaultdict(list)
    for fold in folds:
        head = train_head([r for r in rows if r["group"] not in fold])
        test = [r for r in rows if r["group"] in fold and r["labels"][0].any()]
        for r in test:
            kind = int(r["labels"][0].argmax())
            with torch.no_grad():
                logits = head(
                    r["views"][None].expand(2, -1, -1), r["state"][None].expand(2, -1), r["actions"][:2]
                )
            judgments[r["object"] if objects else ("robot", "bowl")[kind]].append(
                {
                    "file": r["file"],
                    "type_margin": float(logits[0, kind] - logits[1, kind]),
                    "margin": float((functional.softplus(logits[0]) - functional.softplus(logits[1])).mean()),
                }
            )
    scores = {key: sum(x["margin"] > 0 for x in values) for key, values in judgments.items()}
    passed = all(
        judgments[k] and scores.get(k, 0) >= 2 * len(judgments[k]) / 3 for k in (objects or ("bowl", "robot"))
    )
    save_json(out / "contact_probe.json", {"correct": scores, "held_out": judgments, "passed": bool(passed)})
    if passed:
        head = train_head(rows)
        # Frozen visual/state inputs for the corrective and preservation samples.
        inputs = {r["file"]: (r["views"], r["state"]) for r in rows}
        return head, inputs
    return None
