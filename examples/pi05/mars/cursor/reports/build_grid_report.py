"""PDF report on the compositional adapter grid: every method side by side on the users x tasks grid.

    python reports/build_grid_report.py

Reads results/sim/sweeps/{grid_s0,grid_task_check,grid_expert_check}/ and configs/sweeps/grid.yaml; writes
reports/grid_report.pdf. The figures are drawn here and embedded; nothing else is written.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import datetime
import json
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, PathPatch, Rectangle
from matplotlib.path import Path as MPath
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from cursor_sa.paths import CONFIGS, REPORTS, RESULTS  # noqa: E402

SWEEPS = RESULTS / "sim" / "sweeps"
OUT = REPORTS / "grid_report.pdf"
TMP = Path(tempfile.mkdtemp())
PAGE_W = letter[0] - 1.2 * inch

FONTS = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DejaVu", FONTS / "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", FONTS / "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DejaVu-Oblique", FONTS / "DejaVuSans-Oblique.ttf"))
pdfmetrics.registerFont(TTFont("DejaVu-BoldOblique", FONTS / "DejaVuSans-BoldOblique.ttf"))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu-Oblique",
                              boldItalic="DejaVu-BoldOblique")
ss = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="DejaVu-Bold", fontSize=15, spaceBefore=10)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="DejaVu-Bold", fontSize=11.5, spaceBefore=6)
BODY = ParagraphStyle("body", parent=ss["BodyText"], fontName="DejaVu", fontSize=9, leading=12.5)
CAP = ParagraphStyle("cap", parent=BODY, fontSize=7.5, leading=9.5, textColor=colors.HexColor("#444444"))
TITLE = ParagraphStyle("title", parent=ss["Title"], fontName="DejaVu-Bold", fontSize=17)
CODE = ParagraphStyle("code", parent=BODY, fontName="Courier", fontSize=7.5, leading=9.5)

# the figures' ink and surface (cursor_sa.analysis), and one hue per method in the order the bars are drawn:
# every adjacent pair passes the data-viz palette checks (lightness, chroma, CVD and normal-vision separation)
INK, INK_2, SURFACE, GRID_INK, BASE_INK, MUTED = "#1f1f1e", "#5f5e5a", "#fcfcfb", "#e1e0d9", "#c3c2b7", "#898781"
COLOR = {"teleop": "#008300", "flow_control": "#e87ba4", "frs": "#e69f00", "mars_task": "#8a63d2", "mars_user": "#6da7ec",
         "retrained_mars_user": "#2e7da0", "mars": "#2f45c4"}  # the figures' palette (figures/build_figures.py)
LABEL = {"teleop": "Teleop", "flow_control": "Flow control", "frs": "FRS", "mars_user": "MARS-U (user module)",
         "mars_task": "MARS-T (task module)", "mars": "MARS (user + task modules)",
         "retrained_mars_user": "Retrained policy + MARS-U"}
SHORT = {"teleop": "teleop", "flow_control": "flow control", "frs": "FRS", "mars_user": "user module",
         "mars_task": "task module", "mars": "U + T", "retrained_mars_user": "retrained + U"}
FOCUS = "mars"
TASK_SEEDS = (11, 5, 8, 2, 7)  # one sampled layout per task for the illustration
FIGN = {k: i + 1 for i, k in enumerate(("tasks", "means", "heat_success", "methods", "heat_collision", "leads", "depths"))}
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})

fig_no, tab_no = [0], [0]


# ---------------------------------------------------------------- data

grid = json.loads((SWEEPS / "grid_s0" / "test.json").read_text())
TEST, ARGS = grid["test"], grid["args"]
TUNING = json.loads((SWEEPS / "grid_s0" / "tuning.json").read_text())["tuning"]
TASK_CHECK = json.loads((SWEEPS / "grid_task_check" / "test.json").read_text())["test"]
EXPERT_CHECK = json.loads((SWEEPS / "grid_expert_check" / "test.json").read_text())["test"]
GRID_CFG = yaml.safe_load((CONFIGS / "sweeps" / "grid.yaml").read_text())
STEER = json.loads((RESULTS / "research" / "steering_states" / "U1xT4.json").read_text())  # research/steering_states.py
G = ARGS["sweep"]["grid"]
USERS, TASKS, METHODS = G["users"], G["tasks"], list(ARGS["sweep"]["methods"])
NOISE = G["noises"][0]
TRAIN = {tuple(p) for p in GRID_CFG["grid"]["train"]}
OBSTACLE_TASKS = [t for t in TASKS if G["cells"][f"{USERS[0]}x{t}"]["obstacle"]]


def cell(u, t):
    return TEST[f"{u}x{t}|{NOISE}"]


def limit(t):
    return G["cells"][f"{USERS[0]}x{t}"]["time_limit_s"] or 8.0


def corruption_label(spec):
    if "scale" in spec:
        return f"gain ({spec['scale'][0]:g}, {spec['scale'][1]:g})"
    if "rotation_deg" in spec:
        return f"rotation {spec['rotation_deg']:g}°"
    return f"curl {spec['curl']:g}/s"


def task_label(t):
    spec = G["cells"][f"{USERS[0]}x{t}"]["obstacle"]
    if not spec:
        return "no obstacle"
    if spec["kind"] == "ring":
        return f"ring around the {spec['around']}"
    return {"wall": "wall with a gap", "disc": "disc"}[spec["kind"]]


def pooled(cells, m, metric="success"):
    return 100 * np.mean([c["metrics"][m][metric] for c in cells])


def hits(c, m):
    return np.array([o == "hit" for o in c["trials"]["outcome"][m]], float)


def paired(c, a, b, n_boot=2000, seed=1):
    """a minus b on the cell's trials: [mean, lower, upper] of a paired bootstrap."""
    d = hits(c, a) - hits(c, b)
    boot = np.random.default_rng(seed).integers(0, len(d), (n_boot, len(d)))
    return [float(d.mean()), *np.percentile(d[boot].mean(1), [2.5, 97.5])]


pct = lambda x: f"{100 * x:.1f}%"
pts = lambda v: f"{100 * v[0]:+.0f}" + ("*" if v[1] > 0 or v[2] < 0 else "")
pm = lambda vals: f"{min(vals):.0f}–{max(vals):.0f}"
signed = lambda v: "0" if round(v) == 0 else f"{v:+.0f}"
span = lambda vals: f"{signed(min(vals))} to {signed(max(vals))}"


# ---------------------------------------------------------------- figures


def frame(ax, value_axis):
    ax.set_facecolor(SURFACE)
    for side, spine in ax.spines.items():
        spine.set_visible(side == ("left" if value_axis == "x" else "bottom"))
        spine.set_color(BASE_INK)
    ax.grid(False)
    ax.grid(True, axis=value_axis, color=GRID_INK, lw=0.8, zorder=0)
    ax.tick_params(colors=MUTED, length=0, labelsize=8)


def rounded_bar(ax, x0, x1, y0, y1, color, horizontal=False, radius=4):
    """A bar square at its baseline and rounded at its value's end (radius in pixels); set the limits first."""
    (ax0, ay0), (ax1, ay1) = ax.transData.transform([(0, 0), (1, 1)])
    sx, sy = abs(ax1 - ax0), abs(ay1 - ay0)
    if horizontal:
        r = min(radius, (y1 - y0) * sy / 2, (x1 - x0) * sx)
        rx, ry = r / sx, r / sy
        verts = [(x0, y0), (x1 - rx, y0), (x1, y0), (x1, y0 + ry), (x1, y1 - ry), (x1, y1), (x1 - rx, y1), (x0, y1), (x0, y0)]
    else:
        r = min(radius, (x1 - x0) * sx / 2, (y1 - y0) * sy)
        rx, ry = r / sx, r / sy
        verts = [(x1, y0), (x1, y1 - ry), (x1, y1), (x1 - rx, y1), (x0 + rx, y1), (x0, y1), (x0, y1 - ry), (x0, y0), (x1, y0)]
    codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CURVE3, MPath.CURVE3,
             MPath.LINETO, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor="none", lw=0, zorder=2))


def legend(fig, methods, y, ncol):
    fig.legend([Patch(color=COLOR[m]) for m in methods], [LABEL[m] for m in methods], loc="upper center",
               bbox_to_anchor=(0.5, y), ncol=ncol, fontsize=7.8, frameon=False, handlelength=1.0, handleheight=1.0,
               columnspacing=1.3)


def users_note():
    return "   ".join(f"{u} {corruption_label(G['cells'][f'{u}x{TASKS[0]}']['corruption'])}" for u in USERS)


def fig_methods(path):
    """Every method's test success in every cell: a row per task, a group per user, a bar per method."""
    fig, axes = plt.subplots(len(TASKS), 1, figsize=(10.5, 2.05 * len(TASKS)), dpi=200, facecolor=SURFACE)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.93, bottom=0.045, hspace=0.5)
    width, gap = 0.86 / len(METHODS), 0.012
    for ax, t in zip(axes, TASKS):
        frame(ax, "y")
        ax.set(xlim=(-0.5, len(USERS) - 0.5), ylim=(0, 108), yticks=[0, 25, 50, 75, 100], xticks=range(len(USERS)))
        ax.set_yticklabels(["0", "25", "50", "75", "100%"])
        for i, u in enumerate(USERS):
            c = cell(u, t)
            for k, m in enumerate(METHODS):
                s, lo, hi = (100 * v for v in c["success_ci"][m])
                x0 = i - 0.43 + k * width
                rounded_bar(ax, x0 + gap / 2, x0 + width - gap / 2, 0, s, COLOR[m])
                ax.plot([x0 + width / 2] * 2, [lo, hi], color=INK_2, lw=0.8, zorder=3, solid_capstyle="butt")
                if m == FOCUS:
                    ax.text(x0 + width / 2, hi + 1.5, f"{s:.0f}", ha="center", va="bottom", fontsize=6.5, color=INK,
                            weight="semibold")
        ax.set_xticklabels([u + ("*" if (u, t) in TRAIN and t in OBSTACLE_TASKS else "") for u in USERS], fontsize=8.5, color=INK)
        ax.set_title(f"{t}  ·  {task_label(t)}, {limit(t):g} s", loc="left", fontsize=9, color=INK, pad=4)
    legend(fig, METHODS, 0.995, 4)
    fig.text(0.055, 0.004, f"Users: {users_note()}.   * the user the task's module was trained with.   "
             "Lines: 95% bootstrap CIs; the number is U + T's success.", fontsize=7, color=INK_2)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def fig_tasks(path):
    """One sampled layout per task: the targets (the goal filled), the obstacle's discs, the start, and the
    expert's route, which the demonstrations follow."""
    from cursor_sa.analysis import draw_scene
    from cursor_sa.env import ROUTE_CLEARANCE, TaskConfig, obstacle_spec, plan_route, sample_layouts
    fig, axes = plt.subplots(1, len(TASKS), figsize=(12.5, 2.95), dpi=200, facecolor=SURFACE)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.03, wspace=0.08)
    for ax, t, seed in zip(axes, TASKS, TASK_SEEDS):
        spec = G["cells"][f"{USERS[0]}x{t}"]["obstacle"]
        cfg = TaskConfig(min_reach=0.35, max_reach=0.5, obstacle=obstacle_spec(spec) if spec else None)
        layout = sample_layouts(np.random.default_rng(seed), 1, (4, 4), cfg)
        draw_scene(ax, layout, cfg)
        route = plan_route(layout.start[0], layout.goal_pos[0], layout.real_obstacles(0), ROUTE_CLEARANCE)
        pts = np.vstack([layout.start[0][None], np.asarray(route if route is not None and len(route) else layout.goal_pos[0][None])])
        ax.plot(pts[:, 0], pts[:, 1], color=COLOR[FOCUS], lw=1.6, zorder=3, solid_capstyle="round")
        ax.set_facecolor(SURFACE)
        ax.set_title(f"{t}  ·  {task_label(t)}, {limit(t):g} s", fontsize=8.5, color=INK, pad=5)
        for spine in ax.spines.values():
            spine.set_color(GRID_INK)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def fig_means(path):
    """Every method's success averaged over the obstacle cells and over the open-field cells."""
    groups = [(f"Obstacle tasks ({', '.join(OBSTACLE_TASKS)}): mean of {len(USERS) * len(OBSTACLE_TASKS)} cells",
               [cell(u, t) for u in USERS for t in OBSTACLE_TASKS]),
              (f"No obstacle ({', '.join(t for t in TASKS if t not in OBSTACLE_TASKS)}): mean of {len(USERS)} cells",
               [cell(u, t) for u in USERS for t in TASKS if t not in OBSTACLE_TASKS])]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 2.9), dpi=200, facecolor=SURFACE)
    fig.subplots_adjust(left=0.24, right=0.99, top=0.86, bottom=0.12, wspace=0.35)
    for g, (ax, (title, cs)) in enumerate(zip(axes, groups)):
        frame(ax, "x")
        ax.set(xlim=(0, 112), ylim=(-0.6, len(METHODS) - 0.4), xticks=[0, 25, 50, 75, 100], yticks=[])
        ax.set_xticklabels(["0", "25", "50", "75", "100%"])
        for i, m in enumerate(METHODS):
            h = [hits(c, m) for c in cs]
            boot = np.mean([x[rng.integers(0, len(x), (2000, len(x)))].mean(1) for x in h], 0)
            s, lo, hi = 100 * np.mean([x.mean() for x in h]), *np.percentile(100 * boot, [2.5, 97.5])
            y = len(METHODS) - 1 - i
            rounded_bar(ax, 0, s, y - 0.3, y + 0.3, COLOR[m], horizontal=True)
            ax.plot([lo, hi], [y, y], color=INK_2, lw=0.9, zorder=3, solid_capstyle="butt")
            ax.text(hi + 1.5, y, f"{s:.0f}%", va="center", fontsize=8, color=INK, weight="semibold" if m == FOCUS else "normal")
            if g == 0:
                ax.text(-2, y, LABEL[m], ha="right", va="center", fontsize=8.5, color=INK,
                        weight="semibold" if m == FOCUS else "normal")
        ax.set_title(title, loc="left", fontsize=9, color=INK, pad=6)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def fig_heat(path, metric, title, hue):
    """One users x tasks map per method, all on one row."""
    fig, axes = plt.subplots(1, len(METHODS), figsize=(12.5, 2.55), dpi=200, facecolor=SURFACE)
    fig.subplots_adjust(left=0.03, right=0.995, top=0.74, bottom=0.12, wspace=0.12)
    cmap = LinearSegmentedColormap.from_list("seq", ["#f3f4f6", hue])
    for k, (ax, m) in enumerate(zip(axes, METHODS)):
        M = np.array([[100 * cell(u, t)["metrics"][m][metric] for t in TASKS] for u in USERS])
        ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto")
        for i in range(len(USERS)):
            for j in range(len(TASKS)):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] > 55 else INK)
                if (USERS[i], TASKS[j]) in TRAIN:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, lw=1.1))
        ax.set(xticks=range(len(TASKS)), xticklabels=TASKS, yticks=range(len(USERS)), yticklabels=USERS if k == 0 else [])
        ax.tick_params(length=0, labelsize=7.5, colors=INK_2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(LABEL[m].replace(": ", ":\n").replace(" + its", "\n+ its"), fontsize=7.8, color=INK, pad=4)
    fig.text(0.03, 0.95, title, fontsize=9.5, color=INK, va="top")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def fig_leads(path):
    """U + T's paired lead over every other method, cell by cell, on the obstacle tasks."""
    others = [m for m in METHODS if m != FOCUS]
    fig, axes = plt.subplots(1, len(OBSTACLE_TASKS), figsize=(11, 3.7), dpi=200, facecolor=SURFACE, sharey=True)
    fig.subplots_adjust(left=0.04, right=0.995, top=0.8, bottom=0.17, wspace=0.08)
    step = 0.115
    for ax, t in zip(axes, OBSTACLE_TASKS):
        frame(ax, "x")
        ax.axvline(0, color=INK_2, lw=0.8, zorder=1)
        for i, u in enumerate(USERS):
            for k, m in enumerate(others):
                mean, lo, hi = (100 * v for v in cell(u, t)["lead"][m])
                y = (len(USERS) - 1 - i) + (len(others) - 1 - k) * step - step * (len(others) - 1) / 2
                ax.plot([lo, hi], [y, y], color=COLOR[m], lw=1.3, solid_capstyle="butt", zorder=2)
                ax.plot(mean, y, "o", color=COLOR[m], ms=4.2, mec=SURFACE, mew=0.6, zorder=3)
        ax.set(yticks=range(len(USERS)), yticklabels=USERS[::-1], xlim=(-20, 100), ylim=(-0.55, len(USERS) - 0.45),
               xticks=[0, 25, 50, 75, 100])
        ax.tick_params(axis="y", labelsize=8.5, colors=INK)
        ax.set_title(f"{t}  ·  {task_label(t)}", loc="left", fontsize=9, color=INK, pad=4)
    fig.text(0.5, 0.03, "U + T minus the method, points (paired bootstrap 95% CI)", ha="center", fontsize=8.5, color=INK_2)
    legend(fig, others, 0.995, 6)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def depth_range(mode):
    """The depths the sweep tried for a mode, as 'first–last' (one list, or a list per mode)."""
    spec = ARGS["sweep"]["depths"]
    d = spec[mode] if isinstance(spec, dict) else spec
    return f"{min(d)}–{max(d)}"


def fig_depths(path, cells):
    """Success and collisions against the reversal depth on the tuning trials, for the methods with a depth."""
    spec = ARGS["sweep"]["depths"]  # one list, or a list per mode (flow control's tau counts chunk steps of 16)
    mode_of = lambda m: ARGS["sweep"]["methods"][m].get("mode")
    depths_of = lambda m: spec[mode_of(m)] if isinstance(spec, dict) else spec
    ticks = sorted({d for m in METHODS if not isinstance(spec, dict) or mode_of(m) in spec for d in depths_of(m)})
    fig, axes = plt.subplots(2, len(cells), figsize=(10.5, 5.4), dpi=200, facecolor=SURFACE, sharex=True)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.86, bottom=0.09, hspace=0.3, wspace=0.18)
    for j, key in enumerate(cells):
        tune = TUNING[f"{key}|{NOISE}"]
        for i, metric in enumerate(("success", "collision")):
            ax = axes[i, j]
            frame(ax, "y")
            for m in METHODS:
                if "None" in tune[m]:
                    ax.axhline(100 * tune[m]["None"][metric], color=COLOR[m], lw=1.4, ls=(0, (4, 2)))
                else:
                    depths = depths_of(m)
                    ax.plot(depths, [100 * tune[m][str(d)][metric] for d in depths], color=COLOR[m], lw=1.8,
                            marker="o", ms=3.5, mec=SURFACE, mew=0.5)
            ax.set(xticks=ticks, ylim=(0, 104), yticks=[0, 25, 50, 75, 100])
            ax.set_yticklabels(["0", "25", "50", "75", "100%"] if j == 0 else [])
            u, t = key.split("x")
            ax.set_title(f"{key}  ·  {corruption_label(G['cells'][key]['corruption'])}, {task_label(t)}: "
                         f"{'success' if metric == 'success' else 'collisions'}", loc="left", fontsize=8.5, color=INK, pad=4)
            if i == 1:
                ax.set_xlabel("depth (flow control: chunk steps of 16 carrying the command; FRS and MARS: steps reversed of 10)",
                              fontsize=8, color=INK_2)
    legend(fig, METHODS, 0.995, 4)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- document pieces


def img(path, caption, max_h=4.6 * inch):
    fig_no[0] += 1
    w, h = PILImage.open(path).size
    s = min(PAGE_W / w, max_h / h)
    return KeepTogether([Image(str(path), w * s, h * s),
                         Paragraph(f"<b>Figure {fig_no[0]}.</b> {caption}", CAP), Spacer(1, 8)])


def table(rows, caption, col_widths=None, font=7.2, shade=(), bold_rows=()):
    tab_no[0] += 1
    style = lambda r: ParagraphStyle("tc", parent=BODY, fontName="DejaVu-Bold" if r in bold_rows else "DejaVu",
                                     fontSize=font, leading=font + 2)
    tb = Table([[Paragraph(str(c), style(r)) for c in row] for r, row in enumerate(rows)], colWidths=col_widths, repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef6")),
          ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#bbbbbb")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
    st += [("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f4f4f0")) for r in shade]
    tb.setStyle(TableStyle(st))
    return KeepTogether([tb, Spacer(1, 3), Paragraph(f"<b>Table {tab_no[0]}.</b> {caption}", CAP), Spacer(1, 8)])


def bullets(items):
    return ListFlowable([ListItem(Paragraph(t, BODY), leftIndent=12) for t in items], bulletType="bullet",
                        start="•", leftIndent=12)


P = lambda text: Paragraph(text, BODY)


def main():
    paths = {name: TMP / f"{name}.png" for name in FIGN}
    fig_tasks(paths["tasks"])
    fig_methods(paths["methods"])
    fig_means(paths["means"])
    fig_heat(paths["heat_success"], "success", "Test success (%) of every method, users × tasks; outlined: the cells trained on",
             "#1c5cab")
    fig_heat(paths["heat_collision"], "collision", "Collisions (% of trials) of every method, users × tasks", "#8a4a1e")
    fig_leads(paths["leads"])
    fig_depths(paths["depths"], ["U1xT4", "U3xT5"])

    # numbers the text quotes
    obstacle = [cell(u, t) for u in USERS for t in OBSTACLE_TASKS]
    open_field = [cell(u, t) for u in USERS for t in TASKS if t not in OBSTACLE_TASKS]
    beats = sum(c["beats_all"] for c in obstacle)
    lead_best = [100 * c["lead_over_best"][0] for c in obstacle]
    mean_obs = {m: pooled(obstacle, m) for m in METHODS}
    mean_open = {m: pooled(open_field, m) for m in METHODS}
    coll = {m: [100 * c["metrics"][m]["collision"] for c in obstacle] for m in METHODS}
    by_task = {t: [100 * cell(u, t)["metrics"][FOCUS]["success"] for u in USERS] for t in OBSTACLE_TASKS}
    add_task = {(u, t): cell(u, t)["lead"]["mars_user"] for u in USERS for t in OBSTACLE_TASKS}
    add_user = {(u, t): cell(u, t)["lead"]["mars_task"] for u in USERS for t in OBSTACLE_TASKS}
    vs_retr = {(u, t): cell(u, t)["lead"]["retrained_mars_user"] for u in USERS for t in OBSTACLE_TASKS}
    unseen = [k for k in add_task if k not in TRAIN]
    seen = [k for k in add_task if k in TRAIN]
    alone = {m: [paired(cell(u, t), m, cell(u, t)["best_other"]) for u in USERS for t in OBSTACLE_TASKS]
             for m in ("mars_user", "mars_task", "retrained_mars_user")}
    retr_alone = {t: TASK_CHECK[f"U0x{t}|0.0"]["metrics"] for t in OBSTACLE_TASKS}
    t1 = TASKS[0]
    t1_lead = {u: cell(u, t1)["lead"]["teleop"] for u in USERS}
    tune = ARGS["tune_trials"]
    test_n = ARGS["test_trials"]

    doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch,
                            topMargin=0.6 * inch, bottomMargin=0.6 * inch,
                            title="Composed user × task reversal adapter: the grid")
    s = [Paragraph("Composed user × task reversal adapter: the grid", TITLE),
         Paragraph(f"Rice BCI Lab · lerobot examples/pi05/mars/cursor · results/sim/sweeps/grid_s0 · "
                   f"{datetime.date.today():%B %-d, %Y}", CAP), Spacer(1, 8)]

    # ------------------------------------------------------------ summary
    s += [Paragraph("Summary", H1), P(
        f"Four simulated users with different command corruptions × five tasks (the open field and four obstacle "
        f"layouts), seven methods in every cell, {test_n} test trials per cell with every method at the depth it "
        f"tuned on {tune} other trials. The reversal adapter of MARS has two modules trained apart: a "
        f"<b>user module</b> that undoes one user's corruption, calibrated on 100 open-field demonstrations, and a "
        f"<b>task module</b> that steers the frozen open-field policy around one task's obstacles, trained on 300 of "
        f"the task's demonstrations with one user's module in the loop. Composed at test time (U + T) they meet 12 user × task combinations that "
        f"never met in training."), Spacer(1, 4), bullets([
        f"<b>U + T beats teleop, flow control and FRS in {beats} of the {len(obstacle)} obstacle cells</b>, by "
        f"{pm(lead_best)} points, with {pm(coll[FOCUS])}% collisions where every other method collides on "
        f"{min(min(v) for m, v in coll.items() if m != FOCUS):.0f}–{max(max(v) for m, v in coll.items() if m != FOCUS):.0f}% "
        f"of trials. Averaged over the obstacle cells it hits {mean_obs[FOCUS]:.0f}%; no other method is above "
        f"{max(v for m, v in mean_obs.items() if m != FOCUS):.0f}%.",
        f"<b>The task modules carry over to users they never met.</b> In the {len(unseen)} unseen cells adding the "
        f"task module to the user module gains {pm([100 * add_task[k][0] for k in unseen])} points, as much as for the "
        f"users the modules were trained with ({', '.join(f'{100 * add_task[k][0]:+.0f}' for k in seen)}).",
        f"<b>Neither module does it alone.</b> Against the best baseline of each cell, the user module alone is "
        f"{span([100 * v[0] for v in alone['mars_user']])} points and the task module alone "
        f"{span([100 * v[0] for v in alone['mars_task']])}; adding the user module to the task module gains "
        f"{pm([100 * v[0] for v in add_user.values()])}.",
        f"<b>A policy retrained for each task with obstacle tokens, on 20 000 demonstrations against the task "
        f"module's 300, and steered through its own user module, trails U + T:</b> "
        f"{pm([100 * c['metrics']['retrained_mars_user']['success'] for c in obstacle])}% with "
        f"{pm(coll['retrained_mars_user'])}% collisions; U + T leads it by {span([100 * v[0] for v in vs_retr.values()])} "
        f"points, beyond noise in {sum(v[1] > 0 for v in vs_retr.values())} of the {len(vs_retr)} cells, although alone "
        f"with the goal given the same policies hit "
        f"{pm([100 * retr_alone[t]['oracle_retrained']['success'] for t in OBSTACLE_TASKS])}%. Its collisions stay "
        f"high at every reversal depth (Figure {FIGN['depths']}): a per-replan collision hazard that compounds over a trial once its "
        f"trajectory leaves the demonstrated routes (section 3).",
        f"<b>On the open field</b> ({t1}, the study's 5 s) the user module against teleop: "
        + ", ".join(f"{u} {pts(t1_lead[u])}" for u in USERS) + " points; it times out less and goes to wrong targets "
        f"more (Table 7).",
    ])]

    # ------------------------------------------------------------ 1 setup
    s += [Paragraph("1. The grid", H1), P(
        f"The study geometry (4 targets, reach 0.35–0.5) at input noise {NOISE:g} on every command. Every obstacle is "
        f"placed relative to the reach and its side is drawn at random per trial, so the side to pass is part of the "
        f"scene. The simulated user is the expert, the same planner whose paths the demonstrations follow: it plans "
        f"a route around the obstacles and pushes at full deflection toward the farthest point of the route it can "
        f"see, re-aiming every 20 ms, with no reaction lag; the channel corrupts its push (the user's corruption, "
        f"then the noise), whereas the demonstrations were recorded without a channel. The cursor runs at half the "
        f"demonstrations' speed by slower time (<font face='Courier'>gain_mode: time</font>: each planned step is "
        f"played over two control steps), so the policy sees the motion it was trained on; the open-field study's "
        f"controller scales every step instead, which breaks route following around obstacles (section 4).")]
    s += [img(paths["tasks"], "The five tasks, one sampled layout each: four targets (the goal filled), the start (dot), the "
              "obstacle's discs and the expert's route (blue), which the demonstrations follow. The disc sits 15–30% of its "
              "radius off the reach, the wall's gap 0.14 to one side, and a ring's gap is turned 15–30° from the side facing "
              "the reach; every side is drawn at random per trial.", max_h=2.9 * inch)]
    rows = [["User", "Corruption of the command", "Its module", "Task", "Obstacle", "Limit", "Its module, trained with"]]
    for u, t in zip(USERS, TASKS[:len(USERS)]):
        pass
    tasks_rows = [(t, task_label(t), f"{limit(t):g} s",
                   next((f"{u}'s module in the loop" for (u, tt) in TRAIN if tt == t and t in OBSTACLE_TASKS), "none (identity)"))
                  for t in TASKS]
    user_rows = [(u, corruption_label(G["cells"][f"{u}x{TASKS[0]}"]["corruption"]), G["cells"][f"{u}x{TASKS[0]}"]["user_adapter"])
                 for u in USERS]
    for i in range(max(len(tasks_rows), len(user_rows))):
        ur = user_rows[i] if i < len(user_rows) else ("", "", "")
        tr = tasks_rows[i] if i < len(tasks_rows) else ("", "", "", "")
        rows.append([*ur, *tr])
    s += [table(rows, "Users and tasks. A user module is calibrated open loop on the open field from 100 of the "
                "expert's demonstrations through that user's channel; a task module is trained on 300 of the task's "
                "demonstrations with one user's frozen module in the loop (the pairs U1–T2, U2–T3, U3–T4, U4–T5 and the T1 column "
                "are the cells trained on; the other 12 are unseen).",
                col_widths=[0.42 * inch, 1.4 * inch, 1.3 * inch, 0.42 * inch, 1.25 * inch, 0.45 * inch, 2.0 * inch])]
    rows = [["Method", "What runs", "Depth tuned"]]
    desc = {"teleop": "the served command moves the cursor, every step", "flow_control": "Kao et al.'s flow control: the command, at the noise prior's scale, as the initial condition of the first tau of the chunk's 16 steps; the flow then runs unmodified",
            "frs": "Flow Reversal Steering: the command's chunk reversed n steps through the frozen policy's field, then regenerated",
            "mars_user": "FRS with the user's module in the backward pass (no task module)",
            "mars_task": "FRS with the task's module in both passes (no user module)",
            "mars": "FRS with both modules: the composed adapter",
            "retrained_mars_user": "MARS on a policy retrained from scratch for the task (an obstacle head, 20 000 of the task's demonstrations), with a user module calibrated on that policy; on T1 the frozen policy and the user module"}
    for m in METHODS:
        rows.append([LABEL[m], desc[m], "—" if m == "teleop" else f"tau {depth_range('flow_control')}" if m == "flow_control"
                     else f"n {depth_range('frs')}"])
    s += [table(rows, f"The seven methods. Every method in a cell plays the same {test_n} layouts and the same channel noise; "
                "on T1, which has no task module and no retrained policy, the task-module arm is FRS and the U + T and "
                "retrained arms are the user module.", col_widths=[1.7 * inch, 4.8 * inch, 0.8 * inch], bold_rows=(6,))]

    # ------------------------------------------------------------ 2 results
    s += [Paragraph("2. Results: every method side by side", H1),
          img(paths["means"], f"Test success of every method at its tuned depth, averaged over the {len(obstacle)} obstacle cells "
              f"and over the {len(open_field)} open-field cells (95% CIs from resampling every cell's trials).", max_h=2.9 * inch),
          img(paths["heat_success"], "Success per method as a users × tasks map; outlined cells are the ones trained on "
              "(the T1 column for every user module, and each task module's training pair).", max_h=2.6 * inch),
          img(paths["methods"], f"Every cell ({test_n} test trials each, every method at its tuned depth): a row per task, a "
              "group per user, a bar per method in the legend's order; lines are 95% bootstrap CIs. The number above the "
              "dark-blue bar is U + T's success.", max_h=8.2 * inch),
          img(paths["heat_collision"], "Collisions per method: the share of trials that ended on an obstacle (T1 has none).",
              max_h=2.6 * inch)]
    for u in USERS:
        rows = [[u + ": " + corruption_label(G["cells"][f"{u}x{TASKS[0]}"]["corruption"])]
                + [t + ("*" if (u, t) in TRAIN and t in OBSTACLE_TASKS else "") for t in TASKS]]
        for m in METHODS:
            rows.append([LABEL[m]] + [pct(cell(u, t)["metrics"][m]["success"])
                                      + (f" ({100 * cell(u, t)['metrics'][m]['collision']:.0f})" if t in OBSTACLE_TASKS else "")
                                      for t in TASKS])
        s += [table(rows, f"{u}: test success, collisions in brackets (% of trials). * the task whose module was trained with "
                    f"{u}.", col_widths=[2.2 * inch] + [1.0 * inch] * len(TASKS), bold_rows=(6,))]
    s += [img(paths["leads"], "U + T's paired lead over every other method in every obstacle cell: the difference in success "
              "on the same trials, with a paired bootstrap 95% CI. Every interval lies above zero.", max_h=3.6 * inch)]
    rows = [["Effect (points, T2–T5; * CI excludes 0; (s) the training pair)"] + USERS]
    rows.append(["U + T over the best of teleop, flow control and FRS"]
                + [", ".join(pts(cell(u, t)["lead_over_best"]) for t in OBSTACLE_TASKS) for u in USERS])
    rows.append(["adding the task module to the user module"]
                + [", ".join(pts(add_task[(u, t)]) + (" (s)" if (u, t) in TRAIN else "") for t in OBSTACLE_TASKS) for u in USERS])
    rows.append(["adding the user module to the task module"]
                + [", ".join(pts(add_user[(u, t)]) + (" (s)" if (u, t) in TRAIN else "") for t in OBSTACLE_TASKS) for u in USERS])
    rows.append(["U + T over the retrained policy + its user module"]
                + [", ".join(pts(vs_retr[(u, t)]) for t in OBSTACLE_TASKS) for u in USERS])
    s += [table(rows, "Paired effects on the obstacle tasks (T2, T3, T4, T5 in that order).",
                col_widths=[2.5 * inch] + [1.2 * inch] * len(USERS))]

    # the open field
    rows = [["T1, 5 s"] + [f"{u}: hit / wrong / timeout, median hit" for u in USERS]]
    for m in METHODS:
        row = [LABEL[m]]
        for u in USERS:
            c, mt = cell(u, t1), cell(u, t1)["metrics"][m]
            med = np.median([d for d, o in zip(c["trials"]["duration_s"][m], c["trials"]["outcome"][m]) if o == "hit"])
            row.append(f"{100 * mt['success']:.0f} / {100 * mt['wrong']:.0f} / {100 * mt['timeout']:.0f}%, {med:.1f} s")
        rows.append(row)
    s += [table(rows, "The open field at the study's 5 s: outcomes in % of trials and the median time of the hits. Teleop's "
                "only failure is the clock; the policy methods trade timeouts for wrong targets, which the policy picks "
                "when the reversed command does not identify the goal.", col_widths=[1.9 * inch] + [1.3 * inch] * len(USERS),
                bold_rows=(6,))]

    # by side
    sided = [u for u in USERS if {"rotation_deg", "curl"} & set(G["cells"][f"{u}x{t1}"]["corruption"])]  # channels that turn pushes
    for u in sided:
        rows = [[u + ": " + corruption_label(G["cells"][f"{u}x{t1}"]["corruption"])]
                + [f"{t} {side}" for t in OBSTACLE_TASKS for side in ("left", "right")]]
        for m in METHODS:
            rows.append([LABEL[m]] + [f"{100 * cell(u, t)['by_side'][side]['metrics'][m]['success']:.0f}"
                                      for t in OBSTACLE_TASKS for side in ("left", "right")])
        s += [table(rows, f"{u}: success (%) by the side of the reach the expert's route passes the obstacle on. "
                    + ("The rotation bends pushes left" if "rotation_deg" in G["cells"][f"{u}x{t1}"]["corruption"] else
                       "The curl bends pushes right") + ", so one side is the one this channel cannot take on its own.",
                    col_widths=[1.9 * inch] + [0.6 * inch] * (2 * len(OBSTACLE_TASKS)), bold_rows=(6,))]

    # ------------------------------------------------------------ 3 why the retrained policy fails
    FIELDS = list(next(iter(STEER["hazard"].values())))
    DEPTHS = sorted(set(STEER["depths"].values()))
    hz = lambda name, field: f"{STEER['hazard'][name][field][str(STEER['depths']['mars'])][0]:.1f}%"
    steer_rows = [["States", "n", "Clearance, median", "Within 0.02"]
                  + [f"Hazard, {f} (depth " + " / ".join(map(str, DEPTHS)) + ")" for f in FIELDS]]
    for name, st_ in STEER["states"].items():
        row_name = name.replace("retrained_mars_user", "retrained + U").replace("mars", "U + T")
        steer_rows.append([row_name, st_["n"], f"{st_['clearance_median']:.3f}", f"{100 * st_['share_within_0.02']:.1f}%"]
                          + [" / ".join(f"{STEER['hazard'][name][f][str(d)][0]:.1f}%" for d in DEPTHS) for f in FIELDS])
    floor = {m: [100 * min(v["collision"] for v in TUNING[f"{u}x{t}|{NOISE}"][m].values()) for u in USERS for t in OBSTACLE_TASKS]
             for m in ("mars", "retrained_mars_user")}  # each cell's lowest collision rate over the depths
    alone_retr = ", ".join(f"{t} {pct(retr_alone[t]['oracle_retrained']['success'])}" for t in OBSTACLE_TASKS)
    alone_task = ", ".join(pct(retr_alone[t]["oracle_task"]["success"]) for t in OBSTACLE_TASKS)
    s += [Paragraph("3. Steering a retrained policy", H1), P(
        "The retrained arm asks whether the task module is needed at all: for each obstacle task a policy with an "
        "obstacle head was trained from scratch on 20 000 of the task's demonstrations (the task module had 300), and a user module calibrated "
        "on it exactly as on the frozen policy. Alone, with the goal given, these policies follow their routes "
        f"({alone_retr}; the frozen policy with the task module {alone_task}). "
        "Steered through MARS by the same expert, whose pushes toward the route now arrive through the user's "
        "channel and are read as a constant-velocity chunk, they keep colliding at every reversal depth, whereas the "
        "frozen policy with the task module recovers the route as the depth grows: each cell's lowest collision rate "
        f"over the depths is {pm(floor['retrained_mars_user'])}% for the retrained arm and {pm(floor['mars'])}% for U + T "
        f"(Figure {FIGN['depths']})."),
          img(paths["depths"], f"Tuning trials ({tune} per cell): success and collisions against the depth for two cells, "
              "the ring around the goal with U1 and the ring around the start with U3. Dashed: teleop, which has no depth.",
              max_h=4.6 * inch),
          P("Open loop from demonstration states the retrained field is as good as the frozen policy with the task "
            "module: a clean straight push reversed and regenerated through it collides in none of the executed steps, "
            "nor does the push corrupted as the channel corrupts it with the arm's own user module, and a push turned "
            "60° with no user module leaves it the more forgiving field (0.1% against 7.4%). The difference is in the "
            "states each arm drives itself into. The table below records, on U1 × T4, the live states each arm visits in "
            "closed loop and each field's per-replan hazard from those states: the share of chunks, regenerated from the "
            "recorded command, whose executed two steps run into a disc. From the states the U + T arm visits, the "
            f"frozen policy with the task module collides in {hz('states of mars', FIELDS[0])} of the executed "
            f"steps per replan and the retrained field in {hz('states of mars', FIELDS[1])}. That extra hazard "
            "takes the retrained arm's trajectory off the demonstrated routes: "
            f"{100 * STEER['states']['states of retrained_mars_user']['share_within_0.02']:.1f}% of its live states lie "
            f"within 0.02 of a disc, against {100 * STEER['states']['states of mars']['share_within_0.02']:.1f}% "
            f"for U + T and {100 * STEER['states']['demonstration states']['share_within_0.02']:.0f}% in the demonstrations, "
            f"and from those states its hazard is {hz('states of retrained_mars_user', FIELDS[1])} per replan, where even "
            f"the task module's field would collide in {hz('states of retrained_mars_user', FIELDS[0])}. Over a trial the "
            f"hazards compound: {hz('states of mars', FIELDS[0])} over the U + T arm's "
            f"{STEER['closed_loop']['mars']['replans_per_trial']:.0f} replans per trial is its "
            f"{100 * STEER['closed_loop']['mars']['collision']:.0f}% of trials; the retrained arm's rising hazard "
            f"ends {100 * STEER['closed_loop']['retrained_mars_user']['collision']:.0f}% of its trials within "
            f"{STEER['closed_loop']['retrained_mars_user']['replans_per_trial']:.0f} replans. The task module's avoidance "
            "holds at steered states as it does at demonstration states; the retrained policy's degrades as soon as the "
            "trajectory leaves the routes it was trained on. The histories are alike (median step "
            f"{STEER['states']['states of retrained_mars_user']['step_median']:.3f} against the demonstrations' "
            f"{STEER['states']['demonstration states']['step_median']:.3f}), so it is position, not speed. Their user "
            "modules also fit less well open loop (heading cosine 0.66–0.93 against 0.97 on the frozen policy)."),
          table(steer_rows, "research/steering_states.py on " + STEER["cell"] + ": the live states each arm visits in closed "
                "loop (" + str(STEER["trials"]) + " trials at the arms' tuned depths, "
                + ", ".join(SHORT[m] + " " + str(d) for m, d in STEER["depths"].items()) + ") and the demonstrations' states "
                "with the expert's push through the user's channel; the cursor's clearance from the nearest disc (the expert "
                "keeps at least 0.03); and from each state, regenerating from the recorded command through each field, the "
                "share of chunks whose executed two steps run into a disc, at each arm's depth.",
                col_widths=[1.5 * inch, 0.5 * inch, 0.9 * inch, 0.8 * inch, 1.8 * inch, 1.8 * inch]),
          P("<b>In plain terms.</b> Both the task module and the retrained policy learned from the same kind of demonstrations, "
            "the expert's routes, always at a comfortable distance from the discs: the task module from 300 of them, the "
            "retrained policy from 20 000. Steering through a noisy, corrupted channel "
            "never keeps the cursor exactly on those routes; small deviations happen at every replan. The retrained policy "
            "learned everything it knows from the routes, so once the cursor is a little off them, closer to a disc than any "
            "demonstration ever went, it has nothing to go on, makes a larger error, drifts further off, and a few replans "
            "later it hits the disc: the compounding error of imitation learning. The task module saw far less of the same narrow data "
            "and does not fail, so data diversity alone is not the explanation; the difference is how much each had to learn "
            "from that data. The task module learns only a small correction, to bend the chunk away from the discs it sees, "
            "on top of the frozen open-field policy, which was trained on a wide variety of positions and reaches a target "
            "from anywhere. Where the cursor is somewhere unfamiliar, the frozen policy still produces a sensible reach and "
            "the correction still pushes it away from the discs. The retrained policy had to learn reaching and avoiding "
            "together, from routes only, and inherited nothing that holds off the routes. The modular design therefore needs "
            "far less of the task's data: it learns only the task-specific part from it and keeps the rest from a policy "
            "that already generalizes. The usual remedy for the retrained policy would be training on its own closed-loop "
            "rollouts (DAgger-style) or on perturbed demonstrations; the composed adapter gets there without it.")]

    # ------------------------------------------------------------ 4 checks
    s += [Paragraph("4. Checks behind the grid", H1)]
    rows = [["Alone, goal given, no user (500 trials)"] + [f"{t} {task_label(t)}" for t in OBSTACLE_TASKS]]
    for m, name in (("oracle", "frozen policy's own chunks"), ("oracle_task", "+ the task module"),
                    ("oracle_retrained", "the retrained policy"), ("policy", "frozen policy, goal hidden"),
                    ("policy_task", "+ the task module, goal hidden"), ("policy_retrained", "the retrained policy, goal hidden")):
        rows.append([name] + [f"{pct(retr_alone[t][m]['success'])} ({100 * retr_alone[t][m]['collision']:.0f})" for t in OBSTACLE_TASKS])
    rows.append(["the expert alone, clean channel"] + [pct(EXPERT_CHECK[f"U0x{t}|0.0"]["metrics"]["teleop"]["success"]) for t in OBSTACLE_TASKS])
    s += [table(rows, "grid_task_check and grid_expert_check under the grid's controller: the goal-given policy's own chunks "
                "executed with and without each task module, each task's retrained policy alone, and the expert on a clean "
                "channel (collisions in brackets). "
                + (lambda ok: "Every task module passes" if len(ok) == len(OBSTACLE_TASKS) else
                   f"The task modules of {', '.join(ok)} pass")([t for t in OBSTACLE_TASKS if retr_alone[t]["oracle_task"]["success"] >= 0.8])
                + " the 80% gate set before the grid"
                + "".join(f"; {t}'s reaches {pct(retr_alone[t]['oracle_task']['success'])}" for t in OBSTACLE_TASKS
                          if retr_alone[t]["oracle_task"]["success"] < 0.8) + ". With the goal "
                "hidden, the wall and the goal ring give the goal away, since obstacles sit on the reach.",
                col_widths=[2.2 * inch] + [1.25 * inch] * len(OBSTACLE_TASKS)),
          P("<b>The controller.</b> The study's controller halves every executed step but still moves one chunk step "
            "per control step and divides the policy's history by the gain, so the cursor advances at half the rate its "
            "history says. On the open field that is harmless; on a route around an obstacle the plan and the cursor "
            "drift apart and the cursor cuts into the obstacle: under it the ring task modules of that check (20 000 "
            "demonstrations) reached 67% and 43% alone "
            "and the retrained policies 56%, 93%, 51% and 8%. The grid therefore plays each planned step over two "
            "control steps, which keeps the on-screen speed and gives the policy the history it was trained on. The "
            "price is on the open field, where steering acts once per chunk and chunks come half as often per second: "
            "there the policy methods lose part of their speed edge over teleop. The open-field study keeps the "
            "controller its sessions were played on."),
          P("<b>The simulated teleop user</b> is an upper bound on teleop: it knows the goal, re-aims every 20 ms and "
            "pushes at full deflection, so a 60° rotation only makes it spiral in (90% on T1) and a gain only slows it. "
            "People in the study, under the same channel, reached 48% in teleop against 76% with the MARS "
            "adapter. With a 200 ms reaction lag the same expert's teleop at 60° drops to 9%.")]

    # ------------------------------------------------------------ 5 caveats, reproducing
    s += [Paragraph("5. Caveats", H1), bullets([
        "Simulated users: a lag-free expert with a fixed corruption and Gaussian noise on its command. One training "
        "seed per module and per retrained policy.",
        "The grid's controller differs from the study's; the open-field study's results are not to be read off this grid.",
        "The task modules were trained with one user's module each; whether a module trained without any user, or with "
        "several, carries over as well is untested here.",
        "The steering-states analysis covers one cell (U1 × T4), with each arm at its tuned depth.",
    ]), Paragraph("Reproducing", H2), P("From the project folder (GPU; the grid's cells are CPU-bound on the expert's "
                                         "route planning, so run them in parallel):")]
    for line in ("python train/train_adapter.py --kind user --corruption '{scale: [1.0, 0.25]}' --name user_gain025y_expert_n100_s0 \\",
                 "    --n-demos 100 --steps 500 --frozen-noise",
                 "python train/train_adapter.py --kind task --obstacle '<spec>' --user-module user_gain025y_expert_n100_s0 \\",
                 "    --corruption '{scale: [1.0, 0.25]}' --lambda-field 0.03 --name task_disc_large_n300_s0_with_gain025y \\",
                 "    --n-demos 300 --frozen-noise",
                 "python train/train_policy.py --name flow_goaldrop_disc_large --goal_dropout 0.25 --targets 4 4 --reach 0.35 0.5 \\",
                 "    --obstacle '<spec>'   # then its user modules as above with --model flow_goaldrop_disc_large",
                 "python run_sweep.py --sweep grid_task_check; python run_sweep.py --sweep grid_expert_check",
                 "python run_sweep.py --sweep grid --cells U1xT1 --out grid_part_U1xT1   # one process per cell, then",
                 "python run_sweep.py --sweep grid --merge grid_part_U1xT1 grid_part_U2xT1 ... --out grid_s0",
                 "python research/steering_states.py   # the closed-loop states and per-replan hazards (section 3)",
                 "python reports/build_grid_report.py"):
        s.append(Paragraph(line.replace("<", "&lt;").replace(">", "&gt;").replace(" ", "&nbsp;"), CODE))
    doc.build(s)
    print(f"wrote {OUT}; figures in {TMP}")


if __name__ == "__main__":
    main()
