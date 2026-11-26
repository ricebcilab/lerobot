"""The paper's figures: MARS (Modular Adapters for Reversal Steering, the display name of mars) illustrated, the simulated
cursor grid with success against the time spent demonstrating a task, the LIBERO study, and the cursor user study.

    uv run --extra mars python examples/pi05/mars/paper/figures/build_figures.py                         # all four, into examples/pi05/mars/paper/figures/
    uv run --extra mars python examples/pi05/mars/paper/figures/build_figures.py --only fig1 fig3        # some
    uv run --extra mars python examples/pi05/mars/paper/figures/build_figures.py --out /tmp/figs --device cpu

Writes figures/fig1_method, fig2_sim, fig3_libero, fig4_study (.pdf and .png; 7.0 in wide, 300 dpi, Type 42 fonts) and
figures/stats.json, the numbers the captions quote (one entry per figure, so --only keeps the rest).

Names: MARS is MARS with both modules (user and task; MARS-UT in Figure 2, beside its ablations), MARS-U the user
module alone, MARS-T the task module alone. The notation is the Method's: the command x_1 = f(a_1), the noise a_0, the
policy's field v_theta, the user module M_U, the task module M_T, b_T.

Reads (results/ is git-ignored, so these are local):
    fig1  nothing (panel a's reaches come from the readout's rule; panel b's flows are exact flow-matching fields of
          made-up 1-D mixtures; panel c's densities are made up)
    fig2  the grid's five training seeds: results/cursor_sim/sweeps/grid_s<k>/test.json (configs/grid.yaml; seed 0 tuned, seeds
          1-4 tested at seed 0's settings: teleop, flow control and FRS on each task's fine-tuned policy, the vision task
          module alone and with the user module, and the fine-tuned policy with the user module refitted on it), and seed
          0's checkpoints for one cell's rollouts (the vision module's scene tokens through checkpoints/scene/fig3); for
          panel c's curve the demonstration-time runs, results/cursor_sim/sweeps/demo_time_s<k>/test.json (configs/grid_demo_time.yaml:
          the vision module, Fine-tuned + MARS-U, flow control and FRS on the fine-tuned policies, each per budget; on the
          grid's trials), and the fine-tuned policies' training logs (checkpoints/policies/demo_time/: each budget's
          demonstration count and training length)
    fig3  the LIBERO study's users: results/libero_sim/replication/users/<user>/test/<arm>/<run>/trial_*.npz and the trial
          videos beside them (users.sh: MARS, FRS at MARS's setting without the module, flow control, FRS; ffmpeg reads one frame)
    fig4  the cursor user study's sessions: results/cursor_study/p<k>/study_T<t>_U<k>_s<visit>/ and the T1 sessions recorded
          with the earlier short protocols, results/cursor_study_short/p<k>/study_short_U<k>_s<visit>_T1/
"""

import argparse
import json
import shutil
import subprocess  # nosec B404
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cursor"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cursor" / "tools"))  # analyze_results lives there

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.colors import to_rgb
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Polygon, Rectangle
from scipy import stats as sps

from analyze_results import find_runs, load_trials
from cursor_sa import config
from cursor_sa.agents import ExpertAgent
from cursor_sa.analysis import GRID, INK, INK_2, cumulative_hits, draw_scene
from lerobot.envs.cursor import sample_layouts
from cursor_sa.paths import CHECKPOINTS, RESULTS
from cursor_sa.session import arm_task, readout_for, load_checkpoints, run_trials
from lerobot.policies.steering.processor_steering import SCENE_VLM_PATH, SceneEncoder
from run_sweep import attach_scene, method_arm, scene_sources, with_setting

FIG_DIR = Path(__file__).resolve().parent
MARS = "MARS"
MARS_KEY = "mars"
ARMS = ["teleop", "flow_control", "frs", MARS_KEY]
ARM_LABEL = {"teleop": "Teleop", "flow_control": "Flow control", "frs": "FRS", MARS_KEY: MARS}
ARM_COLOR = {"teleop": "#008300", "flow_control": "#e87ba4", "frs": "#e69f00", MARS_KEY: "#2f45c4"}  # every pair CVD-safe
SERVED = "#8a3a5b"  # the served (corrupted) command
OUTCOMES = ("hit", "wrong_target", "timeout")
OUTCOME_LABEL = {"hit": "Success", "wrong_target": "Wrong target", "collision": "Collision", "timeout": "Timeout"}
OUTCOME_GRAY = {"hit": "#1f1f1e", "collision": "#1f1f1e", "wrong_target": "#767571", "timeout": "#aeaca4"}  # ordinal ramp
GRID_METHODS = ["teleop", "ft_flow_control", "ft_frs", "mars_task", "ft_mars_user", "mars"]
GRID_LABEL = {"teleop": "Teleop", "ft_flow_control": "Fine-tuned + flow control", "ft_frs": "Fine-tuned + FRS",
              "mars_user": f"{MARS}-U", "mars_task": f"{MARS}-T", "mars": f"{MARS}-UT",
              "ft_mars_user": f"Fine-tuned + {MARS}-U"}  # MARS-UT: both modules; the fine-tuned arms play each paper's
# baseline on the policy fine-tuned on the task's demonstrations (on T1, with nothing to fine-tune on, the frozen policy)
GRID_COLOR = {"teleop": "#008300", "ft_flow_control": "#e87ba4", "ft_frs": "#e69f00", "mars_task": "#8a63d2", "mars_user": "#6da7ec",
              "ft_mars_user": "#2e7da0", "mars": ARM_COLOR[MARS_KEY]}  # one palette with Figures 1 and 2: MARS a
# royal blue, MARS-U a sky blue, the fine-tuned arm a cerulean, MARS-T a lighter purple; every pair of the family is 15 or more
# colour units apart (OKLab x 100), the limit the colour checks ask for
T1_METHODS = ("teleop", "ft_flow_control", "ft_frs", "mars")  # on T1 MARS-U is MARS, MARS-T is FRS, the fine-tuned arm MARS
FULL_W = 7.0  # in, a two-column figure
SURFACE = "white"
FILL = "#ebeae4"


FS_PANEL, FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND, FS_TEXT = 10, 8, 8, 7, 7, 7  # pt: panel letters, titles, axis labels,
# tick labels, legends, text in the plots (callouts, equations, numbers on marks)


def paper_style():
    # figure.dpi as savefig's: rich_text and callout measure text at the figure's dpi, and at matplotlib's default 100 the
    # glyphs come out ~9% wide (gaps around a line's coloured pieces, callout boxes too wide)
    plt.rcParams.update({"font.size": FS_TEXT, "font.family": "Liberation Sans", "axes.edgecolor": INK_2, "axes.labelcolor": INK,
                         "xtick.color": INK_2, "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": False, "legend.frameon": False, "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": None,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "text.color": INK, "axes.titlesize": FS_TITLE, "axes.labelsize": FS_LABEL,
                         "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "legend.fontsize": FS_LEGEND, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                         "lines.solid_capstyle": "round", "mathtext.fontset": "stixsans"})


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", facecolor=SURFACE)
    plt.close(fig)
    return out / f"{name}.png"


def write_stats(out, key, value):
    out.mkdir(parents=True, exist_ok=True)
    path = out / "stats.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[key] = value
    path.write_text(json.dumps(data, indent=1))


def panel_label(fig, x, y, text):
    """A bold panel letter at figure coordinates (outside the panels)."""
    fig.text(x, y, text, fontsize=FS_PANEL, fontweight="bold", ha="left", va="top", color=INK)


def frame(ax, color=GRID):
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(color)
        s.set_linewidth(0.6)


def text_on(color):
    """Ink that reads on a fill: white on a dark one, the text colour on a light one (relative luminance 0.25)."""
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in to_rgb(color)]
    return "white" if 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2] < 0.25 else INK


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def bearing(v, ref_dir):
    """Degrees of v relative to ref_dir, signed, in (-180, 180]."""
    a = float(np.degrees(np.arctan2(v[1], v[0]) - np.arctan2(ref_dir[1], ref_dir[0])))
    a = (a + 180) % 360 - 180
    return 180.0 if a == -180 else a


def t_ci(values):
    """[mean, lower, upper]: the mean with a 95% t confidence interval over the values (participants)."""
    v = np.asarray(values, float)
    h = sps.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
    return [float(v.mean()), float(v.mean() - h), float(v.mean() + h)]


def row_major(handles, labels, ncol):
    """Reorder legend entries so matplotlib's column-major fill reads row by row."""
    n = len(handles)
    rows = int(np.ceil(n / ncol))
    order = [r + c * rows for c in range(ncol) for r in range(rows) if r + c * rows < n]
    # the entries placed at those slots, so that slot (row r, col c) shows entry r * ncol + c
    slots = [None] * n
    for k, pos in enumerate(order):
        slots[pos] = k
    grid = [[None] * ncol for _ in range(rows)]
    k = 0
    for r in range(rows):
        for c in range(ncol):
            if k < n:
                grid[r][c] = k
                k += 1
    seq = [grid[r][c] for c in range(ncol) for r in range(rows) if grid[r][c] is not None]
    return [handles[i] for i in seq], [labels[i] for i in seq]


def arrow(ax, pos, cmd, color, ls="-", scale=0.14):
    ax.annotate("", xy=pos + scale * cmd, xytext=pos, zorder=5,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9, ls=ls, shrinkA=0, shrinkB=0, mutation_scale=6))


# ================================================================ The method schematic (Figures 1b and 3a)


SCHEMATIC_X = np.linspace(-3.2, 3.2, 800)  # the schematic's axis: actions, or noise, in made-up units
PANEL_EDGE, CALLOUT_EDGE, CALLOUT_FILL = "#b9b7af", "#cfcdc5", "#f6f5f1"
DENSITY_FILL, READOUT_FILL = "#dedcd5", "#efdde5"  # the policy's density; the user readout's commands (a tint of SERVED)
BAND_HATCH, BAND_FILL = "#a5a49e", "#efeee9"  # the actions that hit the obstacle
OBSTACLE_STYLE = dict(facecolor="#8f8d86", edgecolor="#5f5e5a", lw=0.6)  # the scenes' obstacles: darker than the other targets
SCHEMATIC_TITLES = {"readout": "user commands  $x$", "noise": "noise  $a_0 \\sim \\mathcal{N}(0, I)$",
                    "policy": "robot actions  $a$"}
SCHEMATIC_VARS = {"readout": "x", "noise": "a_0", "policy": "a"}  # each panel's variable, at its baseline's right end
TASK_COLOR = GRID_COLOR["mars_task"]  # the task flow (Figure 1b's lines, 1c's density); the user flow is the served colour
TASK_FILL = "#e7e0f6"  # a tint of TASK_COLOR
INTENT_DASH = (0, (2.5, 1.5))  # the user's intent density
EMPHASIS = dict(color=ARM_COLOR[MARS_KEY], fontweight="bold")
FLOW_NAME = dict(color=INK_2, fontstyle="italic")


def inches(fig, x, y):
    """Figure coordinates of a point (or a size) given in inches from the figure's bottom left."""
    return x / fig.get_figwidth(), y / fig.get_figheight()


def rich_text(fig, x, y, pieces, size, ha="center", va="baseline"):
    """One line at (x, y) inches made of differently styled pieces [(text, text keywords)], set side by side."""
    r = fig.canvas.get_renderer()
    texts = [fig.text(0, y / fig.get_figheight(), s, fontsize=size, ha="left", va=va, **style) for s, style in pieces]
    widths = [t.get_window_extent(r).width for t in texts]
    px = x * fig.dpi - {"center": 0.5, "left": 0.0, "right": 1.0}[ha] * sum(widths)
    for t, w in zip(texts, widths):
        t.set_x(px / fig.bbox.width)
        px += w
    return texts


def callout(fig, x, y, lines, size=None, pad=0.045):
    size = size or FS_TEXT
    """A boxed note of rich-text lines centred at (x, y) inches, as FRS's Figure 1 annotates its panels."""
    step = 1.32 * size / 72
    texts = [t for i, pieces in enumerate(lines)
             for t in rich_text(fig, x, y + ((len(lines) - 1) / 2 - i) * step - 0.3 * size / 72, pieces, size)]
    r = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(r).transformed(fig.dpi_scale_trans.inverted()) for t in texts]
    x0, y0 = min(b.x0 for b in boxes) - pad, min(b.y0 for b in boxes) - pad
    x1, y1 = max(b.x1 for b in boxes) + pad, max(b.y1 for b in boxes) + pad
    fig.patches.append(Rectangle(inches(fig, x0, y0), *inches(fig, x1 - x0, y1 - y0), transform=fig.transFigure,
                                 facecolor=CALLOUT_FILL, edgecolor=CALLOUT_EDGE, lw=0.6, zorder=0))


def density_axes(fig, x, y, w, h, top, var):
    """A density plot at (x, y, w, h) inches: a baseline, no ticks, the variable at the baseline's right end."""
    ax = fig.add_axes([*inches(fig, x, y), *inches(fig, w, h)])
    ax.set_axis_off()
    ax.set(xlim=(SCHEMATIC_X[0], SCHEMATIC_X[-1]), ylim=(-0.04 * top, top))
    ax.axhline(0, color=INK_2, lw=0.6, zorder=1)
    ax.text(SCHEMATIC_X[-1], -0.02 * top, f"${var}$", fontsize=FS_TICK, color=INK_2, ha="right", va="top")
    return ax


def density(ax, y, fill, edge=INK, lw=0.9, ls="-", support_only=False):
    """A density, filled (unless fill is None) and outlined (only where it is not negligible, with support_only)."""
    if fill:
        ax.fill_between(SCHEMATIC_X, 0, y, color=fill, lw=0, zorder=2)
    ax.plot(SCHEMATIC_X, np.where(y > 2e-3 * y.max(), y, np.nan) if support_only else y, color=edge, lw=lw, ls=ls, zorder=2.5)


def on_density(ax, x, y, marker, color, hollow=False):
    """A marker at x on the density y (on the axis where y vanishes); returns its height."""
    h = float(np.interp(x, SCHEMATIC_X, y))
    ax.plot(x, h, marker, color=color, mfc="white" if hollow else color, mew=1.0, ms=4.6, zorder=6, clip_on=False)
    return h


def obstacle_band(ax, lo, hi, above=False):
    """The actions that hit the obstacle: a hatched band, drawn over the densities with `above`."""
    top = 0.97 * ax.get_ylim()[1]
    ax.axvspan(lo, hi, ymax=0.97, color=BAND_FILL, lw=0, zorder=0)
    ax.add_patch(Rectangle((lo, 0), hi - lo, top, facecolor="none", edgecolor=BAND_HATCH, lw=0, hatch="//////", zorder=4 if above else 1))


def intent_and_command(ax, pol, readout, u, uh, intent=None):
    """The user commands' panel: the policy's density for reference, the commands' density, the intended action a_1 (on the
    user's intent density, dashed, when given, else on the axis; it need not be in the policy's distribution) and the command
    x_1 = f(a_1) the interface turns it into, joined by f."""
    density(ax, pol, DENSITY_FILL)
    if intent is not None:
        density(ax, intent, None, edge=INK, lw=0.9, ls=INTENT_DASH, support_only=True)
    density(ax, readout, READOUT_FILL, edge=SERVED, support_only=True)
    yu = on_density(ax, u, np.zeros_like(readout) if intent is None else intent, "v", INK, hollow=True)
    yh = on_density(ax, uh, readout, "v", SERVED)
    ax.annotate("", xy=(uh, yh), xytext=(u, yu), arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=0.7, mutation_scale=5, shrinkA=4, shrinkB=4))
    ax.text((u + uh) / 2 - 0.08, (yu + yh) / 2 + 0.03, "$f$", fontsize=FS_TEXT, color=INK_2, ha="right", va="bottom")
    ax.text(u - 0.12, yu + 0.07, "$a_1$", fontsize=FS_TEXT, ha="right", va="bottom")
    ax.text(uh + 0.17, yh, "$x_1$", fontsize=FS_TEXT, ha="left", va="center", color=SERVED)


def noise_panel(ax, z, other, other_color):
    """The prior with MARS's noise a-hat_0 (in its bulk) and another method's (hollow)."""
    noise = sps.norm.pdf(SCHEMATIC_X)
    density(ax, noise, DENSITY_FILL)
    h = on_density(ax, z, noise, "o", ARM_COLOR[MARS_KEY])
    on_density(ax, other, noise, "o", other_color, hollow=True)
    ax.text(z + 0.2, h + 0.01, "$\\hat a_0$", fontsize=FS_TEXT, color=ARM_COLOR[MARS_KEY], ha="left", va="bottom")


READOUT_CALLOUT = [[("The interface ", {}), ("distorts", dict(color=SERVED, fontweight="bold")), (" the intent $a_1$", {})],
                   [("into an off-policy command $x_1$ …", {})]]


def schematic(fig, x0, y0, spec):
    """The method as FRS's Figure 1 draws it, with made-up 1-D densities, (x0, y0) inches the bottom left of the block's
    lower panel: the user readout (top left) -> flow reversal -> noise (right) -> flow matching -> policy actions (bottom left),
    each flow a straight arrow with its equation above and its name below, MARS's net effect a dashed bracket on the left,
    one callout per panel. spec: draw and callouts per panel, top (each density plot's height in density units), eq and
    notes per flow, and optionally gap (between the panels, wide enough for the equations) and noise_w."""
    pw, ph, rows_gap, band = 1.72, 0.95, 0.12, 0.34  # an action panel; between the rows; the callout band at a panel's top
    gap, nw = spec.get("gap", 1.17), spec.get("noise_w", 1.3)
    ax0 = x0 + 0.33                                  # room on the left for the bracket
    top_y = y0 + ph + rows_gap
    nx = ax0 + pw + gap
    ry, by = top_y + 0.27, y0 + ph - 0.27            # the flows, on the panels' inner halves (inside the noise panel's span)
    ny = (ry + by) / 2 - ph / 2
    for key, (px, py, w) in {"readout": (ax0, top_y, pw), "noise": (nx, ny, nw), "policy": (ax0, y0, pw)}.items():
        fig.patches.append(Rectangle(inches(fig, px, py), *inches(fig, w, ph), transform=fig.transFigure, facecolor="none",
                                     edgecolor=PANEL_EDGE, lw=0.8))
        spec["draw"][key](density_axes(fig, px + 0.07, py + 0.1, w - 0.14, ph - band - 0.15, spec["top"][key], SCHEMATIC_VARS[key]))
        callout(fig, px + w / 2, py + ph - band / 2 - 0.02, spec["callouts"][key])
    fig.text(*inches(fig, ax0 + 0.02, top_y + ph + 0.05), SCHEMATIC_TITLES["readout"], fontsize=FS_TITLE, ha="left", va="bottom")
    fig.text(*inches(fig, nx + 0.02, ny + ph + 0.05), SCHEMATIC_TITLES["noise"], fontsize=FS_TITLE, ha="left", va="bottom")
    fig.text(*inches(fig, ax0 + 0.02, y0 - 0.05), SCHEMATIC_TITLES["policy"], fontsize=FS_TITLE, ha="left", va="top")
    xa, xb = ax0 + pw + 0.06, nx - 0.06
    for (p0, p1), eq, notes in ((((xa, ry), (xb, ry)), *spec["flows"][0]), (((xb, by), (xa, by)), *spec["flows"][1])):
        fig.patches.append(FancyArrowPatch(inches(fig, *p0), inches(fig, *p1), transform=fig.transFigure, arrowstyle="-|>",
                                           mutation_scale=7, color=INK_2, lw=0.9, shrinkA=0, shrinkB=0))
        rich_text(fig, (xa + xb) / 2, p0[1] + 0.055, [(eq, {})], FS_TEXT, va="bottom")
        for i, note in enumerate(notes):
            rich_text(fig, (xa + xb) / 2, p0[1] - 0.05 - 1.3 * FS_TEXT / 72 * i, [note], FS_TEXT, va="top")
    # the net effect: the user readout -> policy actions
    bx, dash, mars = ax0 - 0.13, (0, (3.5, 2)), ARM_COLOR[MARS_KEY]
    fig.add_artist(plt.Line2D(*zip(*(inches(fig, x, y) for x, y in ((ax0 - 0.02, ry), (bx, ry), (bx, by)))), transform=fig.transFigure,
                              color=mars, lw=1.0, ls=dash))
    fig.patches.append(FancyArrowPatch(inches(fig, bx, by), inches(fig, ax0 - 0.02, by), transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=6, color=mars, lw=1.0, ls=dash, shrinkA=0, shrinkB=0))
    fig.text(*inches(fig, bx - 0.05, (ry + by) / 2), MARS, rotation=90, ha="right", va="center", fontsize=FS_TITLE, color=mars, fontweight="bold")


def pipeline_schematic():
    """Figure 1c, the whole method on made-up 1-D densities: the policy's two modes, the one the user means running through
    the task's constraint (the hatched band of actions); the user's intent a_1 clear of it, and the command x_1 = f(a_1) off
    the policy's support. MARS reverses x_1 through M_U(M_T v + b_T) into the bulk of the noise, and the task's flow
    M_T v + b_T, whose mode the task module moved clear of the constraint, generates the intended action there. FRS
    (M_U = M_T = I, b_T = 0) reverses through the policy's own field into the noise's tail and comes back near the command,
    off the policy's support."""
    X, mars, frs = SCHEMATIC_X, ARM_COLOR[MARS_KEY], ARM_COLOR["frs"]
    lo, hi = -0.85, 0.05
    pol = 0.45 * sps.norm.pdf(X, -2.0, 0.36) + 0.55 * sps.norm.pdf(X, -0.4, 0.34)
    task = 0.45 * sps.norm.pdf(X, -2.0, 0.36) + 0.55 * sps.norm.pdf(X, 0.75, 0.3)
    intent = 0.36 * sps.norm.pdf(X, 0.75, 0.3)
    readout = 0.36 * sps.norm.pdf(X, 2.4, 0.22)

    def readout_panel(ax):
        intent_and_command(ax, pol, readout, 0.75, 2.4, intent)
        obstacle_band(ax, lo, hi, above=True)

    def policy_panel(ax):
        obstacle_band(ax, lo, hi)
        density(ax, pol, DENSITY_FILL)
        density(ax, task, TASK_FILL, edge=TASK_COLOR, lw=0.9)
        ax.fill_between(X, 0, np.minimum(pol, task), color=DENSITY_FILL, lw=0, zorder=2.2)   # where the two agree, the policy's
        h = on_density(ax, 0.75, task, "o", mars)
        on_density(ax, 2.4, task, "o", frs, hollow=True)
        ax.text(0.95, h, "$\\hat a_1$", fontsize=FS_TEXT, color=mars, ha="left", va="center")
        ax.text((lo + hi) / 2, -0.028, "task constraint", fontsize=FS_TEXT, color=INK_2, ha="center", va="top")

    return dict(
        gap=1.5, noise_w=1.45,
        draw={"readout": readout_panel, "noise": lambda ax: noise_panel(ax, 0.78, 2.75, frs), "policy": policy_panel},
        top={"readout": 0.84, "noise": 0.46, "policy": 0.8},
        callouts={"readout": READOUT_CALLOUT,
                  "noise": [[("… ", {}), ("$M_U$", dict(color=SERVED)), (" and ", {}), ("$M_T, b_T$", dict(color=TASK_COLOR)), (" steer the", {})],
                            [("reversal into ", {}), ("typical noise", EMPHASIS), (" …", {})]],
                  "policy": [[("… which the task's flow maps to an", {})], [("action meeting the ", {}), ("task constraint", EMPHASIS), (".", {})]]},
        flows=[("$\\hat a_0 = x_1 - \\int_0^1 M_U (M_T\\, v_\\theta + b_T)\\, dt$",
                [("flow reversal", FLOW_NAME), ("FRS: $M_U = M_T = I,\\ b_T = 0$", dict(color=frs))]),
               ("$\\hat a_1 = \\hat a_0 + \\int_0^1 (M_T\\, v_\\theta + b_T)\\, dt$", [("flow matching", FLOW_NAME)])])


def method_legend(entries):
    """Legend handles and labels for the schematic's and the scenes' marks, by name, in the given order."""
    hatch = Rectangle((0, 0), 1, 1, facecolor=BAND_FILL, edgecolor=BAND_HATCH, lw=0, hatch="//////")
    disc = lambda face, edge, mew=0.8: plt.Line2D([], [], ls="none", marker="o", ms=6.5, mfc=face, mec=edge, mew=mew)
    marks = {"intent": (plt.Line2D([], [], color=INK, lw=0.9), "intent $u$"),
             "command": (plt.Line2D([], [], color=SERVED, lw=0.9), "command $\\hat u$"),
             "policy": (Rectangle((0, 0), 1, 1, facecolor=DENSITY_FILL, edgecolor=INK, lw=0.7), "policy density"),
             "intent_density": (Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=INK, lw=0.8, ls=INTENT_DASH), "intent density"),
             "readout": (Rectangle((0, 0), 1, 1, facecolor=READOUT_FILL, edgecolor=SERVED, lw=0.7), "command density"),
             "task": (Rectangle((0, 0), 1, 1, facecolor=TASK_FILL, edgecolor=TASK_COLOR, lw=0.7), "task flow density"),
             "frs": (plt.Line2D([], [], color=ARM_COLOR["frs"], lw=1.2, marker="o", mfc="white", ms=4.2), "FRS"),
             "mars": (plt.Line2D([], [], color=ARM_COLOR[MARS_KEY], lw=1.2, marker="o", ms=4.2), MARS),
             "start": (plt.Line2D([], [], ls="none", color=INK, marker="o", ms=4.2), "start"),
             "cursor": (plt.Line2D([], [], ls="none", color=INK, marker="o", ms=2.6), "cursor"),
             "target": (disc("#fbe3a1", INK, 1.0), "target"),
             "other": (disc("#ecebe6", "#c9c8c1"), "other targets"),
             "obstacle": ((hatch, disc(OBSTACLE_STYLE["facecolor"], OBSTACLE_STYLE["edgecolor"], 0.6)), "task constraint"),
             "constraint": (hatch, "task constraint")}
    return [marks[e][0] for e in entries], [marks[e][1] for e in entries]


# ================================================================ Figure 1b: the main insight as flow lines


FLOW_T = np.linspace(0, 1, 401)  # the flow time, noise at 0 and actions at 1, as the Method writes it
MODE_SD = 0.22
POLICY_MODES = (-1.3, 1.3)   # the policy's two behaviours (two objects, say), in made-up action units
TASK_MODES = (-1.3, 2.2)     # the task's demonstrations: the right behaviour moved clear of the constraint
CONSTRAINT = (0.8, 1.8)      # the actions the task rules out (the policy's right behaviour)
USER_F = (0.35, 2.2)         # the interface, f(a) = 0.35 a + 2.2: commands compressed and shifted above both behaviours
FLOW_NOISES = sps.norm.ppf(np.linspace(0.04, 0.96, 12))  # the drawn lines' noises, evenly spaced in probability (none at
# the median, whose line would balance between the two behaviours)
MARS_NOISE = {"user": sps.norm.ppf(0.25), "task": sps.norm.ppf(0.75)}  # each panel's MARS example: the left behaviour meant
# through the interface (left), the right one under the task's constraint (right)
FLOW_A = (-2.35, 3.15)       # the drawn action range


def gmm_velocity(a, t, modes, sd=MODE_SD):
    """The flow-matching field that carries noise N(0, 1) to an equal mixture of N(m, sd^2) along the straight paths
    a_t = t a_1 + (1 - t) a_0: E[a_1 - a_0 | a_t = a], in closed form (a Gaussian posterior per mode, weighted by the
    mode's likelihood of a_t)."""
    a = np.asarray(a, float)[..., None]
    m = np.asarray(modes, float)
    var = t ** 2 * sd ** 2 + (1 - t) ** 2
    logw = -0.5 * (a - t * m) ** 2 / var
    w = np.exp(logw - logw.max(-1, keepdims=True))
    w /= w.sum(-1, keepdims=True)
    return (w * (m + (t * sd ** 2 - (1 - t)) * (a - t * m) / var)).sum(-1)


def flow_lines(a0, modes):
    """The paths a_t of the noises a0 under gmm_velocity, [len(FLOW_T), len(a0)] (Euler on FLOW_T)."""
    a = np.array(a0, float, ndmin=1)
    out = [a.copy()]
    for t0, t1 in zip(FLOW_T[:-1], FLOW_T[1:]):
        a = a + (t1 - t0) * gmm_velocity(a, t0, modes)
        out.append(a.copy())
    return np.array(out)


def user_lines(a0):
    """The user's flow as the module is trained on it: from each noise a_0 the straight path to the command its action
    gives, x_t = (1 - t) a_0 + t f(mu_theta(a_0))."""
    a1 = flow_lines(a0, POLICY_MODES)[-1]
    x1 = USER_F[0] * a1 + USER_F[1]
    return (1 - FLOW_T[:, None]) * np.array(a0, float, ndmin=1) + FLOW_T[:, None] * x1


def edge_density(ax, centers, sd, t, side, color, fill, width=0.13):
    """A density of actions drawn sideways at flow time t (t = 1: to the right, t = 0: to the left), its peak `width` wide."""
    y = np.linspace(*FLOW_A, 400)
    d = sum(sps.norm.pdf(y, c, s) for c, s in zip(centers, sd))
    d = width * d / d.max()
    x = t + side * d
    ax.fill_betweenx(y, t, x, color=fill, lw=0, zorder=1, clip_on=False)
    ax.plot(np.where(d > 2e-3 * width, x, np.nan), y, color=color, lw=0.7, zorder=1.5, clip_on=False)


def along(ax, path, color, ls="-", lw=1.4, toward=0.5, backward=False):
    """A highlighted path over FLOW_T with an arrowhead at flow time `toward`, pointing backward in time if asked."""
    ax.plot(FLOW_T, path, color=color, lw=lw, ls=ls, zorder=5, solid_capstyle="round", dash_capstyle="round")
    i = int(toward * (len(FLOW_T) - 1))
    j = i - 8 if backward else i + 8
    ax.annotate("", xy=(FLOW_T[j], path[j]), xytext=(FLOW_T[i], path[i]), zorder=6,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=0.8, mutation_scale=9, shrinkA=0, shrinkB=0))


def flow_axes(fig, rect, title):
    ax = fig.add_axes(rect)
    ax.set(xlim=(-0.16, 1.16), ylim=FLOW_A, xticks=[0, 1], yticks=[])
    ax.set_xticklabels(["0\nnoise", "1\naction"], fontsize=FS_TICK, linespacing=1.0)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_bounds(0, 1)
    ax.tick_params(axis="x", pad=1.5)
    ax.set_xlabel("flow time  $t$", fontsize=FS_LABEL, labelpad=1)
    rich_text(fig, (rect[0] + rect[2] / 2) * fig.get_figwidth(), (rect[1] + rect[3]) * fig.get_figheight() + 0.06, title, FS_TITLE, va="bottom")
    return ax


def user_flow_panel(fig, rect):
    """Figure 1b, left: the policy's flow (grey) from the noise into its two behaviours, and the user's flow (pink), which
    joins the same noises to the commands their actions give, the policy's velocity reshaped by M_U. MARS takes the command
    back along the user's flow to the noise of the meant action, then forward along the policy's flow to that action."""
    mars = ARM_COLOR[MARS_KEY]
    ax = flow_axes(fig, rect, [("user flow  ", dict(color=SERVED, fontweight="bold")), ("$v_U = M_U\\, v_\\theta$", dict(color=SERVED))])
    grey, pink = flow_lines(FLOW_NOISES, POLICY_MODES), user_lines(FLOW_NOISES)
    ax.plot(FLOW_T, grey, color="#b4b2ab", lw=0.6, zorder=2)
    ax.plot(FLOW_T, pink, color=SERVED, lw=0.6, alpha=0.55, zorder=3)
    edge_density(ax, [0], [1], 0, -1, INK_2, DENSITY_FILL)
    edge_density(ax, POLICY_MODES, [MODE_SD] * 2, 1, 1, INK_2, DENSITY_FILL)
    edge_density(ax, [USER_F[0] * m + USER_F[1] for m in POLICY_MODES], [USER_F[0] * MODE_SD] * 2, 1, 1, SERVED, READOUT_FILL, width=0.1)
    a0 = MARS_NOISE["user"]
    back, fwd = user_lines([a0])[:, 0], flow_lines([a0], POLICY_MODES)[:, 0]
    along(ax, back, mars, ls=(0, (2.2, 1.4)), toward=0.55, backward=True)
    along(ax, fwd, mars, toward=0.45)
    ax.plot(1, back[-1], "v", color=SERVED, ms=4.8, zorder=7, clip_on=False)
    ax.plot([0, 1], [a0, fwd[-1]], "o", color=mars, ms=3.8, zorder=7, clip_on=False)
    ax.text(0.97, back[-1] + 0.12, "$x_1$", fontsize=FS_TEXT, color=SERVED, ha="right", va="bottom")
    ax.text(-0.03, a0 - 0.12, "$\\hat a_0$", fontsize=FS_TEXT, color=mars, ha="right", va="top")
    ax.text(1.03, fwd[-1] - 0.12, "$\\hat a_1$", fontsize=FS_TEXT, color=mars, ha="left", va="top")
    ax.text(0.5, -2.1, "policy flow  $v_\\theta$", fontsize=FS_TEXT, color=INK_2, ha="center", va="center")
    return ax


def task_flow_panel(fig, rect):
    """Figure 1b, right: the policy's flow (grey) and the task's (purple), the policy's velocity reshaped by M_T and shifted
    by b_T, which carries the same noises to the task's behaviours: the left one kept, the right one moved clear of the
    actions the task rules out. MARS generates along the task's flow; the policy's own flow ends in the constraint."""
    mars = ARM_COLOR[MARS_KEY]
    ax = flow_axes(fig, rect, [("task flow  ", dict(color=TASK_COLOR, fontweight="bold")),
                               ("$v_T = M_T\\, v_\\theta + b_T$", dict(color=TASK_COLOR))])
    grey, purple = flow_lines(FLOW_NOISES, POLICY_MODES), flow_lines(FLOW_NOISES, TASK_MODES)
    ax.plot(FLOW_T, grey, color="#b4b2ab", lw=0.6, zorder=2)
    ax.plot(FLOW_T, purple, color=TASK_COLOR, lw=0.6, alpha=0.6, zorder=3)
    edge_density(ax, [0], [1], 0, -1, INK_2, DENSITY_FILL)
    edge_density(ax, POLICY_MODES, [MODE_SD] * 2, 1, 1, INK_2, DENSITY_FILL)
    edge_density(ax, TASK_MODES[1:], [MODE_SD], 1, 1, TASK_COLOR, TASK_FILL)
    lo, hi = CONSTRAINT
    ax.add_patch(Rectangle((0.9, lo), 0.1, hi - lo, facecolor=BAND_FILL, edgecolor=BAND_HATCH, lw=0, hatch="//////", zorder=4))
    ax.text(0.86, (lo + hi) / 2, "task\nconstraint", fontsize=FS_TEXT, color=INK_2, ha="right", va="center", linespacing=1.0, zorder=6,
            bbox=dict(boxstyle="square,pad=0.08", facecolor="white", edgecolor="none", alpha=0.85))
    a0 = MARS_NOISE["task"]
    own, task = flow_lines([a0], POLICY_MODES)[:, 0], flow_lines([a0], TASK_MODES)[:, 0]
    ax.plot(FLOW_T, own, color=INK_2, lw=0.9, ls=(0, (2.2, 1.4)), zorder=4.5)
    ax.plot(1, own[-1], "x", color=INK, ms=4, mew=1.1, zorder=7, clip_on=False)
    along(ax, task, mars, toward=0.5)
    ax.plot([0, 1], [a0, task[-1]], "o", color=mars, ms=3.8, zorder=7, clip_on=False)
    ax.text(-0.03, a0 + 0.12, "$\\hat a_0$", fontsize=FS_TEXT, color=mars, ha="right", va="bottom")
    ax.text(1.03, task[-1] + 0.1, "$\\hat a_1$", fontsize=FS_TEXT, color=mars, ha="left", va="bottom")
    ax.text(0.5, -2.1, "policy flow  $v_\\theta$", fontsize=FS_TEXT, color=INK_2, ha="center", va="center")
    return ax


# ================================================================ Figure 1: the method


TARGET_STYLE = dict(facecolor="#fbe3a1", edgecolor=INK, lw=0.8)  # the scenes' target, as Figures 2 and 3 draw it
TILE_W, TILE_H = 1.6, 1.0  # a tile's data extent: the pictogram on the left half, a reach on the right
START, GOAL = np.array([0.98, 0.2]), np.array([1.42, 0.78])


ASSETS = FIG_DIR / "assets"  # Figure 1a's pictograms: hand-drawn SVGs, each rendered beside it as a PNG (rsvg-convert -w 480)


def icon(ax, name):
    """Place the pictogram assets/<name>.png on the tile's left half (its SVG's 80 x 100 aspect), re-rendering it first
    when its SVG is newer and rsvg-convert is there."""
    svg, png = ASSETS / f"{name}.svg", ASSETS / f"{name}.png"
    if (not png.exists() or svg.stat().st_mtime > png.stat().st_mtime) and shutil.which("rsvg-convert"):
        subprocess.run(["rsvg-convert", "-w", "480", str(svg), "-o", str(png)], check=True)  # nosec B603 B607
    h = 0.9
    ax.imshow(plt.imread(png), extent=(0.4 - 0.4 * h, 0.4 + 0.4 * h, 0.5 - h / 2, 0.5 + h / 2), interpolation="antialiased", zorder=2)


PLANE = [(0.8, 0.08), (1.5, 0.08), (1.58, 0.48), (0.9, 0.48)]  # the plane a two-axis device moves in, in perspective
SHADOW = np.array([GOAL[0], 0.36])  # the target's projection onto it
GHOST = GOAL + [-0.26, -0.04]  # where the user believes the target is


def served_path(kind):
    """The reach the served commands give, from START, for the user's straight push to GOAL."""
    d = GOAL - START
    if kind == "rotate":
        c, s = np.cos(np.radians(45)), np.sin(np.radians(45))
        return np.array([START, START + 0.9 * np.array([[c, -s], [s, c]]) @ d])
    if kind == "project":  # the height above the plane is lost
        return np.array([START, SHADOW])
    if kind == "misjudge":  # straight at the believed target, corrected toward the real one only after a reaction time
        t = np.linspace(0, 1, 30)[:, None]
        turn = START + 0.8 * (GHOST - START)
        bend = START + 0.97 * (GHOST - START)
        return (1 - t) ** 2 * turn + 2 * t * (1 - t) * bend + t ** 2 * (GOAL - [0.05, 0.05])
    steps = [START]  # quantized: one axis at a time
    for k in range(6):
        steps.append(steps[-1] + (np.array([d[0], 0]) if k % 2 == 0 else np.array([0, d[1]])) / 3)
    return np.array(steps)


def reach(ax, kind):
    """The tile's reach: the target, the straight push the user means (ink) and where the served commands go (the served colour)."""
    if kind == "project":  # the target floats above the device's plane
        ax.add_patch(Polygon(PLANE, closed=True, facecolor=CALLOUT_FILL, edgecolor=PANEL_EDGE, lw=0.6, zorder=1))
        ax.plot(*np.c_[GOAL - [0, 0.07], SHADOW], color=INK_2, lw=0.6, ls=(0, (2, 1.5)), zorder=1.5)
        ax.add_patch(Ellipse(SHADOW, 0.1, 0.04, facecolor=PANEL_EDGE, lw=0, zorder=1.5))
    if kind == "misjudge":
        ax.add_patch(Circle(GHOST, 0.07, facecolor="none", edgecolor=INK_2, lw=0.7, ls=(0, (2, 1.5)), zorder=2))
    ax.add_patch(Circle(GOAL, 0.07, zorder=2, **TARGET_STYLE))
    ax.annotate("", xy=GOAL - 0.08 * unit(GOAL - START), xytext=START, zorder=3,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, mutation_scale=5, shrinkA=0, shrinkB=0))
    p = served_path(kind)
    if kind == "misjudge":
        p = np.r_[[START], p]
    ax.plot(*p[:-1].T if len(p) > 2 else p[:1].T, color=SERVED, lw=1.0, zorder=4)
    ax.annotate("", xy=p[-1], xytext=p[-2], zorder=4,
                arrowprops=dict(arrowstyle="-|>", color=SERVED, lw=1.0, mutation_scale=5, shrinkA=0, shrinkB=0))
    ax.plot(*START, "o", color=INK, ms=2.6, zorder=5)


# the corrupted inputs: (label, pictogram, how the commands go wrong); sources of input corruption in assistive
# teleoperation, BCI and EMG work: decoder misalignment and drift (Sadtler et al. 2014, Jarosiewicz et al. 2015; for EMG,
# electrode shift, sweat and fatigue), low-DOF interfaces for high-DOF robots (Herlant et al. 2016, Losey et al. 2020),
# the user's misperception and reaction time, and discrete interfaces (sip-and-puff, switches, motor-imagery classes)
FIG1_INPUTS = (("decoding error", "decoding", "rotate"), ("low-DOF control", "low_dof", "project"),
               ("misperception & delay", "misperception", "misjudge"), ("discrete input", "discrete", "quantize"))


def inputs_panel(fig, x0, top, tile_w=1.0, gap_x=0.1, row_h=0.78):
    """Figure 1a: one tile per source of corrupted input, two to a row (a last odd one centred), (x0, top) inches the
    panel's top left."""
    tile_h = tile_w * TILE_H / TILE_W
    for i, (label, name, kind) in enumerate(FIG1_INPUTS):
        lone = i == len(FIG1_INPUTS) - 1 and i % 2 == 0
        x, y = x0 + (0.5 if lone else i % 2) * (tile_w + gap_x), top - (i // 2) * row_h - tile_h
        ax = fig.add_axes([*inches(fig, x, y), *inches(fig, tile_w, tile_h)])
        icon(ax, name)
        reach(ax, kind)
        ax.set(xlim=(0, TILE_W), ylim=(0, TILE_H), xticks=[], yticks=[], aspect="auto")
        frame(ax)
        fig.text(*inches(fig, x + tile_w / 2, y - 0.03), label, fontsize=FS_TEXT, color=INK, ha="center", va="top")


FIG1_H = 5.0  # in: a and b on top, c below


def build_fig1(out):
    paper_style()
    fig = plt.figure(figsize=(FULL_W, FIG1_H))
    # a: where distorted commands come from
    inputs_panel(fig, 0.1, FIG1_H - 0.5)
    # b: the main insight, the user's and the task's flows as reshapings of the policy's
    w, h, bottom = 1.8, 1.4, 2.98
    user_flow_panel(fig, [*inches(fig, 2.78, bottom), *inches(fig, w, h)])
    task_flow_panel(fig, [*inches(fig, 2.78 + w + 0.5, bottom), *inches(fig, w, h)])
    # c: the method end to end, a schematic with made-up densities
    schematic(fig, 0.05, 0.22, pipeline_schematic())
    handles, labels = method_legend(["policy", "task", "intent_density", "readout", "constraint", "mars", "frs"])
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=inches(fig, 5.35, 1.3), ncol=1, fontsize=FS_LEGEND,
               handlelength=1.6, handletextpad=0.5, borderaxespad=0, labelspacing=0.55,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)})
    for x, y, letter in ((0.005, 0.995, "a"), (2.45 / FULL_W, 0.995, "b"), (0.005, 2.62 / FIG1_H, "c")):
        panel_label(fig, x, y, letter)
    png = save(fig, out, "fig1_method")
    write_stats(out, "fig1", dict(inputs=[label for label, _, _ in FIG1_INPUTS],
                                  schematic="panel a's reaches are illustrations; panel b's flows are exact flow-matching fields "
                                            "of made-up 1-D mixtures (the user's flow the straight paths the module is trained "
                                            "on); panel c's densities are made up",
                                  interface=f"panel b: f(a) = {USER_F[0]} a + {USER_F[1]}"))
    return png


# ================================================================ Figure 4: the cursor user study


STUDY_DIRS = ("cursor_study_short", "cursor_study")  # under results/: the sessions recorded with the short protocols
# (study_short_U<k>, T1 under condition_study.yaml, 2026-11-24/25) and the study's (study_T<t>_U<k>, condition_grid.yaml)
STUDY_GROUPS = tuple(((t,), t) for t in ("T1", "T2", "T3", "T4", "T5"))  # panel a's groups, as Figure 2b's tasks
CURVE_GROUPS = ((("T1",), "T1", "-"), (("T2", "T3", "T4", "T5"), "T2–T5", (0, (3, 1.6))))  # panel b's


def study_sessions(results):
    """Every logged trial of the study, by session: {(folder, participant, protocol, visit): [trial]}. A session is one
    participant's visit under one protocol (a user profile, and a task), counted separately when a participant plays
    several; the calibration runs are left out."""
    sessions = {}
    for study in STUDY_DIRS:
        root = Path(results) / study
        if not root.exists():
            continue
        runs = [r for r in find_runs(root) if r[0] != "sim" and not r[1].endswith("_calibration")]
        trials, _ = load_trials(runs)
        for r in trials:
            protocol, visit = r["run"].rsplit("_s", 1)[0], r["run"].rsplit("_s", 1)[1].split("_")[0]
            sessions.setdefault((study, r["participant"], protocol, int(visit)), []).append(r)
    return sessions


def group_rates(sessions, tasks):
    """Per session that played the tasks: each arm's success (%) over those tasks' trials."""
    out = {}
    for key, trials in sessions.items():
        own = [r for r in trials if r["task"] in tasks]
        if not own:
            continue
        out[key] = {arm: 100 * float(np.mean([r["outcome"] == "hit" for r in own if r["arm"] == arm])) for arm in ARMS
                    if any(r["arm"] == arm for r in own)}
    return out


def build_fig4(out, results):
    paper_style()
    sessions = study_sessions(results)
    if not sessions:
        raise FileNotFoundError(f"no study sessions under {results}/cursor_study*")
    slots = [(tasks, label, group_rates(sessions, tasks)) for tasks, label in STUDY_GROUPS]
    fig = plt.figure(figsize=(FULL_W, 2.75))
    # a: success per arm and task, as Figure 2b, the session as the unit (a dot each, joined across the arms); each task
    # keeps its slot, an empty one marked as such
    ax = fig.add_axes([*inches(fig, 0.45, 0.72), *inches(fig, 4.2, 1.8)])
    width = 0.85 / len(ARMS)
    stats = {}
    for i, (_, label, rates) in enumerate(slots):
        if not rates:
            ax.text(i, 50, "no sessions yet", ha="center", va="center", fontsize=FS_TEXT, color=INK_2, style="italic")
            continue
        left = i - width * len(ARMS) / 2
        xs = left + width * (np.arange(len(ARMS)) + 0.5)
        for per_arm in rates.values():
            ax.plot(xs, [per_arm.get(a, np.nan) for a in ARMS], "-", color="#c9c8c1", lw=0.6, zorder=2)
        stats[label] = {}
        for k, arm in enumerate(ARMS):
            vals = [per_arm[arm] for per_arm in rates.values() if arm in per_arm]
            m, lo, hi = t_ci(vals)
            stats[label][arm] = dict(mean_ci=[round(v, 1) for v in (m, lo, hi)], sessions=len(vals))
            x0 = left + k * width
            ax.add_patch(Rectangle((x0 + 0.06 * width, 0), 0.88 * width, m, facecolor=ARM_COLOR[arm], alpha=0.18, lw=0, zorder=1))
            ax.plot([x0 + 0.06 * width, x0 + 0.94 * width], [m, m], color=ARM_COLOR[arm], lw=2.0, zorder=3)
            if len(vals) > 1:
                ax.plot([xs[k], xs[k]], [lo, hi], color=INK, lw=0.8, zorder=3.5, solid_capstyle="butt")
            ax.scatter([xs[k]] * len(vals), vals, s=7, color=ARM_COLOR[arm], edgecolor="white", lw=0.4, zorder=4)
            ax.text(xs[k], min(max(max(vals), hi if len(vals) > 1 else m), 104) + 3, f"{m:.0f}", ha="center", va="bottom",
                    fontsize=FS_TEXT, color=INK, weight="bold" if arm == MARS_KEY else "normal", clip_on=False)
    ax.set(xlim=(-0.5, len(slots) - 0.5), ylim=(0, 112), yticks=[0, 25, 50, 75, 100], xticks=range(len(slots)))
    ax.set_xticklabels([f"{label}\n{len(rates)} session{'' if len(rates) == 1 else 's'}" for _, label, rates in slots],
                       fontsize=FS_TICK, linespacing=1.1)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_ylabel("Success (%)", labelpad=1)
    # b: the share of trials successful by each time, the mean of the sessions' curves; T1 solid, T2-T5 dashed
    ax = fig.add_axes([*inches(fig, 5.35, 0.72), *inches(fig, 1.6, 1.8)])
    curves = {}
    limit = 0.0
    groups = [(tasks, label, dash, rates) for tasks, label, dash in CURVE_GROUPS if (rates := group_rates(sessions, tasks))]
    for tasks, label, dash, rates in groups:
        group_trials = {key: [r for r in sessions[key] if r["task"] in tasks] for key in rates}
        lim = max(r["time_limit_s"] for ts in group_trials.values() for r in ts)
        limit = max(limit, lim)
        t = np.arange(0, lim + 1e-9, 0.02)
        curves[label] = {}
        for arm in ARMS:
            cs = [cumulative_hits([r for r in ts if r["arm"] == arm], t) for ts in group_trials.values() if any(r["arm"] == arm for r in ts)]
            c = 100 * np.mean(cs, 0)
            curves[label][arm] = {f"{s:g}": round(float(np.interp(s, t, c)), 1) for s in (2, 3, 4)}
            ax.step(t, c, where="post", color=ARM_COLOR[arm], lw=1.2, ls=dash, zorder=3 + (arm == MARS_KEY))
    ax.set(xlim=(0, limit), ylim=(0, 100), yticks=range(0, 101, 25))
    ax.set_xlabel("Time since trial start (s)", labelpad=1)
    ax.set_ylabel("Trials successful by then (%)", labelpad=1)
    if len(groups) > 1:
        styles = [plt.Line2D([], [], color=INK_2, lw=1.0, ls=dash) for _, _, dash, _ in groups]
        ax.legend(styles, [label for _, label, _, _ in groups], loc="upper left", fontsize=FS_LEGEND, handlelength=1.8, borderaxespad=0.2)
    handles = [Rectangle((0, 0), 1, 1, facecolor=ARM_COLOR[a], lw=0) for a in ARMS]
    fig.legend(handles, [ARM_LABEL[a] for a in ARMS], loc="center", bbox_to_anchor=(0.5, 0.11 / 2.75), ncol=len(ARMS),
               fontsize=FS_LEGEND, handlelength=1.2, handleheight=0.9, columnspacing=1.6, handletextpad=0.6, borderaxespad=0)
    for x, letter in ((0.005, "a"), (4.85 / FULL_W, "b")):
        panel_label(fig, x, 0.995, letter)
    png = save(fig, out, "fig4_study")
    write_stats(out, "fig4", dict(
        sessions=[dict(folder=k[0], participant=k[1], protocol=k[2], visit=k[3]) for k in sessions],
        success=stats, success_per_session={label: {"|".join(map(str, k)): {a: round(v, 1) for a, v in r.items()}
                                                    for k, r in rates.items()} for _, label, rates in slots if rates},
        success_by_s=curves,
        note="the session is the unit (a participant playing several user profiles counts once per profile); the sessions "
             "in cursor_study_short played T1 under condition_study.yaml (step gain mode, 40 ms replans), those in "
             "cursor_study every task under condition_grid.yaml (time gain mode, 80 ms); 95% t CIs over the sessions; panel b "
             "the mean of the sessions' cumulative success curves"))
    return png


# ================================================================ Figure 3: LIBERO


LIBERO_RUNS = Path("libero_sim") / "replication" / "users"  # under results/: <user>/test/<arm>/<run>/trial_*.npz (users.sh)
LIBERO_USERS = {"rotation": "U1\nrotation 60°", "lowdof": "U2\nno vertical", "lag": "U3\nlag + turn"}  # the users, as
# libero/configs/study_object_<user>.yaml name them: a fixed 3-D rotation of 60 degrees, the command's vertical component
# dropped, a lag-and-turn field
LIBERO_ARMS = ("fc", "frs", "frs_shallow", "mars")  # drawing order; the ablation, FRS at MARS's depth and prefix without the
# module, beside MARS
LIBERO_LABEL = {"fc": "Flow control", "frs": "FRS", "frs_shallow": f"FRS at {MARS}'s setting, no module", "mars": MARS}
LIBERO_COLOR = {"fc": ARM_COLOR["flow_control"], "frs": ARM_COLOR["frs"], "frs_shallow": "#a39e93", "mars": ARM_COLOR[MARS_KEY]}
LIBERO_EXAMPLES = ("3d9c5dda", "d27c18ea", "4832be44", "9eb1e0a4")  # panel c's resets (pair ids' first 8 characters), the
# rotation user's, one per task: MARS succeeds and every other arm fails on each (tasks 1, 6, 4, 7)
LIBERO_OBJECTS = {"akita_black_bowl_1": "bowl", "cream_cheese_1": "cheese", "wine_bottle_1": "bottle", "plate_1": "plate",
                  "wooden_cabinet_1": "cabinet", "flat_stove_1": "stove", "wine_rack_1": "rack"}
LIBERO_TARGETS = {8: ("akita_black_bowl_1", "plate_1"), 6: ("cream_cheese_1", "akita_black_bowl_1"),
                  1: ("akita_black_bowl_1", "flat_stove_1"), 4: ("akita_black_bowl_1", "wooden_cabinet_1"),
                  5: ("plate_1", "flat_stove_1"), 7: ("flat_stove_1",)}  # each task's object and destination
LIBERO_TASK = {1: "bowl on the stove", 4: "bowl on the cabinet", 5: "plate to the stove", 6: "cheese in the bowl",
               7: "turn on the stove", 8: "bowl on the plate"}  # panel c's titles
LIBERO_BOX = ((-0.47, 0.16), (-0.3, 0.27), (0.88, 1.2))  # panel c's extent (m): x, y, z
FIG3_H = 3.05


def libero_data(results):
    """Every test trial of the LIBERO study, per user and arm: {user: {arm: [trial]}}, a trial a dict of its record's
    fields the figure uses, its pair id (the shared reset) and the video beside it."""
    root = Path(results) / LIBERO_RUNS
    keep = ("task_id", "task_description", "success", "eef_pos", "object_names", "object_pos", "pair_id", "init_state_id")
    out = {}
    for user in LIBERO_USERS:
        out[user] = {}
        for arm in LIBERO_ARMS:
            runs = sorted((root / user / "test" / arm).glob("*/"))
            if len(runs) != 1:
                raise FileNotFoundError(f"{root / user / 'test' / arm}: expected one run, found {len(runs)}")
            trials = []
            for f in sorted(runs[0].glob("trial_*.npz")):
                z = np.load(f, allow_pickle=False)
                trials.append({k: z[k].item() if z[k].ndim == 0 else z[k] for k in keep} | {"video": f.with_suffix(".mp4")})
            out[user][arm] = trials
    return out


def wilson(k, n, z=1.96):
    """[rate, lower, upper] (%), the Wilson 95% interval."""
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [100 * p, 100 * (c - h), 100 * (c + h)]


def video_frame(path, k=0):
    """Frame k of a trial's video (the agent view), as an image array, through ffmpeg."""
    png = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{k})", "-frames:v", "1",  # nosec B603 B607
                          "-f", "image2pipe", "-vcodec", "png", "-"], check=True, capture_output=True).stdout
    import io
    return plt.imread(io.BytesIO(png), format="png")


def draw_libero_bars(ax, L):
    """Success per user and arm: a light bar to the pooled success (60 trials), its Wilson 95% interval in ink, each task's
    success (10 trials) as a dot, the arms of a user side by side; MARS's value in bold."""
    users = list(LIBERO_USERS)
    width = 0.85 / len(LIBERO_ARMS)
    stats = {}
    for i, user in enumerate(users):
        left = i - width * len(LIBERO_ARMS) / 2
        for k, arm in enumerate(LIBERO_ARMS):
            trials = L[user][arm]
            m, lo, hi = wilson(sum(t["success"] for t in trials), len(trials))
            tasks = sorted({t["task_id"] for t in trials})
            per_task = [100 * np.mean([t["success"] for t in trials if t["task_id"] == task]) for task in tasks]
            stats[f"{user}|{arm}"] = dict(success_wilson=[round(v, 1) for v in (m, lo, hi)], n=len(trials),
                                          per_task={str(task): round(v, 1) for task, v in zip(tasks, per_task)})
            x0 = left + k * width
            xc = x0 + width / 2
            color = LIBERO_COLOR[arm]
            ax.add_patch(Rectangle((x0 + 0.06 * width, 0), 0.88 * width, m, facecolor=color, alpha=0.18, lw=0, zorder=1))
            ax.plot([x0 + 0.06 * width, x0 + 0.94 * width], [m, m], color=color, lw=2.0, zorder=3)
            ax.plot([xc, xc], [lo, hi], color=INK, lw=0.8, zorder=3.5, solid_capstyle="butt")
            jitter = np.linspace(-0.22, 0.22, len(per_task)) * width
            ax.scatter(xc + jitter, per_task, s=5, color=color, edgecolor="white", lw=0.3, zorder=4)
            ax.text(xc, max(hi, max(per_task)) + 3, f"{m:.0f}", ha="center", va="bottom", fontsize=FS_TEXT, color=INK,
                    weight="bold" if arm == "mars" else "normal")
    ax.set(xlim=(-0.5, len(users) - 0.5), ylim=(0, 112), yticks=[0, 25, 50, 75, 100], xticks=range(len(users)))
    ax.set_xticklabels([LIBERO_USERS[u] for u in users], fontsize=FS_TICK, linespacing=1.1)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_ylabel("Success (%)", labelpad=1)
    return stats


def draw_libero_paths(ax, L, pid, user="rotation"):
    """One shared reset in 3-D, seen as panel a's frame shows the scene (the robot at the back; that frame is mirrored
    left to right against the simulator's y, so y is drawn negated to match it): the table, the objects where they start
    (the task's object and its destination as the scenes' target, named), and each arm's gripper path with its shadow on
    the table, MARS on top."""
    by_arm = {arm: next(t for t in L[user][arm] if t["pair_id"].startswith(pid)) for arm in LIBERO_ARMS}
    first = by_arm["mars"]
    names = [str(n) for n in first["object_names"]]
    targets = LIBERO_TARGETS.get(first["task_id"], ())
    (x0, x1), (y0, y1), (z0, z1) = LIBERO_BOX
    table = z0 + 0.02
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    ax.add_collection3d(Poly3DCollection([[(x0, -y0, table), (x1, -y0, table), (x1, -y1, table), (x0, -y1, table)]],
                                         facecolor="#f3f1ea", edgecolor=GRID, lw=0.5, zorder=0))
    pos = dict(zip(names, first["object_pos"][0]))
    for name, p in pos.items():
        target = name in targets
        style = TARGET_STYLE if target else dict(facecolor="#ecebe6", edgecolor="#b5b3ab", lw=0.6)
        ax.scatter([p[0]], [-p[1]], [table], s=28 if target else 18, c=[style["facecolor"]], edgecolors=[style["edgecolor"]],
                   linewidths=style["lw"], depthshade=False, zorder=2)
    for arm in LIBERO_ARMS:
        e = by_arm[arm]["eef_pos"]
        mars = arm == "mars"
        ax.plot(e[:, 0], -e[:, 1], np.full(len(e), table), color=LIBERO_COLOR[arm], lw=0.5, alpha=0.3, zorder=1)
        ax.plot(e[:, 0], -e[:, 1], e[:, 2], color=LIBERO_COLOR[arm], lw=1.3 if mars else 0.8, alpha=1 if mars else 0.85,
                zorder=5 if mars else 3)
    e = first["eef_pos"]
    ax.scatter([e[0, 0]], [-e[0, 1]], [e[0, 2]], s=9, c=[INK], depthshade=False, zorder=6)
    ax.set(xlim=(x0, x1), ylim=(-y1, -y0), zlim=(z0, z1))
    ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0), zoom=1.25)
    ax.view_init(elev=38, azim=0)
    ax.set_axis_off()
    ax.text2D(0.5, 0.0, LIBERO_TASK.get(first["task_id"], first["task_description"]), transform=ax.transAxes, fontsize=FS_TICK,
              color=INK, ha="center", va="top")   # under the table
    paths = [np.c_[t["eef_pos"][:, 0], -t["eef_pos"][:, 1], z] for t in by_arm.values() for z in (t["eef_pos"][:, 2], np.full(len(t["eef_pos"]), table))]
    place_labels(ax, [(np.array([pos[n][0], -pos[n][1], table]), LIBERO_OBJECTS.get(n, n)) for n in targets], np.concatenate(paths))
    return dict(task=first["task_id"], init_state=first["init_state_id"], success={arm: bool(t["success"]) for arm, t in by_arm.items()})


def place_labels(ax, marks, points, gap=0.065):
    """Name each mark ((x, y, z), text) of a 3-D axes beside it, on whichever of eight sides covers the fewest of `points`
    ([N, 3], the drawn paths) and the other marks and names, as figure text (drawn over every panel). gap: inches."""
    from mpl_toolkits.mplot3d import proj3d
    fig = ax.figure
    r = fig.canvas.get_renderer()
    proj = ax.get_proj()
    screen = lambda xyz: ax.transData.transform(np.c_[proj3d.proj_transform(*np.asarray(xyz, float).T, proj)[:2]])  # noqa: E731
    pts = screen(points)
    anchors = screen([m for m, _ in marks])
    g = gap * fig.dpi
    sides = [((1, 0), "left", "center"), ((-1, 0), "right", "center"), ((0, 1), "center", "bottom"), ((0, -1), "center", "top"),
             ((0.8, 0.8), "left", "bottom"), ((-0.8, 0.8), "right", "bottom"), ((0.8, -0.8), "left", "top"), ((-0.8, -0.8), "right", "top")]
    taken = [(a[0] - g, a[1] - g, a[0] + g, a[1] + g) for a in anchors]   # the marks themselves
    for (_, text), a in zip(marks, anchors):
        probe = fig.text(0, 0, text, fontsize=FS_TICK)
        bb = probe.get_window_extent(r)
        probe.remove()
        w, h = bb.width, bb.height
        best = None
        for k, ((dx, dy), ha, va) in enumerate(sides):
            x, y = a[0] + dx * g, a[1] + dy * g
            x0 = x - {"left": 0, "right": w, "center": w / 2}[ha]
            y0 = y - {"bottom": 0, "top": h, "center": h / 2}[va]
            box = (x0, y0, x0 + w, y0 + h)
            hits = ((pts[:, 0] >= box[0]) & (pts[:, 0] <= box[2]) & (pts[:, 1] >= box[1]) & (pts[:, 1] <= box[3])).sum()
            clash = sum(box[0] < t[2] and t[0] < box[2] and box[1] < t[3] and t[1] < box[3] for t in taken)
            score = (hits + 50 * clash, k)   # ties keep the side listed first
            if best is None or score < best[0]:
                best = (score, x, y, ha, va, box)
        _, x, y, ha, va, box = best
        taken.append(box)
        fig.text(*fig.transFigure.inverted().transform((x, y)), text, fontsize=FS_TICK, color=INK, ha=ha, va=va, zorder=10)


def build_fig3(out, results):
    paper_style()
    L = libero_data(results)
    fig = plt.figure(figsize=(FULL_W, FIG3_H))
    # a: the scene, the first frame of panel c's first MARS trial
    first = next(t for t in L["rotation"]["mars"] if t["pair_id"].startswith(LIBERO_EXAMPLES[0]))
    ax = fig.add_axes([*inches(fig, 0.1, 0.72), *inches(fig, 1.85, 1.85)])
    ax.imshow(video_frame(first["video"]))
    ax.set(xticks=[], yticks=[])
    frame(ax)
    # b: success per user and arm, each user's glyph under its name
    rect = [*inches(fig, 2.5, 0.78), *inches(fig, 2.05, 1.95)]
    ax = fig.add_axes(rect)
    stats = draw_libero_bars(ax, L)
    # c: four shared resets in 3-D, the rotation user's, two to a row
    examples = {}
    for k, pid in enumerate(LIBERO_EXAMPLES):
        r, c = divmod(k, 2)
        ax = fig.add_axes([*inches(fig, 4.74 + c * 1.13, 1.84 - r * 1.12), *inches(fig, 1.1, 1.08)], projection="3d", computed_zorder=False)
        ax.set_facecolor("none")   # so a neighbour's labels are not covered: the left column drawn over the right
        ax.set_zorder(2 - c)
        examples[pid] = draw_libero_paths(ax, L, pid)
    handles = [plt.Line2D([], [], color=LIBERO_COLOR[a], lw=4, solid_capstyle="butt") for a in LIBERO_ARMS]
    fig.legend(handles, [LIBERO_LABEL[a] for a in LIBERO_ARMS], loc="center", bbox_to_anchor=(0.5, 0.12 / FIG3_H), ncol=len(handles),
               fontsize=FS_LEGEND, handlelength=1.1, columnspacing=1.3, handletextpad=0.45, borderaxespad=0)
    for x, letter in ((0.005, "a"), (2.08 / FULL_W, "b"), (4.72 / FULL_W, "c")):
        panel_label(fig, x, 0.99, letter)
    png = save(fig, out, "fig3_libero")
    write_stats(out, "fig3", dict(
        users={u: v.replace("\n", " ") for u, v in LIBERO_USERS.items()}, arms=LIBERO_LABEL, success=stats, examples=examples,
        note="LIBERO-Goal tasks 1, 4, 5, 6, 7, 8, 10 test trials each per arm on shared resets (users.sh); success with a "
             "Wilson 95% interval over the 60 trials, a dot per task; FC at gain 1.5, each arm at its tuned setting"))
    return png


# ================================================================ Figure 2: the simulated cursor grid


TASK_SEEDS = (11, 5, 8, 2, 7)  # one sampled layout per task for the illustration
SEED_RUNS = tuple(f"grid_s{k}" for k in range(5))  # the grid's runs (configs/grid.yaml), under results/cursor_sim/sweeps, one per
# training seed: 0 is the grid of record, tuned; 1-4 retrain every module and fine-tuned policy on its own draw of the
# demonstrations and are tested at seed 0's settings on the grid's trials (teleop trains nothing, so every seed has the grid's)
FT_ARM = "ft_mars_user"   # the policy fine-tuned on each obstacle task's 300 demonstrations (as many as its task module's),
# with the user modules refitted on it
GRID_BASELINES = ("teleop", "ft_flow_control", "ft_frs")
N_BOOT = 4000  # bootstrap draws for the 95% CIs over the seeds, then the trials
ROLLOUT_SEED = 7  # panel a's trials
FIG2_H = 4.2  # in: a (the users x tasks matrix) on top, the methods' legend in one line, then b and c (positions in inches from the bottom)


def grid_data(results):
    """The grid's runs, one per training seed found (seed 0 is required; D["cell"](u, t, k) is run k's cell, the seed
    D["seeds"][k]). Every run is on the grid's trials, so methods compare trial for trial within a seed and a cell's
    numbers average over the seeds."""
    sweeps = Path(results) / "cursor_sim" / "sweeps"
    seeds = [k for k, run in enumerate(SEED_RUNS) if k == 0 or (sweeps / run / "test.json").exists()]
    runs = [json.loads((sweeps / SEED_RUNS[k] / "test.json").read_text()) for k in seeds]
    grid = runs[0]
    for k, run in zip(seeds[1:], runs[1:]):
        if run["test"].keys() != grid["test"].keys() or any(c["trials"]["side"] != grid["test"][key]["trials"]["side"]
                                                             for key, c in run["test"].items()):
            raise ValueError(f"seed {k}'s run ({SEED_RUNS[k]}) is not on the grid's trials")
    G = grid["args"]["sweep"]["grid"]
    users, tasks, noise = G["users"], G["tasks"], G["noises"][0]
    train = yaml.safe_load((config.CONFIG_DIR / "grid.yaml").read_text())["grid"]["train"]
    return dict(G=G, runs=runs, seeds=seeds, users=users, tasks=tasks, noise=noise, leads={},
                methods=[m for m in GRID_METHODS if m in grid["args"]["sweep"]["methods"]],
                obstacle_tasks=[t for t in tasks if G["cells"][f"{users[0]}x{t}"]["obstacle"]],
                cell=lambda u, t, k=0: runs[k]["test"][f"{u}x{t}|{noise}"], train={tuple(p) for p in train},
                test_n=grid["args"]["test_trials"])


def seed_values(D, u, t, m, key="success"):
    """A method's metric in a cell (%), one value per seed."""
    return [100 * D["cell"](u, t, k)["metrics"][m][key] for k in range(len(D["runs"]))]


def seed_mean(D, u, t, m, key="success"):
    return float(np.mean(seed_values(D, u, t, m, key)))


def lead_ci(D, u, t, a, b, n_boot=N_BOOT):
    """a's success minus b's in a cell (points): [mean over the seeds, 95% CI]. The trials pair within a seed; the CI
    resamples the seeds, then the trials within each (the README's five-seed comparisons, with the same draws)."""
    key = (u, t, a, b, n_boot)
    if key not in D["leads"]:
        hit = lambda k, m: np.array([o == "hit" for o in D["cell"](u, t, k)["trials"]["outcome"][m]], float)
        d = [hit(k, a) - hit(k, b) for k in range(len(D["runs"]))]
        rng = np.random.default_rng(zlib.crc32(f"{u}x{t}{a}{b}".encode()))
        boot = [np.mean([d[i][rng.integers(0, len(d[i]), len(d[i]))].mean() for i in rng.integers(0, len(d), len(d))])
                for _ in range(n_boot)]
        D["leads"][key] = [100 * float(np.mean([x.mean() for x in d])), 100 * float(np.percentile(boot, 2.5)),
                           100 * float(np.percentile(boot, 97.5))]
    return D["leads"][key]


def corruption_label(spec):
    if "scale" in spec:
        return f"gain ({spec['scale'][0]:g}, {spec['scale'][1]:g})"
    if "rotation_deg" in spec:
        return f"rotation {spec['rotation_deg']:g}°"
    return f"curl {spec['curl']:g}/s"


def task_label(spec):
    if not spec:
        return "no obstacle"
    if spec["kind"] == "ring":
        return f"ring at the {spec['around']}"
    return {"wall": "wall with a gap", "disc": "disc"}[spec["kind"]]


def cell_arms(D, cell, methods, k=0):
    """Run k's arms in a cell (its sweep names that seed's modules) at the settings that run tuned."""
    u, t = cell.split("x")
    return {m: with_setting(method_arm(D["runs"][k]["args"]["sweep"], m, cell), tuned(D["cell"](u, t, k))[m]) for m in methods}


def tuned(cell):
    """A test cell's tuned settings per method (runs before the joint tuning recorded `depths`)."""
    return cell.get("settings", cell.get("depths"))


CELL_FILL = {"U": "#dbe6f5", "T": "#e5dff7", "": "white"}  # a cell whose user module, task module, or neither was trained there


def user_glyph(ax, spec):
    """A user's readout in Figure 1a's terms: the straight push the user means (ink) and where the readout sends the
    cursor (the served colour): a gain, a rotation, or for a curl force field a clockwise-bending arrow."""
    start, aim = np.array([0.3, 0.2]), np.array([0.5, 0.5])
    ax.annotate("", xy=start + aim, xytext=start, arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.7, mutation_scale=4.5, shrinkA=0, shrinkB=0))
    served = dict(arrowstyle="-|>", color=SERVED, lw=0.9, mutation_scale=4.5, shrinkA=0, shrinkB=0)
    if "scale" in spec:
        end = start + np.array(spec["scale"]) * aim
    elif "rotation_deg" in spec:
        c, s = np.cos(np.radians(spec["rotation_deg"])), np.sin(np.radians(spec["rotation_deg"]))
        end = start + 0.9 * np.array([[c, -s], [s, c]]) @ aim   # anticlockwise, as the readout turns it
    else:   # the curl turns the cursor clockwise, by atan(tau * curl) at steady state, and slows it
        turn = -np.arctan(spec["tau"] * spec["curl"])
        c, s = np.cos(turn), np.sin(turn)
        end = start + 0.8 * np.array([[c, -s], [s, c]]) @ aim
        served["connectionstyle"] = "arc3,rad=-0.45"
    ax.annotate("", xy=end, xytext=start, arrowprops=served)
    ax.plot(*start, "o", color=INK, ms=1.8)
    ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[], aspect="equal")
    frame(ax)


def draw_users_tasks(fig, D, shown, top, side=0.74, pitch=0.8, row_h=0.25, glyph_gap=0.06, key_gap=0.16):
    """Figure 2a: the tasks as columns, each headed by its scene with a successful MARS trial (task_rollouts); the users as
    rows, each named beside its readout's glyph; each cell MARS-UT's success (mean over the seeds), filled by what was
    trained there (the user modules on T1, each task module on one task with one user's module in the loop; the rest
    compositions never trained together); the key on the rows to the right. The block (names to key) is centred on the
    figure; top inches its top. Returns the cells' numbers."""
    users, tasks = D["users"], D["tasks"]
    names = [f"{u}  {corruption_label(D['G']['cells'][f'{u}x{tasks[0]}']['corruption'])}" for u in users]
    key = [("", f"cells: {MARS}-UT success (%)"), ("U", "user module trained"), ("T", "task module trained"),
           ("", "composed at test time")]
    r = fig.canvas.get_renderer()

    def width(text):   # inches, measured on a text removed again
        probe = fig.text(0, 0, text, fontsize=FS_TICK)
        w = probe.get_window_extent(r).width / fig.dpi
        probe.remove()
        return w

    name_w = max(width(n) for n in names)
    key_w = max(width(t) + (0.22 if i else 0) for i, (_, t) in enumerate(key))
    glyph = row_h - 0.03
    grid_w = (len(tasks) - 1) * pitch + side
    block = name_w + glyph_gap + glyph + 0.1 + grid_w + key_gap + key_w
    x0 = (FULL_W - block) / 2
    gx = x0 + name_w + glyph_gap          # the glyphs' left edge
    cx = gx + glyph + 0.1                 # the first column's left edge
    kx = cx + grid_w + key_gap            # the key's left edge
    for j, t in enumerate(tasks):
        ax = fig.add_axes([*inches(fig, cx + j * pitch, top - side), *inches(fig, side, side)])
        s = shown[t]
        draw_scene(ax, s["layouts"], s["task"], obstacle_style=OBSTACLE_STYLE)
        if s["path"] is not None:
            ax.plot(s["path"][:, 0], s["path"][:, 1], color=GRID_COLOR["mars"], lw=1.0, zorder=3)
        ax.text(0.04, 0.96, t, transform=ax.transAxes, fontsize=FS_TEXT, color=INK, ha="left", va="top", zorder=5,
                bbox=dict(boxstyle="square,pad=0.12", facecolor="white", edgecolor="none"))
        frame(ax)
    values = {}
    row_y = [top - side - 0.08 - (i + 1) * row_h for i in range(len(users))]   # each row's bottom
    for i, (u, y) in enumerate(zip(users, row_y)):
        spec = D["G"]["cells"][f"{u}x{tasks[0]}"]["corruption"]
        fig.text(*inches(fig, gx - glyph_gap, y + row_h / 2), names[i], fontsize=FS_TICK, color=INK, ha="right", va="center")
        user_glyph(fig.add_axes([*inches(fig, gx, y + 0.015), *inches(fig, glyph, glyph)]), spec)
        for j, t in enumerate(tasks):
            kind = "U" if t == tasks[0] else "T" if (u, t) in D["train"] else ""
            x = cx + j * pitch
            fig.patches.append(Rectangle(inches(fig, x, y + 0.01), *inches(fig, side, row_h - 0.02), transform=fig.transFigure,
                                         facecolor=CELL_FILL[kind], edgecolor=GRID, lw=0.6))
            v = seed_mean(D, u, t, "mars")
            values[f"{u}x{t}"] = round(v, 1)
            fig.text(*inches(fig, x + side / 2, y + row_h / 2), f"{v:.0f}", fontsize=FS_TEXT, color=INK, ha="center", va="center")
            if kind:
                fig.text(*inches(fig, x + 0.05, y + row_h / 2), kind, fontsize=FS_TICK, color=INK_2, ha="left", va="center")
    for i, ((kind, text), y) in enumerate(zip(key, row_y)):   # the key, a line per row
        yy = y + row_h / 2
        if i == 0:
            fig.text(*inches(fig, kx, yy), text, fontsize=FS_LEGEND, color=INK, ha="left", va="center")
            continue
        fig.patches.append(Rectangle(inches(fig, kx, yy - 0.08), *inches(fig, 0.16, 0.16), transform=fig.transFigure,
                                     facecolor=CELL_FILL[kind], edgecolor=GRID, lw=0.6))
        fig.text(*inches(fig, kx + 0.08, yy), kind, fontsize=FS_TICK, color=INK_2, ha="center", va="center")
        fig.text(*inches(fig, kx + 0.22, yy), text, fontsize=FS_LEGEND, color=INK, ha="left", va="center")
    return values


def task_rollouts(D, device, n=8, seed=ROLLOUT_SEED):
    """One sampled layout per task and a successful MARS trial on it: the task's trained cell (T1 with the first user),
    seed 0's modules at their tuned depth, the first success of n trials with the grid's readout noise. A vision task
    module's scene tokens are attached as the sweep attaches them (cached under checkpoints/scene/fig3)."""
    out = {}
    for t, layout_seed in zip(D["tasks"], TASK_SEEDS):
        u = next((u for u in D["users"] if (u, t) in D["train"] and t != D["tasks"][0]), D["users"][0])
        arms = cell_arms(D, f"{u}x{t}", ["mars"])
        policies, adapters = load_checkpoints(arms, device)
        arm = arms["mars"]
        task = arm_task(arm)
        layouts = sample_layouts(np.random.default_rng(layout_seed), 1, (4, 4), task)
        attach_scene(layouts, scene_sources(adapters), CHECKPOINTS / "scene" / "fig3", f"{u}x{t}",
                     lambda: SceneEncoder(SCENE_VLM_PATH, device=device))
        torch.manual_seed(seed)
        recs = run_trials(arm, layouts.tile(n), ExpertAgent(), readout_for(arm, np.random.default_rng(seed + 1), D["noise"]),
                          policies[arm["model"]], adapters, device)
        hits = [i for i, r in enumerate(recs) if r["outcome"] == "hit"]
        out[t] = dict(cell=f"{u}x{t}", layouts=layouts, task=task, path=np.asarray(recs[hits[0]]["path"], float) if hits else None,
                      trial=hits[0] if hits else None, hits=len(hits), of=n)
    return out


def draw_grid_bars(fig, D, rect):
    """Success per task and method in Figure 2b's style, over the seeds: a light bar to the mean over the four users,
    the mean as a line with its 95% t CI over the users in ink (the user is the unit, as the participant in Figure 2b), each user's mean over the seeds as a dot
    (hollow: a cell the arm's modules were trained on), a cell's dots joined across the methods. On T1 only the four
    arms that differ."""
    ax = fig.add_axes(rect)
    users, tasks, methods = D["users"], D["tasks"], D["methods"]
    width = 0.9 / len(methods)
    ax.set(xlim=(-0.5, len(tasks) - 0.5), ylim=(0, 108), yticks=[0, 25, 50, 75, 100], xticks=range(len(tasks)))
    ax.set_yticklabels(["0", "25", "50", "75", "100"])
    ax.set_ylabel("Success (%)", labelpad=1)
    ax.tick_params(axis="x", length=0)
    trained = {"mars_user", "mars_task", "mars"}
    means = {}
    r = fig.canvas.get_renderer()
    line = 1.15 * FS_TEXT / 72 / (rect[3] * fig.get_figheight()) * 108   # a value label's height, in success points
    for i, t in enumerate(tasks):
        shown = [m for m in methods if t != "T1" or m in T1_METHODS]
        left = i - width * len(shown) / 2
        labels = []
        for u in users:   # one grid cell, method to method
            ax.plot(left + width * (np.arange(len(shown)) + 0.5), [seed_mean(D, u, t, m) for m in shown], "-", color="#c9c8c1",
                    lw=0.6, zorder=2)
        for k, m in enumerate(shown):
            vals = [seed_mean(D, u, t, m) for u in users]
            mean, lo, hi = t_ci(vals)
            means[f"{t}|{m}"] = [round(mean, 1), round(lo, 1), round(hi, 1)]
            x0 = left + k * width
            xc = x0 + width / 2
            ax.add_patch(Rectangle((x0 + 0.06 * width, 0), 0.88 * width, mean, facecolor=GRID_COLOR[m], alpha=0.18, lw=0, zorder=1))
            ax.plot([x0 + 0.06 * width, x0 + 0.94 * width], [mean, mean], color=GRID_COLOR[m], lw=2.0, zorder=3)
            ax.plot([xc, xc], [lo, hi], color=INK, lw=0.8, zorder=3.5, solid_capstyle="butt")
            for u, v in zip(users, vals):
                hollow = m in trained and (u, t) in D["train"]
                ax.plot(xc, v, "o", ms=2.6, color=GRID_COLOR[m], mfc="white" if hollow else GRID_COLOR[m], mew=0.8, zorder=4)
            labels.append(ax.text(xc, min(max(max(vals), hi), 108) + 2.5, f"{mean:.0f}", ha="center", va="bottom", fontsize=FS_TEXT,
                                  color=INK, weight="semibold" if m == "mars" else "normal", clip_on=False))   # above the CI (cut at the axis top)
        for prev, cur in zip(labels, labels[1:]):   # a label that would touch its left neighbour's goes a line above it
            a, b = prev.get_window_extent(r), cur.get_window_extent(r)
            if b.x0 < a.x1 + 1.5 / 72 * fig.dpi and b.y0 < a.y1 and a.y0 < b.y1:
                cur.set_y(prev.get_position()[1] + line)
    ax.set_xticklabels([f"{t}" for t in tasks], fontsize=FS_TICK)
    return means


def build_fig2(out, results, device):
    paper_style()
    D = grid_data(results)
    fig = plt.figure(figsize=(FULL_W, FIG2_H))
    y = lambda inch: inch / FIG2_H
    # a: the users x tasks matrix, each task headed by its scene and a MARS trial, each cell MARS-UT's success
    shown = task_rollouts(D, device)
    cell_values = draw_users_tasks(fig, D, shown, FIG2_H - 0.1)
    # the methods' legend, for b and c below
    handles = [plt.Line2D([], [], color=GRID_COLOR[m], lw=4, solid_capstyle="butt") for m in D["methods"]]
    fig.legend(handles, [GRID_LABEL[m] for m in D["methods"]], loc="center", bbox_to_anchor=(0.5, y(2.0)), ncol=len(handles),
               fontsize=FS_LEGEND, handlelength=1.1, columnspacing=1.3, handletextpad=0.45, borderaxespad=0)
    # b: every method on every task; c: on the obstacle tasks, the arms against the time spent demonstrating each
    task_cis = draw_grid_bars(fig, D, (0.065, y(0.42), 4.45 / FULL_W, y(1.2)))
    demo_time = draw_demo_time(fig, demo_time_data(results), (5.5 / FULL_W, y(0.42), 1.45 / FULL_W, y(1.2)))
    for x, yy, letter in ((0.005, 0.995, "a"), (0.005, y(1.74), "b"), (5.0 / FULL_W, y(1.74), "c")):
        panel_label(fig, x, yy, letter)
    png = save(fig, out, "fig2_sim")
    users, tasks, methods, seeds = D["users"], D["tasks"], D["methods"], D["seeds"]
    cells = [(u, t) for u in users for t in tasks]
    obstacle = [(u, t) for u in users for t in D["obstacle_tasks"]]
    r1 = lambda v: [round(x, 1) for x in v]
    per_cell = {f"{u}x{t}": {m: round(seed_mean(D, u, t, m), 1) for m in methods} for u, t in cells}
    best_base = {f"{u}x{t}": max(GRID_BASELINES, key=lambda b: seed_mean(D, u, t, b)) for u, t in obstacle}
    lead_best = {f"{u}x{t}": r1(lead_ci(D, u, t, "mars", best_base[f"{u}x{t}"])) for u, t in obstacle}
    over_fine_tuned = {f"{u}x{t}": r1(lead_ci(D, u, t, "mars", FT_ARM)) for u, t in obstacle}
    obstacle_per_seed = {m: [float(np.mean([100 * D["cell"](u, t, k)["metrics"][m]["success"] for u, t in obstacle]))
                             for k in range(len(seeds))] for m in methods}
    beyond = lambda leads: dict(cells=int(sum(v[1] > 0 for v in leads.values())), of=len(leads),
                                level=[c for c, v in leads.items() if v[1] <= 0])
    write_stats(out, "fig2", dict(
        names={"MARS-UT": "user + task modules (mars)", "MARS-T": "task module alone (mars_task)"},
        seeds=seeds, seed_runs=[f"results/cursor_sim/sweeps/{SEED_RUNS[k]}" for k in seeds],
        task_module_note="the task module is the vision module: pi0.5's vision tokens of a picture of the scene (vlm_image, "
                         "relative geometry), trained on 300 demonstrations with 32 rigid copies of each, rank 256",
        seed_note="every number is over the seeds: seed 0 is the grid of record; seeds 1-4 retrain every module (user modules "
                  "from 100 demonstrations, task modules from 300) and the fine-tuned policies (each obstacle task's policy "
                  "fine-tuned on 300 demonstrations, its user modules refitted on it; flow control and FRS play on them) on "
                  "their own draws and rerun them on the grid's trials; teleop trains nothing, so its is the grid's in every seed",
        ci_note=f"panel b: 95% t CIs over the four users (each user's success averaged over the seeds), the user the unit as "
                f"the session in Figure 4 (the drawn lines stop at the axis); panel c: 95% t CIs over the training "
                f"seeds (demo_time); the leads: 95% CIs resampling the seeds, then the trials within each ({N_BOOT} draws)",
        users={u: corruption_label(D["G"]["cells"][f"{u}x{tasks[0]}"]["corruption"]) for u in users},
        tasks={t: dict(label=task_label(D["G"]["cells"][f"{users[0]}x{t}"]["obstacle"]), time_limit_s=D["G"]["cells"][f"{users[0]}x{t}"]["time_limit_s"] or 8.0) for t in tasks},
        test_trials_per_cell=D["test_n"], success_per_cell=per_cell,
        success_per_cell_per_seed={f"{u}x{t}": {m: r1(seed_values(D, u, t, m)) for m in methods} for u, t in cells},
        collision_per_cell={f"{u}x{t}": {m: round(seed_mean(D, u, t, m, "collision"), 1) for m in methods} for u, t in cells},
        task_means={t: {m: round(float(np.mean([per_cell[f"{u}x{t}"][m] for u in users])), 1) for m in methods} for t in tasks},
        task_means_ci=task_cis,
        obstacle_means={m: round(float(np.mean(v)), 1) for m, v in obstacle_per_seed.items()},
        obstacle_means_per_seed={m: r1(v) for m, v in obstacle_per_seed.items()},
        obstacle_means_sd_over_seeds={m: round(float(np.std(v, ddof=1)), 1) if len(v) > 1 else 0.0 for m, v in obstacle_per_seed.items()},
        demo_time=demo_time,
        composed_beats_all_baselines=beyond(lead_best), composed_lead_over_best_baseline=lead_best, best_baseline=best_base,
        composed_lead_range=[min(v[0] for v in lead_best.values()), max(v[0] for v in lead_best.values())],
        composed_over_fine_tuned=over_fine_tuned, composed_beats_fine_tuned=beyond(over_fine_tuned),
        t1_user_module_over_teleop={f"{u}x{tasks[0]}": r1(lead_ci(D, u, tasks[0], "mars", "teleop")) for u in users},
        tuned_settings={f"{u}x{t}": {f"s{s}": tuned(D["cell"](u, t, k)) for k, s in enumerate(seeds)} for u, t in cells},
        mars_ut_per_cell=cell_values,
        task_rollouts={t: {k: v[k] for k in ("cell", "trial", "hits", "of")} for t, v in shown.items()},
        t1_note="on T1 MARS-U is MARS, MARS-T is FRS and the fine-tuned arm is MARS, and flow control and FRS play the frozen "
                "policy (no obstacle, nothing to fine-tune on), so only four arms are drawn",
        flow_control_note="Kao et al.'s flow control on each task's fine-tuned policy, ic_gain 2.0, tau tuned per cell; FRS on "
                          "the same policy, pinned_steps tuned per cell"))
    return png


# ================================================================ Figure 3e: success against the time spent demonstrating


DEMO_SECONDS = (30, 60, 120, 300, 600, 1200, 3600, 10800)  # the budgets, time spent demonstrating a task (s; _t<s> in names)
DEMO_RUNS = "demo_time"  # results/cursor_sim/sweeps/demo_time_s<k> (configs/grid_demo_time.yaml), one per training seed
DEMO_ARMS = {"ft_frs": "ft_frs_t{s}", "ft_flow_control": "ft_flow_control_t{s}", FT_ARM: "ft_mars_user_t{s}",
             "mars": "mars_t{s}"}  # panel b's arm: its method per budget (_t<s>), in drawing order (MARS-UT on top);
# colours and labels are panel b's
DEMO_TASKS = {"T2": "disc_large", "T3": "wall_gap", "T4": "ring_goal", "T5": "ring_start"}  # the tasks' checkpoint names
GRID_DEMOS = 300  # the grid's task modules (and panel b's fine-tuned policies): 300 demonstrations of the task


def demo_time_data(results):
    """The demonstration-time runs, one per training seed found (seed 0 is required). An arm's value at a budget is its
    success averaged over the 16 obstacle cells, one value per seed. The runs play the grid's cells and trials: their
    teleop's outcomes must be the grid's (grid_s0's), trial for trial."""
    sweeps = Path(results) / "cursor_sim" / "sweeps"
    load = lambda name: json.loads((sweeps / name / "test.json").read_text())  # noqa: E731
    seeds = [k for k in range(5) if k == 0 or (sweeps / f"{DEMO_RUNS}_s{k}" / "test.json").exists()]
    runs = {k: load(f"{DEMO_RUNS}_s{k}") for k in seeds}
    grid = load("grid_s0")
    cells = list(runs[0]["test"])
    teleop = lambda run, c: run["test"][c]["trials"]["outcome"]["teleop"]  # noqa: E731
    for k, run in runs.items():
        if list(run["test"]) != cells or any(teleop(run, c) != teleop(grid, c) for c in cells):
            raise ValueError(f"{DEMO_RUNS}_s{k} is not on the grid's cells and trials")
    pooled = lambda run, m: 100 * float(np.mean([run["test"][c]["metrics"][m]["success"] for c in cells]))  # noqa: E731
    curve = {arm: {s: [pooled(run, method.format(s=s)) for run in runs.values()] for s in DEMO_SECONDS}
             for arm, method in DEMO_ARMS.items()}
    return dict(cells=cells, curve=curve, seeds={arm: seeds for arm in DEMO_ARMS})


def demo_log(task, seconds, seed=0):
    """The training log of the policy fine-tuned on a budget of a task's demonstrations (its args hold the count the
    budget held, the same draw as the task module's; its eval the training length picked)."""
    path = CHECKPOINTS / "policies" / "demo_time" / f"flow_goaldrop_{task}_ft_t{seconds}_s{seed}" / "train_log.json"
    return json.loads(path.read_text())


def mean_ci(values):
    """[mean, lower, upper]: the 95% t CI over the seeds, or the one value thrice."""
    return t_ci(values) if len(values) > 1 else [float(np.mean(values))] * 3


def draw_demo_time(fig, D, rect):
    """Panel e: each arm's success on the 16 obstacle cells against the time spent demonstrating each task (log): MARS-UT
    with the vision task module trained on the budget, and the three arms on the policy fine-tuned on it (Fine-tuned +
    MARS-U, flow control, FRS); the mean over the training seeds with its 95% t CI as a band, on panel b's success scale,
    and in grey the time the grid's 300 demonstrations take (panel b's task modules and fine-tuned policies; each task's
    at its mean demonstration, from the largest budget's draw). The methods' legend is the figure's. Returns the panel's
    numbers."""
    ax = fig.add_axes(rect)
    minutes = np.array(DEMO_SECONDS) / 60
    per_demo = {t: DEMO_SECONDS[-1] / demo_log(task, DEMO_SECONDS[-1])["args"]["n_demos"] for t, task in DEMO_TASKS.items()}
    grid_min = {t: GRID_DEMOS * v / 60 for t, v in per_demo.items()}
    lo, hi = min(grid_min.values()), max(grid_min.values())
    ax.axvspan(lo, hi, color=GRID, alpha=0.7, lw=0, zorder=1)
    ax.text(np.sqrt(lo * hi), 107, f"{GRID_DEMOS} demos (b)", ha="center", va="top", fontsize=FS_TEXT, color=INK_2)
    curve = {arm: np.array([mean_ci(D["curve"][arm][s]) for s in DEMO_SECONDS]) for arm in DEMO_ARMS}
    for arm in DEMO_ARMS:
        if len(D["seeds"][arm]) > 1:
            ax.fill_between(minutes, curve[arm][:, 1], curve[arm][:, 2], color=GRID_COLOR[arm], alpha=0.2, lw=0, zorder=2)
        ax.plot(minutes, curve[arm][:, 0], "-o", color=GRID_COLOR[arm], lw=1.3, ms=2.6, zorder=3)
    ax.set_xscale("log")
    ax.set_xticks(minutes, [f"{v:g}" for v in minutes])
    ax.minorticks_off()
    ax.set(xlim=(minutes[0] / 1.4, minutes[-1] * 1.4), ylim=(0, 108), yticks=[0, 25, 50, 75, 100])
    ax.set_xlabel("Demonstration time (min)", labelpad=1)
    ax.set_ylabel("Success, T2–T5 (%)", labelpad=1)
    r1 = lambda v: [round(float(x), 1) for x in v]
    return dict(
        seconds=list(DEMO_SECONDS), seeds=D["seeds"], cells=len(D["cells"]),
        success={arm: {str(s): r1(c) for s, c in zip(DEMO_SECONDS, curve[arm])} for arm in DEMO_ARMS},
        success_per_seed={arm: {str(s): r1(D["curve"][arm][s]) for s in DEMO_SECONDS} for arm in DEMO_ARMS},
        demonstrations_per_budget={t: {str(s): demo_log(task, s)["args"]["n_demos"] for s in DEMO_SECONDS}
                                   for t, task in DEMO_TASKS.items()},
        fine_tuned_training_steps={t: {str(s): demo_log(task, s)["best_step"] for s in DEMO_SECONDS} for t, task in DEMO_TASKS.items()},
        grid_demonstrations_minutes={t: round(v, 1) for t, v in grid_min.items()},
        time_note="a budget's demonstrations each take their duration up to the goal's selection (the reach and the 0.3 s "
                  "hold), with no time between them; the grid's 300 at each task's mean duration from the largest budget's "
                  "draw")


# ================================================================


FIGURES = ("fig1", "fig2", "fig3", "fig4")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", choices=list(FIGURES), default=list(FIGURES))
    p.add_argument("--out", default=str(FIG_DIR), help="output folder (default: figures/)")
    p.add_argument("--results", default=str(RESULTS))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    out, results = Path(args.out), Path(args.results)
    builders = {"fig1": lambda: build_fig1(out), "fig2": lambda: build_fig2(out, results, args.device), "fig3": lambda: build_fig3(out, results),
                "fig4": lambda: build_fig4(out, results)}
    for name in args.only:
        print(f"{name} -> {builders[name]()}", flush=True)


if __name__ == "__main__":
    main()
