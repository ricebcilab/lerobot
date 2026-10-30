"""The paper's figures: MARS (Modular Adapters for Reversal Steering, the display name of mars) illustrated,
the human study, and the simulated user x task grid.

    uv run --extra mars python examples/pi05/mars/figures/build_figures.py                         # all three, into examples/pi05/mars/figures/
    uv run --extra mars python examples/pi05/mars/figures/build_figures.py --only fig1 fig2        # some
    uv run --extra mars python examples/pi05/mars/figures/build_figures.py --out /tmp/figs --device cpu

Writes figures/fig1_method.{pdf,png}, fig2_study.{pdf,png}, fig3_grid.{pdf,png} (7.0 in wide, 300 dpi, Type 42
fonts) and figures/stats.json, the numbers the captions quote (one entry per figure, so --only keeps the rest).

Names: MARS is MARS with both modules (user and task), MARS-U the user module alone, MARS-T the task module alone.

Reads (results/ is git-ignored, so these are local):
    fig1  nothing (its reaches come from the channel's rule, its densities are made up)
    fig2  results/p*/study_s*/ (the recorded sessions) and results/sim/study_s1 (the expert's reference)
    fig3  the grid's five training seeds: results/sim/sweeps/grid_s0/test.json (seed 0, the grid of record) and
          results/sim/sweeps/grid_s<k>/combined/test.json (seeds 1-4), and each seed's checkpoints for one cell's rollouts

Figures 1b and 3a are schematics in FRS's manner: their 1-D densities are made up to show the mechanism.
"""

import argparse
import json
import shutil
import subprocess  # nosec B404
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cursor"))

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
from cursor_sa.analysis import GRID, INK, INK_2, aligned_path, cumulative_hits, draw_scene, paired_lead
from cursor_sa.env import Layouts, TaskConfig, sample_layouts
from cursor_sa.paths import RESULTS
from cursor_sa.records import load_run
from cursor_sa.session import arm_task, channel_for, load_checkpoints, run_trials
from run_sweep import method_arm, with_depth

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
GRID_METHODS = ["teleop", "flow_control", "frs", "mars_task", "mars_user", "retrained_mars_user", "mars"]
GRID_LABEL = {"teleop": "Teleop", "flow_control": "Flow control", "frs": "FRS", "mars_user": f"{MARS}-U", "mars_task": f"{MARS}-T",
              "mars": MARS, "retrained_mars_user": f"Retrained policy + {MARS}-U"}
GRID_COLOR = {"teleop": "#008300", "flow_control": "#e87ba4", "frs": "#e69f00", "mars_task": "#8a63d2", "mars_user": "#6da7ec",
              "retrained_mars_user": "#2e7da0", "mars": ARM_COLOR[MARS_KEY]}  # one palette with Figures 1 and 2: MARS a
# royal blue, MARS-U a sky blue, the retrained arm a cerulean, MARS-T a lighter purple; every pair of the family is 15 or more
# colour units apart (OKLab x 100), the limit the colour checks ask for
T1_METHODS = ("teleop", "flow_control", "frs", "mars")  # on T1 MARS-U is MARS, MARS-T is FRS and the retrained arm is MARS
FULL_W = 7.0  # in, a two-column figure
SURFACE = "white"
FILL = "#ebeae4"


FS_PANEL, FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND, FS_TEXT = 10, 8, 8, 7, 7, 7  # pt: panel letters, titles, axis labels,
# tick labels, legends, text in the plots (callouts, equations, numbers on marks)


def paper_style():
    plt.rcParams.update({"font.size": FS_TEXT, "font.family": "Liberation Sans", "axes.edgecolor": INK_2, "axes.labelcolor": INK,
                         "xtick.color": INK_2, "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": False, "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": None,
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
SCHEMATIC_TITLES = {"readout": "user readout  $a_{\\mathrm{readout}}$", "noise": "noise  $z \\sim \\mathcal{N}(0, I)$",
                    "policy": "policy actions  $a_{\\mathrm{policy}}$"}
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
    """The user readout's panel: the policy's density for reference, the readout's commands, the intent u (on the user's intent
    density, dashed, when given, else on the axis; it need not be in the policy's distribution) and the command u-hat it is
    corrupted into, joined by C."""
    density(ax, pol, DENSITY_FILL)
    if intent is not None:
        density(ax, intent, None, edge=INK, lw=0.9, ls=INTENT_DASH, support_only=True)
    density(ax, readout, READOUT_FILL, edge=SERVED, support_only=True)
    yu = on_density(ax, u, np.zeros_like(readout) if intent is None else intent, "v", INK, hollow=True)
    yh = on_density(ax, uh, readout, "v", SERVED)
    ax.annotate("", xy=(uh, yh), xytext=(u, yu), arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=0.7, mutation_scale=5, shrinkA=4, shrinkB=4))
    ax.text((u + uh) / 2 - 0.08, (yu + yh) / 2 + 0.03, "$C$", fontsize=FS_TEXT, color=INK_2, ha="right", va="bottom")
    ax.text(u - 0.12, yu + 0.07, "$u$", fontsize=FS_TEXT, ha="right", va="bottom")
    ax.text(uh + 0.17, yh, "$\\hat u$", fontsize=FS_TEXT, ha="left", va="center", color=SERVED)


def noise_panel(ax, z, other, other_color):
    """The prior with MARS's latent z-hat (in its bulk) and another method's (hollow)."""
    noise = sps.norm.pdf(SCHEMATIC_X)
    density(ax, noise, DENSITY_FILL)
    h = on_density(ax, z, noise, "o", ARM_COLOR[MARS_KEY])
    on_density(ax, other, noise, "o", other_color, hollow=True)
    ax.text(z + 0.2, h + 0.01, "$\\hat z$", fontsize=FS_TEXT, color=ARM_COLOR[MARS_KEY], ha="left", va="bottom")


READOUT_CALLOUT = [[("The readout ", {}), ("corrupts", dict(color=SERVED, fontweight="bold")), (" the intent $u$", {})],
               [("into an off-policy command $\\hat u$ …", {})]]


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
        spec["draw"][key](density_axes(fig, px + 0.07, py + 0.1, w - 0.14, ph - band - 0.15, spec["top"][key], "z" if key == "noise" else "a"))
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


def fig1_schematic():
    """Figure 1b: the policy's two modes (two targets); the user's intent near the right one, partly off its support; the
    readout's commands, the intent corrupted, off the policy's support. MARS's reversal (through A_U) lands in the bulk of the
    noise and the flow brings it to the mode the user means, FRS's (A_U = I) lands far in the tail and comes back near the
    command."""
    X, mars, frs = SCHEMATIC_X, ARM_COLOR[MARS_KEY], ARM_COLOR["frs"]
    pol = 0.45 * sps.norm.pdf(X, -2.0, 0.36) + 0.55 * sps.norm.pdf(X, -0.35, 0.34)
    intent = 0.42 * sps.norm.pdf(X, 0.45, 0.32)
    readout = 0.42 * sps.norm.pdf(X, 2.1, 0.3)

    def policy_panel(ax):
        density(ax, pol, DENSITY_FILL)
        h = on_density(ax, -0.35, pol, "o", mars)
        on_density(ax, 2.05, pol, "o", frs, hollow=True)
        ax.text(-0.15, h, "$\\hat a$", fontsize=FS_TEXT, color=mars, ha="left", va="center")

    return dict(
        draw={"readout": lambda ax: intent_and_command(ax, pol, readout, 0.45, 2.1, intent), "noise": lambda ax: noise_panel(ax, 0.62, 2.75, frs),
              "policy": policy_panel},
        top={"readout": 0.84, "noise": 0.46, "policy": 0.8},
        callouts={"readout": READOUT_CALLOUT,
                  "noise": [[("… ", {}), ("$A_U$", dict(color=mars)), (" steers its reversal", {})],
                            [("into ", {}), ("typical noise", EMPHASIS), (" …", {})]],
                  "policy": [[("… which the policy maps to", {})], [("an ", {}), ("in-distribution", EMPHASIS), (" action $\\hat a$.", {})]]},
        flows=[("$\\hat z = \\hat u - \\int_0^1 A_U\\, v_P\\, dt$", [("flow reversal", FLOW_NAME), ("FRS: $A_U = I$", dict(color=frs))]),
               ("$\\hat a = \\hat z + \\int_0^1 v_P\\, dt$", [("flow matching", FLOW_NAME)])])


def fig3_schematic():
    """Figure 3a: the original policy P's intended mode runs through the obstacle (the hatched band of actions); the task
    module (A_T, b_T) moves the flow's mode clear of it; MARS lands there, MARS-U (A_T = I, b_T = 0) on P's mode."""
    X, mars, mars_u = SCHEMATIC_X, ARM_COLOR[MARS_KEY], GRID_COLOR["mars_user"]
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
        density(ax, pol, None, edge=INK_2, lw=0.8, ls=(0, (2.5, 1.5)))
        density(ax, task, DENSITY_FILL)
        h = on_density(ax, 0.75, task, "o", mars)
        hu = on_density(ax, -0.4, pol, "o", mars_u, hollow=True)
        ax.plot(-0.67, hu + 0.02, "x", color=INK, ms=3.6, mew=1.0, zorder=7, clip_on=False)
        ax.text(0.95, h, "$\\hat a$", fontsize=FS_TEXT, color=mars, ha="left", va="center")
        ax.text((lo + hi) / 2, -0.028, "obstacle", fontsize=FS_TEXT, color=INK_2, ha="center", va="top")

    return dict(
        gap=1.42, noise_w=1.45,
        draw={"readout": readout_panel, "noise": lambda ax: noise_panel(ax, 0.78, 0.5, mars_u), "policy": policy_panel},
        top={"readout": 0.84, "noise": 0.46, "policy": 0.8},
        callouts={"readout": READOUT_CALLOUT,
                  "noise": [[("… ", {}), ("$A_U$", dict(color=mars)), (" and ", {}), ("$A_T, b_T$", dict(color=mars)), (" steer the", {})],
                            [("reversal into ", {}), ("typical noise", EMPHASIS), (" …", {})]],
                  "policy": [[("… which the adapted flow maps to", {})], [("an action that ", {}), ("avoids the obstacle", EMPHASIS), (".", {})]]},
        flows=[("$\\hat z = \\hat u - \\int_0^1 A_U (A_T\\, v_P + b_T)\\, dt$",
                [("flow reversal", FLOW_NAME), (f"{MARS}-U: $A_T = I,\\ b_T = 0$", dict(color=mars_u))]),
               ("$\\hat a = \\hat z + \\int_0^1 (A_T\\, v_P + b_T)\\, dt$", [("flow matching", FLOW_NAME)])])


def method_legend(entries):
    """Legend handles and labels for the schematic's and the scenes' marks, by name, in the given order."""
    hatch = Rectangle((0, 0), 1, 1, facecolor=BAND_FILL, edgecolor=BAND_HATCH, lw=0, hatch="//////")
    disc = lambda face, edge, mew=0.8: plt.Line2D([], [], ls="none", marker="o", ms=6.5, mfc=face, mec=edge, mew=mew)
    marks = {"intent": (plt.Line2D([], [], color=INK, lw=0.9), "intent $u$"),
             "command": (plt.Line2D([], [], color=SERVED, lw=0.9), "command $\\hat u$"),
             "policy": (Rectangle((0, 0), 1, 1, facecolor=DENSITY_FILL, edgecolor=INK, lw=0.7), "policy density"),
             "intent_density": (Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=INK, lw=0.8, ls=INTENT_DASH), "intent density"),
             "readout": (Rectangle((0, 0), 1, 1, facecolor=READOUT_FILL, edgecolor=SERVED, lw=0.7), "readout density"),
             "frs": (plt.Line2D([], [], color=ARM_COLOR["frs"], lw=1.2, marker="o", mfc="white", ms=4.2), "FRS"),
             "mars": (plt.Line2D([], [], color=ARM_COLOR[MARS_KEY], lw=1.2, marker="o", ms=4.2), MARS),
             "start": (plt.Line2D([], [], ls="none", color=INK, marker="o", ms=4.2), "start"),
             "cursor": (plt.Line2D([], [], ls="none", color=INK, marker="o", ms=2.6), "cursor"),
             "target": (disc("#fbe3a1", INK, 1.0), "target"),
             "other": (disc("#ecebe6", "#c9c8c1"), "other targets"),
             "obstacle": ((hatch, disc(OBSTACLE_STYLE["facecolor"], OBSTACLE_STYLE["edgecolor"], 0.6)), "obstacle")}
    return [marks[e][0] for e in entries], [marks[e][1] for e in entries]


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


def build_fig1(out):
    paper_style()
    fig = plt.figure(figsize=(FULL_W, 2.62))
    # a: where corrupted inputs come from
    inputs_panel(fig, 0.1, 2.08)
    panel_label(fig, 0.005, 0.995, "a")
    # b: the method, a schematic with made-up densities
    x0 = 2.36
    schematic(fig, x0, 0.2, fig1_schematic())
    panel_label(fig, (x0 + 0.02) / FULL_W, 0.995, "b")
    handles, labels = method_legend(["policy", "intent_density", "readout"])
    fig.legend(handles, labels, loc="center", bbox_to_anchor=inches(fig, 5.68, 0.42), ncol=len(handles), fontsize=FS_LEGEND,
               handlelength=1.1, columnspacing=0.6, handletextpad=0.35, borderaxespad=0)
    png = save(fig, out, "fig1_method")
    write_stats(out, "fig1", dict(inputs=[label for label, _, _ in FIG1_INPUTS],
                                  schematic="panel a's reaches are illustrations; panel b's densities are made up"))
    return png


# ================================================================ Figure 2: the human study


def study_trials(results):
    runs = [r for r in find_runs(results, protocol="study") if r[0] != "sim"]
    trials, arms = load_trials(runs)
    return trials, [a for a in ARMS if a in arms]


def participant_stats(trials, arms):
    """Per participant, sessions pooled: n, success rate, outcome shares and the median time of the successes per arm."""
    by_p = {}
    for r in trials:
        by_p.setdefault(r["participant"], []).append(r)
    out = {}
    for p, rs in sorted(by_p.items(), key=lambda kv: int(kv[0][1:])):
        out[p] = {"sessions": sorted({r["run"] for r in rs}, key=lambda run: int(run.rsplit("_s", 1)[1]))}
        for arm in arms:
            a = [r for r in rs if r["arm"] == arm]
            times = [r["duration_s"] for r in a if r["outcome"] == "hit"]
            out[p][arm] = dict(n=len(a), **{o: float(np.mean([r["outcome"] == o for r in a])) for o in OUTCOMES},
                               time_s=float(np.median(times)) if times else None)
    return out


def example_trial(trials, participant="p1", arm=MARS_KEY):
    """The participant's success under the arm of median duration, and the step whose push/served angle is the median."""
    hits = sorted((r for r in trials if r["participant"] == participant and r["arm"] == arm and r["outcome"] == "hit"),
                  key=lambda r: r["duration_s"])
    r = hits[len(hits) // 2]
    raw, served = np.asarray(r["raw_commands"], float), np.asarray(r["served_commands"], float)
    moving = np.flatnonzero((np.linalg.norm(raw, axis=1) > 0.3) & (np.linalg.norm(served, axis=1) > 0.05))
    angles = np.array([abs(bearing(served[i], raw[i])) for i in moving])
    step = int(moving[np.argsort(angles)[len(angles) // 2]])
    return r, step


def draw_example(ax, r, step):
    targets = np.asarray(r["targets"], float)
    lay = Layouts(np.asarray(r["start"], float)[None], targets[None], np.ones((1, len(targets)), bool), np.array([r["goal"]]))
    draw_scene(ax, lay, TaskConfig(target_radius=r["target_radius"]))
    path = np.asarray(r["path"], float)
    ax.plot(path[:, 0], path[:, 1], color=ARM_COLOR[MARS_KEY], lw=1.1, zorder=3)
    p = path[step]
    arrow(ax, p, np.asarray(r["raw_commands"][step]), INK, "-", 0.16)
    arrow(ax, p, np.asarray(r["served_commands"][step]), SERVED, "-", 0.16)
    ax.plot(*p, "o", color=INK, ms=2.5, zorder=6)
    frame(ax)


def draw_aligned(ax, trials, arm, radius):
    """Every recorded trial of an arm in the reach's frame (start at the origin, goal at (1, 0)): successes in the arm's
    colour, the failures in the outcome ramp; the start and the goal marked as in the scenes."""
    for outcome, color, alpha, z in (("timeout", OUTCOME_GRAY["timeout"], 0.35, 2), ("wrong_target", OUTCOME_GRAY["wrong_target"], 0.3, 3),
                                     ("hit", ARM_COLOR[arm], 0.18, 4)):
        for r in (r for r in trials if r["arm"] == arm and r["outcome"] == outcome):
            p = aligned_path(r)[0]
            ax.plot(p[:, 0], p[:, 1], color=color, lw=0.35, alpha=alpha, zorder=z)
    ax.add_patch(plt.Circle((1, 0), radius, facecolor="#fbe3a1", edgecolor="none", zorder=1))
    ax.add_patch(plt.Circle((1, 0), radius, facecolor="none", edgecolor=INK, lw=1.0, zorder=7))
    ax.plot(0, 0, "o", color=INK, ms=3.5, zorder=6)
    ax.set(xlim=(-0.5, 1.55), ylim=(-0.85, 0.85), aspect="equal", xticks=[], yticks=[])
    frame(ax)


def build_fig2(out, results):
    paper_style()
    trials, arms = study_trials(results)
    if not trials:
        raise FileNotFoundError(f"no study sessions under {results}")
    stats = participant_stats(trials, arms)
    people = list(stats)
    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(FULL_W, 4.5))
    # (a) the task and the corruption: an example MARS success
    ax = fig.add_axes([0.03, 0.6, 0.19, 0.34])
    r, step = example_trial(trials)
    draw_example(ax, r, step)
    handles, labels = row_major(*method_legend(["intent", "command", "start", "cursor", "target", "other"]), 2)
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=FS_LEGEND, handlelength=1.3,
              columnspacing=0.9, handletextpad=0.5, borderaxespad=0, labelspacing=0.45)
    # (b) success per arm, the participant as the unit
    ax = fig.add_axes([0.325, 0.6, 0.27, 0.34])
    x = np.arange(len(arms))
    rates = {arm: [100 * stats[p][arm]["hit"] for p in people] for arm in arms}
    for i, p in enumerate(people):
        y = [rates[arm][i] for arm in arms]
        ax.plot(x, y, "-", color="#c9c8c1", lw=0.6, zorder=2)
        ax.scatter(x, y, s=11, color=[ARM_COLOR[a] for a in arms], edgecolor="white", lw=0.5, zorder=4)
    ci = {arm: t_ci(rates[arm]) for arm in arms}
    for i, arm in enumerate(arms):
        m, lo, hi = ci[arm]
        ax.add_patch(Rectangle((i - 0.3, 0), 0.6, m, facecolor=ARM_COLOR[arm], alpha=0.18, lw=0, zorder=1))
        ax.plot([i - 0.3, i + 0.3], [m, m], color=ARM_COLOR[arm], lw=2.2, zorder=5)
        ax.plot([i, i], [lo, hi], color=INK, lw=0.9, zorder=5)
        ax.text(i, 104, f"{m:.0f}", ha="center", va="bottom", fontsize=FS_TEXT, color=INK)
    ax.set(xticks=x, xticklabels=[], ylim=(0, 115), yticks=range(0, 101, 25), xlim=(-0.6, len(arms) - 0.4))
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Success (%)", labelpad=1)
    # (c) time to success: the share of trials succeeded by each time, mean of the participants' curves
    ax = fig.add_axes([0.675, 0.6, 0.31, 0.34])
    limit = trials[0]["time_limit_s"]
    t = np.arange(0, limit + 1e-9, 0.02)
    curves = {}
    for arm in arms:
        cs = [cumulative_hits([r for r in trials if r["participant"] == p and r["arm"] == arm], t) for p in people]
        curves[arm] = 100 * np.mean(cs, 0)
        ax.step(t, curves[arm], where="post", color=ARM_COLOR[arm], lw=1.4, zorder=3 + (arm == MARS_KEY))
    ax.set(xlim=(0, limit), ylim=(0, 100), xticks=range(0, int(limit) + 1), yticks=range(0, 101, 25))
    ax.set_xlabel("Time since trial start (s)", labelpad=1)
    ax.set_ylabel("Trials successful by then (%)", labelpad=1)
    # the arms' legend, shared by every panel, between the rows
    handles = [Rectangle((0, 0), 1, 1, facecolor=ARM_COLOR[a], lw=0) for a in arms]
    fig.legend(handles, [ARM_LABEL[a] for a in arms], loc="center", bbox_to_anchor=(0.5, 0.505), ncol=4, fontsize=FS_LEGEND,
               handlelength=1.2, handleheight=0.9, columnspacing=1.6, handletextpad=0.6, borderaxespad=0)
    # (d) every recorded trial per arm, in the reach's frame
    radius = float(np.median([aligned_path(r)[2] for r in trials]))
    for i, arm in enumerate(arms):
        ax = fig.add_axes([0.03 + i * 0.1675, 0.09, 0.15, 0.36])
        draw_aligned(ax, trials, arm, radius)
        ax.set_title(ARM_LABEL[arm], fontsize=FS_TITLE, pad=3)
        if i == 0:
            ax.set_xlabel("progress", fontsize=FS_LABEL, labelpad=2)
            ax.set_ylabel("deviation", fontsize=FS_LABEL, labelpad=2)
    # (e) outcomes: the mean of the participants' shares; successes in the arm's colour, the failures in the outcome ramp
    ax = fig.add_axes([0.80, 0.14, 0.185, 0.31])
    shares = {arm: {o: 100 * np.mean([stats[p][arm][o] for p in people]) for o in OUTCOMES} for arm in arms}
    for i, arm in enumerate(arms):
        y0 = 0
        for o in OUTCOMES:
            v = shares[arm][o]
            color = ARM_COLOR[arm] if o == "hit" else OUTCOME_GRAY[o]
            ax.add_patch(Rectangle((i - 0.32, y0), 0.64, max(v - 0.8, 0), facecolor=color, lw=0, zorder=2))
            if v >= 9:
                ax.text(i, y0 + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=FS_TEXT, color=text_on(color))
            y0 += v
    ax.set(xticks=x, xticklabels=[], ylim=(0, 100), yticks=range(0, 101, 25), xlim=(-0.6, len(arms) - 0.4))
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Trials (%)", labelpad=1)
    success = tuple(Rectangle((0, 0), 1, 1, facecolor=ARM_COLOR[a]) for a in arms)   # one swatch of the arms' colours
    ax.legend([success] + [Rectangle((0, 0), 1, 1, facecolor=OUTCOME_GRAY[o]) for o in OUTCOMES[1:]],
              [OUTCOME_LABEL[o] for o in OUTCOMES], handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
              loc="upper right", bbox_to_anchor=(1.03, -0.1), ncol=3, fontsize=FS_LEGEND, handlelength=1.0, handleheight=1.0,
              columnspacing=0.6, handletextpad=0.35, borderaxespad=0)
    for xx, yy, letter in ((0.005, 0.995, "a"), (0.265, 0.995, "b"), (0.615, 0.995, "c"), (0.005, 0.485, "d"), (0.745, 0.485, "e")):
        panel_label(fig, xx, yy, letter)
    png = save(fig, out, "fig2_study")
    # the numbers for the caption
    leads = {}
    for arm in arms:
        if arm == MARS_KEY:
            continue
        per_p = [rates[MARS_KEY][i] - rates[arm][i] for i in range(len(people))]
        leads[arm] = dict(mean_ci=t_ci(per_p), positive=int(sum(v > 0 for v in per_p)), of=len(people),
                          trial_level=[100 * v for v in paired_lead(trials, MARS_KEY, arm, rng)])
    best = {arm: 0 for arm in arms}
    for i in range(len(people)):
        top = max(arms, key=lambda a: rates[a][i])
        best[top] += 1
    sim = {}
    sim_dir = Path(results) / "sim" / "study_s1"
    if sim_dir.exists():
        _, recs = load_run(sim_dir)
        for arm in arms:
            a = [r for r in recs if r["arm"] == arm and not r["practice"]]
            sim[arm] = 100 * float(np.mean([r["outcome"] == "hit" for r in a])) if a else None
    write_stats(out, "fig2", dict(
        participants=len(people), sessions={p: stats[p]["sessions"] for p in people},
        trials_per_arm={arm: int(sum(stats[p][arm]["n"] for p in people)) for arm in arms},
        success_mean_ci={arm: ci[arm] for arm in arms}, success_per_participant=rates,
        success_pooled_over_trials={arm: 100 * float(np.mean([r["outcome"] == "hit" for r in trials if r["arm"] == arm])) for arm in arms},
        leads=leads, best_arm_for_k_participants=best,
        median_success_time_s={arm: float(np.median([stats[p][arm]["time_s"] for p in people if stats[p][arm]["time_s"] is not None])) for arm in arms},
        success_by_s={arm: {f"{s:g}": round(float(np.interp(s, t, curves[arm])), 1) for s in (2, 3, 4)} for arm in arms},
        outcome_shares=shares, expert_reference_success=sim,
        condition=dict(targets=trials[0]["targets_range"], corruption=trials[0]["corruption"], input_noise=trials[0]["input_noise"],
                       reach=trials[0]["reach"], time_limit_s=trials[0]["time_limit_s"]),
        flow_control_note="Kao et al.'s flow control with every chunk step written (tau 16, lambda 1) and the command in the policy's "
                          "action units (ic_gain 2.08), executed 2 of 16 steps every 40 ms",
        example_trial=dict(participant=r["participant"], run=r["run"], arm=r["arm"], trial=r["trial"], duration_s=r["duration_s"],
                           arrow_step=step), durations_include_dwell_s=trials[0]["dwell_s"], aligned_goal_radius=radius))
    return png


# ================================================================ Figure 3: the simulated grid


TASK_SEEDS = (11, 5, 8, 2, 7)  # one sampled layout per task for the illustration (as in reports/build_grid_report.py)
SEED_RUNS = ("grid_s0",) + tuple(f"grid_s{k}/combined" for k in (1, 2, 3, 4))  # the grid's runs under results/sim/sweeps,
# one per training seed: 0 is the grid of record; 1-4 retrain every module on its own draw of the demonstrations and rerun
# the module methods on the grid's trials (the baselines train nothing, so every seed has the grid's)
GRID_BASELINES = ("teleop", "flow_control", "frs")
N_BOOT = 4000  # bootstrap draws for the 95% CIs over the seeds, then the trials
ROLLOUT_CELL, ROLLOUT_TRIALS, ROLLOUT_SEED = "U1xT4", 40, 7  # the trials split evenly over the seeds' modules
CORNER = 0.42  # the second method's corner in panel c's lead matrix, as a share of the cell's side
COMPARED = [("retrained_mars_user", f"Retrained policy + {MARS}-U"), ("mars", MARS)]
GRID_OUTCOMES = ("hit", "collision", "wrong_target", "timeout")
METRIC_KEY = {"hit": "success", "collision": "collision", "wrong_target": "wrong", "timeout": "timeout"}  # a cell's metrics
FIG3_H = 8.4  # in: a (the schematic) and b (the tasks) side by side on top, rows c-f below them (positions in inches from the bottom)


def grid_data(results):
    """The grid's runs, one per training seed found (seed 0 is required; D["cell"](u, t, k) is run k's cell, the seed
    D["seeds"][k]). Every run is on the grid's trials, so methods compare trial for trial within a seed and a cell's
    numbers average over the seeds."""
    sweeps = Path(results) / "sim" / "sweeps"
    seeds = [k for k, run in enumerate(SEED_RUNS) if k == 0 or (sweeps / run / "test.json").exists()]
    runs = [json.loads((sweeps / SEED_RUNS[k] / "test.json").read_text()) for k in seeds]
    grid = runs[0]
    for k, run in zip(seeds[1:], runs[1:]):
        if run["test"].keys() != grid["test"].keys() or any(c["trials"]["side"] != grid["test"][key]["trials"]["side"]
                                                             for key, c in run["test"].items()):
            raise ValueError(f"seed {k}'s run ({SEED_RUNS[k]}) is not on the grid's trials")
    G = grid["args"]["sweep"]["grid"]
    users, tasks, noise = G["users"], G["tasks"], G["noises"][0]
    train = yaml.safe_load((config.CONFIG_DIR / "sweeps" / "grid.yaml").read_text())["grid"]["train"]
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


def ranked(D, u, t):
    """The methods drawn in a cell, best first by their success averaged over the seeds (on T1 the four that differ)."""
    shown = [m for m in D["methods"] if t != D["tasks"][0] or m in T1_METHODS]
    return sorted(shown, key=lambda m: -seed_mean(D, u, t, m))


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
    """Run k's arms in a cell (its sweep names that seed's modules) at the depths that run tuned."""
    u, t = cell.split("x")
    return {m: with_depth(method_arm(D["runs"][k]["args"]["sweep"], m, cell), D["cell"](u, t, k)["depths"][m]) for m in methods}


def lead_cell(ax, x, y, best, second, level, text, size=1.0):
    """A cell of panel c's lead matrix centred at (x, y), the y axis running down: the best method's colour, the second's
    in the lower right corner (half the cell when the two are level), the lead (in the best one's half when level)."""
    h = size / 2
    ax.add_patch(Rectangle((x - h, y - h), size, size, facecolor=best, edgecolor="white", lw=0.8, zorder=1))
    s = size if level else CORNER * size
    ax.add_patch(Polygon([(x + h, y + h), (x + h, y + h - s), (x + h - s, y + h)], closed=True, facecolor=second,
                         edgecolor="white", lw=0.6, zorder=2))
    off, fontsize = (0.2, FS_TEXT) if level else (0.04, FS_TEXT)
    ax.text(x - off * size, y - off * size, text, ha="center", va="center", fontsize=fontsize, color=text_on(best), zorder=3)


def draw_training(fig, D, rect):
    """How the modules were trained, as a users x tasks matrix (the user modules on the open field through each
    user's channel, the task modules on one task each with one user's module in the loop, the other cells
    compositions never trained together), and beside it, over the seeds, the best method in every cell, the second
    and the best one's lead over it."""
    x0, y0, w, h = rect
    users, tasks = D["users"], D["tasks"]
    nU, nT = len(users), len(tasks)
    t0 = tasks[0]

    def matrix(x, labels):
        ax = fig.add_axes([x, y0, 0.19 * w, h])
        ax.set(xlim=(-0.5, nT - 0.5), ylim=(nU - 0.5, -0.5), xticks=range(nT), yticks=range(nU), aspect="equal")
        ax.xaxis.tick_top()
        ax.set_xticklabels(tasks, fontsize=FS_TICK)
        ax.set_yticklabels(labels, fontsize=FS_TICK)
        ax.tick_params(length=0, pad=2)
        for sp in ax.spines.values():
            sp.set_visible(False)
        return ax

    styles = {"U": "#dbe6f5", "T": "#e5dff7", "": "white"}
    ax = matrix(x0, [f"{u}  {corruption_label(D['G']['cells'][u + 'x' + t0]['corruption'])}" for u in users])
    for i, u in enumerate(users):
        for j, t in enumerate(tasks):
            kind = "U" if t == t0 else "T" if (u, t) in D["train"] else ""
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=styles[kind], edgecolor=GRID, lw=0.6, zorder=1))
            if kind:
                ax.text(j, i, kind, ha="center", va="center", fontsize=FS_TEXT, color=INK, zorder=2)
    lines = [("U", "User module training"), ("T", "Task module training"), ("", "Composed at test time")]
    xt = x0 + 0.235 * w
    for k, (kind, text) in enumerate(lines):
        yy = y0 + h * (0.84 - 0.36 * k)
        sw, sh = inches(fig, 0.112, 0.172)
        fig.patches.append(Rectangle((xt, yy - sh / 2), sw, sh, transform=fig.transFigure, facecolor=styles[kind], edgecolor=GRID, lw=0.6))
        fig.text(xt + 0.008, yy, kind, ha="center", va="center", fontsize=FS_TEXT, color=INK)
        fig.text(xt + 0.026, yy, text, ha="left", va="center", fontsize=FS_LEGEND, color=INK, linespacing=1.2)
    # the best method per cell by its success over the seeds (fill), the second (the corner; half the cell when the two
    # are level, the 95% CI of the lead holding 0) and the lead in points
    ax = matrix(x0 + 0.56 * w, users)
    winners = {}
    for i, u in enumerate(users):
        for j, t in enumerate(tasks):
            best, second = ranked(D, u, t)[:2]
            lead = lead_ci(D, u, t, best, second)
            level = lead[1] <= 0
            winners[f"{u}x{t}"] = dict(method=best, success=round(seed_mean(D, u, t, best), 1), second=second,
                                       second_success=round(seed_mean(D, u, t, second), 1), lead=[round(v, 1) for v in lead],
                                       level=level)
            lead_cell(ax, j, i, GRID_COLOR[best], GRID_COLOR[second], level, f"+{lead[0]:.0f}")
    ax.set_title("Best method (fill), second (corner), lead (points)", fontsize=FS_TITLE, pad=12, loc="center")
    return winners


def task_rollouts(D, device, n=8, seed=ROLLOUT_SEED):
    """One sampled layout per task and a successful MARS trial on it: the task's trained cell (T1 with the first user),
    seed 0's modules at their tuned depth, the first success of n trials with the grid's channel noise."""
    out = {}
    for t, layout_seed in zip(D["tasks"], TASK_SEEDS):
        u = next((u for u in D["users"] if (u, t) in D["train"] and t != D["tasks"][0]), D["users"][0])
        arms = cell_arms(D, f"{u}x{t}", ["mars"])
        policies, adapters = load_checkpoints(arms, device)
        arm = arms["mars"]
        task = arm_task(arm)
        layouts = sample_layouts(np.random.default_rng(layout_seed), 1, (4, 4), task)
        torch.manual_seed(seed)
        recs = run_trials(arm, layouts.tile(n), ExpertAgent(), channel_for(arm, np.random.default_rng(seed + 1), D["noise"]),
                          policies[arm["model"]], adapters, device)
        hits = [i for i, r in enumerate(recs) if r["outcome"] == "hit"]
        out[t] = dict(cell=f"{u}x{t}", layouts=layouts, task=task, path=np.asarray(recs[hits[0]]["path"], float) if hits else None,
                      trial=hits[0] if hits else None, hits=len(hits), of=n)
    return out


def draw_tasks(fig, D, shown, rect, cols=2, gap=0.08):
    """Each task's layout with its successful MARS trial (task_rollouts): square scenes, `cols` to a row, each row centred in
    rect (the last one too), gap in inches."""
    x0, y0, w, h = rect[0] * fig.get_figwidth(), rect[1] * fig.get_figheight(), rect[2] * fig.get_figwidth(), rect[3] * fig.get_figheight()
    n = len(D["tasks"])
    rows = -(-n // cols)
    side = min((w - (cols - 1) * gap) / cols, (h - (rows - 1) * gap) / rows)
    for i, t in enumerate(D["tasks"]):
        r, c = divmod(i, cols)
        in_row = min(cols, n - r * cols)
        left = x0 + (w - in_row * side - (in_row - 1) * gap) / 2
        ax = fig.add_axes([*inches(fig, left + c * (side + gap), y0 + h - (r + 1) * side - r * gap), *inches(fig, side, side)])
        s = shown[t]
        draw_scene(ax, s["layouts"], s["task"], obstacle_style=OBSTACLE_STYLE)
        if s["path"] is not None:
            ax.plot(s["path"][:, 0], s["path"][:, 1], color=GRID_COLOR["mars"], lw=1.1, zorder=3)
        ax.text(0.03, 0.97, t, transform=ax.transAxes, fontsize=FS_TEXT, color=INK, ha="left", va="top", zorder=5,
                bbox=dict(boxstyle="square,pad=0.12", facecolor="white", edgecolor="none"))   # over a disc that sits in the corner
        frame(ax)


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
    for i, t in enumerate(tasks):
        shown = [m for m in methods if t != "T1" or m in T1_METHODS]
        left = i - width * len(shown) / 2
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
            if m == "mars":
                ax.text(xc, max(max(vals), min(hi, 100)) + 2.5, f"{mean:.0f}", ha="center", va="bottom", fontsize=FS_TEXT, color=INK, weight="semibold")
    ax.set_xticklabels([f"{t}" for t in tasks], fontsize=FS_TICK)
    return means


def rollouts(D, device, cell=ROLLOUT_CELL, n=ROLLOUT_TRIALS, seed=ROLLOUT_SEED):
    """n trials of the two compared arms on ONE layout of a cell (the same scene), split evenly over the seeds: each
    seed's modules at the depths its run tuned, with that seed's own channel-noise draws (the same for both arms). The
    paths show where each arm goes and where the retrained policy's trials end."""
    per = n // len(D["runs"])
    records, depths, per_seed = {m: [] for m, _ in COMPARED}, {}, {m: {} for m, _ in COMPARED}
    for k, s in enumerate(D["seeds"]):
        arms = cell_arms(D, cell, [m for m, _ in COMPARED], k)
        policies, adapters = load_checkpoints(arms, device)
        task = arm_task(next(iter(arms.values())))
        layouts = sample_layouts(np.random.default_rng(seed), 1, (4, 4), task).tile(per)
        for m, arm in arms.items():
            torch.manual_seed(seed + 10 * s)
            channel = channel_for(arm, np.random.default_rng(seed + 10 * s + 1), D["noise"])
            recs = run_trials(arm, layouts, ExpertAgent(), channel, policies[arm["model"]], adapters, device)
            records[m] += recs
            per_seed[m][f"s{s}"] = {o: int(sum(r["outcome"] == o for r in recs)) for o in GRID_OUTCOMES}
        depths[f"s{s}"] = {m: arm["n_reverse_steps"] for m, arm in arms.items()}
    return dict(cell=cell, layouts=layouts, task=task, records=records, depths=depths, per_seed=per, outcomes_per_seed=per_seed)


def draw_rollouts(fig, D, roll, rect):
    """The compared arms on the same scene: successes in the arm's colour, collisions in ink ending at an x."""
    x0, y0, w, h = rect
    pw = 0.46 * w
    counts = {}
    for i, (m, label) in enumerate(COMPARED):
        ax = fig.add_axes([x0 + i * (w - pw), y0, pw, h])
        draw_scene(ax, roll["layouts"][0:1], roll["task"], obstacle_style=OBSTACLE_STYLE)
        recs = roll["records"][m]
        for outcome, color, alpha, z in (("timeout", OUTCOME_GRAY["timeout"], 0.9, 2), ("wrong_target", OUTCOME_GRAY["wrong_target"], 0.9, 2),
                                         ("hit", GRID_COLOR[m], 0.55, 3),
                                         ("collision", INK, 0.6, 4)):
            for r in (r for r in recs if r["outcome"] == outcome):
                p = np.asarray(r["path"], float)
                ax.plot(p[:, 0], p[:, 1], color=color, lw=0.6, alpha=alpha, zorder=z)
                if outcome == "collision":
                    ax.plot(*p[-1], "x", color=INK, ms=3, mew=0.8, zorder=5)
        counts[m] = {o: int(sum(r["outcome"] == o for r in recs)) for o in GRID_OUTCOMES}
        frame(ax)
    return counts


def draw_outcomes(fig, D, rect):
    """How the compared arms' trials end on each obstacle task (mean over the users and the seeds): successes in the arm's
    colour, the failures in the outcome ramp."""
    ax = fig.add_axes(rect)
    users, tasks = D["users"], D["obstacle_tasks"]
    bw = 0.36
    shares = {}
    for i, t in enumerate(tasks):
        for k, (m, _) in enumerate(COMPARED):
            s = {o: float(np.mean([seed_mean(D, u, t, m, METRIC_KEY[o]) for u in users])) for o in GRID_OUTCOMES}
            shares[f"{t}|{m}"] = s
            xc, y0 = i + (k - 0.5) * (bw + 0.04), 0
            for o in GRID_OUTCOMES:
                v = s[o]
                ax.add_patch(Rectangle((xc - bw / 2, y0), bw, max(v - 0.8, 0), facecolor=GRID_COLOR[m] if o == "hit" else OUTCOME_GRAY[o], lw=0, zorder=2))
                if v >= 12 and o in ("hit", "collision"):
                    ax.text(xc, y0 + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=FS_TEXT, color="white")
                y0 += v
    ax.set(xticks=range(len(tasks)), xticklabels=tasks, ylim=(0, 100), yticks=range(0, 101, 25), xlim=(-0.6, len(tasks) - 0.4))
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Trials (%)", labelpad=1)
    return shares


def build_fig3(out, results, device):
    paper_style()
    D = grid_data(results)
    roll = rollouts(D, device)
    fig = plt.figure(figsize=(FULL_W, FIG3_H))
    y = lambda inch: inch / FIG3_H
    # a: the method with the task module, a schematic with made-up densities (as Figure 1b)
    schematic(fig, 0.05, 6.05, fig3_schematic())
    # b: the tasks, two to a row, beside it
    shown = task_rollouts(D, device)
    draw_tasks(fig, D, shown, (5.12 / FULL_W, y(5.9), 1.85 / FULL_W, y(2.35)))
    # the marks of a and b, one legend as Figure 1's
    handles, labels = method_legend(["policy", "intent_density", "readout", "obstacle", "start", "target", "other"])
    fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.5, y(5.62)), ncol=len(handles), fontsize=FS_LEGEND, handlelength=1.7, columnspacing=1.1,
               handletextpad=0.5, borderaxespad=0, labelspacing=0.4, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)})
    # c: how the modules were trained, and who wins where
    winners = draw_training(fig, D, (0.193, y(4.25), 0.85, y(0.9)))
    # the methods' legend, shared by c and d, in one line
    handles = [plt.Line2D([], [], color=GRID_COLOR[m], lw=4, solid_capstyle="butt") for m in D["methods"]]
    fig.legend(handles, [GRID_LABEL[m] for m in D["methods"]], loc="center", bbox_to_anchor=(0.5, y(3.92)), ncol=len(handles), fontsize=FS_LEGEND,
               handlelength=1.1, columnspacing=1.0, handletextpad=0.5, borderaxespad=0)
    # d: every method on every task
    task_cis = draw_grid_bars(fig, D, (0.065, y(2.3), 0.925, y(1.37)))
    # e, f: the retrained policy against MARS
    # e's two square scenes and f's plot share a bottom and a height; the row (e's scenes to f's right edge) is centred
    side, bottom, f_w, between = 1.3, 0.42, 2.6, 0.9         # in; between: e's right edge to f's axis (f's ticks and label)
    e_w = side / 0.46                                          # draw_rollouts gives each scene 0.46 of its width
    ex = (FULL_W - (e_w + between + f_w)) / 2
    fx = ex + e_w + between
    counts = draw_rollouts(fig, D, roll, (ex / FULL_W, y(bottom), e_w / FULL_W, y(side)))
    shares = draw_outcomes(fig, D, (fx / FULL_W, y(bottom), f_w / FULL_W, y(side)))
    # the outcomes' legend, shared by e and f, centred under both (as Fig 2e's)
    success = tuple(Rectangle((0, 0), 1, 1, facecolor=GRID_COLOR[m]) for m, _ in COMPARED)   # one swatch of the arms' colours
    fig.legend([success] + [Rectangle((0, 0), 1, 1, facecolor=OUTCOME_GRAY[o]) for o in GRID_OUTCOMES[1:]],
               [OUTCOME_LABEL[o] for o in GRID_OUTCOMES], handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
               loc="center", bbox_to_anchor=(0.5, y(0.1)), ncol=len(GRID_OUTCOMES), fontsize=FS_LEGEND, handlelength=1.0,
               handleheight=1.0, columnspacing=1.0, handletextpad=0.4, borderaxespad=0)
    for x, yy, letter in ((0.005, 0.995, "a"), (5.1 / FULL_W, 0.995, "b"), (0.005, y(5.3), "c"), (0.005, y(3.82), "d"), (0.005, y(1.83), "e"),
                          ((fx - 0.55) / FULL_W, y(1.83), "f")):
        panel_label(fig, x, yy, letter)
    png = save(fig, out, "fig3_grid")
    users, tasks, methods, seeds = D["users"], D["tasks"], D["methods"], D["seeds"]
    cells = [(u, t) for u in users for t in tasks]
    obstacle = [(u, t) for u in users for t in D["obstacle_tasks"]]
    r1 = lambda v: [round(x, 1) for x in v]
    per_cell = {f"{u}x{t}": {m: round(seed_mean(D, u, t, m), 1) for m in methods} for u, t in cells}
    best_base = {f"{u}x{t}": max(GRID_BASELINES, key=lambda b: seed_mean(D, u, t, b)) for u, t in obstacle}
    lead_best = {f"{u}x{t}": r1(lead_ci(D, u, t, "mars", best_base[f"{u}x{t}"])) for u, t in obstacle}
    over_retrained = {f"{u}x{t}": r1(lead_ci(D, u, t, "mars", "retrained_mars_user")) for u, t in obstacle}
    gain = {}
    for u, t in obstacle:
        g = lead_ci(D, u, t, "mars", "mars_user")
        gain[f"{u}x{t}"] = dict(points=round(g[0], 1), ci=r1(g[1:]), trained=(u, t) in D["train"])
    obstacle_per_seed = {m: [float(np.mean([100 * D["cell"](u, t, k)["metrics"][m]["success"] for u, t in obstacle]))
                             for k in range(len(seeds))] for m in methods}
    beyond = lambda leads: dict(cells=int(sum(v[1] > 0 for v in leads.values())), of=len(leads),
                                level=[c for c, v in leads.items() if v[1] <= 0])
    write_stats(out, "fig3", dict(
        names={"MARS": "user + task modules (mars)", "MARS-U": "user module alone (mars_user)", "MARS-T": "task module alone (mars_task)"},
        seeds=seeds, seed_runs=[f"results/sim/sweeps/{SEED_RUNS[k]}" for k in seeds],
        seed_note="every number is over the seeds: seed 0 is the grid of record; seeds 1-4 retrain every module (user modules "
                  "from 100 demonstrations, the retrained arm's user modules, task modules from 300) on their own draws and rerun "
                  "the module methods on the grid's trials; the retrained policies are one seed, and the baselines train nothing, "
                  "so theirs are the grid's in every seed",
        ci_note=f"panel d: 95% t CIs over the four users (each user's success averaged over the seeds), the user the unit as "
                f"the participant in Figure 2b (the drawn lines stop at the axis); panel c and the leads: 95% CIs resampling the "
                f"seeds, then the trials within each ({N_BOOT} draws)",
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
        best_per_cell=winners,
        best_per_cell_note="fill: the best method by its success averaged over the seeds; corner: the second; lead: the best's "
                           "success minus the second's (points) with its 95% CI; level (the cell split between the two): the CI "
                           "includes 0",
        composed_beats_all_baselines=beyond(lead_best), composed_lead_over_best_baseline=lead_best, best_baseline=best_base,
        composed_lead_range=[min(v[0] for v in lead_best.values()), max(v[0] for v in lead_best.values())],
        task_module_gain=gain, task_module_gain_range=dict(trained=[g["points"] for g in gain.values() if g["trained"]],
                                                           unseen=[g["points"] for g in gain.values() if not g["trained"]]),
        composed_over_retrained=over_retrained, composed_beats_retrained=beyond(over_retrained),
        t1_user_module_over_teleop={f"{u}x{tasks[0]}": r1(lead_ci(D, u, tasks[0], "mars", "teleop")) for u in users},
        outcome_shares_compared=shares,
        tuned_depths={f"{u}x{t}": {f"s{s}": D["cell"](u, t, k)["depths"] for k, s in enumerate(seeds)} for u, t in cells},
        schematic="panel a's densities are made up",
        task_rollouts={t: {k: v[k] for k in ("cell", "trial", "hits", "of")} for t, v in shown.items()},
        rollouts=dict(cell=roll["cell"], trials=roll["per_seed"] * len(seeds), per_seed=roll["per_seed"], seed=ROLLOUT_SEED,
                      depths=roll["depths"], outcomes=counts, outcomes_per_seed=roll["outcomes_per_seed"]),
        t1_note="on T1 MARS-U is MARS, MARS-T is FRS and the retrained arm is MARS (no task module, nothing to retrain), so only four arms are drawn",
        flow_control_note="Kao et al.'s flow control, ic_gain 2.0, tau tuned per cell"))
    return png


# ================================================================


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", choices=["fig1", "fig2", "fig3"], default=["fig1", "fig2", "fig3"])
    p.add_argument("--out", default=str(FIG_DIR), help="output folder (default: figures/)")
    p.add_argument("--results", default=str(RESULTS))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    out, results = Path(args.out), Path(args.results)
    builders = {"fig1": lambda: build_fig1(out), "fig2": lambda: build_fig2(out, results),
                "fig3": lambda: build_fig3(out, results, args.device)}
    for name in args.only:
        print(f"{name} -> {builders[name]()}", flush=True)


if __name__ == "__main__":
    main()
