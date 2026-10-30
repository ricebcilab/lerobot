"""PDF report of the user study, with conference-style figures: one participant, or the group.

    python reports/build_study_report.py --participants p1
    python reports/build_study_report.py --participants p1 p2 p3 p4

Reads every session of each participant under results/<participant>/<protocol>_s<k>/ and, when
present, the expert agent's test result for the same condition from
results/sim/sweeps/task_space_refine/test.json. One participant gives reports/study_<participant>.pdf;
several give reports/study_group.pdf, where the unit of analysis is the participant (each one's
sessions pooled first). Figures go to results/analysis/study_<participant|group>/ as PDF and PNG.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image as PILImage  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import letter  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import inch  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, Paragraph, SimpleDocTemplate,  # noqa: E402
                                Spacer, Table, TableStyle)

from analyze_results import find_runs, load_trials  # noqa: E402
from cursor_sa.records import ARM_ALIASES
from cursor_sa.analysis import (GRID, INK, INK_2, aligned_path, bootstrap_mean, cumulative_hits, paired_lead,  # noqa: E402
                                path_efficiency, style)
from cursor_sa.paths import REPORTS, RESULTS  # noqa: E402

# Arms in report order (baselines first, the focus last), their labels and colors: the project's
# method colors, the same in every figure of the project (teleop is the gray baseline).
ARM_ORDER = ["teleop", "flow_control", "frs", "mars"]
ARM_LABEL = {"teleop": "Teleop", "flow_control": "Flow control", "frs": "FRS", "mars": "MARS"}
ARM_COLOR = {"teleop": "#008300", "flow_control": "#e87ba4", "frs": "#e69f00", "mars": "#2f45c4"}  # the figures' palette
OUTCOME_COLOR = {"hit": "#0ca30c", "wrong_target": "#6250d6", "collision": "#1f1f1e", "timeout": "#898781"}  # CVD-distinct
OUTCOME_LABEL = {"hit": "Hit", "wrong_target": "Wrong target", "collision": "Collision", "timeout": "Timeout"}
OUTCOMES = ("hit", "wrong_target", "collision", "timeout")
REFERENCE = RESULTS / "sim" / "sweeps" / "task_space_refine" / "test.json"  # the expert agent, same condition
FULL_W, HALF_W = 7.2, 3.5  # figure widths in inches (the page's text width, and a column)

FONTS = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DejaVu", FONTS / "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", FONTS / "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DejaVu-Oblique", FONTS / "DejaVuSans-Oblique.ttf"))
pdfmetrics.registerFont(TTFont("DejaVu-BoldOblique", FONTS / "DejaVuSans-BoldOblique.ttf"))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu-Oblique",
                              boldItalic="DejaVu-BoldOblique")
ss = getSampleStyleSheet()
TITLE = ParagraphStyle("title", parent=ss["Title"], fontName="DejaVu-Bold", fontSize=16, leading=20, spaceAfter=2)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="DejaVu-Bold", fontSize=11.5, spaceBefore=8, spaceAfter=3)
BODY = ParagraphStyle("body", parent=ss["BodyText"], fontName="DejaVu", fontSize=9, leading=12.5)
CAP = ParagraphStyle("cap", parent=BODY, fontSize=7.5, leading=9.5, textColor=colors.HexColor("#444444"))
PAGE_W = letter[0] - 1.4 * inch
fig_no, tab_no = [0], [0]


# ---------------------------------------------------------------- data


def sessions_of(trials):
    """Run names in session order."""
    return sorted({r["run"] for r in trials}, key=lambda run: int(run.rsplit("_s", 1)[1]))


def reference_success(trials, arms):
    """The expert agent's test success per arm in this condition, from the refine sweep (or {})."""
    if not REFERENCE.exists() or not trials:
        return {}
    a = trials[0]
    fmt = lambda v: "-".join(str(x) for x in v)
    key = f"0.25|{float(a['input_noise'])}|targets={fmt(a['targets_range'])}|reach={fmt(a['reach'])}|dwell_s={a['dwell_s']}"
    cell = json.loads(REFERENCE.read_text())["test"].get(key)
    return {arm: cell["metrics"][arm]["success"] for arm in arms if cell and arm in cell["metrics"]}


def arm_stats(trials, arms, focus, rng):
    """Per arm: n, success [mean, lo, hi], wrong, timeout, median time of hits, median path efficiency,
    and the focus arm's paired lead over it."""
    out = {}
    for arm in arms:
        rs = [r for r in trials if r["arm"] == arm]
        hits = [r["outcome"] == "hit" for r in rs]
        times = [r["duration_s"] for r in rs if r["outcome"] == "hit"]
        eff = [e for e in (path_efficiency(r) for r in rs) if e is not None]
        out[arm] = dict(n=len(rs), success=bootstrap_mean(hits, rng), wrong=np.mean([r["outcome"] == "wrong_target" for r in rs]),
                        collision=np.mean([r["outcome"] == "collision" for r in rs]),
                        timeout=np.mean([r["outcome"] == "timeout" for r in rs]), time_s=float(np.median(times)) if times else None,
                        efficiency=float(np.median(eff)) if eff else None,
                        lead=paired_lead(trials, focus, arm, rng) if arm != focus else None)
    return out


def session_table(trials, arms):
    """Hit rate per session x arm."""
    rows = []
    for run in sessions_of(trials):
        rows.append([run.rsplit("_s", 1)[1]] + [np.mean([r["outcome"] == "hit" for r in trials if r["run"] == run and r["arm"] == arm])
                                                for arm in arms])
    return rows


def mean_sem(values):
    v = np.asarray(values, float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")


# ---------------------------------------------------------------- figures


def paper_style():
    style()
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.grid.axis": "y",
                         "grid.linewidth": 0.6, "axes.linewidth": 0.6, "savefig.dpi": 300, "pdf.fonttype": 42})


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png")
    plt.close(fig)
    return out / f"{name}.png"


def outcome_bars(ax, stats, arms):
    """How the trials ended, one horizontal 100% bar per arm."""
    outcomes = [o for o in OUTCOMES if o == "hit" or any(stats[a][{"wrong_target": "wrong"}.get(o, o)] > 0 for a in arms)]
    for j, arm in enumerate(arms):
        left = 0
        parts = [100 * (stats[arm]["success"][0] if o == "hit" else stats[arm][{"wrong_target": "wrong"}.get(o, o)]) for o in outcomes]
        for o, v in zip(outcomes, parts):
            ax.barh(len(arms) - 1 - j, v, left=left, height=0.56, color=OUTCOME_COLOR[o], edgecolor="white", lw=1.2,
                    zorder=3, label=OUTCOME_LABEL[o] if j == 0 else None)
            if v >= 9:
                ax.text(left + v / 2, len(arms) - 1 - j, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white")
            left += v
    ax.set(yticks=range(len(arms)), yticklabels=[ARM_LABEL[m] for m in arms[::-1]], xlim=(0, 100), xlabel="Trials (%)")
    ax.grid(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(outcomes), handlelength=1.0, columnspacing=1.2, borderaxespad=0)


def spread(values, min_gap):
    """Label positions at least min_gap apart, in the order of the values."""
    order = np.argsort(values)
    out = np.array(values, float)
    for a, b in zip(order[:-1], order[1:]):
        out[b] = max(out[b], out[a] + min_gap)
    return out


def reference_markers(ax, arms, reference):
    if reference:
        ax.plot([i for i, arm in enumerate(arms) if arm in reference], [100 * reference[arm] for arm in arms if arm in reference],
                "D", ms=4.5, mfc="white", mec=INK, mew=0.9, ls="none", zorder=6, label="expert agent (simulation)")


def fig_outcomes(stats, arms, reference, out):
    """(a) hit rate with its 95% CI, the expert agent as a hollow marker; (b) the three outcomes."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_W, 2.5), gridspec_kw=dict(width_ratios=[1, 1.25], wspace=0.35))
    for i, arm in enumerate(arms):
        m, lo, hi = (100 * v for v in stats[arm]["success"])
        a.bar(i, m, width=0.56, color=ARM_COLOR[arm], zorder=3)
        a.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=INK, elinewidth=0.9, capsize=2.5, zorder=4)
        a.text(i, hi + 2.5, f"{m:.0f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    reference_markers(a, arms, reference)
    if reference:
        a.legend(loc="lower left", bbox_to_anchor=(0, 1.0), handletextpad=0.4, borderaxespad=0)
    a.set(xticks=range(len(arms)), xticklabels=[ARM_LABEL[m].replace(" ", "\n") for m in arms], ylim=(0, 108),
          yticks=range(0, 101, 25), ylabel="Hit rate (%)")
    a.set_title("(a)", loc="left", x=-0.2, fontweight="bold")
    outcome_bars(b, stats, arms)
    b.set_title("(b)", loc="left", x=-0.36, fontweight="bold")
    return save(fig, out, "fig1_outcomes")


def fig_outcomes_group(by_participant, pooled, arms, reference, out, unit="participant"):
    """(a) every unit's (participant's or session's) hit rate per arm, one line each, and the mean over
    units; (b) the three outcomes, all trials pooled."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_W, 2.4), gridspec_kw=dict(width_ratios=[1, 1.25], wspace=0.35))
    x = np.arange(len(arms))
    last = [100 * stats[arms[-1]]["success"][0] for stats in by_participant.values()]
    for (p, stats), ly in zip(by_participant.items(), spread(last, 4.5)):
        y = [100 * stats[arm]["success"][0] for arm in arms]
        a.plot(x, y, "-", color="#c3c2b7", lw=0.9, zorder=2)
        a.scatter(x, y, s=22, color=[ARM_COLOR[arm] for arm in arms], edgecolor="white", lw=0.8, zorder=4)
        if len(by_participant) <= 6:
            a.text(x[-1] + 0.14, ly, p, fontsize=6.5, color=INK_2, va="center")
    means = [100 * np.mean([stats[arm]["success"][0] for stats in by_participant.values()]) for arm in arms]
    for i, m in enumerate(means):
        a.plot([i - 0.3, i + 0.3], [m, m], color=INK, lw=2.2, zorder=5,
               label=f"mean over {len(by_participant)} {unit}s" if i == 0 else None)
        a.text(i, 103, f"{m:.0f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    a.plot([], [], "-", color="#c3c2b7", lw=0.9, label=f"one {unit}")
    reference_markers(a, arms, reference)
    a.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1, handletextpad=0.4, borderaxespad=0, labelspacing=0.2)
    a.set(xticks=x, xticklabels=[ARM_LABEL[m].replace(" ", "\n") for m in arms], ylim=(0, 112), yticks=range(0, 101, 25),
          ylabel="Hit rate (%)", xlim=(-0.5, len(arms) - 0.3))
    a.set_title("(a)", loc="left", x=-0.2, fontweight="bold")
    outcome_bars(b, pooled, arms)
    b.set_title("(b)", loc="left", x=-0.36, fontweight="bold")
    return save(fig, out, "fig1_outcomes")


def fig_lead(by_participant, arms, focus, out):
    """The focus arm's lead over each other arm: one dot per participant, the mean as a bar."""
    others = [a for a in arms if a != focus]
    fig, ax = plt.subplots(figsize=(HALF_W, 2.3))
    for j, arm in enumerate(others):
        y = len(others) - 1 - j
        leads = {p: 100 * stats[arm]["lead"][0] for p, stats in by_participant.items()}
        ax.scatter(list(leads.values()), [y] * len(leads), s=26, color=ARM_COLOR[arm], edgecolor="white", lw=0.8, zorder=4)
        stacks = {}  # tied values take turns above and below the dot, then further out
        for p, v in leads.items():
            k = len(stacks.setdefault(round(v, 3), []))
            above, dy = k % 2 == 0, 0.14 + 0.3 * (k // 2)
            ax.text(v, y + dy if above else y - dy, p.replace("·", "\n") if "·" in p else p, fontsize=5.5 if "·" in p else 6,
                    color=INK_2, ha="center", va="bottom" if above else "top", linespacing=0.9)
            stacks[round(v, 3)].append(p)
        m = np.mean(list(leads.values()))
        ax.plot([m, m], [y - 0.28, y + 0.28], color=INK, lw=2.2, zorder=5, label="mean" if j == 0 else None)
    ax.axvline(0, color=INK_2, lw=0.8, zorder=1)
    ax.set(yticks=range(len(others)), yticklabels=[f"over {ARM_LABEL[a]}" for a in others[::-1]],
           xlabel=f"{ARM_LABEL[focus]} lead in hit rate (points)", ylim=(-0.75, len(others) - 0.5 + 0.4))
    ax.grid(False)
    ax.grid(True, axis="x")
    ax.legend(loc="lower right")
    return save(fig, out, "fig2_lead")


def fig_sessions(trials, arms, out):
    rows = session_table(trials, arms)
    fig, ax = plt.subplots(figsize=(HALF_W, 2.2))
    for j, arm in enumerate(arms):
        ax.plot([int(r[0]) for r in rows], [100 * r[j + 1] for r in rows], "-o", color=ARM_COLOR[arm], lw=1.8, ms=5,
                mec="white", mew=0.8, label=ARM_LABEL[arm], zorder=3 + (arm == arms[-1]))
    ax.set(xlabel="Session", ylabel="Hit rate (%)", ylim=(0, 100), xticks=[int(r[0]) for r in rows])
    ax.legend(loc="lower left", ncol=2, columnspacing=1.0, handlelength=1.6)
    return save(fig, out, "fig2_sessions")


def fig_cumulative(trials_by_participant, arms, out, name="fig3_cumulative"):
    """Share of trials hit by each time, per arm; with several participants, the mean of their curves."""
    some = next(iter(trials_by_participant.values()))
    limit = some[0]["time_limit_s"]
    t = np.arange(0, limit + 1e-9, 0.02)
    fig, ax = plt.subplots(figsize=(HALF_W, 2.05))
    for arm in arms:
        curves = [cumulative_hits([r for r in trials if r["arm"] == arm], t) for trials in trials_by_participant.values()]
        ax.step(t, 100 * np.mean(curves, 0), where="post", color=ARM_COLOR[arm], lw=1.8, label=ARM_LABEL[arm],
                zorder=3 + (arm == arms[-1]))
    ax.set(xlabel="Time since trial start (s)", ylabel="Trials hit by then (%)", xlim=(0, limit), ylim=(0, 100))
    ax.legend(loc="upper left", handlelength=1.6)
    return save(fig, out, name)


def fig_paths(trials, arms, out):
    fig, axes = plt.subplots(1, len(arms), figsize=(FULL_W, 1.75), gridspec_kw=dict(wspace=0.06))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.79, bottom=0.03)
    radius = np.median([aligned_path(r)[2] for r in trials])
    for ax, arm in zip(axes, arms):
        rs = [r for r in trials if r["arm"] == arm]
        for outcome, alpha, z in (("hit", 0.25, 2), ("timeout", 0.55, 3), ("wrong_target", 0.65, 4), ("collision", 0.7, 5)):
            for r in (r for r in rs if r["outcome"] == outcome):
                p = aligned_path(r)[0]
                ax.plot(p[:, 0], p[:, 1], color=OUTCOME_COLOR[outcome], lw=0.55, alpha=alpha, zorder=z, solid_capstyle="round")
        ax.add_patch(plt.Circle((1, 0), radius, facecolor="none", edgecolor=INK, lw=0.8, zorder=5))
        ax.plot(0, 0, "o", color=INK, ms=3.5, zorder=6)
        hits = sum(r["outcome"] == "hit" for r in rs)
        ax.set_title(f"{ARM_LABEL[arm]}\n{hits}/{len(rs)} hit", fontsize=8, linespacing=1.1, pad=4)
        ax.set(xlim=(-0.45, 1.5), ylim=(-0.8, 0.8), aspect="equal", xticks=[], yticks=[])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(GRID)
    shown = [o for o in OUTCOMES if o != "collision" or any(r["outcome"] == "collision" for r in trials)]
    handles = [plt.Line2D([], [], color=OUTCOME_COLOR[o], lw=1.6) for o in shown]
    fig.legend(handles, [OUTCOME_LABEL[o] for o in shown], loc="upper center", ncol=len(shown),
               bbox_to_anchor=(0.5, 0.0), handlelength=1.4, columnspacing=1.5)
    return save(fig, out, "fig4_paths")


# ---------------------------------------------------------------- PDF


def img(path, caption, max_h=3.6 * inch):
    fig_no[0] += 1
    w, h = PILImage.open(path).size
    s = min(PAGE_W / w, max_h / h)
    return KeepTogether([Image(str(path), w * s, h * s), Spacer(1, 2),
                         Paragraph(f"<b>Figure {fig_no[0]}.</b> {caption}", CAP), Spacer(1, 8)])


def img_pair(left, right, cap_left, cap_right):
    """Two half-width figures side by side, each with its caption."""
    cells, images = [], []
    for path in (left, right):
        w, h = PILImage.open(path).size
        s = min((PAGE_W / 2 - 4) / w, 2.6 * inch / h)
        images.append(Image(str(path), w * s, h * s))
    fig_no[0] += 2
    cells = [images, [Paragraph(f"<b>Figure {fig_no[0] - 1}.</b> {cap_left}", CAP),
                      Paragraph(f"<b>Figure {fig_no[0]}.</b> {cap_right}", CAP)]]
    t = Table(cells, colWidths=[PAGE_W / 2] * 2)
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 2),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 2)]))
    return KeepTogether([t, Spacer(1, 8)])


def table(rows, caption, col_widths=None, font=7.5):
    tab_no[0] += 1
    cells = [[Paragraph(str(c), ParagraphStyle("tc", parent=BODY, fontSize=font, leading=font + 2)) for c in row] for row in rows]
    t = Table(cells, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, 0), "DejaVu-Bold"), ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
                           ("LINEBELOW", (0, -1), (-1, -1), 0.6, colors.black), ("LINEABOVE", (0, 0), (-1, 0), 0.6, colors.black),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    return KeepTogether([Paragraph(f"<b>Table {tab_no[0]}.</b> {caption}", CAP), Spacer(1, 2), t, Spacer(1, 8)])


def bullets(items):
    return ListFlowable([ListItem(Paragraph(t, BODY), leftIndent=12) for t in items], bulletType="bullet", leftIndent=12)


def pct(x):
    return f"{100 * x:.1f}%"


def ci(v):
    return f"{100 * v[0]:+.1f} [{100 * v[1]:+.1f}, {100 * v[2]:+.1f}]"


def join_and(parts):
    """'a, b and c'."""
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def condition_text(a0):
    corruption = a0["corruption"]
    scale = corruption["scale"] if isinstance(corruption, dict) and "scale" in corruption else corruption
    return (f"command gain diag({scale[0]}, {scale[1]}) with Gaussian input noise {a0['input_noise']}, "
            f"{a0['targets_range'][0]} targets per trial, reach {a0['reach'][0]}–{a0['reach'][1]}, {a0['dwell_s']} s dwell, "
            f"{a0['time_limit_s']:g} s limit; the participant did not see the arm")


def depth_text(trials, arms, focus):
    n_frs = next((r["n_reverse_steps"] for r in trials if r["arm"] == "frs"), None)
    return (f"Depths: flow control tau {next(r['tau'] for r in trials if r['arm'] == 'flow_control') if 'flow_control' in arms else '–'}, "
            f"FRS {('n %d' % n_frs) if n_frs else 'full reversal'}, "
            f"{ARM_LABEL[focus]} n {next(r['n_reverse_steps'] for r in trials if r['arm'] == focus)}.")


def repro_paragraph(args, participants, fig_dir):
    code = lambda line: f'<font face="Courier">{line}</font>'
    who = " ".join(participants)
    return Paragraph("<b>Reproducing.</b> " + " · ".join(code(line) for line in (
        f"python run_experiment.py --protocol {args.protocol}.yaml --participant p&lt;N&gt; --session &lt;k&gt;",
        f"python analyze_results.py --participants {who}",
        f"python reports/build_study_report.py --participants {who}"))
        + f". Figures as PDF and PNG: {code(str(fig_dir.relative_to(Path(args.results).parent)) + '/')}", CAP)


def document(out, title):
    return SimpleDocTemplate(str(out), pagesize=letter, leftMargin=0.7 * inch, rightMargin=0.7 * inch, topMargin=0.7 * inch,
                             bottomMargin=0.7 * inch, title=title)


def subtitle(commits):
    return Paragraph(f"Rice BCI Lab · lerobot examples/pi05/mars/cursor · {datetime.date.today():%Y-%m-%d} · "
                     f"code {', '.join(sorted(set(commits)))}", CAP)


def load(args, participant):
    runs = find_runs(args.results, [participant], args.protocol)
    if not runs:
        sys.exit(f"no {args.protocol} runs for {participant} under {args.results}")
    trials, arms_seen = load_trials(runs)
    manifests = [json.loads((r[2] / "manifest.json").read_text()) for r in runs]
    for m in manifests:  # sessions recorded before the rename name the arm differently; the files keep their names
        m["block_order"] = [ARM_ALIASES.get(a, a) for a in m["block_order"]]
    return trials, arms_seen, manifests


def arm_order(arms_seen):
    arms = [a for a in ARM_ORDER if a in arms_seen] + [a for a in arms_seen if a not in ARM_ORDER]
    for a in arms:
        ARM_LABEL.setdefault(a, a)
        ARM_COLOR.setdefault(a, "#6250d6")
    return arms


def stats_table_rows(stats, arms, focus):
    rows = [["Arm", "Trials", "Hit", "Wrong target", "Timeout", "Median time of hits (s)", "Path efficiency (median)",
             f"{ARM_LABEL[focus]} lead, points [95% CI]"]]
    for a in arms:
        st = stats[a]
        rows.append([ARM_LABEL[a], st["n"], f"{pct(st['success'][0])} [{100 * st['success'][1]:.0f}, {100 * st['success'][2]:.0f}]",
                     pct(st["wrong"]), pct(st["timeout"]), f"{st['time_s']:.2f}" if st["time_s"] else "–",
                     f"{st['efficiency']:.2f}" if st["efficiency"] else "–", ci(st["lead"]) if st["lead"] else "–"])
    return rows


STATS_WIDTHS = [1.15 * inch, 0.45 * inch, 0.95 * inch, 0.7 * inch, 0.6 * inch, 0.9 * inch, 0.9 * inch, 1.45 * inch]


# ---------------------------------------------------------------- one participant


def single_report(args, participant):
    trials, arms_seen, manifests = load(args, participant)
    arms = arm_order(arms_seen)
    focus, rng = arms[-1], np.random.default_rng(0)
    stats = arm_stats(trials, arms, focus, rng)
    reference = reference_success(trials, arms)
    sessions = sessions_of(trials)
    per_session = session_table(trials, arms)
    a0, manifest0 = trials[0], manifests[0]
    same_order = len({tuple(m["block_order"]) for m in manifests}) == 1
    fig_dir = Path(args.results) / "analysis" / f"study_{participant}"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else REPORTS / f"study_{participant}.pdf"

    paper_style()
    f1 = fig_outcomes(stats, arms, reference, fig_dir)
    f2 = fig_sessions(trials, arms, fig_dir)
    f3 = fig_cumulative({participant: trials}, arms, fig_dir)
    f4 = fig_paths(trials, arms, fig_dir)

    others = [a for a in arms if a != focus]
    best_other = max(others, key=lambda a: stats[a]["success"][0])
    per_trial = sum(1 for r in trials if r["arm"] == focus and r["run"] == sessions[0])
    wins = sum(1 for row in per_session if row[1 + arms.index(focus)] >= max(row[1:]))
    fs = [100 * row[1 + arms.index(focus)] for row in per_session]
    gap = {a: 100 * (stats[a]["success"][0] - reference[a]) for a in arms if a in reference}
    above, below = [a for a, g in gap.items() if g > 5], [a for a, g in gap.items() if g < -5]
    dates = sorted({r["t_start"][:10] for r in trials})
    setup = (f"Participant {participant} completed {len(sessions)} sessions of the <i>{args.protocol}</i> protocol "
             f"({', '.join(dates)}): {len(arms)} arms per session, one block of {per_trial} logged trials each after "
             f"{manifest0['protocol']['blocks']['practice']} practice trials, {len(trials)} trials in all. Every arm of a "
             f"session played the same {per_trial} layouts in a different order; the block order was "
             + (f"{' → '.join(ARM_LABEL[a] for a in manifest0['block_order'])} in every session" if same_order
                else "a different Latin-square row each session")
             + f". Condition: {condition_text(a0)}. {depth_text(trials, arms, focus)}")
    findings = [
        f"<b>{ARM_LABEL[focus]} hit {pct(stats[focus]['success'][0])} of trials</b>, "
        f"{100 * (stats[focus]['success'][0] - stats[best_other]['success'][0]):.1f} points above the best other arm "
        f"({ARM_LABEL[best_other]}, {pct(stats[best_other]['success'][0])}). Paired by layout, its lead is "
        + join_and([f"{ci(stats[a]['lead'])}{' points' if i == 0 else ''} over {ARM_LABEL[a]}"
                    for i, a in enumerate(sorted(others, key=lambda a: -stats[a]['success'][0]))]) + " (95% CI).",
        "The arms fail differently (Fig. 1b, 4): " + "; ".join(
            f"{ARM_LABEL[a]} mostly by {'wrong targets' if stats[a]['wrong'] > stats[a]['timeout'] else 'timeouts'} "
            f"({pct(max(stats[a]['wrong'], stats[a]['timeout']))})" for a in arms) + ".",
        f"Session to session (Table 2, Fig. 2) {ARM_LABEL[focus]} stayed within {min(fs):.1f}–{max(fs):.1f}% and was the best arm in "
        f"{wins} of {len(sessions)} sessions. Its hits were also the fastest (median {stats[focus]['time_s']:.2f} s; Fig. 3).",
    ]
    if reference:
        findings.append("Against the expert agent in simulation (hollow markers in Fig. 1a): "
                        + ", ".join(f"{ARM_LABEL[a]} {gap[a]:+.0f} points" for a in arms if a in gap)
                        + (f" — the participant did better than the agent with {', '.join(ARM_LABEL[a] for a in above)}" if above else "")
                        + (f"{' and' if above else ' —'} worse with {', '.join(ARM_LABEL[a] for a in below)}" if below else "") + ".")
    caveats = [
        f"One participant; {len(sessions)} sessions on {'one day' if len(dates) == 1 else f'{len(dates)} days'}.",
        ("The block order was the same in every session, so arm and position within a session are confounded."
         if same_order else "The block order rotated through the Latin square across sessions (one row per visit), so "
         "position within a session is balanced over the sessions.") if len(sessions) > 1
        else "A single session: arm and block position are confounded.",
        "The lead's CI pairs the two arms' trials on the same layout within a session and bootstraps over layouts; "
        "it is a within-participant statistic, not a population one.",
    ]

    s = [Paragraph(f"Cursor shared autonomy study: participant {participant}", TITLE),
         subtitle([m.get("git_commit") for m in manifests]), Spacer(1, 8), Paragraph(setup, BODY), Spacer(1, 6),
         Paragraph("Findings", H1), bullets(findings), Spacer(1, 6),
         table(stats_table_rows(stats, arms, focus), "Every session pooled. Hit rate with a bootstrap 95% CI; the lead pairs "
               "trials by layout (session × layout).", col_widths=STATS_WIDTHS),
         img(f1, "(a) Hit rate per arm, all sessions pooled, with a bootstrap 95% CI; hollow markers: the expert agent on "
                 "the same condition in simulation (500 trials, run_sweep.py --sweep task_space_refine). (b) How the trials "
                 "ended.", max_h=2.8 * inch)]
    rows = [["Session"] + [ARM_LABEL[a] for a in arms]] + [[r[0]] + [pct(v) for v in r[1:]] for r in per_session]
    rows.append(["Mean"] + [pct(np.mean([r[j + 1] for r in per_session])) for j in range(len(arms))])
    s += [table(rows, f"Hit rate per session ({per_trial} trials per cell).", col_widths=[0.7 * inch] + [1.15 * inch] * len(arms)),
          img_pair(f2, f3, "Hit rate per session and arm.", "Share of all trials that had selected the goal by each time, "
                   "per arm: the curves' end values are the hit rates, their rise the speed."),
          img(f4, "Every trial's cursor path in reach coordinates (the start at the dot, the goal at the circle; distances "
                  "in units of the reach), colored by outcome, one panel per arm. Wrong-target paths leave toward a "
                  "distractor; timeouts stall short of the goal.", max_h=2.4 * inch),
          KeepTogether([Paragraph("Caveats", H1), bullets(caveats)]), Spacer(1, 8), repro_paragraph(args, [participant], fig_dir)]
    document(out, f"Cursor shared autonomy study: {participant}").build(s)
    print(f"wrote {out}; figures in {fig_dir}")


# ---------------------------------------------------------------- the group


def group_report(args, participants):
    """The session (one participant's visit) is the unit: each session's 40 trials per arm give a hit
    rate and a paired lead; the report averages over sessions (mean ± s.e.m.)."""
    trials_by, manifests, arms_seen = {}, [], []
    for p in participants:
        trials, seen, ms = load(args, p)
        trials_by[p], manifests = trials, manifests + ms
        arms_seen += [a for a in seen if a not in arms_seen]
    arms = arm_order(arms_seen)
    focus, others, rng = arms[-1], arms[:-1], np.random.default_rng(0)
    units = {f"{p}·s{run.rsplit('_s', 1)[1]}": [r for r in trials if r["run"] == run]
             for p, trials in trials_by.items() for run in sessions_of(trials)}
    by_unit = {u: arm_stats(ts, arms, focus, rng) for u, ts in units.items()}
    all_trials = [r for trials in trials_by.values() for r in trials]
    pooled = arm_stats(all_trials, arms, focus, rng)
    first = {p: arm_stats([r for r in trials if r["run"] == sessions_of(trials)[0]], arms, focus, rng)
             for p, trials in trials_by.items()}
    reference = reference_success(all_trials, arms)
    a0 = all_trials[0]
    conditions = {(r["input_noise"], tuple(r["targets_range"]), tuple(r["reach"]), r["dwell_s"]) for r in all_trials}
    fig_dir = Path(args.results) / "analysis" / "study_group"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else REPORTS / "study_group.pdf"

    paper_style()
    f1 = fig_outcomes_group(by_unit, pooled, arms, reference, fig_dir, unit="session")
    f2 = fig_lead(by_unit, arms, focus, fig_dir)
    f3 = fig_cumulative(units, arms, fig_dir)
    f4 = fig_paths(all_trials, arms, fig_dir)

    rate = lambda u, a: 100 * by_unit[u][a]["success"][0]
    n = len(units)
    means = {a: mean_sem([rate(u, a) for u in units]) for a in arms}
    leads = {a: [100 * by_unit[u][a]["lead"][0] for u in units] for a in others}
    best_in = sum(1 for u in units if rate(u, focus) >= max(rate(u, a) for a in arms))
    positive = {a: sum(v > 0 for v in leads[a]) for a in others}
    n_sessions = {p: len(sessions_of(t)) for p, t in trials_by.items()}
    first_means = {a: np.mean([100 * first[p][a]["success"][0] for p in participants]) for a in arms}
    dates = sorted({r["t_start"][:10] for r in all_trials})
    per_block = by_unit[next(iter(units))][focus]["n"]
    gap = {a: means[a][0] - 100 * reference[a] for a in arms if a in reference}
    setup = (f"{len(participants)} participants ({', '.join(participants)}) ran the <i>{args.protocol}</i> protocol between "
             f"{dates[0]} and {dates[-1]}: "
             + join_and([f"{p} {n_sessions[p]} session{'s' if n_sessions[p] > 1 else ''}" for p in participants])
             + f", {n} sessions and {len(all_trials)} logged trials in all ({len(arms)} arms per session, {per_block} trials "
             f"per arm and session, the same layouts for every arm of a session, the block order a Latin-square row). "
             f"Condition{'' if len(conditions) == 1 else 's (NOT identical across runs)'}: {condition_text(a0)}. "
             f"{depth_text(all_trials, arms, focus)} The session is the unit of analysis: each session gives a hit rate "
             f"per arm and a lead paired by layout; the statistics below are means ± s.e.m. over the {n} sessions.")
    findings = [
        f"<b>{ARM_LABEL[focus]} was the best arm in {best_in} of {n} sessions</b>: mean hit rate "
        + join_and([f"{ARM_LABEL[a]} {means[a][0]:.1f} ± {means[a][1]:.1f}%" for a in sorted(arms, key=lambda a: -means[a][0])])
        + ".",
        f"Its lead in hit rate was positive in "
        + join_and([f"{positive[a]} of {n} sessions over {ARM_LABEL[a]}" for a in others])
        + "; mean lead " + join_and([f"{mean_sem(leads[a])[0]:+.1f} ± {mean_sem(leads[a])[1]:.1f} points over {ARM_LABEL[a]}"
                                     for a in others]) + ". Pooling all trials, "
        + join_and([f"{ci(pooled[a]['lead'])} over {ARM_LABEL[a]}" for a in others]) + " (95% CI).",
        "The arms fail differently (Fig. 1b, 4): " + "; ".join(
            f"{ARM_LABEL[a]} mostly by {'wrong targets' if pooled[a]['wrong'] > pooled[a]['timeout'] else 'timeouts'} "
            f"({pct(max(pooled[a]['wrong'], pooled[a]['timeout']))})" for a in arms) + ".",
        f"Each participant's first session only: " + join_and(
            [f"{ARM_LABEL[a]} {first_means[a]:.1f}%" for a in sorted(arms, key=lambda a: -first_means[a])])
        + f". {ARM_LABEL[focus]}'s hits were the fastest (median {pooled[focus]['time_s']:.2f} s; Fig. 3).",
    ]
    if reference:
        findings.append("Against the expert agent in simulation (hollow markers in Fig. 1a), the session means differ by "
                        + ", ".join(f"{gap[a]:+.0f} points for {ARM_LABEL[a]}" for a in arms if a in gap) + ".")
    cover = [len({m["block_order"].index(a) for m in manifests}) for a in arms]  # positions each arm occupied
    caveats = [
        f"{n} sessions from {len(participants)} participants ({join_and([f'{p}: {n_sessions[p]}' for p in participants])}); "
        f"weighting by session counts a participant's repeated visits as independent units, so the s.e.m. understates "
        f"between-person variability. With this few sessions it is descriptive; no population-level test is attempted.",
        "Block orders (Latin-square rows): " + join_and(
            [f"{p} {len({tuple(m['block_order']) for m in ms})} distinct over {len(ms)} session{'s' if len(ms) > 1 else ''}"
             for p, ms in ((p, [m for m in manifests if m["participant"] == p]) for p in participants)])
        + f"; across all runs each arm appeared in {' to '.join(str(v) for v in sorted({min(cover), max(cover)}))} of "
        f"{len(arms)} positions. Within a session arm and position are confounded.",
    ]

    rows = [["Session"] + [ARM_LABEL[a] for a in arms] + [f"{ARM_LABEL[focus]} lead over best other"]]
    for u, st in by_unit.items():
        bo = max(others, key=lambda a: st[a]["success"][0])
        rows.append([u] + [pct(st[a]["success"][0]) for a in arms] + [f"{ci(st[bo]['lead'])} ({ARM_LABEL[bo]})"])
    rows.append(["Mean ± s.e.m."] + [f"{means[a][0]:.1f} ± {means[a][1]:.1f}%" for a in arms] + [""])
    s = [Paragraph("Cursor shared autonomy study: group results", TITLE),
         subtitle([m.get("git_commit") for m in manifests]), Spacer(1, 8), Paragraph(setup, BODY), Spacer(1, 6),
         Paragraph("Findings", H1), bullets(findings), Spacer(1, 6),
         table(rows, f"Hit rate per session and arm ({per_block} trials per cell), and {ARM_LABEL[focus]}'s paired lead over "
                     "that session's best other arm [95% CI over layouts].",
               col_widths=[0.8 * inch] + [0.95 * inch] * len(arms) + [2.0 * inch]),
         table(stats_table_rows(pooled, arms, focus), "All trials of all sessions pooled. Hit rate with a bootstrap 95% CI; "
               "the lead pairs trials by layout within session.", col_widths=STATS_WIDTHS),
         img(f1, "(a) Hit rate per arm: one connected line per session (dots in the arm's color), the bar the mean over "
                 "sessions; hollow markers: the expert agent on the same condition in simulation (500 trials, "
                 "run_sweep.py --sweep task_space_refine). (b) How all trials ended.", max_h=2.9 * inch),
         img_pair(f2, f3, f"{ARM_LABEL[focus]}'s lead in hit rate over each other arm, one dot per session (trials paired "
                  "by layout), the bar the mean.", "Share of trials that had selected the goal by each time, per arm: the mean "
                  "of the sessions' curves; end values are hit rates, the rise is speed."),
         img(f4, "Every trial of every session in reach coordinates (the start at the dot, the goal at the circle; "
                 "distances in units of the reach), colored by outcome, one panel per arm.", max_h=2.4 * inch),
         KeepTogether([Paragraph("Caveats", H1), bullets(caveats)]), Spacer(1, 8), repro_paragraph(args, participants, fig_dir)]
    document(out, "Cursor shared autonomy study: group results").build(s)
    print(f"wrote {out}; figures in {fig_dir}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--participants", nargs="+", required=True, help="p<N> ...; one gives that participant's report, several the group's")
    p.add_argument("--protocol", default="study", help="protocol name as in the manifests")
    p.add_argument("--results", default=str(RESULTS))
    p.add_argument("--out", help="PDF path (default reports/study_<participant>.pdf or reports/study_group.pdf)")
    args = p.parse_args(argv)
    if len(args.participants) == 1:
        single_report(args, args.participants[0])
    else:
        group_report(args, args.participants)


if __name__ == "__main__":
    main()
