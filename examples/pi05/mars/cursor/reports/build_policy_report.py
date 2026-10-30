"""PDF report for the cursor-control flow policy.

    python reports/build_policy_report.py

Reads results/research/{policy_goal_given,policy_goal_given_dropout,policy_goal_hidden,
policy_goal_hidden_nodropout,design_checks}/ and results/sim/sweeps/{gain,rotation}/; writes
reports/cursor_flow_policy_report.pdf.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import datetime
import json
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

from cursor_sa.paths import REPORTS, RESULTS, policy_log  # noqa: E402

RES = RESULTS / "research"          # policy evaluations and design checks
SWEEPS = RESULTS / "sim" / "sweeps"  # run_sweep.py outputs
OUT = REPORTS / "cursor_flow_policy_report.pdf"
PAGE_W = letter[0] - 1.2 * inch
ss = getSampleStyleSheet()
H1, H2, BODY = ss["Heading1"], ss["Heading2"], ss["BodyText"]
BODY.fontSize, BODY.leading = 9.5, 12.5
CAP = ParagraphStyle("cap", parent=BODY, fontSize=8, leading=10, textColor=colors.HexColor("#444444"))
CODE = ParagraphStyle("code", parent=BODY, fontName="Courier", fontSize=7.5, leading=9.5)

figure_no = [0]


def img(path, caption, max_h=7.5 * inch):
    figure_no[0] += 1
    w, h = PILImage.open(path).size
    s = min(PAGE_W / w, max_h / h)
    return KeepTogether([Image(str(path), w * s, h * s),
                         Paragraph(f"<b>Figure {figure_no[0]}.</b> {caption}", CAP), Spacer(1, 10)])


def table(rows, col_widths=None, font=7.5, shade_rows=()):
    tb = Table(rows, colWidths=col_widths, repeatRows=1)
    st = [("FONT", (0, 0), (-1, -1), "Helvetica", font), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", font),
          ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef6")),
          ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#bbbbbb")), ("ALIGN", (1, 0), (-1, -1), "CENTER"),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
    st += [("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f4f4f0")) for r in shade_rows]
    tb.setStyle(TableStyle(st))
    return tb


def bullets(items):
    return ListFlowable([ListItem(Paragraph(t, BODY), leftIndent=12) for t in items],
                        bulletType="bullet", start="•", leftIndent=12)


def pct(x):
    return f"{100 * x:.1f}".rstrip("0").rstrip(".") + "%"


def load(name):
    return {int(k): v for k, v in json.loads((RES / name / "metrics.json").read_text())["summary"].items()}


def pct1(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def sweep_table(test, level_label):
    """All cells of a tuned sweep: test success with tuned depth, and MARS's lead over the best other."""
    rows = [[level_label, "Noise", "Teleop", "Flow control\n(tau)", "FRS\n(n)", "MARS\n(n)",
             "MARS lead over\nbest other [95% CI]"]]
    shade = []
    for k, t in test.items():
        level, noise = k.split("|")
        d, m = t["depths"], t["metrics"]
        lead = t["lead_over_best"]
        rows.append([level, noise, pct1(m["teleop"]["success"])]
                    + [f"{pct1(m[x]['success'])} ({d[x]})" for x in ("flow_control", "frs", "mars_fixed")]
                    + [f"{100 * lead[0]:+.1f} [{100 * lead[1]:+.1f}, {100 * lead[2]:+.1f}]"
                       + (" *" if t["beats_all"] else "")])
        if t["beats_all"]:
            shade.append(len(rows) - 1)
    return table(rows, font=7, shade_rows=shade)


def steering_sections(n_trials):
    ck = json.loads((RES / "design_checks" / "checks.json").read_text())
    gain = json.loads((SWEEPS / "gain" / "test.json").read_text())
    rot = json.loads((SWEEPS / "rotation" / "test.json").read_text())
    tuning_args = json.loads((SWEEPS / "gain" / "tuning.json").read_text())["args"]
    eu, fc, mem, lat = ck["euler_steps"], ck["first_chunk"], ck["memory"], ck["latency"]
    rp, sg, co = ck["replanning"], ck["speed_gain"], ck["corruption"]
    two = lambda x: "—" if x is None or x != x else f"{x:.2f}"
    s = []

    # ------------------------------------------------ 4. evaluations and memory
    g = eu["given"]
    rows = [["Euler steps\nper chunk", "Success", "Median\ntime (s)", "Bow sd", "Evaluations to\nenter target",
             "Evaluations to\nselect target"]]
    for k in ("10", "4", "2", "1"):
        r = g[k]
        rows.append([k, pct1(r["success"]), two(r["time_s"]), two(r["bow_sd"]), f"{r['evals_to_entry']:.0f}",
                     f"{r['evals_to_select']:.0f}"])
    r = g["regression"]
    rows.append(["Regression", pct1(r["success"]), two(r["time_s"]), two(r["bow_sd"]), "(1 per chunk)", ""])
    h = eu["hidden"]
    rows2 = [["Euler steps\nper chunk", "Selected\na target", "Spread of\nchoices", "Ideal\nspread",
              "Spread of the\nfirst chunk", "Median\ntime (s)", "Path\nefficiency", "Evaluations\nto select"]]
    for k in ("10", "4", "2", "1"):
        r = h[k]
        rows2.append([k, pct1(r["selected_any"]), two(r["spread"]), two(r["spread_ideal"]), two(fc[k]["spread"]),
                      two(r["time_s"]), f"{r['path_efficiency']:.3f}", f"{r['evals_to_select']:.0f}"])
    rows3 = [["Condition (goal hidden, 10 Euler steps)", "Selected\na target", "Spread of\nchoices",
              "Median\ntime (s)", "Path\nefficiency"]]
    for name, r in mem.items():
        rows3.append([name, pct1(r["selected_any"]), two(r["spread"]), two(r["time_s"]),
                      "—" if r["path_efficiency"] is None else f"{r['path_efficiency']:.3f}"])
    s += [PageBreak(), Paragraph("4. Network evaluations and memory", H1), Paragraph(
        "Each chunk costs one network evaluation per Euler step, 10 by default, and the policy plans a new "
        "chunk every 8 steps (160 ms). Table 4 counts the evaluations a trial takes and tests fewer Euler "
        "steps (4 targets, goal given, 500 layouts).", BODY), Spacer(1, 4),
        Paragraph("<b>Table 4.</b> Goal given, flow policy. Evaluations are medians, counted until the cursor "
                  "first enters the target, and until the 0.3 s hold selects it.", CAP), Spacer(1, 3),
        table(rows), Spacer(1, 6), Paragraph(
        f"A single Euler step still reaches the goal every time ({pct1(g['1']['success'])}, 6 evaluations per "
        f"trial instead of {g['10']['evals_to_select']:.0f}); what fewer steps lose is the variety of paths "
        f"(bow sd {g['1']['bow_sd']:.2f} at 1 step, {g['10']['bow_sd']:.2f} at 10, "
        f"{g['regression']['bow_sd']:.2f} for regression). With the goal hidden (Table 5, 200 layouts × 20 "
        f"runs), even 1 step spreads the final choices almost ideally, but the first chunk itself does not: "
        f"64 first chunks from the same start divide among the targets with spread {fc['1']['spread']:.2f} at 1 "
        f"step against {fc['10']['spread']:.2f} at 10. At 1 step the closed loop creates the spread, by "
        f"amplifying small noise-driven differences, at the cost of slower and less straight reaches. Four "
        f"steps keep the full first-chunk spread at 40% of the cost.", BODY), Spacer(1, 4),
        Paragraph("<b>Table 5.</b> Goal hidden, goal-dropout flow policy. Spread of the first chunk: the same "
                  "statistic for 64 first chunks sampled from each start, each assigned to the target it "
                  "heads for.", CAP), Spacer(1, 3), table(rows2), Spacer(1, 8),
        KeepTogether([Paragraph(
        "The policy has no memory between chunks: there is no recurrent state, and the noise is redrawn for "
        "every chunk. Two things carry its choice of target forward: the cursor's last 4 displacements in "
        "its input, and the 8 steps of each chunk that run without replanning. Table 6 removes each.", BODY),
        Spacer(1, 4), Paragraph("<b>Table 6.</b> What keeps the goal-free policy on its target (4 targets, "
                                "200 layouts × 20 runs).", CAP), Spacer(1, 3), table(rows3)]), Spacer(1, 6),
        Paragraph(
        "The velocity history is the memory: without it, every replan looks like a fresh start at rest, the "
        "policy re-picks a target each chunk, and it mostly drifts. Open-loop execution cannot hold the "
        "choice alone but makes the reaches straighter. With neither, the cursor never selects anything. "
        "This memory lasts only while the cursor moves: when an operator stops pushing, or a decoder outputs "
        "near zero, the policy's next choice starts fresh.", BODY)]

    # ------------------------------------------------ 5. steering with an operator
    lat_txt = ", ".join(f"{v:.0f} ms ({k.split()[0].upper()}, {'flow control' if 'control' in k else 'reversal'})"
                        for k, v in lat.items())
    s += [PageBreak(), Paragraph("5. Steering with a simulated operator", H1), Paragraph(
        "The goal-free policy is steered by an operator's 2D command, using the shared-autonomy modes of the "
        "lab's LeRobot pi0.5 setup (src/lerobot/policies/pi05/steering.py), with the same names and knobs. "
        "The policy is never told the goal; the operator's command tells it.", BODY), Spacer(1, 4),
        table([["Mode", "What moves the cursor"],
               ["teleop", "the command alone: velocity = command × max speed"],
               ["flow_control (tau 16,\nlambda, action units)", "x_t is set to the command's constant-velocity chunk for the first lambda of\n"
                                             "the 10 Euler steps; the policy then denoises freely (the lab's variant; these\n"
                                             "sweeps tuned lambda, then called tau, and wrote the command in action units)"],
               ["frs\n(FRS, n_reverse_steps)",
                "the command's chunk is run backward n steps through the flow, then forward"],
               ["MARS", "FRS whose backward pass follows F·v, with F the known corruption matrix"]],
              col_widths=[2.0 * inch, 4.6 * inch]), Spacer(1, 6), Paragraph(
        "In the shared modes the executed action is always the policy's own output, and a command inside a "
        "0.05 deadband means the plain policy. The <b>simulated operator</b> pushes straight at the goal, "
        "tapering inside 0.15 of it, and never adapts. Its command passes through the same <b>input "
        "channel</b> as a person's: raw → M·raw (corruption) → + Gaussian noise of std <i>input_noise</i>, "
        "both only while it pushes. A <b>speed gain</b> scales every executed displacement. Configs in "
        "configs/ set all of this for the live app (run_experiment.py). A steered chunk takes "
        f"{lat_txt}, against a 160 ms budget.", BODY)]

    rows = [["Arm (4 targets, 200 trials,\nspeed gain 1)", "Noise 0.3:\nsuccess", "timeouts", "time (s)",
             "Noise 0.6:\nsuccess", "timeouts", "time (s)"]]
    for name, r in rp.items():
        a, b = r["0.3"], r["0.6"]
        rows.append([name, pct1(a["success"]), pct1(a["timeout"]), two(a["time_s"]),
                     pct1(b["success"]), pct1(b["timeout"]), two(b["time_s"])])
    fc8, fc2, tel = rp["flow control, tau 5, execute 8"]["0.6"], rp["flow control, tau 5, execute 2"]["0.6"], rp["teleop"]["0.6"]
    s += [Paragraph("Reading the operator often enough", H2), Paragraph(
        f"LeRobot reads the operator once per chunk. Here that made steering much worse than teleop: at input "
        f"noise 0.6, flow control replanning every 8 steps succeeds on {pct1(fc8['success'])} of trials, "
        f"mostly failing by timeout ({pct1(fc8['timeout'])}) because one noisy command drives 160 ms of "
        f"motion and the cursor cannot hold still on the target; teleop gets {pct1(tel['success'])}. "
        f"Replanning every 2 steps brings flow control to {pct1(fc2['success'])} (Table 7). The live app and "
        f"all later results therefore replan every 2 steps.", BODY), Spacer(1, 4),
        Paragraph("<b>Table 7.</b> Simulated operator, no corruption, input noise 0.3 and 0.6.", CAP),
        Spacer(1, 3), table(rows)]

    rows = [["Mode | speed gain, policy history", "Noise 0.3:\nsuccess", "selected\nany", "time (s)",
             "peak speed\n(screens/s)", "Noise 0.6:\nsuccess"]]
    for name, r in sg.items():
        a = r["0.3"]
        rows.append([name.replace(" | ", " | "), pct1(a["success"]), pct1(a["selected_any"]), two(a["time_s"]),
                     two(a["peak_speed"]), pct1(r["0.6"]["success"])])
    pa = sg["policy | gain 0.5, actual history"]["0.3"]
    s += [KeepTogether([Paragraph("Slowing the cursor down", H2), Paragraph(
        f"The speed gain halves the cursor's on-screen speed in every mode. The policy's velocity history gets "
        f"the unscaled displacement, so it keeps planning as if it moved at its normal speed. Feeding it the "
        f"slowed displacement instead makes the policy alone stall (it selects a target on "
        f"{pct1(pa['selected_any'])} of trials) and slows flow control down (Table 8).", BODY), Spacer(1, 4),
        Paragraph("<b>Table 8.</b> Speed gain 1 against 0.5, with the policy's history unscaled or not "
                  "(4 targets, 200 trials, replanning every 2 steps).", CAP), Spacer(1, 3), table(rows, font=7)])]

    rows = [["Config", "Corruption", "Noise 0:\nsuccess", "wrong target", "time (s)", "Noise 0.3:\nsuccess",
             "wrong target", "time (s)"]]
    for arm, r in co.items():
        a, b = r["0.0"], r["0.3"]
        rows.append([arm, "gain [1, 0.5]" if arm.endswith(("_gain05y", "_adapted")) else "none",
                     pct1(a["success"]), pct1(a["wrong"]), two(a["time_s"]),
                     pct1(b["success"]), pct1(b["wrong"]), two(b["time_s"])])
    frs, frsra = co["frs_gain05y"]["0.0"], co["mars_fixed_gain05y"]["0.0"]
    s += [KeepTogether([Paragraph("A first look at a corrupted command", H2), Paragraph(
        f"With the shipped configs (speed gain 0.5, 2–8 targets, 300 trials), a gain corruption that halves "
        f"the vertical axis costs every mode a few points. The adapter cuts FRS's wrong-target selections "
        f"from {pct1(frs['wrong'])} to {pct1(frsra['wrong'])} at noise 0, but at this mild level teleop "
        f"stays at 97% (Table 9). Times are not comparable across modes: flow control's chunks move faster "
        f"than teleop at the same speed gain.", BODY), Spacer(1, 4),
        Paragraph("<b>Table 9.</b> Shipped clean and gain-corrupted configs at input noise 0 and 0.3, depths "
                  "as shipped (tau 5, full reversal).", CAP), Spacer(1, 3), table(rows, font=7)])]

    # ------------------------------------------------ 6. tuned sweeps
    best = gain["test"][gain["showcase"]]
    rbest = rot["test"][rot["showcase"]]
    n_rot_wins = sum(t["beats_all"] for t in rot["test"].values())
    s += [PageBreak(), Paragraph("6. Where MARS wins: tuned sweeps", H1), Paragraph(
        f"To compare the methods fairly, each method's depth (tau for flow control, n_reverse_steps for FRS "
        f"and MARS, 1–10) was tuned per setting on {tuning_args['tune_trials']} trials, and every method was "
        f"then tested at its tuned depth on {tuning_args['test_trials']} new trials (new layouts and noise; "
        f"2–8 targets, speed gain 0.5, replanning every 2 steps). MARS's lead over each method comes with a "
        f"paired bootstrap 95% CI, since every method sees the same layouts. The setting to highlight was "
        f"chosen on the tuning trials only.", BODY),
        Paragraph("Gain corruption", H2), Paragraph(
        "A gain g on the vertical axis of the command (corruption {scale: [1, g]}) × input noise. MARS "
        "beats every other method in none of the 25 settings. With weak or no gain corruption (g ≥ 0.5) "
        "teleop is best, and its lead grows with noise. With strong gain corruption (g ≤ 0.25) shared "
        "control beats teleop and the adapter adds 8–38 points over plain FRS, but flow control (Kao et al.'s, "
        "at tau 14–16 of its 16 chunk steps) is level with it or ahead in all but one of those settings. That "
        f"setting, which the tuning trials also picked (g {gain['showcase'].split('|')[0]}, noise "
        f"{gain['showcase'].split('|')[1]}), is MARS's best and not beyond noise: "
        f"MARS {pct1(best['metrics']['mars_fixed']['success'])} against flow control "
        f"{pct1(best['metrics']['flow_control']['success'])} "
        f"({100 * best['lead_over_best'][0]:+.1f} points, CI {100 * best['lead_over_best'][1]:+.1f} "
        f"to {100 * best['lead_over_best'][2]:+.1f}).", BODY), Spacer(1, 4),
        Paragraph("<b>Table 10.</b> Gain sweep, test trials, each method at its tuned depth (in brackets). "
                  "* and shading: MARS beats each other method (CI above 0).", CAP), Spacer(1, 3),
        sweep_table(gain["test"], "Gain g"), Spacer(1, 8),
        img(SWEEPS / "gain" / "figs" / "test.png",
            "Gain sweep, test trials: teleop success, and MARS's lead over the best other method.",
            max_h=3.0 * inch),
        img(SWEEPS / "gain" / "figs" / "depth_curves.png",
            "Gain sweep, tuning trials: success against steering depth for every setting. With no "
            "corruption (top row) the FRS and MARS curves coincide, since F = I. Both FRS curves dip "
            "at n_reverse_steps 9 almost everywhere (see text).", max_h=5.3 * inch),
        PageBreak(), Paragraph("Rotation corruption", H2), Paragraph(
        f"The same protocol with the command rotated by 10°, 20°, 45° or 90° (the LeRobot study's corruption "
        f"is a 20° rotation) × input noise. At 45° or less teleop wins, as with the gain. <b>At 90°, tuned "
        f"MARS beats every other method at every noise level</b> ({n_rot_wins} of the 16 settings, all at "
        f"90°): at the setting the tuning trials picked (90°, noise {rot['showcase'].split('|')[1]}), MARS "
        f"succeeds on {pct1(rbest['metrics']['mars_fixed']['success'])} of trials, against "
        f"{pct1(rbest['metrics']['flow_control']['success'])} for flow control, "
        f"{pct1(rbest['metrics']['frs']['success'])} for FRS and {pct1(rbest['metrics']['teleop']['success'])} "
        f"for teleop. At 90° an operator pushing straight at the goal makes the cursor circle it, so teleop "
        f"fails; plain FRS steers toward the rotated command and does worse than the policy alone; flow "
        f"control's best, at tau 1, is about what the policy gets by guessing among the targets. MARS's "
        f"adapter, the known rotation, recovers the intent, peaking at n_reverse_steps 6.", BODY), Spacer(1, 4),
        Paragraph("<b>Table 11.</b> Rotation sweep, test trials, each method at its tuned depth.", CAP),
        Spacer(1, 3), sweep_table(rot["test"], "Rotation (°)"), Spacer(1, 8),
        img(SWEEPS / "rotation" / "figs" / "showcase.png",
            "The selected setting (90°, noise 0.25), test trials with 95% CIs, each method at its tuned depth.",
            max_h=2.6 * inch),
        img(SWEEPS / "rotation" / "figs" / "test.png",
            "Rotation sweep, test trials: teleop success, and MARS's lead over the best other method.",
            max_h=3.0 * inch),
        img(SWEEPS / "rotation" / "figs" / "depth_curves.png",
            "Rotation sweep, tuning trials: success against steering depth. At 90° (bottom row) MARS "
            "peaks at n_reverse_steps 6–8 while the other methods stay near their floor.", max_h=5.3 * inch),
        Paragraph(
        "<b>A schedule artifact at n_reverse_steps 9.</b> In both sweeps FRS and MARS drop sharply at 9. "
        "With LeRobot's schedule, reversing n of N steps leaves N − n forward steps for the rest of the flow, "
        "so n = 9 covers 90% of it in a single Euler step. LeRobot runs with n_reversal_steps: 9 are likely "
        "affected the same way.", BODY)]
    return s


def main():
    ev, egd = load("policy_goal_given"), load("policy_goal_given_dropout")
    gf, gfn = load("policy_goal_hidden"), load("policy_goal_hidden_nodropout")
    train = json.loads(policy_log("flow").read_text())["args"]
    dropout = json.loads(policy_log("flow_goaldrop").read_text())["args"]["goal_dropout"]
    n_trials = json.loads((RES / "policy_goal_given" / "metrics.json").read_text())["args"]["trials"]
    gf_args = json.loads((RES / "policy_goal_hidden" / "metrics.json").read_text())["args"]
    given_counts, free_counts = sorted(ev), sorted(gf)
    mean = lambda xs: sum(xs) / len(xs)
    rng_ = lambda xs, f: f"{f(min(xs))}–{f(max(xs))}" if f(min(xs)) != f(max(xs)) else f(min(xs))

    # Numbers quoted in the text.
    given_success = [ev[n][m]["success"] for n in given_counts for m in ("flow", "regress")]
    bow = {m: mean([ev[n][m]["bow_sd"] for n in given_counts]) for m in ("demos", "flow", "regress")}
    gd_success = [egd[n][m]["success"] for n in given_counts for m in ("flow_goaldrop", "regress_goaldrop")]
    gd_wrong = max(egd[n][m]["wrong_target"] for n in given_counts for m in ("flow_goaldrop", "regress_goaldrop"))
    f_sel = [gf[n]["flow_goaldrop"]["selected_any"] for n in free_counts]
    f_spread = [gf[n]["flow_goaldrop"]["spread"] for n in free_counts]
    ideal = [gf[n]["flow_goaldrop"]["spread_ideal"] for n in free_counts]
    f_inel = [gf[n]["flow_goaldrop"]["ineligible"] for n in free_counts]
    r_sel = [gf[n]["regress_goaldrop"]["selected_any"] for n in free_counts]
    r_inel = [gf[n]["regress_goaldrop"]["ineligible"] for n in free_counts]
    nd_flow = [gfn[n]["flow"]["selected_any"] for n in free_counts]
    nd_reg = [gfn[n]["regress"]["selected_any"] for n in free_counts]
    f_time = [gf[n]["flow_goaldrop"]["time_s"] for n in free_counts]
    f_given_time = [egd[n]["flow_goaldrop"]["time_s"] for n in given_counts]
    two = lambda x: f"{x:.2f}"

    doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch,
                            topMargin=0.6 * inch, bottomMargin=0.6 * inch,
                            title="Cursor control with a transformer flow-matching policy")
    s = [Paragraph("Cursor control with a transformer flow-matching policy", ss["Title"]),
         Paragraph(f"Rice BCI Lab · lerobot examples/pi05/mars/cursor · "
                   f"{datetime.date.today():%B %-d, %Y}", CAP), Spacer(1, 10)]

    # ---------------------------------------------------------------- summary
    s += [Paragraph("Summary", H1), Paragraph(
        "We built a policy that moves a cursor from wherever it is to a target on screen. Like "
        "vision-language-action models such as pi0, it predicts a chunk of future actions (16 steps, "
        "320 ms) with a flow-matching head instead of one action at a time. A transformer with one "
        "token per target lets the same network handle any number of targets. It was trained on "
        "synthetic human-like reaches and tested in closed loop against a regression baseline with "
        "the same network.", BODY), Spacer(1, 4), bullets([
        f"<b>Goal given:</b> both the flow policy and regression select the correct target in "
        f"{rng_(given_success, pct)} of {n_trials} test trials at every target count from "
        f"{given_counts[0]} to {given_counts[-1]}, including counts above the 1–8 seen in training.",
        f"<b>The flow head learns the variety of reaches; regression learns only the average.</b> "
        f"The spread of sideways path bow is {bow['flow']:.2f} for the flow policy, against "
        f"{bow['demos']:.2f} for the demos and {bow['regress']:.2f} for regression (sd, as a fraction "
        f"of reach distance).",
        f"<b>Goal hidden:</b> after training with the goal flag hidden on {pct(dropout)} of samples, the "
        f"flow policy given no goal picks a target and commits to it on {rng_(f_sel, pct)} of runs. Its "
        f"choices spread across the possible targets about as evenly as an ideal random choice "
        f"(normalized entropy {rng_(f_spread, two)}, ideal {rng_(ideal, two)}). Regression always picks "
        f"the same target, and models trained without hidden goals mostly wander "
        f"({rng_(nd_flow, pct)} of flow runs select anything).",
        f"<b>Hiding the goal in training costs little when the goal is given:</b> success stays at "
        f"{rng_(gd_success, pct)} (at most {pct(gd_wrong)} wrong-target selections).",
        "<b>Steering with a simulated operator:</b> the LeRobot pi0.5 shared-autonomy modes (flow control, "
        "Flow Reversal Steering, and MARS with its reversal adapter) work with this policy in real time. "
        "Reading the operator once per 8-step chunk, as LeRobot does, makes steering fail under input noise; "
        "replanning every 2 steps fixes it (Section 5).",
        "<b>Where MARS wins (Section 6):</b> with every method's depth tuned and tested on new trials, a "
        "gain corruption of the operator's vertical axis never made MARS beat all the others (flow control "
        "or teleop wins); a 90° rotation of the command did, at every noise level tested, with the "
        "simulated operator never compensating for the rotation.",
    ]), Spacer(1, 4), Paragraph(
        "This goal-free flow policy is the base needed for shared autonomy: without a goal, its output "
        "already holds one option per target, so a user's (BCI) signal only has to choose among them.",
        BODY)]

    # ---------------------------------------------------------------- setup
    s += [Paragraph("1. Setup", H1), Paragraph("Task", H2), Paragraph(
        "A 2D cursor task in screen coordinates [0, 1]<super>2</super> at 50 Hz (20 ms steps). Each trial "
        "has a random cursor start and random targets (radius 0.05, at least 0.12 apart); one target is "
        "the goal, at least 0.2 from the start. Holding the cursor on any target for 0.3 s selects it "
        "and ends the trial; selecting a non-goal target is a failure, and nothing selected after 5 s is "
        "a timeout. Movement rules match CursorControlTask in shared_autonomy_benchmark.", BODY),
        Paragraph("Demonstrations", H2), Paragraph(
        "Each reach is a minimum-jerk movement whose duration follows Fitts' law (0.3 s + 0.2 s per bit, "
        "with log-normal jitter), bowed sideways by a uniform random amount up to ±25% of the reach "
        "distance. On 40% of reaches the primary movement misses (error sd 8% of distance along the "
        "reach, 5% across) and an overlapping corrective sub-movement brings the cursor onto the goal. "
        "The cursor then holds still until the goal is selected. Plans that leave the screen or select "
        "a distractor are redrawn. Median trial length is 0.9 s including the hold, with median peak "
        "speed 1.1 screens/s.", BODY), Paragraph("Policy", H2), Paragraph(
        "A transformer (4 layers, width 128, 4 heads; 0.9M parameters) over three kinds of tokens:", BODY),
        Spacer(1, 4),
        table([["Token", "Count", "Input features"],
               ["Cursor", "1", "position; last 4 per-step displacements"],
               ["Target", "one per target", "position; offset from the cursor; goal flag (1 for the goal, else 0)"],
               ["Action", "16", "that step's noisy action (2D) and the flow time t"]],
              col_widths=[0.9 * inch, 1.1 * inch, 4.6 * inch]), Spacer(1, 6), Paragraph(
        "Targets have no order embedding and padded target slots are masked out of attention, so the "
        "output is identical however the targets are ordered or padded (checked in the unit tests). As "
        "in pi0, cursor and target tokens do not attend to the action tokens. The flow head is trained "
        "with the loss ||v(x<sub>t</sub>, t | o) − (a − z)||<super>2</super>, where "
        "x<sub>t</sub> = (1 − t)z + t·a, z is Gaussian noise and a is the demonstrated chunk; sampling "
        "runs 10 Euler steps from noise (t = 0) to an action chunk (t = 1). In closed loop the cursor "
        "follows the first 8 steps of each chunk, then the policy plans again. The regression baseline "
        "is the same network predicting the chunk directly with a mean-squared-error loss.", BODY),
        Paragraph("Training and evaluation", H2), Paragraph(
        f"Each model was trained on {train['n_demos']:,} demos with {train['targets'][0]}–"
        f"{train['targets'][1]} targets per trial, for {train['steps']:,} steps (batch {train['batch']}, "
        f"AdamW, learning rate {train['lr']:g} with warmup and cosine decay; exponential moving average "
        f"of weights). Training takes about 10 minutes on one RTX 6000 Ada (training curves in the "
        f"appendix). The test sets use new layouts: {n_trials} per target count with the goal given, "
        f"and {gf_args['layouts']} layouts × {gf_args['runs']} runs per target count with the goal "
        f"hidden. All tests are noise-free.", BODY)]

    # ---------------------------------------------------------------- goal given
    s += [Paragraph("2. Reaching with the goal given", H1)]
    rows = [["Targets", "Success\nflow", "Success\nregression", "Time (s)\ndemos", "Time (s)\nflow",
             "Time (s)\nregression", "Path eff.\nflow", "Bow sd\ndemos", "Bow sd\nflow", "Bow sd\nregression"]]
    for n in given_counts:
        d, f, r = ev[n]["demos"], ev[n]["flow"], ev[n]["regress"]
        rows.append([str(n), pct(f["success"]), pct(r["success"]), two(d["time_s"]), two(f["time_s"]),
                     two(r["time_s"]), f"{f['path_efficiency']:.3f}", two(d["bow_sd"]), two(f["bow_sd"]),
                     two(r["bow_sd"])])
    s += [Paragraph(f"<b>Table 1.</b> Closed-loop results with the goal given ({n_trials} layouts per "
                    f"target count; demos succeed in 100% by construction). Time is the median time to "
                    f"select the goal, including the 0.3 s hold. Path efficiency is straight-line "
                    f"distance divided by path length up to entering the goal. Bow sd is the spread of "
                    f"peak sideways deviation as a fraction of reach distance. Shaded rows exceed the "
                    f"training range of 1–8 targets.", CAP), Spacer(1, 3),
          table(rows, shade_rows=[i + 1 for i, n in enumerate(given_counts) if n > train["targets"][1]]),
          Spacer(1, 8),
          Paragraph(
        "Both policies reach the goal every time, at every target count, and as fast as the demos. "
        "Adding distractors, even beyond the training range, does not slow them down or cause wrong "
        "selections. The difference between the two lies in the shape of the paths (Figures 1 and 3): "
        "the flow policy bows left and right across runs much as the demos do, while regression "
        "produces one nearly straight path per start. Because the goal is given, the average path is "
        "already a good path, so this task cannot separate the methods on success.", BODY), Spacer(1, 6),
          img(RES / "policy_goal_given" / "figs" / "rollouts.png",
              "30 closed-loop runs from the same start for each method, at 1, 4, 8 and 16 targets (goal "
              "outlined). The flow policy's paths vary like the demos'; regression repeats one path.",
              max_h=3.5 * inch),
          img(RES / "policy_goal_given" / "figs" / "by_target_count.png",
              "Success rate and median time to select the goal against the number of targets on screen.",
              max_h=2.8 * inch),
          img(RES / "policy_goal_given" / "figs" / "kinematics.png",
              "Left: distribution of peak sideways deviation (successful trials within the training "
              "range). The flow policy's spread is close to the demos', but its distribution is "
              "bell-shaped where the demos' is flat, so it under-produces the most curved reaches. "
              "Right: speed profiles (median and interquartile range) match.", max_h=2.8 * inch)]

    # ---------------------------------------------------------------- goal hidden
    s += [PageBreak(), Paragraph("3. Reaching with the goal hidden", H1), Paragraph(
        f"For shared autonomy the policy will not be told the goal. To get a policy that works without "
        f"it, two new models were trained exactly as above except that the goal flag was cleared on "
        f"{pct(dropout)} of training samples (goal dropout). In the training data the goal is a random "
        f"target at least 0.2 from the start, so with no flag an ideal policy should pick one of those "
        f"targets at random and commit to it. Each test layout was run {gf_args['runs']} times with "
        f"every goal flag cleared, and we recorded which target each run selected. The original models, "
        f"which never saw a hidden flag in training, were tested the same way as a baseline.", BODY),
        Spacer(1, 6)]
    labels = [("goal_free", "flow_goaldrop", "Flow, goal dropout"),
              ("goal_free", "regress_goaldrop", "Regression, goal dropout"),
              ("goal_free_nodropout", "flow", "Flow, no dropout"),
              ("goal_free_nodropout", "regress", "Regression, no dropout")]
    src = {"goal_free": gf, "goal_free_nodropout": gfn}
    rows = [["Targets", "Model", "Selected\na target", "Spread of\nchoices", "Ideal\nspread",
             "Picked a target\ncloser than 0.2", "Median\ntime (s)"]]
    for n in free_counts:
        for key, m, label in labels:
            r = src[key][n][m]
            spread = "—" if r["spread"] != r["spread"] else two(r["spread"])  # NaN: nothing selected
            rows.append([str(n), label, pct(r["selected_any"]), spread, two(r["spread_ideal"]),
                         pct(r["ineligible"]), "—" if r["time_s"] is None else two(r["time_s"])])
    s += [Paragraph(f"<b>Table 2.</b> Goal hidden ({gf_args['layouts']} layouts × {gf_args['runs']} runs per "
                    f"target count). Spread of choices is the entropy of each layout's choices divided by "
                    f"log(number of targets at least 0.2 from the start): 1 is an even spread over those "
                    f"targets, 0 is always the same one. Ideal spread is the same statistic for a uniform "
                    f"random choice with the same number of runs. Targets closer than 0.2 to the start "
                    f"were never the goal in training.", CAP), Spacer(1, 3),
          table(rows, col_widths=[0.6 * inch, 1.6 * inch] + [0.9 * inch] * 5,
                shade_rows=[i + 1 for i in range(len(rows) - 1) if (i // 4) % 2 == 1]),
          Spacer(1, 8), bullets([
        f"<b>The flow policy commits to one target per run and spreads its choices across runs.</b> It "
        f"selects a target on {rng_(f_sel, pct)} of runs, in a median {rng_(f_time, two)} s, the same as "
        f"with the goal given ({rng_(f_given_time, two)} s), so it does not hesitate. Its spread matches "
        f"the ideal; it is slightly above the ideal because {rng_(f_inel, pct)} of picks land on targets "
        f"closer than 0.2, which add spread the ideal does not have. The flow head's noise decides which "
        f"target a run goes to, and the cursor's recent movement keeps it on that target after replanning.",
        f"<b>Regression also commits, but always to the same target.</b> It selects a target on "
        f"{rng_(r_sel, pct)} of runs, but being deterministic it repeats the same path every run "
        f"(spread 0). We expected it to average the directions and stall between targets; instead its "
        f"averaged first chunk heads somewhere and its movement history then locks it onto whichever "
        f"target lies that way, often a nearby one ({rng_(r_inel, pct)} of its picks are closer than 0.2).",
        f"<b>Without goal dropout, hiding the goal breaks the policy.</b> The original models wander: "
        f"flow selects a target on {rng_(nd_flow, pct)} of runs and regression on {rng_(nd_reg, pct)}.",
    ]), Spacer(1, 6),
          img(RES / "policy_goal_hidden" / "figs" / "goal_free_rollouts.png",
              f"{gf_args['runs']} runs from the same start with no goal flag. Numbers inside targets count "
              f"the runs that selected each target; the dashed circle (radius 0.2) marks where the goal "
              f"never was in training.", max_h=3.9 * inch),
          img(RES / "policy_goal_hidden" / "figs" / "goal_free_summary.png",
              "Goal hidden, dropout-trained models: share of runs that selected a target, and spread of "
              "choices against the ideal for a uniform random choice.", max_h=2.8 * inch)]
    rows = [["Targets", "Success\nflow, dropout", "Success\nregression, dropout", "Wrong target\nflow, dropout",
             "Wrong target\nregression, dropout", "Success\nflow, no dropout"]]
    for n in given_counts:
        f, r = egd[n]["flow_goaldrop"], egd[n]["regress_goaldrop"]
        rows.append([str(n), pct(f["success"]), pct(r["success"]), pct(f["wrong_target"]),
                     pct(r["wrong_target"]), pct(ev[n]["flow"]["success"])])
    s += [KeepTogether([Paragraph("Cost when the goal is given", H2), Paragraph(
        f"The dropout-trained models were also run on the goal-given test set of Section 2. Success "
        f"stays at {rng_(gd_success, pct)}; the only cost is a few wrong-target selections at 8 or more "
        f"targets (at most {pct(gd_wrong)}), against none for the original models. Figures for these runs "
        f"are in results/research/policy_goal_given_dropout/figs/.", BODY), Spacer(1, 4),
        Paragraph(f"<b>Table 3.</b> Goal given, dropout-trained models ({n_trials} layouts per target "
                  f"count).", CAP), Spacer(1, 3), table(rows)])]

    s += steering_sections(n_trials)

    # ---------------------------------------------------------------- caveats, next steps
    s += [Paragraph("7. Caveats", H1), bullets([
        "Demonstrations are synthetic. Real BCI cursor movements are slower and noisier, and the "
        "policy has not been trained or tested on them.",
        f"Each model was trained once (seed 0), so there are no between-seed error bars. With {n_trials} "
        f"trials, a 100% success rate bounds the true failure rate below about {pct(3 / n_trials)} (95%, "
        f"rule of three).",
        "With the goal given, success saturates at 100%, so the task does not rank methods; differences "
        "show only in the distribution of paths.",
        "The flow policy under-produces the most curved reaches, and with the goal hidden it picks "
        "targets that were never goals in training on a few percent of runs.",
        "All steering results use a simulated operator that pushes straight at the goal and never adapts. "
        "A person would learn to compensate for a fixed corruption, in particular a 90° rotation within a "
        "few trials, which would rescue teleop and change what MARS receives; the MARS result applies "
        "to the uncompensated regime.",
        "MARS's adapter is the known corruption matrix; how it behaves when the corruption is unknown or "
        "misestimated is untested.",
        "The sweeps compare 25 and 16 settings without correcting for multiple comparisons; the 90° "
        "rotation leads (21–62 points, CIs well clear of 0) are far beyond that concern, the gain-sweep tie "
        "is not.",
    ]), Paragraph("8. Next steps", H1), bullets([
        "<b>SpaceMouse sessions</b> (run_experiment.py with the *_rot90 and *_gain05y arms), recording whether "
        "and how fast a person compensates for the corruption in teleop, and how each mode compares.",
        "<b>Fix the partial-reversal schedule</b> so that no setting runs most of the flow in one forward "
        "step, and re-tune FRS and MARS; check the same in the LeRobot runs.",
        "<b>Rotations between 45° and 90°</b>, and an adapter that is estimated rather than known.",
        "<b>Classifier-free guidance</b>: mix the goal-conditioned and goal-free outputs of the same network "
        "once a goal guess exists.",
    ])]

    # ---------------------------------------------------------------- appendix
    s += [PageBreak(), Paragraph("Appendix", H1),
          img(RES / "policy_goal_given" / "figs" / "training.png",
              "Training and validation loss (validation uses the averaged weights), original models.",
              max_h=2.2 * inch),
          img(RES / "policy_goal_given_dropout" / "figs" / "training.png",
              f"Training and validation loss, models trained with goal dropout ({pct(dropout)}).",
              max_h=2.2 * inch),
          Paragraph("Reproducing", H2), Paragraph(
        "From examples/pi05/mars/cursor, in the lerobot environment (uv run --extra mars):", BODY),
          Spacer(1, 3)]
    for line in ["python -m pytest",
                 "python train/train_policy.py --mode flow",
                 "python train/train_policy.py --mode regress",
                 f"python train/train_policy.py --mode flow --name flow_goaldrop --goal_dropout {dropout}",
                 f"python train/train_policy.py --mode regress --name regress_goaldrop --goal_dropout {dropout}",
                 "python research/evaluate_policy_goal_given.py",
                 "python research/evaluate_policy_goal_given.py --models flow_goaldrop regress_goaldrop --out policy_goal_given_dropout",
                 "python research/evaluate_policy_goal_hidden.py",
                 "python research/evaluate_policy_goal_hidden.py --models flow regress --out policy_goal_hidden_nodropout",
                 "python research/design_checks.py",
                 "python run_sweep.py --sweep gain",
                 "python run_sweep.py --sweep rotation",
                 "python run_experiment.py --arm mars_fixed_rot90.yaml   # live, SpaceMouse",
                 "python reports/build_policy_report.py   # this report"]:
        s.append(Paragraph(line.replace(" ", "&nbsp;"), CODE))

    doc.build(s)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
