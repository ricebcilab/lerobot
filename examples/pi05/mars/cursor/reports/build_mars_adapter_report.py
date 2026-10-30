"""PDF report on the learned reversal adapter (MARS), with the math.

    python reports/build_mars_adapter_report.py

Reads results/sim/sweeps/{learned_adapter_rot90,learned_adapter_gain025y,study_condition}/,
results/research/adapter_diagnostics/diagnostics.json, the adapters' training logs in
checkpoints/adapters/ and checkpoints/policies/flow_goaldrop/train_log.json; writes
reports/mars_adapter_report.pdf. Equations are rendered with
matplotlib's mathtext.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import datetime
import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from cursor_sa.paths import REPORTS, RESULTS, adapter_dir, policy_log  # noqa: E402

SWEEPS = RESULTS / "sim" / "sweeps"  # run_sweep.py outputs
RESEARCH = RESULTS / "research"      # adapter_diagnostics.py output
OUT = REPORTS / "mars_adapter_report.pdf"
METHOD_NAMES = {"oracle": "oracle", "user": "user", "policy": "policy", "frs": "frs", "mars_fixed": "mars_fixed",
                "mars_goal": "mars_goal", "mars_free": "mars_free"}


def closed_loop_results():
    """Every method at the full reversal (depth 10) from the learned_adapter_* sweeps' tuning trials,
    as {args: {trials}, results: {"<label> | noise <s>": {method: metrics}}}."""
    out = {"args": {}, "results": {}}
    for label in ("rot90", "gain025y"):
        tj = json.loads((SWEEPS / f"learned_adapter_{label}" / "tuning.json").read_text())
        out["args"]["trials"] = tj["args"]["tune_trials"]
        for key, cell in tj["tuning"].items():
            noise = key.split("|")[1]
            out["results"][f"{label} | noise {noise}"] = {
                new: cell[old]["10"] if "10" in cell[old] else cell[old]["None"] for old, new in METHOD_NAMES.items()}
    return out
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
plt.rcParams.update({"mathtext.fontset": "cm", "font.size": 11})

eq_no = [0]
fig_no = [0]


def eq(tex, size=11.5, numbered=True):
    """A display equation rendered by mathtext (a list of strings gives one line each),
    with its number on the right."""
    eq_no[0] += numbered
    lines = []
    for line in [tex] if isinstance(tex, str) else tex:
        path = TMP / f"eq{len(list(TMP.glob('eq*.png')))}.png"
        fig = plt.figure()
        fig.text(0, 0, f"${line}$", fontsize=size)
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03, transparent=True)
        plt.close(fig)
        w, h = PILImage.open(path).size
        lines.append([Image(str(path), w * 72 / 300, h * 72 / 300)])
    img = Table(lines) if len(lines) > 1 else lines[0][0]
    cells = [[img, Paragraph(f"({eq_no[0]})" if numbered else "", BODY)]]
    t = Table(cells, colWidths=[PAGE_W - 0.5 * inch, 0.5 * inch])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (0, 0), (0, 0), "CENTER"),
                           ("ALIGN", (1, 0), (1, 0), "RIGHT")]))
    return t


def img(path, caption, max_h=4 * inch):
    fig_no[0] += 1
    w, h = PILImage.open(path).size
    s = min(PAGE_W / w, max_h / h)
    return KeepTogether([Image(str(path), w * s, h * s),
                         Paragraph(f"<b>Figure {fig_no[0]}.</b> {caption}", CAP), Spacer(1, 8)])


def table(rows, col_widths=None, font=7.5, shade=()):
    tb = Table([[Paragraph(str(c), ParagraphStyle("tc", parent=BODY, fontSize=font, leading=font + 2))
                 for c in r] for r in rows], colWidths=col_widths, repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef6")),
          ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#bbbbbb")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
    st += [("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f4f4f0")) for r in shade]
    tb.setStyle(TableStyle(st))
    return tb


def bullets(items):
    return ListFlowable([ListItem(Paragraph(t, BODY), leftIndent=12) for t in items], bulletType="bullet",
                        start="•", leftIndent=12)


P = lambda text: Paragraph(text, BODY)
pct = lambda x: "—" if x is None else f"{100 * x:.1f}%"


def training_figure(logs):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    plt.rcParams.update({"mathtext.fontset": "dejavusans"})
    look = {"rot90": "#2a78d6", "gain025y": "#eb6834"}
    for name, log in logs.items():
        label, data = name.rsplit("_", 1)
        ls = "-" if data == "free" else "--"
        steps = [e["step"] for e in log]
        axes[0].plot(steps, [e["train"] for e in log], ls, color=look[label], lw=1.8, label=f"{label}, {data} pairs")
        axes[1].plot(steps, [e["heading_cos_learned"] for e in log], ls, color=look[label], lw=1.8)
        if data == "free":
            axes[1].plot(steps, [e["heading_cos_fixed"] for e in log], ":", color=look[label], lw=1.5)
    axes[0].set(xlabel="Training step", ylabel="Loss (2)", title="Training loss")
    axes[1].set(xlabel="Training step", ylabel="Mean cosine", ylim=(0.8, 1.0),
                title="Regenerated chunk heading vs intended\n(held-out states; dotted: fixed F = M)")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#e6e5df")
    axes[0].legend(fontsize=7, frameon=False)
    path = TMP / "training.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    plt.rcParams.update({"mathtext.fontset": "cm"})
    return path


def main():
    ev = closed_loop_results()
    dg = json.loads((RESEARCH / "adapter_diagnostics" / "diagnostics.json").read_text())
    study = json.loads((SWEEPS / "study_condition" / "test.json").read_text())["test"]["0.25|1.0"]
    logs = {name: json.loads((adapter_dir(name) / "train_log.json").read_text())["log"]  # this report's four adapters
            for name in ("gain025y_free", "gain025y_goal", "rot90_free", "rot90_goal")}
    targs = json.loads((adapter_dir("rot90_free") / "train_log.json").read_text())["args"]
    s_a = json.loads(policy_log("flow_goaldrop").read_text())["action_scale"]
    dropout = json.loads(policy_log("flow_goaldrop").read_text())["args"]["goal_dropout"]
    kappa = 1.2 * 0.02 / s_a
    r, ca = ev["results"], dg["constant_adapter"]
    rs = dg["reach_start"]
    live = {m: study["metrics"][m] for m in ("teleop", "frs", "mars_fixed", "mars")}
    live_depths = study["depths"]
    finals = [log[-1] for log in logs.values()]
    lat = {k: (min(f[f"latent_{k}"] for f in finals), max(f[f"latent_{k}"] for f in finals))
           for k in ("none", "fixed", "learned")}
    rot_f = ca["rot90"]["F"]
    rot_scale = (rot_f[1][0] - rot_f[0][1]) / 2           # F* ~ scale * R(90 deg)
    gain_f, gain_m = ca["gain025y"]["F"], ca["gain025y"]["M"]
    gain_scale = (gain_f[0][0] / gain_m[0][0] + gain_f[1][1] / gain_m[1][1]) / 2
    fmt_m = lambda m: "[[" + "], [".join(", ".join(f"{v:+.2f}" for v in row) for row in m) + "]]"

    doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch,
                            topMargin=0.6 * inch, bottomMargin=0.6 * inch,
                            title="Learned reversal adapter for Flow Reversal Steering")
    s = [Paragraph("Learned reversal adapter for Flow Reversal Steering", TITLE),
         Paragraph(f"Rice BCI Lab · lerobot examples/pi05/mars/cursor · "
                   f"{datetime.date.today():%B %-d, %Y}", CAP), Spacer(1, 8)]

    # ------------------------------------------------------------ summary
    s += [Paragraph("Summary", H1), bullets([
        "<b>How H is learned.</b> By regression on the objective H(x<sub>t</sub>, t)·P(x<sub>t</sub>, t) = "
        "a<sub>bci</sub> − a<sub>noise</sub> along the straight path x<sub>t</sub> = (1 − t)·a<sub>noise</sub> + "
        "t·a<sub>bci</sub> (eq. 6–7). Training samples are generated on the fly: a demo state, a noise chunk, the "
        "policy's chunk from that noise, its corrupted BCI version, and a random t. P is the frozen policy's "
        "goal-free velocity field.",
        "<b>What H is conditioned on.</b> H takes the current point x<sub>t</sub> and the flow time t (through a "
        "sinusoidal embedding), and outputs one 2×2 matrix per chunk step, starting at the identity. It does not "
        "see the observation or the goal; those reach the adapted field only through P.",
        "<b>The goal is never used to infer the action from the operator's input.</b> At deployment the reversal "
        "and the forward pass both run the policy with every goal flag cleared. The goal enters only (i) the "
        "training pairs of one variant (goal-conditioned pairs) and (ii) the simulated user that stands in for a "
        "person in the closed-loop simulation (Table 1).",
        f"<b>Results.</b> Under a gain corruption diag(1, 0.25), the learned adapter matches the fixed MARS "
        f"adapter F = M at low noise and beats it at BCI noise 1.0 "
        f"({pct(r['gain025y | noise 1.0']['mars_free']['success'])} against "
        f"{pct(r['gain025y | noise 1.0']['mars_fixed']['success'])}); in the live setting (the expert pushing at "
        f"the goal, every method at its tuned depth) it reaches {pct(live['mars']['success'])}, against "
        f"{pct(live['teleop']['success'])} for teleop. Under a 90° rotation it fails, doing no better "
        f"than the policy alone.",
        "<b>Why it fails under rotation.</b> The regression's solution is the conditional-mean velocity of the "
        "coupling (eq. 11–12), which is shorter than any single a<sub>bci</sub> − a<sub>noise</sub>: restricted to "
        f"a constant matrix it learns {fmt_m(ca['rot90']['F'])}, about {rot_scale:.2f} × the 90° rotation. The reversal "
        "undershoots and no longer reaches the noise that encodes the intended target.",
    ])]

    # ------------------------------------------------------------ 1 policy
    s += [Paragraph("1. The policy and its flow", H1), P(
        "Actions are chunks a ∈ ℝ<super>L×2</super> of L = 16 per-step cursor displacements, in normalized units: "
        f"a = a<sub>screen</sub>/s<sub>a</sub> with s<sub>a</sub> = {s_a:.4f} screens per step (the RMS step of the "
        "demonstrations). An observation o = (p, h, T, m, g) holds the cursor position p, its last 4 displacements h, "
        "the target positions T, a mask m of real targets, and goal flags g (one-hot on the goal). o<super>∅</super> "
        "denotes the same observation with g = 0, the goal-free input used at deployment. A transformer v<sub>θ</sub> "
        "is trained by flow matching, with the goal flags cleared on "
        f"{100 * dropout:.0f}% of samples (goal dropout), so that one network serves both inputs:"),
        eq(r"x_t=(1-t)\,z+t\,a,\qquad z\sim\mathcal{N}(0,I),\qquad "
           r"\mathcal{L}(\theta)=\mathbb{E}\,\|v_\theta(x_t,t\,|\,o)-(a-z)\|^2"),
        P("Sampling integrates the learned field from noise (t = 0) to an action (t = 1) with N = 10 Euler steps; "
          "Φ(z; o) denotes this map:"),
        eq(r"x_0=z,\qquad x_{k+1}=x_k+\frac{1}{N}\,v_\theta\!\left(x_k,\frac{k}{N}\,|\,o\right),\qquad "
           r"\Phi(z;o)=x_N"),
        P("In closed loop the cursor executes the first 2 steps of a chunk and the policy plans again. The executed "
          "displacement is s<sub>a</sub>·a<sub>j</sub>·γ with speed gain γ = 0.5; the policy's history h stores "
          "the displacement divided by γ.")]

    # ------------------------------------------------------------ 2 operator / BCI
    s += [Paragraph("2. The operator and the BCI channel", H1), P(
        "Operator commands are deflections c ∈ [−1, 1]<super>2</super>; full deflection means v<sub>max</sub> = 1.2 "
        f"screens/s, i.e. a step of v<sub>max</sub>Δt = 0.024 screens, which is κ = {kappa:.2f} in normalized "
        "units. The BCI version of an intended chunk a is built step by step: convert to a command, apply the "
        "corruption matrix M, clip, add Gaussian noise of std σ, clip again, convert back:"),
        eq(r"c_j=a_j/\kappa,\qquad \hat c_j=\mathrm{clip}\left(\mathrm{clip}(M\,c_j)+\sigma\,\epsilon_j\right),"
           r"\qquad \epsilon_j\sim\mathcal{N}(0,I_2),\qquad a^{\mathrm{bci}}_j=\kappa\,\hat c_j"),
        P("clip(·) clips each component to [−1, 1]. Steps whose intended command is inside the deadband "
          "(‖c<sub>j</sub>‖<sub>∞</sub> &lt; 0.05) are passed through unchanged, as in the live input channel. "
          "We write a<sub>bci</sub> = f(a; M, σ). The corruptions studied are a rotation R(90°) = [[0, −1], [1, 0]] "
          "and a gain M = diag(1, 0.25)."),
        P("<b>In the live app</b> the operator gives one command per replan, not a chunk. The reference chunk is then "
          "constant: the served command ĉ = clip(clip(M c) + σε) repeated over the chunk,"),
        eq(r"r_j=\kappa\,\hat c\qquad (j=1,\ldots,L)")]

    # ------------------------------------------------------------ 3 FRS
    s += [Paragraph("3. Flow Reversal Steering with a fixed adapter", H1), P(
        "Flow Reversal Steering runs a reference chunk r backward through the goal-free field to a latent ẑ, then "
        "forward from ẑ. MARS multiplies the field by a fixed matrix F during the backward pass only "
        "(F = I is plain FRS; MARS uses F = M, the known corruption). All results here use the full reversal:"),
        eq(r"y_0=r,\qquad y_{k+1}=y_k-\frac{1}{N}\,F\,v_\theta\!\left(y_k,\,1-\frac{k}{N}\,|\,o^{\emptyset}\right),"
           r"\qquad \hat z=y_N,\qquad a=\Phi(\hat z;\,o^{\emptyset})")]

    # ------------------------------------------------------------ 4 learned adapter
    s += [Paragraph("4. The learned adapter", H1), Paragraph("Objective", H2), P(
        "The fixed F is replaced by a learned, state- and time-dependent adapter H<sub>φ</sub>. With "
        "P(x, t) = v<sub>θ</sub>(x, t | o<super>∅</super>) the frozen goal-free field, H is trained so that the "
        "adapted field moves along the straight path from a<sub>noise</sub> to a<sub>bci</sub>:"),
        eq(r"H_\phi(x_t,t)\,P(x_t,t)=a^{\mathrm{bci}}-a^{\mathrm{noise}},\qquad "
           r"x_t=(1-t)\,a^{\mathrm{noise}}+t\,a^{\mathrm{bci}}"),
        P("as the least-squares regression (mean over the L × 2 entries; H<sub>φ,j</sub> is the matrix for step j):"),
        eq(r"\mathcal{L}(\phi)=\mathbb{E}\left[\frac{1}{2L}\sum_{j=1}^{L}\left\|H_{\phi,j}(x_t,t)\,P_j(x_t,t)"
           r"-\left(a^{\mathrm{bci}}_j-a^{\mathrm{noise}}_j\right)\right\|^2\right]"),
        P("If H·P matched the path velocity exactly, integrating it backward from a<sub>bci</sub> would return "
          "a<sub>noise</sub>, and Φ(a<sub>noise</sub>; o<super>∅</super>) is the action the operator meant (for "
          "goal-free pairs, below)."),
        Paragraph("Training samples", H2), P(
        "Each sample is generated on the fly with the policy frozen:"),
        eq([r"o\sim D,\qquad a^{\mathrm{noise}}\sim\mathcal{N}(0,I),\qquad "
            r"a^{\mathrm{policy}}=\Phi(a^{\mathrm{noise}};\,o_{\mathrm{gen}}),",
            r"\sigma\sim\mathcal{U}(0,1),\qquad a^{\mathrm{bci}}=f(a^{\mathrm{policy}};M,\sigma),\qquad "
            r"t\sim\mathcal{U}(0,1)"]),
        bullets([
            f"<i>D</i>: states sampled uniformly over the time steps of {targs['n_demos']:,} synthetic demonstrations with "
            "1–8 targets (2,000 more for validation), the same distribution the policy was trained on.",
            "o<sub>gen</sub> = o<super>∅</super> for <b>goal-free pairs</b> (--data free): a<sub>noise</sub> is then "
            "exactly the latent from which the deployed, goal-free forward pass produces a<sub>policy</sub>.",
            "o<sub>gen</sub> = o (true goal flag set) for <b>goal-conditioned pairs</b> (--data goal), as first "
            "specified. a<sub>policy</sub> is then the goal-directed reach, but a<sub>noise</sub> is its latent "
            "under the goal-conditioned field, not the goal-free one used at deployment.",
            "The regression target and P always use o<super>∅</super>.",
        ]),
        Paragraph("Architecture and conditioning", H2), P(
        "H<sub>φ</sub> is a multilayer perceptron on the flattened point and a sinusoidal embedding of t (the same "
        "embedding the policy uses), whose 4L outputs are reshaped into L matrices and added to the identity:"),
        eq([r"H_\phi(x,t)=I+\mathrm{reshape}_{L\times2\times2}\,\mathrm{MLP}_\phi\left([\,\mathrm{vec}(x);"
            r"\,e(t)\,]\right),",
            r"e(t)=\left[\sin(1000\,t\,\omega_k),\,\cos(1000\,t\,\omega_k)\right]_{k=0}^{31},\qquad "
            r"\omega_k=10^{-4k/32}"]),
        bullets([
            "Inputs: x<sub>t</sub> (32 numbers) and t. No observation and no goal: H is conditioned on (x<sub>t</sub>, "
            "t) only; scene information reaches the adapted field through P.",
            "MLP: 96 → 256 → 256 → 64 with SiLU activations (107,072 parameters). The last layer is initialized to "
            "zero, so training starts from H ≡ I, i.e. plain FRS.",
            f"AdamW, learning rate {targs['lr']:g} with cosine decay, weight decay 10<super>−4</super>, batch "
            f"{targs['batch']}, {targs['steps']:,} steps (about 9 minutes on one RTX 6000 Ada); one adapter per "
            "corruption and pair type.",
        ]),
        Paragraph("Deployment", H2), P(
        "The adapter replaces F in the backward pass, evaluated at each step's flow time; the forward pass is the "
        "policy's own (eq. 10). In simulation the reference is the BCI chunk a<sub>bci</sub>; in the live app it is "
        "the constant chunk r of eq. 4."),
        eq(r"y_0=a^{\mathrm{bci}},\qquad y_{k+1}=y_k-\frac{1}{N}\,H_\phi(y_k,t_k)\,P(y_k,t_k),\quad "
           r"t_k=1-\frac{k}{N},\qquad a=\Phi(y_N;\,o^{\emptyset})")]

    # ------------------------------------------------------------ 5 goal usage
    s += [KeepTogether([Paragraph("5. Where the goal enters", H1), P(
        "The goal is used to describe what the operator wants, never to compute the action from their input:"),
        Spacer(1, 3), table([
            ["Step", "Goal flags", "Why"],
            ["Policy training (flow matching)", f"set, cleared on {100 * dropout:.0f}%", "one network, both inputs"],
            ["Training pairs, goal-conditioned variant", "set (true goal)", "a<sub>policy</sub> is a reach to the goal"],
            ["Training pairs, goal-free variant", "cleared", "latent consistent with deployment"],
            ["P in the training loss", "cleared", "the field used at deployment"],
            ["Simulated user, closed-loop simulation", "set", "stand-in for the operator's intent"],
            ["Expert agent, the live setting", "not used (pushes at the goal)", "stand-in for a person"],
            ["Reversal at deployment", "<b>cleared</b>", "the goal is unknown"],
            ["Forward pass at deployment", "<b>cleared</b>", "the goal is unknown"],
        ], col_widths=[2.6 * inch, 1.7 * inch, 2.3 * inch], shade=(7, 8)),
        Spacer(1, 3), Paragraph("<b>Table 1.</b> Goal conditioning at each stage.", CAP)])]

    # ------------------------------------------------------------ 6 results
    rows = [["Corruption", "BCI noise σ", "Oracle", "User alone", "Policy alone", "FRS", "MARS, F = M",
             "Learned H, goal pairs", "Learned H, goal-free pairs"]]
    for key, v in r.items():
        label, sig = key.split(" | noise ")
        rows.append([{"rot90": "rotation 90°", "gain025y": "gain diag(1, 0.25)"}[label], sig]
                    + [pct(v[m]["success"]) for m in ("oracle", "user", "policy", "frs", "mars_fixed",
                                                      "mars_goal", "mars_free")])
    lrows = [["Method (tuned depth)", "Success", "Wrong target", "Timeout", "Median time (s)"]]
    names = {"teleop": "teleop", "frs": "FRS", "mars_fixed": "MARS, F = M", "mars": "MARS"}
    for m, v in live.items():
        depth = f" ({live_depths[m]})" if live_depths[m] is not None else ""
        lrows.append([names[m] + depth, pct(v["success"]), pct(v["wrong"]), pct(v["timeout"]),
                      "—" if v["time_s"] is None else f"{v['time_s']:.2f}"])
    s += [Paragraph("6. Results", H1), Paragraph("Closed-loop simulation", H2), P(
        f"{ev['args']['trials']} trials per setting, 2–8 targets, replanning every 2 steps, speed gain 0.5, full "
        "reversal. At each replan the simulated user produces the goal-conditioned policy's chunk toward the true "
        "goal (its intent) and the BCI corrupts it (eq. 3). User alone executes a<sub>bci</sub>; the oracle executes "
        "the uncorrupted intent."),
        Spacer(1, 3), table(rows, font=7, col_widths=[1.1 * inch, 0.55 * inch] + [(PAGE_W - 1.65 * inch) / 7] * 7),
        Spacer(1, 3),
        Paragraph("<b>Table 2.</b> Closed-loop success. Adapters for each corruption were trained on that corruption "
                  "with σ drawn from [0, 1].", CAP), Spacer(1, 6),
        img(SWEEPS / "learned_adapter_rot90" / "figs" / "test.png", "Rotation 90°: closed-loop success against "
            "BCI noise on the test trials, every method at its tuned depth (Table 2 reports the full reversal).",
            max_h=2.6 * inch),
        img(SWEEPS / "learned_adapter_gain025y" / "figs" / "test.png", "Gain diag(1, 0.25): the same.",
            max_h=2.6 * inch),
        bullets([
            "<b>Gain:</b> the learned adapter matches F = M at σ ≤ 0.5 and beats it at σ = 1.0. F = M's chunks slow "
            f"down and time out there ({pct(r['gain025y | noise 1.0']['mars_fixed']['timeout'])} timeouts); the "
            "learned adapter's do not (0%), but it picks the wrong target in about half the trials.",
            "<b>Rotation:</b> the learned adapter performs at the level of the policy alone; F = M is the best "
            "steering method.",
            "Goal-conditioned and goal-free pairs differ by at most 3 points.",
        ]),
        Paragraph("The live setting", H2), P(
        "The portal's setting (gain diag(1, 0.25), input noise 1.0, constant reference chunks from one command "
        "per replan), with the expert agent pushing straight at the goal through the same input channel: the "
        "study-condition sweep's test trials, every method at the depth tuned for this noise. The adapter is "
        "the goal-free-pairs adapter trained on BCI chunks, not on constant references."),
        Spacer(1, 3), table(lrows, col_widths=[1.8 * inch, 1.1 * inch, 1.1 * inch, 1.1 * inch, 1.3 * inch]),
        Spacer(1, 3), Paragraph("<b>Table 3.</b> The live setting at input noise 1.0 (run_sweep.py --sweep "
                                "study_condition).", CAP), Spacer(1, 6),
        img(training_figure(logs), "Training. Left: the loss (eq. 7). Right: on held-out states, the mean cosine "
            "between the total displacement of the regenerated chunk Φ(ẑ; o<super>∅</super>) and of the intended "
            "chunk; dotted lines use the fixed F = M. These states are mostly mid-reach, where the target is already "
            "decided, so this metric is high for every adapter.", max_h=2.6 * inch)]

    # ------------------------------------------------------------ 7 analysis
    short = {"policy alone": "Policy alone", "FRS": "FRS", "MARS, F = M": "F = M",
             "constant F from the objective": "Constant F*", "learned H, goal pairs": "Learned H (goal pairs)",
             "learned H, goal-free pairs": "Learned H (goal-free pairs)",
             "learned H (goal-free pairs), norm-preserving": "Norm-kept H (goal-free pairs)"}
    rrows = [["Setting"] + [short[k] for k in next(iter(rs.values()))]]
    for key, v in rs.items():
        rrows.append([key.replace("rot90", "rotation 90°").replace("gain025y", "gain").replace(" | noise", ", σ")]
                     + [pct(x) for x in v.values()])
    s += [Paragraph("7. Why the learned adapter shrinks", H1), P(
        "Write u = a<sub>bci</sub> − a<sub>noise</sub>. Because H depends only on (x, t) while P also depends on "
        "the observation, the minimizer of eq. 7 at each (x, t) and step j is the least-squares matrix"),
        eq(r"H_j^{*}(x,t)=\mathbb{E}\left[u_j\,P_j^{\mathrm{T}}\,|\,x_t=x,t\right]\;"
           r"\mathbb{E}\left[P_j\,P_j^{\mathrm{T}}\,|\,x_t=x,t\right]^{-1}"),
        P("and wherever P is fixed by (x, t), the adapted field is the conditional mean of the path velocity, the "
          "marginal velocity of the coupling (a<sub>noise</sub>, a<sub>bci</sub>) as in flow matching:"),
        eq(r"H^{*}(x,t)\,P(x,t)=\mathbb{E}\left[\,a^{\mathrm{bci}}-a^{\mathrm{noise}}\,|\,x_t=x\,\right]"),
        P("Integrating the marginal velocity backward maps the distribution of a<sub>bci</sub> onto <i>N</i>(0, I), "
          "but it returns a given a<sub>bci</sub>'s own a<sub>noise</sub> only if the coupling is deterministic and "
          "its paths do not cross. Here it is far from that: σ adds noise, and many noise chunks lead to nearly the "
          "same action. On held-out states the latent error ‖ẑ − a<sub>noise</sub>‖² per entry is "
          f"{lat['learned'][0]:.2f}–{lat['learned'][1]:.2f} for the learned adapters, between the error of guessing "
          "zero (1.0) and of an independent random draw (2.0); it is "
          f"{lat['none'][0]:.0f}–{lat['none'][1]:.0f} without an adapter and {lat['fixed'][0]:.1f}–"
          f"{lat['fixed'][1]:.1f} with F = M. The reversal lands in a noise-like region, not on the specific "
          "latent. Given x<sub>t</sub>, the velocity averages over all compatible a<sub>noise</sub> and is shorter "
          "than each one: the reversal undershoots."),
        P("The same shrinkage shows when H is restricted to one constant matrix F, whose optimum is "
          "F* = E[u P<super>T</super>] E[P P<super>T</super>]<super>−1</super> (fitted with the same objective, goal-free pairs, 3,000 steps):"),
        Spacer(1, 3), table([["Corruption", "True M", "Fitted constant F*"],
                             ["rotation 90°", fmt_m(ca["rot90"]["M"]), fmt_m(ca["rot90"]["F"])],
                             ["gain diag(1, 0.25)", fmt_m(ca["gain025y"]["M"]), fmt_m(ca["gain025y"]["F"])]],
                            col_widths=[1.6 * inch, 2.3 * inch, 2.7 * inch]),
        Spacer(1, 3), Paragraph(f"<b>Table 4.</b> The objective recovers the direction of M at {rot_scale:.2f} "
                                f"(rotation) and {gain_scale:.2f} (gain) times its size.", CAP), Spacer(1, 6),
        P("Where it matters is the start of a reach, when the cursor is at rest and the target is chosen. Table 5 "
          "takes 2,000 layouts at their start, generates the intended chunk with the goal-conditioned policy, "
          "corrupts it, and checks which target the regenerated chunk heads for (the target whose direction best "
          "matches its total displacement). The norm-preserving variant rescales each step's adapted velocity "
          "H·P to the length of P."),
        Spacer(1, 3), KeepTogether([
            table(rrows, font=6.5, col_widths=[1.0 * inch] + [(PAGE_W - 1.0 * inch) / 7] * 7), Spacer(1, 3),
            Paragraph("<b>Table 5.</b> At the start of a reach: how often the regenerated first chunk heads to the "
                      "intended target. Norm-kept H rescales each step's adapted velocity H·P to the length of P.",
                      CAP)]), Spacer(1, 6),
        P("Under rotation, the learned adapters transfer no intent at the decision point (the policy alone gets "
          "the same rate by chance), while F = M does; restoring the velocity's length recovers part of the gap. "
          "Under the gain, plain FRS already transfers much of the intent and the learned adapter adds to it. "
          "A second reason the regression is aimed at the wrong place: most training states are mid-reach, where "
          "the cursor's velocity history already fixes the target.")]

    # ------------------------------------------------------------ 8 next steps
    s += [Paragraph("8. Limitations and next steps", H1), bullets([
        "<b>Train end to end on the regenerated action:</b> minimize ‖Φ(reverse<sub>H</sub>(a<sub>bci</sub>); "
        "o<super>∅</super>) − a<sub>policy</sub>‖², backpropagating through both flows (20 policy evaluations). This "
        "targets the action, not the latent, and avoids the conditional-mean shrinkage.",
        "<b>Constrain H's size</b>, e.g. a rotation with per-axis gains of fixed overall norm, so the regression "
        "cannot shrink it.",
        "<b>Condition H on the observation</b>, and oversample states at the start of a reach.",
        "<b>Train on the live reference</b> (constant chunks from one command per replan) rather than on BCI "
        "chunks, for the SpaceMouse app.",
        "All results use simulated users and one training seed per adapter; adapters assume the corruption "
        "family is known at training time.",
    ]), Paragraph("Reproducing", H2), P("From the project folder at git commit 5c2167cd (the adapter trainer has "
                                          "changed since), in the lerobot environment (uv run --extra mars):")]
    for line in ["python train/train_adapter.py --corruption '{rotation_deg: 90}' --data goal --name rot90_goal",
                 "python train/train_adapter.py --corruption '{rotation_deg: 90}' --data free --name rot90_free",
                 "python train/train_adapter.py --corruption '{scale: [1.0, 0.25]}' --data goal --name gain025y_goal",
                 "python train/train_adapter.py --corruption '{scale: [1.0, 0.25]}' --data free --name gain025y_free",
                 "python run_sweep.py --sweep learned_adapter_rot90",
                 "python run_sweep.py --sweep learned_adapter_gain025y",
                 "python run_sweep.py --sweep study_condition",
                 "python research/adapter_diagnostics.py",
                 "python run_experiment.py --arm mars_gain025y.yaml   # live",
                 "python reports/build_mars_adapter_report.py"]:
        s.append(Paragraph(line.replace(" ", "&nbsp;"), CODE))
    doc.build(s)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
