"""The study's analysis: the per-arm table and the paired tests between arms, from the run records.

Phase D's analysis (DESIGN.md, "Phase D: running a study"). ``analyse`` reads the ``trials.jsonl`` files
of one or more run directories -- or of one directory holding them -- groups the trials by ``(run, arm)``
and returns two tables; ``main`` prints them and writes them as CSV beside the runs. It needs no config,
no station, no checkpoint and no scipy: the three statistics are computed here, from the definitions.

**The per-arm table** is one row per ``(run, arm)``: ``n`` (the trials), the success count and rate with
its **Wilson 95% interval**, and the mean authority with its standard error over the trials that had an
engaged step (``n_authority``). The row names its ``n``; a column that could not be measured is empty,
not zero.

**The paired tests** compare every pair of arms within a run, on the trials that **matched**: same
``pair`` (the round) and same ``scene_hash`` (the reset observation's digest). A trial with no match is
excluded and counted in ``unmatched`` -- that is the honest caveat, because a station's ``reset()`` may
not restore the scene. Success is compared with **McNemar's exact two-sided test** (the binomial on the
discordant pairs, computed directly) and authority with the **Wilcoxon signed-rank test** (exact for
``n <= 20`` nonzero differences, the normal approximation with a continuity correction -- and a tie
correction -- beyond).

Records are refused, not guessed at: a field this analysis does not know (a legacy record, or a newer
writer's record) and a missing field are both an error naming the field, because silently ignoring a
column changes what the tables mean. The one shape this module trusts is ``study.py``'s; the writer's
field list is checked in the study's test suite.

Read the numbers as an apparatus check first and a study second: each row is one operator, a handful of
trials, and one scene. A Wilson interval at ``n = 10`` spans most of the range, and a paired test only
sees the discordant pairs, so a block's tables show whether a difference is worth a larger block; they
do not settle it.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

WILSON_Z = NormalDist().inv_cdf(0.975)
"""The two-sided 95% normal quantile (1.959963984540054), from the standard library."""

EXACT_WILCOXON_MAX_N = 20
"""The largest number of nonzero differences the exact signed-rank distribution is computed for."""

RECORD_FIELDS = (
    "run",
    "arm",
    "pair",
    "prompt",
    "success",
    "steps",
    "duration_s",
    "authority",
    "engaged_fraction",
    "mean_r",
    "q0",
    "scene_hash",
    "config",
    "config_sha256",
    "timestamp",
)
"""The fields a study record has (``study.py`` writes exactly these); anything else is a foreign record."""

PER_ARM_FIELDS = (
    "run",
    "arm",
    "n",
    "successes",
    "success_rate",
    "wilson_low",
    "wilson_high",
    "n_authority",
    "mean_authority",
    "authority_se",
)
"""The per-arm CSV's columns, in order."""

PAIRED_FIELDS = (
    "run",
    "a",
    "b",
    "n_matched",
    "unmatched",
    "success_a",
    "success_b",
    "a_only",
    "b_only",
    "discordant",
    "mcnemar_p",
    "n_authority_pairs",
    "authority_a",
    "authority_b",
    "authority_diff",
    "wilcoxon_p",
)
"""The paired-test CSV's columns, in order."""


@dataclass(frozen=True)
class Trial:
    """The fields of one record the analysis reads; the rest are provenance it does not interpret."""

    run: str
    arm: str
    pair: int
    success: bool
    authority: float | None
    scene_hash: str


# ---------------------------------------------------------------- the three statistics


def wilson(successes: int, n: int) -> tuple[float, float]:
    """The Wilson score 95% interval for ``successes / n``; ``(nan, nan)`` when ``n = 0``.

    Wilson's interval is the normal-approximation interval inverted on the logit-free score, so it
    stays inside [0, 1] and does not collapse at 0 or n successes the way the Wald interval does:
    with 0 of 10 successes the lower bound is 0 and the upper bound is still above 0.24.
    """
    if n <= 0:
        return (math.nan, math.nan)
    proportion = successes / n
    z_squared = WILSON_Z * WILSON_Z
    denominator = 1.0 + z_squared / n
    centre = (proportion + z_squared / (2.0 * n)) / denominator
    half = WILSON_Z / denominator * math.sqrt(proportion * (1.0 - proportion) / n + z_squared / (4.0 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar(a: Sequence[bool], b: Sequence[bool]) -> dict[str, Any]:
    """McNemar's exact two-sided test on paired successes ``a`` and ``b`` (same trial keys, same order).

    Only the discordant pairs carry information: ``a_only`` where ``a`` succeeded and ``b`` did not,
    ``b_only`` the reverse. Under the null they are fair coin flips, so the p-value is the two-sided
    binomial tail ``2 * P(X <= min(a_only, b_only) | n = discordant, p = 1/2)``, capped at 1. A table
    with no discordant pair gives ``p = 1`` (there is no evidence of a difference); the exact
    distribution is counted here rather than taken from scipy.
    """
    x = np.asarray(a, dtype=bool)
    y = np.asarray(b, dtype=bool)
    if x.shape != y.shape:
        raise ValueError(f"McNemar needs paired arrays of the same length, got {x.shape} and {y.shape}")
    a_only = int(np.sum(x & ~y))
    b_only = int(np.sum(~x & y))
    discordant = a_only + b_only
    if discordant == 0:
        p = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(a_only, b_only) + 1))
        p = min(1.0, 2.0 * tail / 2**discordant)
    return {"n": int(x.size), "a_only": a_only, "b_only": b_only, "discordant": discordant, "p": p}


def wilcoxon(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    """The Wilcoxon signed-rank test on paired authority values ``a`` and ``b``.

    Pairs with a missing value on either side drop out; zero differences drop out of the rank statistic
    (Wilcoxon's convention, and the only exact reading of "no change"). The statistic is
    ``min(W+, W-)``, and the test is two-sided: exact for at most ``EXACT_WILCOXON_MAX_N`` nonzero
    differences by counting the ``2**n`` sign assignments, else the normal approximation with a
    continuity correction of 0.5 and the tie correction to the variance. Returns the pair and
    difference counts it used, so a table can say how many pairs the ``p`` describes.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"Wilcoxon needs paired arrays of the same length, got {x.shape} and {y.shape}")
    both = np.isfinite(x) & np.isfinite(y)
    differences = (y - x)[both]
    nonzero = differences[differences != 0.0]
    n = int(nonzero.size)
    result: dict[str, Any] = {
        "pairs": int(both.sum()),
        "n": n,
        "w_plus": math.nan,
        "w_minus": math.nan,
        "p": math.nan,
    }
    if n == 0:
        return result
    ranks = _average_ranks(np.abs(nonzero))
    w_plus = float(ranks[nonzero > 0.0].sum())
    w_minus = float(ranks[nonzero < 0.0].sum())
    if n <= EXACT_WILCOXON_MAX_N:
        p = _exact_signed_rank_p(ranks, min(w_plus, w_minus))
    else:
        p = _normal_signed_rank_p(w_plus, ranks)
    result.update(w_plus=w_plus, w_minus=w_minus, p=p)
    return result


def _average_ranks(magnitudes: np.ndarray) -> np.ndarray:
    """Ranks 1..n of ``magnitudes``, ties sharing their average rank (1.5, 1.5, 3, ...)."""
    order = np.argsort(magnitudes, kind="stable")
    ranks = np.empty(len(magnitudes), dtype=np.float64)
    ordered = magnitudes[order]
    start = 0
    while start < len(ordered):
        end = start
        while end + 1 < len(ordered) and ordered[end + 1] == ordered[start]:
            end += 1
        ranks[order[start : end + 1]] = (start + end) / 2.0 + 1.0
        start = end + 1
    return ranks


def _exact_signed_rank_p(ranks: np.ndarray, observed: float) -> float:
    """The exact two-sided signed-rank p for ``min(W+, W-) = observed``.

    Every sign assignment is equally likely under the null, and ``W+`` is the sum of the ranks whose
    difference is positive: count the subsets of the rank multiset whose sum is at most ``observed``
    and double the tail. Average ranks are multiples of 1/2, so they are doubled and counted with
    integers, which makes the count exact.
    """
    scaled = [int(round(2.0 * rank)) for rank in ranks]
    threshold = int(round(2.0 * observed))
    counts: dict[int, int] = {0: 1}
    for value in scaled:
        extended = dict(counts)
        for total, ways in counts.items():
            extended[total + value] = extended.get(total + value, 0) + ways
        counts = extended
    tail = sum(ways for total, ways in counts.items() if total <= threshold)
    return min(1.0, 2.0 * tail / 2 ** len(scaled))


def _normal_signed_rank_p(w_plus: float, ranks: np.ndarray) -> float:
    """The normal approximation, with the continuity and tie corrections, for more than 20 differences."""
    n = len(ranks)
    mean = float(ranks.sum()) / 2.0
    ties = Counter(ranks.tolist())
    variance = n * (n + 1) * (2 * n + 1) / 24.0 - sum(t**3 - t for t in ties.values()) / 48.0
    if variance <= 0.0:
        return math.nan
    z = (abs(w_plus - mean) - 0.5) / math.sqrt(variance)
    if z <= 0.0:
        return 1.0
    return math.erfc(z / math.sqrt(2.0))


# ---------------------------------------------------------------- reading the records


def find_runs(paths: Sequence[Path]) -> list[Path]:
    """Every run directory behind ``paths``: a run itself, or a directory whose children are runs.

    A path that is a run directory (it holds ``trials.jsonl``) is kept as it is; a directory that holds
    such directories expands to them, sorted; anything else is refused, naming the path. Repeated
    runs (a parent and its child given together) are read once.
    """
    runs: list[Path] = []
    seen: set[Path] = set()
    for path in map(Path, paths):
        if (path / "trials.jsonl").is_file():
            found = [path]
        elif path.is_dir():
            found = sorted(
                child for child in path.iterdir() if child.is_dir() and (child / "trials.jsonl").is_file()
            )
            if not found:
                raise ValueError(f"{path}: no trials.jsonl and no run directory holding one under it")
        else:
            raise ValueError(f"{path}: no such run directory or parent directory")
        for run in found:
            resolved = run.resolve()
            if resolved not in seen:
                seen.add(resolved)
                runs.append(run)
    return runs


def load_trials(run_dirs: Sequence[Path]) -> list[Trial]:
    """The trials of ``run_dirs``, one per line: foreign fields, and duplicated pairing keys, are refused.

    A ``(run, arm, pair, scene_hash)`` key identifies one trial of one arm of one round. The study runs
    each arm once per round, so a duplicate means two records describe the same trial -- the pairing
    could not say which was which -- and the analysis refuses it instead of picking one.
    """
    trials: list[Trial] = []
    seen: dict[tuple[str, str, int, str], str] = {}
    for run_dir in run_dirs:
        path = Path(run_dir) / "trials.jsonl"
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as error:
            raise ValueError(f"{path}: cannot read the run's records: {error}") from error
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            where = f"{path}:{number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{where}: not a JSON record: {error}") from error
            _check_record(record, where)
            key = (record["run"], record["arm"], record["pair"], record["scene_hash"])
            if key in seen:
                raise ValueError(
                    f"{where}: arm {record['arm']!r} has two trials at pair {record['pair']} with the "
                    f"same scene (hash {record['scene_hash'][:12]}...; the first is {seen[key]}), so "
                    f"the pairing is ambiguous"
                )
            seen[key] = where
            trials.append(
                Trial(
                    run=record["run"],
                    arm=record["arm"],
                    pair=record["pair"],
                    success=record["success"],
                    authority=_optional_number(record["authority"]),
                    scene_hash=record["scene_hash"],
                )
            )
    return trials


def _check_record(record: Any, where: str) -> None:
    """Refuse a record this analysis cannot read whole: an unknown field, a missing one, a wrong type.

    A legacy record is a different protocol (the LIBERO study's fields, or an earlier draft's
    ``mean_|r|``); reading its rows as if the columns matched would silently change what the tables
    mean, so the analysis says which field it does not know and stops.
    """
    if not isinstance(record, dict):
        raise ValueError(f"{where}: expected a JSON object, got {type(record).__name__}")
    unknown = sorted(set(record) - set(RECORD_FIELDS))
    if unknown:
        raise ValueError(
            f"{where}: unknown record field {unknown[0]!r}; this analysis reads a Phase D study record "
            f"(fields: {', '.join(RECORD_FIELDS)}), so a legacy or newer record is refused rather than "
            f"silently ignored"
        )
    missing = [field for field in RECORD_FIELDS if field not in record]
    if missing:
        raise ValueError(
            f"{where}: missing record field {missing[0]!r}; a Phase D record has all of "
            f"{', '.join(RECORD_FIELDS)}"
        )
    for field in ("run", "arm", "scene_hash"):
        value = record[field]
        if not isinstance(value, str) or not value:
            raise ValueError(f"{where}: {field!r} must be a non-empty string, got {value!r}")
    if not isinstance(record["success"], bool):
        raise ValueError(f"{where}: 'success' must be true or false, got {record['success']!r}")
    if isinstance(record["pair"], bool) or not isinstance(record["pair"], int):
        raise ValueError(f"{where}: 'pair' must be an integer, got {record['pair']!r}")
    _optional_number(record["authority"], where, "authority")


def _optional_number(value: Any, where: str = "", field: str = "") -> float | None:
    """A number or None (NaN also reads as missing); anything else is refused by name."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: {field!r} must be a number or null, got {value!r}")
    return float(value) if math.isfinite(value) else None


# ---------------------------------------------------------------- the tables


def analyse(runs: Sequence[Path]) -> dict[str, Any]:
    """The analysis of ``runs``: which run directories were read and the two tables.

    ``runs`` may name run directories or directories holding them. The returned dict's ``per_arm`` and
    ``paired`` rows are CSV-shaped dicts with the column names above; the tables come back in the
    order the runs were given and the arms first appeared in them.
    """
    run_dirs = find_runs(runs)
    trials = load_trials(run_dirs)
    return {
        "run_dirs": [str(path) for path in run_dirs],
        "n_trials": len(trials),
        "per_arm": per_arm_table(trials),
        "paired": paired_table(trials),
    }


def per_arm_table(trials: Sequence[Trial]) -> list[dict[str, Any]]:
    """One row per ``(run, arm)``: n, the Wilson interval, and the authority's mean and standard error."""
    groups: dict[tuple[str, str], list[Trial]] = {}
    for trial in trials:
        groups.setdefault((trial.run, trial.arm), []).append(trial)
    rows = []
    for (run, arm), group in groups.items():
        n = len(group)
        successes = sum(1 for trial in group if trial.success)
        low, high = wilson(successes, n)
        authorities = [trial.authority for trial in group if trial.authority is not None]
        rows.append(
            {
                "run": run,
                "arm": arm,
                "n": n,
                "successes": successes,
                "success_rate": successes / n,
                "wilson_low": low,
                "wilson_high": high,
                "n_authority": len(authorities),
                "mean_authority": float(np.mean(authorities)) if authorities else None,
                "authority_se": _standard_error(authorities),
            }
        )
    return rows


def _standard_error(values: Sequence[float]) -> float | None:
    """The standard error of the mean: the sample standard deviation over sqrt(n); None below two values."""
    if len(values) < 2:
        return None
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def paired_table(trials: Sequence[Trial]) -> list[dict[str, Any]]:
    """The paired tests: every pair of arms, within a run, on the matched ``(pair, scene_hash)`` trials."""
    by_run: dict[str, list[Trial]] = {}
    for trial in trials:
        by_run.setdefault(trial.run, []).append(trial)
    rows = []
    for run, run_trials in by_run.items():
        arms = list(dict.fromkeys(trial.arm for trial in run_trials))
        for a, b in combinations(arms, 2):
            rows.append(_paired_row(run, run_trials, a, b))
    return rows


def _paired_row(run: str, trials: Sequence[Trial], a: str, b: str) -> dict[str, Any]:
    """One pair's matched trials, its discordant table and its two tests; unmatched trials are counted.

    Every rate and mean in the row is over the matched trials only: ``success_a``/``success_b``, the
    discordant ``a_only``/``b_only`` and the authority columns all describe the same trials, so a row
    is one comparison and ``n_matched`` is its size. ``unmatched`` counts the trials of the two arms
    that had no counterpart (both sides, so four trials that failed to pair contribute 4).
    """
    a_trials = [trial for trial in trials if trial.arm == a]
    b_trials = [trial for trial in trials if trial.arm == b]
    a_by = _by_pair(a_trials)
    b_by = _by_pair(b_trials)
    matched = sorted(key for key in a_by if key in b_by)
    n = len(matched)
    success_a = [a_by[key].success for key in matched]
    success_b = [b_by[key].success for key in matched]
    success = mcnemar(success_a, success_b)
    pairs = [
        (a_by[key].authority, b_by[key].authority)
        for key in matched
        if a_by[key].authority is not None and b_by[key].authority is not None
    ]
    signed_rank = wilcoxon([first for first, _ in pairs], [second for _, second in pairs])
    return {
        "run": run,
        "a": a,
        "b": b,
        "n_matched": n,
        "unmatched": len(a_trials) - n + (len(b_trials) - n),
        "success_a": _mean([1.0 if value else 0.0 for value in success_a]),
        "success_b": _mean([1.0 if value else 0.0 for value in success_b]),
        "a_only": success["a_only"],
        "b_only": success["b_only"],
        "discordant": success["discordant"],
        "mcnemar_p": success["p"],
        "n_authority_pairs": signed_rank["pairs"],
        "authority_a": _mean([first for first, _ in pairs]),
        "authority_b": _mean([second for _, second in pairs]),
        "authority_diff": _mean([second - first for first, second in pairs]),
        "wilcoxon_p": signed_rank["p"],
    }


def _by_pair(trials: Sequence[Trial]) -> dict[tuple[int, str], Trial]:
    """The trials keyed ``(pair, scene_hash)``; a duplicated key is an ambiguous pairing, refused."""
    keyed: dict[tuple[int, str], Trial] = {}
    for trial in trials:
        key = (trial.pair, trial.scene_hash)
        if key in keyed:
            raise ValueError(
                f"run {trial.run!r}, arm {trial.arm!r}: two trials started from pair {trial.pair} and "
                f"the same scene (hash {trial.scene_hash[:12]}...), so the pairing is ambiguous"
            )
        keyed[key] = trial
    return keyed


def _mean(values: Sequence[float]) -> float | None:
    """The mean of a non-empty list, else None."""
    return float(np.mean(values)) if values else None


# ---------------------------------------------------------------- the command


def main(argv: list[str] | None = None) -> int:
    """``mars-franka analyse --run DIR [--run DIR ...]``: print the tables and write them as CSV."""
    parser = argparse.ArgumentParser(
        prog="mars-franka analyse",
        description="Read a study's run records and print (and write) the per-arm and paired tables.",
    )
    parser.add_argument(
        "--run",
        action="append",
        type=Path,
        required=True,
        metavar="DIR",
        dest="runs",
        help="a run directory, or a directory holding run directories (repeatable)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="DIR",
        help="where the CSVs go (default: the directory containing the run directories)",
    )
    args = parser.parse_args(argv)
    run_dirs = find_runs(args.runs)
    result = analyse(run_dirs)
    _print_tables(result)
    output = _output_directory(args.out, run_dirs)
    per_arm, paired = _write_csv(output, result)
    print(f"csv: {per_arm}, {paired}", flush=True)
    return 0


def _output_directory(explicit: Path | None, run_dirs: Sequence[Path]) -> Path:
    """The CSV directory: ``--out``, else the parents' common directory, else the caller's directory."""
    if explicit is not None:
        return Path(explicit)
    parents = {str(Path(run).resolve().parent) for run in run_dirs}
    if len(parents) == 1:
        return Path(parents.pop())
    return Path.cwd()


def _print_tables(result: dict[str, Any]) -> None:
    """The tables as text: every row names its n, and a column that could not be measured says so."""
    print(f"runs: {len(result['run_dirs'])}, trials: {result['n_trials']}", flush=True)
    _print_per_arm(result["per_arm"])
    _print_paired(result["paired"])


def _print_per_arm(rows: Sequence[dict[str, Any]]) -> None:
    print(
        "\nper arm (n = trials; success with its Wilson 95% interval; "
        "authority = mean +/- s.e. over trials with an engaged step)",
        flush=True,
    )
    if not rows:
        print("no trials", flush=True)
        return
    print(f"{'run':<28} {'arm':<14} {'n':>4} {'success':>8} {'95% Wilson':>19} {'authority':>19}", flush=True)
    for row in rows:
        interval = f"[{row['wilson_low']:.3f}, {row['wilson_high']:.3f}]" if row["n"] else ""
        authority = _authority_text(row["mean_authority"], row["authority_se"], row["n_authority"])
        print(
            f"{row['run']:<28} {row['arm']:<14} {row['n']:>4} {row['success_rate']:>8.3f} "
            f"{interval:>19} {authority:>19}",
            flush=True,
        )


def _authority_text(mean: float | None, error: float | None, n: int) -> str:
    """One authority cell: ``mean +/- s.e. (n=...)``, or why it is not measurable."""
    if mean is None:
        return f"n/a (n={n})"
    if error is None:
        return f"{mean:.3f} (n={n})"
    return f"{mean:.3f} +/- {error:.3f} (n={n})"


def _print_paired(rows: Sequence[dict[str, Any]]) -> None:
    print(
        "\npaired tests (matched on pair and scene_hash within a run; unmatched trials excluded)",
        flush=True,
    )
    if not rows:
        print("no pair of arms to compare", flush=True)
        return
    print(
        f"{'run':<28} {'a':<10} {'b':<10} {'n':>3} {'unmatched':>9} {'a-only':>6} {'b-only':>6} "
        f"{'McNemar p':>10} {'auth. diff':>11} {'Wilcoxon p':>11}",
        flush=True,
    )
    for row in rows:
        difference = "" if row["authority_diff"] is None else f"{row['authority_diff']:+.3f}"
        print(
            f"{row['run']:<28} {row['a']:<10} {row['b']:<10} {row['n_matched']:>3} "
            f"{row['unmatched']:>9} {row['a_only']:>6} {row['b_only']:>6} "
            f"{_p_text(row['mcnemar_p']):>10} {difference:>11} {_p_text(row['wilcoxon_p']):>11}",
            flush=True,
        )


def _p_text(value: float) -> str:
    """A p-value in a table cell: three significant digits, empty when the test does not exist."""
    return "" if value is None or not math.isfinite(value) else f"{value:.3g}"


def _write_csv(output: Path, result: dict[str, Any]) -> tuple[Path, Path]:
    """Write ``per_arm.csv`` and ``paired.csv`` under ``output``; return their paths."""
    output.mkdir(parents=True, exist_ok=True)
    per_arm = output / "per_arm.csv"
    _write_rows(per_arm, PER_ARM_FIELDS, result["per_arm"])
    paired = output / "paired.csv"
    _write_rows(paired, PAIRED_FIELDS, result["paired"])
    return per_arm, paired


def _write_rows(path: Path, fields: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    """One CSV table: the columns in ``fields`` order, a missing number as an empty cell."""
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row[field]) for field in fields})


def _csv_value(value: Any) -> Any:
    """A CSV cell: None and non-finite numbers are empty, everything else is written as it is."""
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value
