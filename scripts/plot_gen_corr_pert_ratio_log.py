#!/usr/bin/env python3
"""Paper-quality plots of the GEN correction/perturbation repair ratio (r_repair)
per training round, parsed from console_*minibatch* logs.

One figure is produced per architecture pair (attack / defense), inferred from the
log file name. Series within a figure are distinguished by their d_mb and alpha_KD
values (also read from the file name).
"""
import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# ── naming conventions ───────────────────────────────────────────────────────
_ARCH_TOKENS = frozenset({"dcgan", "dc", "sigmoid", "tanh"})
_ARCH_NORM = {"dc": "dcgan"}
# Display names for the title (Attack / Defense)
_ARCH_DISPLAY = {"tanh": "TEST-TANH", "sigmoid": "TEST-SIGMOID", "dcgan": "DC-GAN"}


def _arch_tokens(name: str) -> list[str]:
    """Return the (normalized) last two architecture tokens found in name."""
    arches: list[str] = []
    for tok in reversed(name.lower().split("_")):
        if tok in _ARCH_TOKENS:
            arches.insert(0, _ARCH_NORM.get(tok, tok))
            if len(arches) == 2:
                break
    return arches


def extract_arch_pair(name: str) -> str:
    """Return normalized 'arch1_arch2' (attack_defense) extracted from name."""
    arches = _arch_tokens(name)
    return "_".join(arches) if len(arches) == 2 else "unknown"


def _parse_frac(tok: str) -> float | None:
    """Interpret a filename numeric token as a fraction: '0'->0.0, '02'->0.2, '07'->0.7."""
    if tok == "0":
        return 0.0
    if tok.startswith("0") and len(tok) > 1 and tok.isdigit():
        return float("0." + tok[1:])
    try:
        return float(tok)
    except ValueError:
        return None


def parse_meta(name: str) -> dict:
    """Extract constraint type, attack/defense arch and (d_mb, alpha_KD) from a log name."""
    lname = name.lower()
    constraint = "Unconstrained" if "unconstrained" in lname else "Constrained"
    arches = _arch_tokens(name)
    attack = _ARCH_DISPLAY.get(arches[0], arches[0]) if len(arches) >= 1 else "?"
    defense = _ARCH_DISPLAY.get(arches[1], arches[1]) if len(arches) >= 2 else "?"

    d_mb = alpha_kd = None
    tokens = lname.split("_")
    if "minibatch" in tokens:
        nums: list[float] = []
        for tok in tokens[tokens.index("minibatch") + 1:]:
            val = _parse_frac(tok)
            if val is None:
                break
            nums.append(val)
            if len(nums) == 2:
                break
        if len(nums) >= 1:
            d_mb = nums[0]
        if len(nums) >= 2:
            alpha_kd = nums[1]
    return {
        "constraint": constraint,
        "attack": attack,
        "defense": defense,
        "d_mb": d_mb,
        "alpha_kd": alpha_kd,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the GEN correction/perturbation repair ratio per round "
            "(log scale) from console_*minibatch* logs, one figure per arch pair."
        )
    )
    parser.add_argument(
        "--run-results-path",
        default="/cephyr/users/garciafe/temp/CIFAR10/run_results",
        help="Directory containing console_*minibatch*.log files.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Base output PDF path; the arch pair is appended per figure.",
    )
    parser.add_argument(
        "--exclude-substr",
        action="append",
        dest="exclude_substrs",
        metavar="SUBSTR",
        help="Exclude logs whose filename contains SUBSTR (repeatable). "
             "Default when not specified: minibatch_discrimination.",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.9,
        help="Quantile to plot per round (0-1). Default is 0.9 (p90).",
    )
    parser.add_argument(
        "--no-attack-gen",
        action="store_true",
        help="Omit the attack-GEN loss overlay (secondary axis).",
    )
    parser.add_argument("--fig-width", type=float, default=7.5, help="Figure width (inches).")
    parser.add_argument("--fig-height", type=float, default=4.8, help="Figure height (inches).")
    parser.add_argument("--dpi", type=int, default=220, help="Output resolution.")
    return parser.parse_args()


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)
    vals = sorted(values)
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def build_series(log_paths: list[Path], q: float) -> list[tuple[str, list[tuple[int, float]]]]:
    stage_re = re.compile(
        r"ID:\s*(\d+), stage:\s*(pre|perturbation|correction)\s*, "
        r"Global/GEN loss:\s*\[[0-9.eE+-]+,\s*([0-9.eE+-]+)\]"
    )
    round_re = re.compile(r"fit progress:\s*\((\d+),")

    series_by_file: list[tuple[str, list[tuple[int, float]]]] = []

    for path in log_paths:
        blocks: list[dict[str, object]] = []
        current_block: dict[str, object] | None = None

        with path.open("r", errors="ignore") as handle:
            for line in handle:
                if "Evaluating MAL clients across stages" in line:
                    current_block = {"round": None, "clients": {}}
                    blocks.append(current_block)
                    continue

                match_round = round_re.search(line)
                if match_round and blocks:
                    if blocks[-1]["round"] is None:
                        blocks[-1]["round"] = int(match_round.group(1))
                    continue

                match_stage = stage_re.search(line)
                if not match_stage or current_block is None:
                    continue

                client_id = int(match_stage.group(1))
                stage = match_stage.group(2)
                gen_loss = float(match_stage.group(3))
                clients = current_block["clients"]
                if isinstance(clients, dict):
                    clients.setdefault(client_id, {})[stage] = gen_loss

        series: list[tuple[int, float]] = []
        for idx, block in enumerate(blocks):
            ratios: list[float] = []
            clients = block.get("clients", {})
            if not isinstance(clients, dict):
                continue
            for stages in clients.values():
                if not isinstance(stages, dict):
                    continue
                if "perturbation" in stages and "correction" in stages:
                    pert = stages["perturbation"]
                    corr = stages["correction"]
                    if pert > 0:
                        ratios.append(corr / pert)
            if ratios:
                q_value = quantile(ratios, q)
                if q_value is None:
                    continue
                round_value = block.get("round")
                x = round_value if isinstance(round_value, int) else idx
                series.append((x, q_value))

        if series:
            series.sort(key=lambda item: item[0])
            name = path.stem.replace("console_", "")
            series_by_file.append((name, series))

    return series_by_file


def build_attack_gen_series(log_paths: list[Path], q: float) -> list[tuple[str, list[tuple[int, float]]]]:
    """Parse Attack gen_updates average losses, anchored to the following fit-progress round."""
    round_re = re.compile(r"fit progress:\s*\((\d+),")
    attack_gen_re = re.compile(r"Attack gen_updates: average loss \[([^\]]*)\]")

    series_by_file: list[tuple[str, list[tuple[int, float]]]] = []

    for path in log_paths:
        round_losses: dict[int, list[float]] = {}
        pending: list[float] = []

        with path.open("r", errors="ignore") as handle:
            for line in handle:
                m_atk = attack_gen_re.search(line)
                if m_atk:
                    for token in m_atk.group(1).strip().split(","):
                        token = token.strip().strip("'\"")
                        if token:
                            try:
                                pending.append(float(token))
                            except ValueError:
                                pass
                    continue

                m_round = round_re.search(line)
                if m_round and pending:
                    rnd = int(m_round.group(1))
                    round_losses.setdefault(rnd, []).extend(pending)
                    pending = []

        series: list[tuple[int, float]] = []
        for rnd, losses in sorted(round_losses.items()):
            q_val = quantile(losses, q)
            if q_val is not None and q_val > 0:
                series.append((rnd, q_val))

        if series:
            name = path.stem.replace("console_", "")
            series_by_file.append((name, series))

    return series_by_file


def _group_by_arch(
    series_list: list[tuple[str, list[tuple[int, float]]]],
) -> dict[str, list[tuple[str, list[tuple[int, float]]]]]:
    groups: dict[str, list] = defaultdict(list)
    for name, series in series_list:
        groups[extract_arch_pair(name)].append((name, series))
    return groups


def _decade_limits(values: list[float]) -> tuple[float, float]:
    """Return (lo, hi) snapped to surrounding powers of ten for a log axis."""
    pos = [v for v in values if v > 0]
    if not pos:
        return 1e-3, 1e0
    lo = 10 ** math.floor(math.log10(min(pos)))
    hi = 10 ** math.ceil(math.log10(max(pos)))
    if lo == hi:
        lo /= 10
        hi *= 10
    return lo, hi


def _series_label(meta: dict, show_constraint: bool) -> str:
    """Legend entry showing d_mb and alpha_KD (and constraint if the panel mixes them)."""
    bits = []
    if meta["d_mb"] is not None:
        bits.append(rf"$d_{{\mathrm{{mb}}}}={meta['d_mb']:g}$")
    if meta["alpha_kd"] is not None:
        bits.append(rf"$\alpha_{{\mathrm{{KD}}}}={meta['alpha_kd']:g}$")
    label = ", ".join(bits) if bits else "?"
    if show_constraint:
        label = f"{meta['constraint']}  " + label
    return label


def _setup_style() -> None:
    plt.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "0.7",
        "savefig.bbox": "tight",
    })


def draw_figure(
    series_by_file: list[tuple[str, list[tuple[int, float]]]],
    attack_gen_series_by_file: list[tuple[str, list[tuple[int, float]]]],
    *,
    arch_pair: str,
    quantile_value: float,
    y_limits: tuple[float, float],
    atk_y_limits: tuple[float, float] | None,
    x_limits: tuple[int, int],
    fig_width: float,
    fig_height: float,
    dpi: int,
    output_path: Path,
) -> None:
    metas = {name: parse_meta(name) for name, _ in series_by_file}
    constraints = {m["constraint"] for m in metas.values()}
    show_constraint = len(constraints) > 1

    # Stable ordering: by constraint, then d_mb, then alpha_KD
    ordered = sorted(
        series_by_file,
        key=lambda ns: (
            metas[ns[0]]["constraint"],
            metas[ns[0]]["d_mb"] if metas[ns[0]]["d_mb"] is not None else -1,
            metas[ns[0]]["alpha_kd"] if metas[ns[0]]["alpha_kd"] is not None else -1,
        ),
    )
    color_for = {name: f"C{i % 10}" for i, (name, _) in enumerate(ordered)}
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

    y_floor = y_limits[0]
    for i, (name, series) in enumerate(ordered):
        xs = [x for x, y in series if y > 0]
        ys = [y for x, y in series if y > 0]
        if not xs:
            continue
        color = color_for[name]
        ax.plot(
            xs, ys,
            color=color, lw=1.8,
            marker=markers[i % len(markers)], markersize=4,
            markeredgecolor="white", markeredgewidth=0.4,
            label=_series_label(metas[name], show_constraint),
            zorder=3,
        )
        ax.fill_between(xs, ys, y_floor, color=color, alpha=0.13, lw=0, zorder=1)

    ax.set_yscale("log")
    ax.set_ylim(*y_limits)
    ax.set_xlim(x_limits[0], x_limits[1])
    ax.set_ylabel(r"$r_{\mathrm{repair}}$")
    ax.set_xlabel("Training round")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=9))
    ax.grid(True, which="major", ls="-", lw=0.45, color="0.75", alpha=0.7)
    ax.grid(True, which="minor", ls=":", lw=0.35, color="0.85", alpha=0.6)
    ax.tick_params(which="both", top=False, right=False)

    # ── optional attack-GEN overlay on a secondary axis ──────────────────────
    atk_handles: list[Line2D] = []
    if attack_gen_series_by_file and atk_y_limits is not None:
        ax2 = ax.twinx()
        ax2.set_yscale("log")
        ax2.set_ylim(*atk_y_limits)
        ax2.set_ylabel(r"attack-GEN loss $\mathcal{L}_{\mathrm{atk}}$", color="0.35")
        ax2.tick_params(axis="y", colors="0.35")
        atk_metas = {name: parse_meta(name) for name, _ in attack_gen_series_by_file}
        for name, series in attack_gen_series_by_file:
            xs = [x for x, y in series if y > 0]
            ys = [y for x, y in series if y > 0]
            if not xs:
                continue
            color = color_for.get(name, "0.4")
            ax2.plot(
                xs, ys, color=color, lw=1.4, ls="--",
                marker="o", markersize=3.5, markerfacecolor="none",
                markeredgewidth=1.0, zorder=2,
            )
        atk_handles = [Line2D([0], [0], color="0.35", lw=1.4, ls="--",
                              marker="o", markerfacecolor="none",
                              label="attack-GEN loss (right axis)")]

    # ── title: Attack / Defense / constraint ─────────────────────────────────
    sample = next(iter(metas.values())) if metas else {"attack": "?", "defense": "?"}
    constraint_str = sorted(constraints)[0] if len(constraints) == 1 else "Constrained + Unconstrained"
    title = (
        f"Attack: {sample['attack']}     "
        f"Defense: {sample['defense']}     "
        f"{constraint_str}"
    )
    ax.set_title(title, pad=10)

    handles, labels = ax.get_legend_handles_labels()
    handles += atk_handles
    labels += [h.get_label() for h in atk_handles]
    if handles:
        ax.legend(handles, labels, loc="best", handlelength=2.0, borderpad=0.6,
                  labelspacing=0.4)

    fig.savefig(str(output_path))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    _setup_style()

    run_results = Path(args.run_results_path)
    exclude_substrs = args.exclude_substrs or ["minibatch_discrimination"]
    log_paths = sorted(
        p for p in run_results.glob("*_minibatch*.log")
        if not any(s in p.name for s in exclude_substrs)
    )
    if not log_paths:
        raise SystemExit("No console_*minibatch*.log files found.")

    series_by_file = build_series(log_paths, args.quantile)
    if not series_by_file:
        raise SystemExit("No GEN corr/pert ratio series found in logs.")

    attack_gen_series_by_file = (
        [] if args.no_attack_gen else build_attack_gen_series(log_paths, args.quantile)
    )

    if args.output:
        base_output = Path(args.output)
    else:
        base_output = run_results / "r_repair_per_round.pdf"
    if base_output.suffix.lower() != ".pdf":
        base_output = base_output.with_suffix(".pdf")
    base_output.parent.mkdir(parents=True, exist_ok=True)

    # Global limits so every per-arch figure shares the same axes (comparable).
    all_y = [y for _, s in series_by_file for _, y in s if y > 0]
    all_x = [x for _, s in series_by_file for x, _ in s]
    y_limits = _decade_limits(all_y)
    x_limits = (min(all_x), max(all_x))
    atk_y_limits = None
    if attack_gen_series_by_file:
        atk_all_y = [y for _, s in attack_gen_series_by_file for _, y in s if y > 0]
        atk_y_limits = _decade_limits(atk_all_y)

    main_groups = _group_by_arch(series_by_file)
    atk_groups = _group_by_arch(attack_gen_series_by_file)

    stem_has_constraint = "constrained" in base_output.stem.lower()

    for arch_pair in sorted(main_groups):
        main_sub = main_groups[arch_pair]
        atk_sub = atk_groups.get(arch_pair, [])

        # Tag the output with the group's constraint type (constrained/unconstrained/mixed),
        # unless the base name already encodes it.
        group_constraints = {parse_meta(name)["constraint"] for name, _ in main_sub}
        if len(group_constraints) == 1:
            constraint_token = next(iter(group_constraints)).lower()
        else:
            constraint_token = "mixed"
        parts = [base_output.stem]
        if not stem_has_constraint:
            parts.append(constraint_token)
        parts.append(arch_pair)
        out = base_output.with_name("_".join(parts) + base_output.suffix)
        draw_figure(
            main_sub,
            atk_sub,
            arch_pair=arch_pair,
            quantile_value=args.quantile,
            y_limits=y_limits,
            atk_y_limits=atk_y_limits,
            x_limits=x_limits,
            fig_width=args.fig_width,
            fig_height=args.fig_height,
            dpi=args.dpi,
            output_path=out,
        )
        print(str(out))


if __name__ == "__main__":
    main()
