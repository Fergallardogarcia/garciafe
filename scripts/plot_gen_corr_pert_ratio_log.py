#!/usr/bin/env python3
import argparse
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot GEN correction/perturbation loss ratio per round "
            "(log scale) from console_minibatch logs."
        )
    )
    parser.add_argument(
        "--run-results-path",
        default="/cephyr/users/garciafe/temp/CIFAR10/run_results",
        help="Directory containing console_minibatch_*.log files.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output PNG path. Defaults to run-results-path with a standard name.",
    )
    parser.add_argument(
        "--exclude-substr",
        default="minibatch_discrimination",
        help="Exclude logs containing this substring.",
    )
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=950)
    parser.add_argument("--left", type=int, default=150)
    parser.add_argument("--right", type=int, default=40)
    parser.add_argument("--top", type=int, default=60)
    parser.add_argument("--bottom", type=int, default=120)
    parser.add_argument("--xticks", type=int, default=7)
    parser.add_argument("--log-base", type=int, default=10)
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.9,
        help="Quantile to plot per round (0-1). Default is 0.9 (p90).",
    )
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


def draw_plot(
    series_by_file: list[tuple[str, list[tuple[int, float]]]],
    width: int,
    height: int,
    left: int,
    right: int,
    top: int,
    bottom: int,
    xticks: int,
    log_base: int,
    quantile_value: float,
) -> Image.Image:
    bg_color = (255, 255, 255)
    axis_color = (0, 0, 0)
    text_color = (0, 0, 0)
    grid_color = (220, 220, 220)
    colors = [
        (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
        (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
        (188, 189, 34), (23, 190, 207)
    ]

    all_x = [x for _, series in series_by_file for x, _ in series]
    all_y = [y for _, series in series_by_file for _, y in series if y > 0]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    if min_x == max_x:
        min_x -= 1
        max_x += 1

    log_min_y = math.log(min_y, log_base)
    log_max_y = math.log(max_y, log_base)
    if log_min_y == log_max_y:
        log_min_y -= 0.5
        log_max_y += 0.5

    plot_left = left
    plot_right = width - right
    plot_top = top
    plot_bottom = height - bottom

    def map_x(x: float) -> float:
        return plot_left + (x - min_x) * (plot_right - plot_left) / (max_x - min_x)

    def map_y(y: float) -> float:
        return plot_bottom - (math.log(y, log_base) - log_min_y) * (plot_bottom - plot_top) / (log_max_y - log_min_y)

    image = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    def text_size(text: str) -> tuple[int, int]:
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            return font.getsize(text)

    # X ticks
    for i in range(xticks):
        tx = min_x + i * (max_x - min_x) / (xticks - 1)
        px = map_x(tx)
        draw.line([(px, plot_bottom), (px, plot_bottom + 6)], fill=axis_color, width=1)
        draw.text((px - 8, plot_bottom + 12), f"{int(round(tx))}", fill=text_color, font=font)

    # Y ticks at decades with subticks (2-9)
    min_dec = int(math.floor(log_min_y))
    max_dec = int(math.ceil(log_max_y))
    minor_grid = (235, 235, 235)
    for dec in range(min_dec, max_dec + 1):
        # Major tick at 10^dec
        val = log_base ** dec
        if min_y <= val <= max_y:
            py = map_y(val)
            draw.line([(plot_left - 6, py), (plot_left, py)], fill=axis_color, width=1)
            draw.text((plot_left - 90, py - 4), f"{val:g}", fill=text_color, font=font)
            draw.line([(plot_left, py), (plot_right, py)], fill=grid_color, width=1)

        # Minor ticks within the decade
        for mult in range(2, 10):
            val = mult * (log_base ** dec)
            if val < min_y or val > max_y:
                continue
            py = map_y(val)
            draw.line([(plot_left - 3, py), (plot_left, py)], fill=axis_color, width=1)
            draw.line([(plot_left, py), (plot_right, py)], fill=minor_grid, width=1)

    # Axes
    draw.line([(plot_left, plot_top), (plot_left, plot_bottom)], fill=axis_color, width=1)
    draw.line([(plot_left, plot_bottom), (plot_right, plot_bottom)], fill=axis_color, width=1)

    # Labels
    draw.text((plot_left, plot_bottom + 40), "round", fill=text_color, font=font)
    percentile_label = int(round(quantile_value * 100))
    draw.text(
        (plot_left - 130, plot_top - 25),
        f"GEN corr/pert ratio (p{percentile_label}, log scale)",
        fill=text_color,
        font=font,
    )
    draw.text(
        (plot_left, 15),
        f"GEN loss correction vs perturbation ratio per round (p{percentile_label}, log scale)",
        fill=text_color,
        font=font,
    )

    # Series
    for idx, (name, series) in enumerate(series_by_file):
        color = colors[idx % len(colors)]
        prev = None
        for x, y in series:
            if y <= 0:
                continue
            px = map_x(x)
            py = map_y(y)
            if prev is not None:
                draw.line([prev, (px, py)], fill=color, width=2)
            draw.ellipse([(px - 2, py - 2), (px + 2, py + 2)], fill=color, outline=color)
            prev = (px, py)

    # Legend (top-right inside plot area)
    legend_pad = 8
    legend_item_h = 14
    legend_marker = 8
    legend_items = [(name, colors[idx % len(colors)]) for idx, (name, _) in enumerate(series_by_file)]
    max_label_w = 0
    for name, _ in legend_items:
        w, _ = text_size(name)
        max_label_w = max(max_label_w, w)
    legend_w = legend_pad * 3 + legend_marker + max_label_w
    legend_h = legend_pad * 2 + legend_item_h * len(legend_items)
    legend_x = plot_right - legend_w - 10
    legend_y = plot_top + 10

    draw.rectangle(
        [legend_x, legend_y, legend_x + legend_w, legend_y + legend_h],
        fill=(255, 255, 255),
        outline=(0, 0, 0),
        width=1,
    )

    for idx, (name, color) in enumerate(legend_items):
        y = legend_y + legend_pad + idx * legend_item_h
        x0 = legend_x + legend_pad
        y_mid = y + legend_item_h // 2
        draw.line([(x0, y_mid), (x0 + legend_marker, y_mid)], fill=color, width=2)
        draw.ellipse(
            [(x0 + legend_marker - 3, y_mid - 3), (x0 + legend_marker + 3, y_mid + 3)],
            fill=color,
            outline=color,
        )
        draw.text((x0 + legend_marker + legend_pad, y), name, fill=color, font=font)

    return image


def main() -> None:
    args = parse_args()
    run_results = Path(args.run_results_path)
    log_paths = sorted(
        p for p in run_results.glob("*_minibatch*.log")
        if args.exclude_substr not in p.name
    )
    if not log_paths:
        raise SystemExit("No console_minibatch_*.log files found.")

    series_by_file = build_series(log_paths, args.quantile)
    if not series_by_file:
        raise SystemExit("No GEN corr/pert ratio series found in logs.")

    image = draw_plot(
        series_by_file=series_by_file,
        width=args.width,
        height=args.height,
        left=args.left,
        right=args.right,
        top=args.top,
        bottom=args.bottom,
        xticks=args.xticks,
        log_base=args.log_base,
        quantile_value=args.quantile,
    )

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = run_results / "filter_gen_corr_pert_ratio_trends_minibatch_axes_no_discrim_log.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(output_path))
    print(str(output_path))


if __name__ == "__main__":
    main()
