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
        default="minibatch_discrim",
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
    return parser.parse_args()


def build_series(log_paths: list[Path]) -> list[tuple[str, list[tuple[int, float]]]]:
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
                round_value = block.get("round")
                x = round_value if isinstance(round_value, int) else idx
                series.append((x, sum(ratios) / len(ratios)))

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

    # X ticks
    for i in range(xticks):
        tx = min_x + i * (max_x - min_x) / (xticks - 1)
        px = map_x(tx)
        draw.line([(px, plot_bottom), (px, plot_bottom + 6)], fill=axis_color, width=1)
        draw.text((px - 8, plot_bottom + 12), f"{int(round(tx))}", fill=text_color, font=font)

    # Y ticks at decades
    min_dec = int(math.floor(log_min_y))
    max_dec = int(math.ceil(log_max_y))
    for dec in range(min_dec, max_dec + 1):
        val = log_base ** dec
        if val < min_y or val > max_y:
            continue
        py = map_y(val)
        draw.line([(plot_left - 6, py), (plot_left, py)], fill=axis_color, width=1)
        draw.text((plot_left - 90, py - 4), f"1e{dec}", fill=text_color, font=font)
        draw.line([(plot_left, py), (plot_right, py)], fill=grid_color, width=1)

    # Axes
    draw.line([(plot_left, plot_top), (plot_left, plot_bottom)], fill=axis_color, width=1)
    draw.line([(plot_left, plot_bottom), (plot_right, plot_bottom)], fill=axis_color, width=1)

    # Labels
    draw.text((plot_left, plot_bottom + 40), "round", fill=text_color, font=font)
    draw.text((plot_left - 110, plot_top - 25), "GEN corr/pert ratio (log10)", fill=text_color, font=font)
    draw.text(
        (plot_left, 15),
        "GEN loss correction vs perturbation ratio per round (log scale)",
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

        lx, ly = map_x(series[-1][0]), map_y(series[-1][1])
        draw.text((lx + 6, ly - 6), name, fill=color, font=font)

    return image


def main() -> None:
    args = parse_args()
    run_results = Path(args.run_results_path)
    log_paths = sorted(
        p for p in run_results.glob("console_minibatch_*.log")
        if args.exclude_substr not in p.name
    )
    if not log_paths:
        raise SystemExit("No console_minibatch_*.log files found.")

    series_by_file = build_series(log_paths)
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
