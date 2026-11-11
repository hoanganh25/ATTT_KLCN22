"""Plot real TRAIN/DEV loss from saved training history; never train a model.

Run from any directory:
    python chart_loss_train_dev.py
    python chart_loss_train_dev.py --run cnn_awgn --mark-best-dev
    python chart_loss_train_dev.py --run se_awgn --format pdf

Use another training history:
    python chart_loss_train_dev.py --history path/to/history.csv \
        --model CNN_Baseline --channel AWGN --output-dir path/to/charts

Required CSV columns by default:
    epoch, train_normalized_mse, val_normalized_mse

DEV is the arithmetic mean normalized MSE over the validation SNR grid,
as recorded by the training scripts. It is not TEST loss. Curves are not
smoothed or modified to make loss decrease monotonically.

Dependency: matplotlib. Output defaults to chart/figures/*.png at 300 DPI.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import matplotlib

# Save charts without opening a GUI or requiring a display server.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "figures"


@dataclass(frozen=True)
class TrainingRun:
    history: Path
    model: str
    channel: str


RUNS = {
    "cnn_awgn": TrainingRun(
        PROJECT_ROOT / "runs/multi_snr/awgn/history.csv",
        "CNN_Baseline",
        "AWGN",
    ),
    "cnn_rayleigh": TrainingRun(
        PROJECT_ROOT / "runs/multi_snr/rayleigh/history.csv",
        "CNN_Baseline",
        "Rayleigh",
    ),
    "se_awgn": TrainingRun(
        PROJECT_ROOT / "runs/se_resnet_deepscs/awgn_repo8_b64/history.csv",
        "SE_ResNet",
        "AWGN",
    ),
        "se_rayleigh": TrainingRun(
        PROJECT_ROOT / "runs/se_resnet_deepscs/rayleigh_repo8_b64/history.csv",
        "SE_ResNet",
        "Rayleigh",
    ),
}


def chart_name(model: str, channel: str) -> str:
    """Create a filesystem-safe model_channel_Loss name and plot title."""
    def component(value: str) -> str:
        ascii_value = unicodedata.normalize("NFKD", value).encode(
            "ascii", "ignore"
        ).decode("ascii")
        result = re.sub(r"[^A-Za-z0-9]+", "_", ascii_value).strip("_")
        if not result:
            raise ValueError("Model and channel must contain letters or digits")
        return result

    return f"{component(model)}_{component(channel)}_Loss"


def read_history(
    path: Path,
    epoch_column: str = "epoch",
    train_column: str = "train_normalized_mse",
    dev_column: str = "val_normalized_mse",
) -> tuple[list[int], list[float], list[float]]:
    """Read and validate existing CSV values without changing the source file."""
    epochs: list[int] = []
    train_losses: list[float] = []
    dev_losses: list[float] = []
    required = {epoch_column, train_column, dev_column}

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing CSV columns {sorted(missing)}")

        for line_number, row in enumerate(reader, start=2):
            try:
                epoch_value = float(row[epoch_column])
                train_loss = float(row[train_column])
                dev_loss = float(row[dev_column])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{path}:{line_number}: epoch/loss must be numeric"
                ) from error

            if not all(math.isfinite(v) for v in (epoch_value, train_loss, dev_loss)):
                raise ValueError(f"{path}:{line_number}: NaN/Inf is not allowed")
            if epoch_value < 1 or not epoch_value.is_integer():
                raise ValueError(f"{path}:{line_number}: epoch must be a positive integer")
            epoch = int(epoch_value)
            if epochs and epoch <= epochs[-1]:
                raise ValueError(
                    f"{path}:{line_number}: epochs must be strictly increasing"
                )
            if min(train_loss, dev_loss) < 0:
                raise ValueError(f"{path}:{line_number}: MSE loss cannot be negative")

            epochs.append(epoch)
            train_losses.append(train_loss)
            dev_losses.append(dev_loss)

    if not epochs:
        raise ValueError(f"{path}: no training-history rows")
    return epochs, train_losses, dev_losses


def plot_loss(
    run: TrainingRun,
    history: tuple[list[int], list[float], list[float]],
    output_dir: Path,
    *,
    image_format: str = "png",
    dpi: int = 300,
    mark_best_dev: bool = False,
) -> Path:
    """Draw two unsmoothed curves using a gray background and white grid."""
    epochs, train_losses, dev_losses = history
    name = chart_name(run.model, run.channel)
    output_path = output_dir / f"{name}.{image_format}"

    # Local settings: no global change to plotting styles used by other scripts.
    with plt.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "axes.labelcolor": "#555555",
        "xtick.color": "#555555",
        "ytick.color": "#555555",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }):
        figure, ax = plt.subplots(figsize=(8, 5.2), layout="constrained")
        try:
            figure.patch.set_facecolor("white")
            ax.set_facecolor("#E5E5E5")
            ax.set_axisbelow(True)
            ax.grid(True, color="white", linewidth=1.0)
            for spine in ax.spines.values():
                spine.set_color("white")

            ax.plot(epochs, train_losses, color="#FF482D", linewidth=1.8,
                    label="Train loss")
            ax.plot(epochs, dev_losses, color="#168DD0", linewidth=1.8,
                    label="DEV loss")
            ax.set_title(name, pad=12)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss (normalized MSE)")

            # Preserve real epoch numbering; do not introduce an artificial epoch 0.
            ticks = sorted({epochs[0], epochs[-1]} | {e for e in epochs if e % 5 == 0})
            ax.set_xticks(ticks)
            if len(epochs) > 1:
                ax.set_xlim(epochs[0], epochs[-1])
            else:
                ax.set_xlim(epochs[0] - 0.5, epochs[0] + 0.5)
            ax.margins(y=0.10)

            if mark_best_dev:
                best_index = min(range(len(dev_losses)), key=dev_losses.__getitem__)
                ax.scatter(epochs[best_index], dev_losses[best_index],
                           color="#168DD0", s=36, zorder=4)
                ax.text(
                    0.03, 0.04,
                    f"Best DEV: {dev_losses[best_index]:.6f} "
                    f"(epoch {epochs[best_index]})",
                    transform=ax.transAxes, fontsize=10,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
                )

            ax.legend(loc="upper right", frameon=True, facecolor="#E5E5E5",
                      edgecolor="white")
            output_dir.mkdir(parents=True, exist_ok=True)
            figure.savefig(output_path, dpi=dpi, format=image_format,
                           facecolor="white")
        finally:
            plt.close(figure)

    return output_path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run", choices=["all", *RUNS], default="all",
                        help="Saved run to plot; default: all three available models")
    parser.add_argument("--history", type=Path,
                        help="Custom history.csv; requires --model and --channel")
    parser.add_argument("--model", help="Custom model name, e.g. CNN_Baseline")
    parser.add_argument("--channel", help="Custom channel name, e.g. AWGN")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--format", choices=["png", "pdf", "svg"], default="png")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--epoch-column", default="epoch")
    parser.add_argument("--train-column", default="train_normalized_mse")
    parser.add_argument("--dev-column", default="val_normalized_mse")
    parser.add_argument("--mark-best-dev", action="store_true",
                        help="Mark the minimum recorded DEV loss and its epoch")
    args = parser.parse_args()
    if args.dpi < 1:
        parser.error("--dpi must be positive")
    if args.history is not None:
        if not args.model or not args.channel:
            parser.error("--history requires --model and --channel")
        if args.run != "all":
            parser.error("Use either --history or --run, not both")
    elif args.model or args.channel:
        parser.error("--model and --channel are only used with --history")
    return args


def main() -> None:
    args = arguments()
    runs = ([TrainingRun(args.history, args.model, args.channel)]
            if args.history is not None
            else list(RUNS.values()) if args.run == "all" else [RUNS[args.run]])

    # Validate all requested histories before writing any chart.
    try:
        histories = [read_history(run.history, args.epoch_column,
                                  args.train_column, args.dev_column) for run in runs]
        for run in runs:
            chart_name(run.model, run.channel)
    except (OSError, ValueError) as error:
        raise SystemExit(f"Cannot read training history: {error}") from error

    for run, history in zip(runs, histories):
        path = plot_loss(run, history, args.output_dir, image_format=args.format,
                         dpi=args.dpi, mark_best_dev=args.mark_best_dev)
        epochs, train_losses, dev_losses = history
        best = min(range(len(epochs)), key=dev_losses.__getitem__)
        print(f"Saved: {path}")
        print(f"  History: {run.history}")
        print(f"  Train: {train_losses[0]:.9f} -> {train_losses[-1]:.9f}; "
              f"DEV: {dev_losses[0]:.9f} -> {dev_losses[-1]:.9f}; "
              f"best DEV={dev_losses[best]:.9f} at epoch {epochs[best]}")


if __name__ == "__main__":
    main()
