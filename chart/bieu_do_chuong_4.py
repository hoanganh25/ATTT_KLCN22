"""Thư viện và điểm vào dùng chung để vẽ các hình Chương 4.

Hình 4.5–4.6 đọc history huấn luyện; các hình TEST đọc summary của evaluator.
Không làm trơn, nội suy hay gộp no-noise vào SNR. Checkpoint chỉ kiểm tra sự
hiện diện, không nạp torch.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SNR = (0, 5, 10, 15, 20)
# key, tên, report, run, model trong CSV, channel, màu, marker, kiểu nét
MODELS = (
    ("A", "CNN–AWGN", "cnn_baseline_awgn", "cnn_baseline/awgn", "cnn", "awgn", "#0072B2", "o", "-"),
    ("B", "CNN–Rayleigh", "cnn_baseline_rayleigh", "cnn_baseline/rayleigh", "cnn", "rayleigh", "#D55E00", "s", "--"),
    ("C", "SE-ResNet–AWGN", "se_resnet_awgn", "se_resnet_deepscs/awgn_repo8_b64", "se", "awgn", "#009E73", "^", "-"),
    ("D", "SE-ResNet–Rayleigh", "se_resnet_rayleigh", "se_resnet_deepscs/rayleigh_repo8_b64", "se", "rayleigh", "#CC79A7", "D", "--"),
)
METRICS = {
    "norm_mse": ("mel", "Normalized log-Mel MSE", "Mean normalized MSE ↓"),
    "raw_logmel_mse": ("mel", "Raw log-Mel MSE", "Mean raw log-Mel MSE ↓"),
    "energy_mask_f1": ("mel", "Energy-Mask F1", "Mean F1 ↑"),
    "pesq_wb": ("wave", "PESQ-WB", "Mean PESQ-WB ↑"),
    "stoi": ("wave", "STOI", "Mean STOI ↑"),
    "waveform_mse": ("wave", "Waveform MSE", "Mean waveform MSE theo sample ↓"),
}
SINGLE = {"4.7": "norm_mse", "4.8": "raw_logmel_mse", "4.9": "energy_mask_f1",
          "4.11": "pesq_wb", "4.12": "stoi", "4.13": "waveform_mse"}

SNR_GRID = SNR
HISTORY_COLUMNS = ["epoch", "train_normalized_mse", "val_normalized_mse"] + [
    f"val_mse_{snr}db" for snr in SNR_GRID
]
HISTORY_MODELS = {
    "A": {"name": "CNN–AWGN", "rel": "runs/cnn_baseline/awgn/history.csv", "zip": "train_cnn_awgn.zip"},
    "B": {"name": "CNN–Rayleigh", "rel": "runs/cnn_baseline/rayleigh/history.csv", "zip": "train_cnn_rayliegh.zip"},
    "C": {"name": "SE-ResNet–AWGN", "rel": "runs/se_resnet/awgn_repo8_b64/history.csv", "zip": "train_se.zip"},
    "D": {"name": "SE-ResNet–Rayleigh", "rel": "runs/se_resnet/rayleigh_repo8_b64/history.csv", "zip": "metric_rayleigh_se.zip"},
}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"CSV trống: {path}")
    return rows


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_data(root):
    """Xác minh provenance, coverage, số mẫu và điều kiện đánh giá A/B/C/D.

    Fail rõ ràng khi thiếu/sai dữ liệu; không âm thầm bỏ đường D hay điền số.
    Đối chiếu checkpoint theo đường dẫn tương đối runs/ để hỗ trợ chuyển máy.
    """
    data, audit, configs = {}, {}, []
    for key, name, report, run, model, channel, *_ in MODELS:
        folder, run_dir = root / "reports" / report, root / "runs" / run
        cfg = read_json(folder / "evaluation_config.json")
        train = read_json(run_dir / "config.json")
        checkpoint = run_dir / "best_model.pth"
        recorded = cfg["checkpoints"][model].replace("\\", "/")
        require(checkpoint.is_file() and checkpoint.stat().st_size > 0,
                f"{key}: Thiếu checkpoint {checkpoint}")
        require(recorded.endswith(f"runs/{run}/best_model.pth"), f"{key}: Sai đường dẫn checkpoint")
        require(cfg["dataset"] == "test" and cfg["mode"] == "both", f"{key}: Không phải TEST mel+wave")
        require(cfg["architectures"][model] == ("cnn" if model == "cnn" else "se_resnet"),
                f"{key}: Sai kiến trúc đánh giá")
        require(train["channel"] == channel, f"{key}: Sai kênh huấn luyện")
        require(set(cfg["channels"]) == {"no_noise", channel}, f"{key}: Sai kênh đánh giá")
        require(sorted(cfg["snrs"]) == list(SNR) and 64 in cfg["iterations"], f"{key}: Sai SNR/iterations")
        train_manifest = train.get("manifest_sha256", train.get("dataset_manifest_sha256"))
        require(train_manifest == cfg["manifest_sha256"], f"{key}: Sai manifest train/eval")
        history = read_csv(run_dir / "history.csv")
        require(len(history) == int(train["epochs"]), f"{key}: History chưa đủ epoch")
        require(all(math.isfinite(float(r["val_normalized_mse"])) for r in history),
                f"{key}: History có DEV MSE không hợp lệ")
        best = min(history, key=lambda r: float(r["val_normalized_mse"]))
        groups = {}
        for kind in ("mel", "wave"):
            path = folder / f"{kind}_summary.csv"
            rows = read_csv(path)
            if kind == "wave":
                rows = [r for r in rows if float(r["iterations"]) == 64]
            indexed = {}
            for row in rows:
                require(row["model"] == model, f"{path}: Sai model")
                require(row["channel"] in (channel, "no_noise"), f"{path}: Sai channel")
                if row["channel"] == "no_noise":
                    require(row["snr_db"].strip().lower() in ("none", "nan", "", "null"),
                            f"{path}: No-noise không được gán SNR hữu hạn")
                    snr = None
                else:
                    snr = float(row["snr_db"])
                    require(snr in SNR, f"{path}: SNR ngoài grid")
                require(snr not in indexed, f"{path}: Trùng điều kiện {snr}")
                require(int(row["n"]) == int(cfg["selected_utterances"]) > 0, f"{path}: Sai số mẫu n")
                for metric, (source, _, _) in METRICS.items():
                    if source != kind:
                        continue
                    value = float(row[metric])
                    require(math.isfinite(value) and value >= 0, f"{path}: {metric} không hợp lệ")
                    if metric in ("energy_mask_f1", "stoi"):
                        require(value <= 1, f"{path}: {metric} ngoài [0, 1]")
                indexed[snr] = row
            require(set(indexed) == {None, *SNR}, f"{path}: Thiếu coverage no-noise/5 SNR")
            groups[kind] = indexed
        for snr in (None, *SNR):
            require(math.isclose(float(groups["mel"][snr]["raw_logmel_mse"]),
                                 float(groups["wave"][snr]["raw_logmel_mse"]), rel_tol=1e-9),
                    f"{key}: Mel/wave không cùng raw log-Mel MSE tại {snr}")
        data[key] = groups
        configs.append(cfg)
        audit[key] = {"name": name, "report": f"reports/{report}",
                      "checkpoint": f"runs/{run}/best_model.pth",
                      "best_epoch_from_dev_history": int(best["epoch"]),
                      "n_per_condition": cfg["selected_utterances"],
                      "coverage": ["no_noise", *SNR]}
    # Các metric phải cùng tập TEST, ngưỡng mask và cấu hình tái tạo waveform.
    comparable = ("manifest_sha256", "selected_utterances", "seed", "repeats",
                  "max_utterances", "experimental_energy_mask_threshold_raw_logmel",
                  "mse_population", "synthesis", "precision", "lambda_mel", "rho",
                  "inner_steps", "init_iterations", "init")
    for field in comparable:
        require(all(c[field] == configs[0][field] for c in configs), f"Khác điều kiện đánh giá: {field}")
    audit["evaluation_conditions"] = {f: configs[0][f] for f in comparable}
    audit["iterations"] = 64
    audit["checkpoint_verification"] = "Path/existence + config/channel/history/CSV coverage; checkpoint tensor metadata not loaded."
    return data, audit


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42,
                         "savefig.dpi": 350})


def value(data, key, metric, snr):
    return float(data[key][METRICS[metric][0]][snr][metric])


def metric_axis(ax, metric):
    ax.set_ylabel(METRICS[metric][2])
    # Dùng cùng phạm vi F1/STOI ở cả hình riêng và hình tổng hợp.
    if metric in ("energy_mask_f1", "stoi"):
        ax.set_ylim(0, 1)
    else:
        ax.set_ylim(bottom=0)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)


def plot_lines(ax, data, metric):
    for key, name, _, _, _, _, color, marker, ls in MODELS:
        ax.plot(SNR, [value(data, key, metric, s) for s in SNR],
                label=f"{key}. {name}", color=color, marker=marker,
                linestyle=ls, linewidth=2, markersize=6)
    ax.set(xlabel="SNR danh định (dB)", xticks=SNR, xlim=(-0.5, 20.5))
    metric_axis(ax, metric)


def plot_figure(number, data):
    if number in SINGLE:
        metric = SINGLE[number]
        fig, ax = plt.subplots(figsize=(10.6, 6.2))
        plot_lines(ax, data, metric)
        ax.legend(loc="best", framealpha=0.95)
        ax.set_title(f"Hình {number}. {METRICS[metric][1]} theo SNR", weight="bold", pad=14)
        fig.tight_layout(rect=(0, 0.055, 1, 1))
        metrics = [metric]
    else:
        metrics = list(METRICS)
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        for ax, metric in zip(axes.flat, metrics):
            if number == "4.10":
                values = [value(data, m[0], metric, None) for m in MODELS]
                bars = ax.bar([m[0] for m in MODELS], values, color=[m[6] for m in MODELS], width=0.58)
                metric_axis(ax, metric)
                if metric not in ("energy_mask_f1", "stoi"):
                    ax.set_ylim(0, max(values) * 1.23)
                ax.bar_label(bars, labels=[f"{v:.4f}" for v in values], padding=4, fontsize=9)
                ax.set_xlabel("Cấu hình")
                ax.grid(False, axis="x")
            else:
                plot_lines(ax, data, metric)
            ax.set_title(METRICS[metric][1], weight="bold")
        handles = [plt.Line2D([], [], color=m[6], marker=m[7], linestyle=m[8],
                             label=f"{m[0]}. {m[1]}") for m in MODELS]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=4)
        title = ("Đối chứng No-noise của bốn cấu hình" if number == "4.10" else
                 "Đối sánh hai kiến trúc trên hai kênh — tập TEST")
        fig.suptitle(f"Hình {number}. {title}", weight="bold", fontsize=15, y=0.99)
        fig.tight_layout(rect=(0, 0.06, 1, 0.89), h_pad=2, w_pad=2)
    footer = ("No-noise là đối chứng riêng; không gán SNR." if number == "4.10" else
              "SNR: 0, 5, 10, 15, 20 dB; đối chứng no-noise xem Hình 4.10.")
    if number in ("4.10", "4.11", "4.12", "4.13", "4.14"):
        footer += "  Waveform: iterations = 64."
    n = int(data["A"]["mel"][None]["n"])
    fig.text(0.5, 0.02, f"TEST: n = {n:,} / điều kiện.  {footer}", ha="center", fontsize=9, color="0.3")
    return fig, metrics


def export_values(out, number, metrics, data):
    fields = ["configuration", "model", "channel", "snr_db", "n", "iterations", "metric", "value", "source"]
    with (out / f"Gia_tri_Hinh_{number.replace('.', '_')}.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for metric in metrics:
            kind = METRICS[metric][0]
            for key, name, report, _, _, channel, *_ in MODELS:
                for snr in ((None,) if number == "4.10" else SNR):
                    row = data[key][kind][snr]
                    writer.writerow(dict(configuration=key, model=name, channel=row["channel"],
                                         snr_db="none" if snr is None else snr, n=row["n"],
                                         iterations=64 if kind == "wave" else "", metric=metric,
                                         value=row[metric], source=f"reports/{report}/{kind}_summary.csv"))


def find_history(key, root, override):
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy history --{key.lower()}: {path}")
        return path
    model = HISTORY_MODELS[key]
    candidates = (root / model["rel"], root / model["zip"], Path(__file__).resolve().parent / model["zip"])
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Thiếu history của {key} ({model['name']}). Đã thử: "
                            + "; ".join(map(str, candidates)))


def read_history(path):
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            matches = [name for name in archive.namelist() if Path(name).name.lower() == "history.csv"]
            if len(matches) != 1:
                raise ValueError(f"ZIP phải chứa đúng một history.csv: {path}")
            with archive.open(matches[0]) as stream:
                rows = list(csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig", newline="")))
    else:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"history.csv trống: {path}")
    missing = [column for column in HISTORY_COLUMNS if column not in rows[0]]
    if missing:
        raise ValueError(f"Thiếu cột trong {path}: {missing}")
    result = {column: [float(row[column]) for row in rows] for column in HISTORY_COLUMNS}
    if any(not math.isfinite(value) for values in result.values() for value in values):
        raise ValueError(f"Dữ liệu NaN/Inf trong history: {path}")
    epochs = result["epoch"]
    if epochs != list(range(1, len(epochs) + 1)):
        raise ValueError(f"Epoch phải liên tiếp từ 1: {path}")
    if len(epochs) != 20:
        raise ValueError(f"Cần đúng 20 epoch cho biểu đồ chương 4; {path} có {len(epochs)}")
    per_snr = [[result[f"val_mse_{snr}db"][i] for snr in SNR_GRID] for i in range(len(epochs))]
    if any(not math.isclose(sum(values) / len(values), result["val_normalized_mse"][i],
                            rel_tol=1e-6, abs_tol=1e-8) for i, values in enumerate(per_snr)):
        raise ValueError(f"DEV MSE không khớp trung bình của 5 SNR: {path}")
    return result


def best_index(history):
    return min(range(len(history["val_normalized_mse"])),
               key=history["val_normalized_mse"].__getitem__)


def plot_history_45(histories, out):
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.6), sharex=True, sharey=True)
    max_y = max(max(max(h["train_normalized_mse"]), max(h["val_normalized_mse"]))
                for h in histories.values())
    for ax, (key, hist) in zip(axes.flat, histories.items()):
        epoch, train, dev = hist["epoch"], hist["train_normalized_mse"], hist["val_normalized_mse"]
        ibest = best_index(hist)
        ax.plot(epoch, train, marker="o", markersize=3, linewidth=1.7, label="TRAIN", color="C0")
        ax.plot(epoch, dev, marker="s", markersize=3, linewidth=1.7, label="DEV", color="C1")
        ax.axvline(epoch[ibest], color="0.45", linestyle="--", linewidth=1, alpha=0.8)
        ax.scatter([epoch[ibest]], [dev[ibest]], marker="*", s=140, zorder=5,
                   color="C3", edgecolors="white", linewidths=0.6)
        ax.set(title=f"{key}. {HISTORY_MODELS[key]['name']}", xlim=(0.5, 20.5),
               xticks=[1, 5, 10, 15, 20], ylim=(0, max_y * 1.13))
        ax.text(0.98, 0.96, f"Epoch chọn: {int(epoch[ibest])}\nDEV MSE: {dev[ibest]:.6f}",
                transform=ax.transAxes, va="top", ha="right", fontsize=8.5,
                bbox={"facecolor": "white", "edgecolor": "0.85", "alpha": 0.9})
        ax.grid(True, linestyle=":", alpha=0.4)
    for ax in axes[1, :]:
        ax.set_xlabel("Chu kỳ huấn luyện (epoch)")
    for ax in axes[:, 0]:
        ax.set_ylabel("MSE log-Mel chuẩn hóa")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("Loss MSE trên TRAIN và DEV", fontsize=13, weight="bold", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2, w_pad=1.4)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"Hinh_4_5_Train_DEV_MSE.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_history_46(histories, sources, out):
    fig, ax = plt.subplots(figsize=(10.6, 5.9))
    styles, markers = ("-", "--", "-.", ":"), ("o", "s", "^", "D")
    for i, (key, hist) in enumerate(histories.items()):
        best = best_index(hist)
        values = [hist[f"val_mse_{snr}db"][best] for snr in SNR_GRID]
        ax.plot(SNR_GRID, values,
                label=f"{key}. {HISTORY_MODELS[key]['name']} (epoch {best + 1})",
                linewidth=2, linestyle=styles[i], marker=markers[i], markersize=7)
    ax.set(xlabel="SNR (dB)", ylabel="MSE log-Mel chuẩn hóa trên DEV",
           xticks=SNR_GRID, xlim=(-0.5, 20.5))
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(loc="upper right", framealpha=0.93)
    ax.set_title("MSE trên DEV theo SNR tại epoch được chọn", fontsize=12, weight="bold", pad=15)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"Hinh_4_6_DEV_MSE_theo_SNR.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    with (out / "Gia_tri_Hinh_4_6.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["model", "best_epoch", "best_val_normalized_mse"] + \
                 [f"dev_mse_{snr}db" for snr in SNR_GRID] + ["history_source"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key, hist in histories.items():
            idx = best_index(hist)
            row = {"model": HISTORY_MODELS[key]["name"], "best_epoch": idx + 1,
                   "best_val_normalized_mse": hist["val_normalized_mse"][idx],
                   "history_source": str(sources[key])}
            row.update({f"dev_mse_{snr}db": hist[f"val_mse_{snr}db"][idx] for snr in SNR_GRID})
            writer.writerow(row)


def main(number, script_dir):
    numbers = (number,) if isinstance(number, str) else tuple(number)
    parser = argparse.ArgumentParser(description="Vẽ biểu đồ Chương 4 từ history huấn luyện hoặc kết quả TEST.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent,
                        help="Thư mục gốc dự án, chứa runs/ và reports/")
    parser.add_argument("--out", type=Path, default=Path(script_dir) / "chart",
                        help="Thư mục xuất PNG/PDF/CSV")
    for key in HISTORY_MODELS:
        parser.add_argument(f"--{key.lower()}", default=None,
                            help=f"Tệp history.csv hoặc ZIP của cấu hình {key} (chỉ hình 4.5/4.6)")
    args = parser.parse_args()
    root, out = args.root.expanduser().resolve(), args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    style()

    if all(n in ("4.5", "4.6") for n in numbers):
        histories, sources = {}, {}
        for key in HISTORY_MODELS:
            source = find_history(key, root, getattr(args, key.lower()))
            sources[key] = source
            histories[key] = read_history(source)
            idx = best_index(histories[key])
            print(f"[{key}] {HISTORY_MODELS[key]['name']}: {source} | best epoch={idx + 1}, "
                  f"DEV MSE={histories[key]['val_normalized_mse'][idx]:.9f}")
        if "4.5" in numbers:
            plot_history_45(histories, out)
        if "4.6" in numbers:
            plot_history_46(histories, sources, out)
        print(f"Đã lưu Hình {', '.join(numbers)} (PNG/PDF/CSV) vào: {out}")
        print("Ghi chú: chọn epoch tốt nhất theo DEV; không sử dụng tập TEST.")
        return

    data, audit = load_data(root)
    for current in numbers:
        fig, metrics = plot_figure(current, data)
        stem = out / f"Hinh_{current.replace('.', '_')}"
        for extension in ("png", "pdf"):
            fig.savefig(stem.with_suffix(f".{extension}"), bbox_inches="tight", facecolor="white")
        plt.close(fig)
        export_values(out, current, metrics, data)
        (out / "Nguon_du_lieu.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Hinh {current}: A/B/C/D, TEST n={audit['A']['n_per_condition']}, PNG/PDF/CSV -> {out}")
