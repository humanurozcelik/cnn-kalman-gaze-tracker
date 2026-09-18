import argparse
import json
import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import median_abs_deviation

try:
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
except ImportError:
    fastdtw = None
    euclidean = None


DEFAULT_CSV = "lab_gaze_tracking_data.csv"

REQUIRED_COLUMNS = {
    "Time_s",
    "EAR",
    "Face_Detected",
    "Target_X",
    "Target_Y",
    "Gaze_X",
    "Gaze_Y",
}


def load_data(filepath):
    """Load gaze-tracking output and perform basic schema checks."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")

    df = pd.read_csv(filepath)

    if df.empty:
        raise ValueError("Input CSV is empty.")

    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        raise ValueError(
            "CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    numeric_columns = [
        "Time_s",
        "EAR",
        "Face_Detected",
        "Target_X",
        "Target_Y",
        "Gaze_X",
        "Gaze_Y",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if not df["Time_s"].is_monotonic_increasing:
        print(
            "Warning: Time_s is not monotonically increasing. "
            "Temporal metrics may be unreliable."
        )

    if "Screen_W" in df.columns and "Screen_H" in df.columns:
        screen_w = int(pd.to_numeric(df["Screen_W"], errors="coerce").dropna().iloc[0])
        screen_h = int(pd.to_numeric(df["Screen_H"], errors="coerce").dropna().iloc[0])
    else:
        screen_w = int(df["Target_X"].max() + 100)
        screen_h = int(df["Target_Y"].max() + 100)

    df["Raw_Error"] = np.sqrt(
        (df["Target_X"] - df["Gaze_X"]) ** 2
        + (df["Target_Y"] - df["Gaze_Y"]) ** 2
    )

    return df, screen_w, screen_h


def detect_artifacts(df, ear_threshold=0.14):
    """Flag missing tracking, blinks, prolonged freezes, and speed spikes."""
    df = df.copy()

    valid_gaze = df["Gaze_X"].notna() & df["Gaze_Y"].notna()
    face_missing = df["Face_Detected"].fillna(0).eq(0)
    blink = (
        df["Face_Detected"].fillna(0).eq(1)
        & df["EAR"].notna()
        & df["EAR"].lt(ear_threshold)
    )
    missing_gaze = ~valid_gaze

    dx = df["Gaze_X"].diff()
    dy = df["Gaze_Y"].diff()

    same_as_previous = valid_gaze & dx.eq(0) & dy.eq(0)
    prolonged_freeze = (
        same_as_previous.astype(int)
        .rolling(window=5, min_periods=5)
        .sum()
        .ge(5)
    )

    dt = df["Time_s"].diff()
    valid_dt = dt.gt(0)

    speed = pd.Series(np.nan, index=df.index, dtype=float)
    speed.loc[valid_gaze & valid_dt] = (
        np.sqrt(dx.loc[valid_gaze & valid_dt] ** 2 + dy.loc[valid_gaze & valid_dt] ** 2)
        / dt.loc[valid_gaze & valid_dt]
    )

    speed_values = speed.dropna()
    speed_spike = pd.Series(False, index=df.index)

    if len(speed_values) >= 10:
        mad = median_abs_deviation(speed_values)
        if np.isfinite(mad) and mad > 1e-9:
            robust_z = 0.6745 * (speed - speed_values.median()) / mad
            speed_spike = robust_z.gt(5.0).fillna(False)

    df["Artifact_Type"] = "None"

    df.loc[missing_gaze, "Artifact_Type"] = "Missing gaze estimate"
    df.loc[blink, "Artifact_Type"] = "Blink / low EAR"
    df.loc[face_missing, "Artifact_Type"] = "Tracking lost"

    freeze_only = prolonged_freeze & df["Artifact_Type"].eq("None")
    df.loc[freeze_only, "Artifact_Type"] = "Repeated gaze samples"

    spike_only = speed_spike & df["Artifact_Type"].eq("None")
    df.loc[spike_only, "Artifact_Type"] = "Speed spike"

    df["Is_Artifact"] = (
        missing_gaze
        | blink
        | face_missing
        | prolonged_freeze
        | speed_spike
    )

    print("\nArtifact summary:")
    print(df["Artifact_Type"].value_counts())

    return df


def align_target_trajectory(df, max_shift_frames=30):
    """
    Find a frame shift that minimizes gaze-to-target RMSE.

    This is a post-hoc temporal alignment step, not a direct measurement of
    end-to-end system latency.
    """
    best_rmse = float("inf")
    best_shift = 0

    usable = (
        ~df["Is_Artifact"]
        & df["Gaze_X"].notna()
        & df["Gaze_Y"].notna()
    )

    for shift in range(-max_shift_frames, max_shift_frames + 1):
        shifted_x = df["Target_X"].shift(shift)
        shifted_y = df["Target_Y"].shift(shift)

        valid = usable & shifted_x.notna() & shifted_y.notna()

        if valid.sum() < 10:
            continue

        error = np.sqrt(
            (shifted_x.loc[valid] - df.loc[valid, "Gaze_X"]) ** 2
            + (shifted_y.loc[valid] - df.loc[valid, "Gaze_Y"]) ** 2
        )
        rmse = float(np.sqrt(np.mean(error ** 2)))

        if rmse < best_rmse:
            best_rmse = rmse
            best_shift = shift

    if not np.isfinite(best_rmse):
        raise ValueError("Not enough valid samples for temporal alignment.")

    df = df.copy()
    df["Target_X_Aligned"] = df["Target_X"].shift(best_shift)
    df["Target_Y_Aligned"] = df["Target_Y"].shift(best_shift)
    df["Aligned_Error"] = np.sqrt(
        (df["Target_X_Aligned"] - df["Gaze_X"]) ** 2
        + (df["Target_Y_Aligned"] - df["Gaze_Y"]) ** 2
    )

    print(
        f"\nTemporal alignment: best target shift = "
        f"{best_shift} frames (RMSE = {best_rmse:.2f} px)"
    )

    return df, best_shift


def compute_metrics(df_clean, best_shift):
    """Compute spatial error, drift, and optional DTW trajectory metrics."""
    raw_error = df_clean["Raw_Error"].dropna()
    aligned_error = df_clean["Aligned_Error"].dropna()

    if raw_error.empty or aligned_error.empty:
        raise ValueError("No valid gaze samples remain after artifact filtering.")

    valid = (
        df_clean["Aligned_Error"].notna()
        & df_clean["Time_s"].notna()
        & df_clean["Target_X_Aligned"].notna()
        & df_clean["Target_Y_Aligned"].notna()
    )

    times = df_clean.loc[valid, "Time_s"].to_numpy()
    errors = df_clean.loc[valid, "Aligned_Error"].to_numpy()

    if len(times) > 1:
        slope, intercept = np.polyfit(times, errors, 1)
    else:
        slope, intercept = 0.0, 0.0

    dtw_score = None

    if fastdtw is not None and valid.sum() > 1:
        target_trajectory = df_clean.loc[
            valid,
            ["Target_X_Aligned", "Target_Y_Aligned"],
        ].to_numpy()

        gaze_trajectory = df_clean.loc[
            valid,
            ["Gaze_X", "Gaze_Y"],
        ].to_numpy()

        try:
            dtw_distance, _ = fastdtw(
                target_trajectory,
                gaze_trajectory,
                dist=euclidean,
            )
            dtw_score = float(
                round(dtw_distance / len(gaze_trajectory), 2)
            )
        except Exception as exc:
            print(f"Warning: DTW calculation failed: {exc}")

    metrics = {
        "Frames_Total": int(len(df_clean)),
        "Temporal_Shift_Frames": int(best_shift),
        "Drift_Slope_px_per_sec": float(round(slope, 3)),
        "DTW_Score_Normalized": dtw_score,
        "Raw": {
            "Mean": float(round(raw_error.mean(), 2)),
            "Median": float(round(raw_error.median(), 2)),
            "RMSE": float(round(np.sqrt(np.mean(raw_error ** 2)), 2)),
            "p95": float(round(np.percentile(raw_error, 95), 2)),
        },
        "Temporally_Aligned": {
            "Mean": float(round(aligned_error.mean(), 2)),
            "Median": float(round(aligned_error.median(), 2)),
            "RMSE": float(
                round(np.sqrt(np.mean(aligned_error ** 2)), 2)
            ),
            "p95": float(round(np.percentile(aligned_error, 95), 2)),
        },
    }

    return metrics, slope, intercept


def make_plots(
    df,
    df_clean,
    metrics,
    slope,
    intercept,
    screen_w,
    screen_h,
    out_dir,
):
    """Generate deterministic plots for one gaze-tracking run."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    plt.figure(figsize=(10, 8))
    plt.title(
        "Gaze vs. target trajectory\n"
        f"Aligned RMSE: {metrics['Temporally_Aligned']['RMSE']} px "
        f"| Run: {timestamp}"
    )

    plt.plot(
        df["Target_X"],
        df["Target_Y"],
        linestyle="--",
        linewidth=1.5,
        alpha=0.5,
        label="Original target path",
    )

    plt.plot(
        df_clean["Target_X_Aligned"],
        df_clean["Target_Y_Aligned"],
        linewidth=2.0,
        label=(
            "Temporally aligned target "
            f"({metrics['Temporal_Shift_Frames']} frames)"
        ),
    )

    scatter = plt.scatter(
        df_clean["Gaze_X"],
        df_clean["Gaze_Y"],
        c=df_clean["Aligned_Error"],
        cmap="viridis_r",
        s=14,
        alpha=0.6,
        edgecolors="none",
        label="Estimated gaze",
    )

    colorbar = plt.colorbar(scatter)
    colorbar.set_label("Aligned spatial error (px)")

    artifacts = df[df["Is_Artifact"]]
    artifacts = artifacts[
        artifacts["Gaze_X"].notna() & artifacts["Gaze_Y"].notna()
    ]

    if not artifacts.empty:
        plt.scatter(
            artifacts["Gaze_X"],
            artifacts["Gaze_Y"],
            marker="x",
            s=20,
            alpha=0.3,
            label="Flagged samples",
        )

    plt.xlim(0, screen_w)
    plt.ylim(screen_h, 0)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend(loc="upper right", fontsize="small")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.xlabel("X coordinate (px)")
    plt.ylabel("Y coordinate (px)")
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "fig_trajectory.png"),
        dpi=300,
    )
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.title(
        "Gaze spatial density\n"
        f"n={len(df_clean)} valid samples"
    )

    hexbin = plt.hexbin(
        df_clean["Gaze_X"],
        df_clean["Gaze_Y"],
        gridsize=40,
        cmap="magma_r",
        mincnt=1,
        alpha=0.35,
    )
    heatmap_colorbar = plt.colorbar(hexbin)
    heatmap_colorbar.set_label("Samples per hexbin")

    if (
        len(df_clean) >= 20
        and df_clean["Gaze_X"].nunique() > 1
        and df_clean["Gaze_Y"].nunique() > 1
    ):
        sns.kdeplot(
            x=df_clean["Gaze_X"],
            y=df_clean["Gaze_Y"],
            fill=True,
            cmap="magma",
            bw_adjust=1.0,
            thresh=0.05,
            alpha=0.75,
            gridsize=150,
        )

    plt.plot(
        df["Target_X"],
        df["Target_Y"],
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Reference target path",
    )

    plt.xlim(0, screen_w)
    plt.ylim(screen_h, 0)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.3)
    plt.xlabel("X coordinate (px)")
    plt.ylabel("Y coordinate (px)")
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "fig_heatmap.png"),
        dpi=300,
    )
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.title(
        "Spatial error over time "
        f"(drift slope: {metrics['Drift_Slope_px_per_sec']} px/s)"
    )
    plt.plot(
        df_clean["Time_s"],
        df_clean["Aligned_Error"],
        ".",
        alpha=0.4,
        label="Aligned spatial error",
    )

    trend_y = df_clean["Time_s"] * slope + intercept
    plt.plot(
        df_clean["Time_s"],
        trend_y,
        "--",
        linewidth=2,
        label="Linear drift trend",
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Error (px)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "fig_error_time.png"),
        dpi=300,
    )
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.title("Spatial error distribution")

    sns.histplot(
        df_clean["Raw_Error"].dropna(),
        alpha=0.5,
        label="Raw error",
        kde=True,
        bins=40,
    )
    sns.histplot(
        df_clean["Aligned_Error"].dropna(),
        alpha=0.5,
        label="Temporally aligned error",
        kde=True,
        bins=40,
    )

    plt.axvline(
        metrics["Temporally_Aligned"]["p95"],
        linestyle="--",
        linewidth=2,
        label=(
            "Aligned p95: "
            f"{metrics['Temporally_Aligned']['p95']} px"
        ),
    )
    plt.axvline(
        metrics["Temporally_Aligned"]["Median"],
        linestyle="-",
        linewidth=2,
        label=(
            "Aligned median: "
            f"{metrics['Temporally_Aligned']['Median']} px"
        ),
    )

    plt.xlabel("Spatial error (px)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "fig_error_hist.png"),
        dpi=300,
    )
    plt.close()


def export_report(df_clean, metrics, out_dir):
    with open(
        os.path.join(out_dir, "summary.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4, ensure_ascii=False)

    df_clean.to_csv(
        os.path.join(out_dir, "cleaned_data.csv"),
        index=False,
    )


def main(csv_path=DEFAULT_CSV, out_root="runs"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_root, f"exp_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    df, screen_w, screen_h = load_data(csv_path)
    df = detect_artifacts(df)
    df, best_shift = align_target_trajectory(df)

    clean_mask = (
        ~df["Is_Artifact"]
        & df["Gaze_X"].notna()
        & df["Gaze_Y"].notna()
        & df["Aligned_Error"].notna()
    )
    df_clean = df.loc[clean_mask].copy()

    if len(df_clean) < 10:
        raise ValueError(
            "Too few valid gaze samples remain after artifact filtering."
        )

    metrics, slope, intercept = compute_metrics(
        df_clean,
        best_shift,
    )

    print("\nAnalysis summary:")
    print(json.dumps(metrics, indent=4))

    make_plots(
        df,
        df_clean,
        metrics,
        slope,
        intercept,
        screen_w,
        screen_h,
        out_dir,
    )
    export_report(df_clean, metrics, out_dir)

    print(f"\nAnalysis outputs saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post-process webcam gaze-tracking CSV output"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=DEFAULT_CSV,
        help=f"Input CSV path (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="runs",
        help="Root folder for analysis outputs (default: runs)",
    )

    args = parser.parse_args()
    main(csv_path=args.csv, out_root=args.out_root)
