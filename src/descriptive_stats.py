"""Descriptive statistics for the music-genre metadata used in the project.

Run from the repository root:
    python src/descriptive_stats.py --metadata data/processed/metadata.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

REQUIRED = {"file_name", "file_path", "genre"}


def analyze(metadata_path: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(metadata_path)
    missing_cols = REQUIRED.difference(df.columns)
    if missing_cols:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing_cols)}")
    df["genre"] = df["genre"].astype("string").str.strip().str.lower()
    df = df.dropna(subset=["genre"])
    if df.empty:
        raise ValueError("No non-missing genre labels found")

    counts = df["genre"].value_counts().sort_index().rename_axis("genre").to_frame("count")
    counts["percentage"] = counts["count"] / counts["count"].sum() * 100
    counts["share"] = counts["count"] / counts["count"].sum()

    # These summarize the five genre-level sample counts, not audio-signal values.
    values = counts["count"].astype(float)
    summary = pd.Series({
        "tracks_total": int(values.sum()),
        "genres_total": int(values.size),
        "mean_tracks_per_genre": values.mean(),
        "median_tracks_per_genre": values.median(),
        "variance_population_tracks_per_genre": values.var(ddof=0),
        "variance_sample_tracks_per_genre": values.var(ddof=1) if len(values) > 1 else float("nan"),
        "std_population_tracks_per_genre": values.std(ddof=0),
        "min_tracks_per_genre": int(values.min()),
        "max_tracks_per_genre": int(values.max()),
    }, name="value")

    output_dir.mkdir(parents=True, exist_ok=True)
    counts.to_csv(output_dir / "genre_descriptive_stats.csv", float_format="%.4f")
    summary.to_csv(output_dir / "summary_statistics.csv", header=["value"], float_format="%.4f")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(counts.index, counts["count"], color="#5267d8")
    ax.bar_label(bars, labels=[f"{n} ({p:.1f}%)" for n, p in zip(counts["count"], counts["percentage"])], padding=3)
    ax.set_title("Audio-file count by genre (n=14)")
    ax.set_xlabel("Genre")
    ax.set_ylabel("Number of audio files")
    ax.set_ylim(0, counts["count"].max() + 1.5)
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(output_dir / "genre_counts.png", dpi=180)
    plt.close(fig)

    return counts, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/metadata.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/plots"))
    args = parser.parse_args()
    counts, summary = analyze(args.metadata, args.output)
    print("Counts and class percentages:")
    print(counts.to_string(float_format=lambda x: f"{x:.2f}"))
    print("\nDescriptive statistics across genre-level counts:")
    print(summary.to_string(float_format=lambda x: f"{x:.2f}"))
    print(f"\nSaved CSV summaries and plot under: {args.output.resolve()}")

if __name__ == "__main__":
    main()
