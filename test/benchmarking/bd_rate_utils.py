#!/usr/bin/env python3
# Copyright(c) 2025 Meta Platforms, Inc. and affiliates.
#
# This source code is subject to the terms of the BSD 2 Clause License and
# the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
# was not distributed with this source code in the LICENSE file, you can
# obtain it at https://www.aomedia.org/license/software-license. If the
# Alliance for Open Media Patent License 1.0 was not distributed with this
# source code in the PATENTS file, you can obtain it at
# https://www.aomedia.org/license/patent-license.

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bd_metric import bd_rate_v2


had_psnr_cbcr_issues = False


def compute_bd_rates_per_image(
    per_image_csv_path: str,
    anchor_encoder: str = "jpegli_enc",
    anchor_speed: int = 3,
    output_csv_path: str = "bd_rates_per_image.csv",
    quality_metric: str = "vmaf",
) -> pd.DataFrame:
    """
    Compute BD-rate per image for each encoder/speed combination against an anchor.

    Args:
        per_image_csv_path: Path to per_image_results.csv
        anchor_encoder: Reference encoder name (default: "jpegli_enc")
        anchor_speed: Reference encoder speed (default: 3)
        output_csv_path: Path to save BD-rate results CSV
        quality_metric: Quality metric to use for BD-rate computation (default: "vmaf")

    Returns:
        DataFrame with BD-rate results per image
    """
    per_image_df = pd.read_csv(per_image_csv_path)

    # Convert file_size_mb to bitrate (for images, file size represents compressed size)
    per_image_df["bitrate_kbps"] = (
        per_image_df["file_size_mb"] * 8 * 1024
    )  # Convert MB to Kbps

    # Get unique images
    images = per_image_df["image"].unique()

    # Get unique encoder/speed combinations
    encoder_speed_combinations = per_image_df[["encoder", "speed"]].drop_duplicates()

    # Prepare results list
    bd_rate_results = []

    for image in images:
        # Get anchor data for this image
        anchor_data = per_image_df[
            (per_image_df["image"] == image)
            & (per_image_df["encoder"] == anchor_encoder)
            & (per_image_df["speed"] == anchor_speed)
        ].copy()

        if anchor_data.empty:
            print(f"  Warning: No anchor data found for image {image}")
            continue

        # Sort anchor data by quality for proper RD curve
        anchor_data = anchor_data.sort_values("quality")
        anchor_bitrates = anchor_data["bitrate_kbps"].to_numpy()
        anchor_qualities = anchor_data[quality_metric].to_numpy()

        if len(anchor_bitrates) < 2:
            print(
                f"  Warning: Not enough anchor points for image {image} (need >=2, got {len(anchor_bitrates)})"
            )
            continue

        # Compute BD-rate for each encoder/speed combination
        for _, (encoder, speed) in encoder_speed_combinations.iterrows():

            def add_result(bd_rate, status, image=image, encoder=encoder, speed=speed):
                bd_rate_results.append(
                    {
                        "image": image,
                        "encoder": encoder,
                        "speed": speed,
                        "bd_rate": bd_rate,
                        "status": status,
                        "quality_metric": quality_metric,
                    }
                )

            # Skip if this is the anchor itself
            if encoder == anchor_encoder and speed == anchor_speed:
                add_result(0.0, "anchor")
                continue

            # Get test data for this encoder/speed/image combination
            test_data = per_image_df[
                (per_image_df["image"] == image)
                & (per_image_df["encoder"] == encoder)
                & (per_image_df["speed"] == speed)
            ].copy()

            if test_data.empty:
                add_result(np.nan, "no_data")
                continue

            # Sort test data by quality
            test_data = test_data.sort_values("quality")
            test_bitrates = test_data["bitrate_kbps"].to_numpy()
            test_qualities = test_data[quality_metric].to_numpy()

            if len(test_bitrates) < 2:
                add_result(np.nan, "insufficient_points")
                continue

            # Compute BD-rate
            try:
                err, bd_rate = bd_rate_v2(
                    quality_metric,
                    anchor_bitrates,
                    anchor_qualities,
                    test_bitrates,
                    test_qualities,
                )

                if err == 0:
                    add_result(bd_rate, "success")
                else:
                    global had_psnr_cbcr_issues
                    if quality_metric not in ["psnr_cb", "psnr_cr"]:
                        print(
                            f"  {encoder} speed {speed}: BD-rate computation failed - {bd_rate}"
                        )
                    elif not had_psnr_cbcr_issues:
                        had_psnr_cbcr_issues = True
                        print(
                            f"  {encoder} speed {speed}: BD-rate computation failed - {quality_metric}"
                        )
                        print("  Omit further such logging as it happens often")
                    add_result(np.nan, f"error: {bd_rate}")
            except Exception as e:
                print(
                    f"  {encoder} speed {speed}: Exception during BD-rate computation - {str(e)}"
                )
                add_result(np.nan, f"exception: {str(e)[:100]}")

    # Convert results to DataFrame
    bd_rate_df = pd.DataFrame(bd_rate_results)

    # Save to CSV only if output path is provided
    if output_csv_path:
        bd_rate_df.to_csv(output_csv_path, index=False)

    return bd_rate_df


def compute_average_bd_rates(
    bd_rate_df: pd.DataFrame,
    per_image_csv_path: str,
    output_csv_path: str = "average_bd_rates.csv",
) -> pd.DataFrame:
    """
    Compute average BD-rates and encoding times per encoder/speed combination across all images.

    Args:
        bd_rate_df: DataFrame with per-image BD-rate results
        per_image_csv_path: Path to per_image_results.csv (needed for encoding times)
        output_csv_path: Path to save average BD-rate results CSV

    Returns:
        DataFrame with average BD-rate and encoding time results
    """
    # Read per-video data to get encoding times
    per_image_df = pd.read_csv(per_image_csv_path)

    # Filter out non-numeric BD-rates (errors, anchor points, etc.)
    valid_data = bd_rate_df[
        (bd_rate_df["status"] == "success") & (pd.notnull(bd_rate_df["bd_rate"]))
    ].copy()

    # Also include anchor points for complete encoder comparison
    anchor_data = bd_rate_df[bd_rate_df["status"] == "anchor"].copy()
    plot_data = pd.concat([valid_data, anchor_data], ignore_index=True)

    if plot_data.empty:
        print("Warning: No valid BD-rate data found for averaging")
        return pd.DataFrame()

    # Group by encoder and speed, compute BD-rate statistics
    bd_rate_stats = plot_data.groupby(["encoder", "speed"]).agg(
        {
            "bd_rate": ["mean", "std", "count", "min", "max"],
            "quality_metric": "first",
        }
    )

    # Flatten column names
    bd_rate_stats.columns = [
        "avg_bd_rate",
        "std_bd_rate",
        "num_images",
        "min_bd_rate",
        "max_bd_rate",
        "quality_metric",
    ]
    bd_rate_stats = bd_rate_stats.reset_index()

    # Compute encoding time statistics from per-video data
    try:
        time_stats = per_image_df.groupby(["encoder", "speed"]).agg(
            {
                "encoding_time": ["mean", "std", "count"],
                "decoding_time": ["mean", "std"],
            }
        )

        # Flatten column names
        time_stats.columns = [
            "avg_encoding_time",
            "std_encoding_time",
            "num_samples",
            "avg_decoding_time",
            "std_decoding_time",
        ]
        time_stats = time_stats.reset_index()

        # Merge BD-rate and encoding time statistics
        avg_results = pd.merge(
            bd_rate_stats, time_stats, on=["encoder", "speed"], how="left"
        )
    except Exception as e:
        print(
            f"Warning: Failed to compute encoding time statistics for average bd rate - {str(e)}"
        )
        avg_results = bd_rate_stats

    # Sort by encoder and speed
    avg_results = avg_results.sort_values(["encoder", "speed"])

    # Save to CSV only if output path is provided
    if output_csv_path:
        avg_results.to_csv(output_csv_path, index=False)

    return avg_results


def plot_average_bd_rates(
    avg_bd_rate_df: pd.DataFrame,
    output_path: str = "average_bd_rates_plot.png",
    title: str = "Average BD-rate vs Encoding Time",
    figsize: Tuple[int, int] = (12, 8),
    show_plot: bool = False,
) -> None:
    """
    Plot average BD-rates vs average encoding time with points labeled by speed/preset.

    Args:
        avg_bd_rate_df: DataFrame with average BD-rate and encoding time results
        output_path: Path to save the plot
        title: Plot title
        figsize: Figure size (width, height)
        show_plot: Whether to display the plot interactively
    """
    if avg_bd_rate_df.empty:
        print("Warning: No data to plot")
        return

    # Check if encoding time data is available
    if "avg_encoding_time" not in avg_bd_rate_df.columns:
        print("Warning: No encoding time data available for plotting")
        return

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Get unique encoders
    encoders = avg_bd_rate_df["encoder"].unique()

    # Define colors and markers for different encoders
    colors = [
        "blue",
        "red",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]

    for i, encoder in enumerate(encoders):
        encoder_data = avg_bd_rate_df[avg_bd_rate_df["encoder"] == encoder].copy()

        # Filter out entries with missing encoding time data
        encoder_data = encoder_data.dropna(subset=["avg_encoding_time"])

        if encoder_data.empty:
            continue

        # Sort by encoding time for proper line connection
        encoder_data = encoder_data.sort_values("avg_encoding_time")

        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        # Plot points and connect with lines if multiple points
        ax.plot(
            encoder_data["avg_encoding_time"],
            encoder_data["avg_bd_rate"],
            color=color,
            marker=marker,
            linewidth=2.5,
            markersize=10,
            label=encoder,
            zorder=5,
            markeredgecolor="black",
            markeredgewidth=1,
        )

        # Add error bars if standard deviation data is available
        if (
            "std_bd_rate" in encoder_data.columns
            and not encoder_data["std_bd_rate"].isna().all()
        ):
            ax.errorbar(
                encoder_data["avg_encoding_time"],
                encoder_data["avg_bd_rate"],
                yerr=encoder_data["std_bd_rate"],
                color=color,
                alpha=0.4,
                capsize=6,
                capthick=2,
                zorder=3,
                linestyle="none",
            )

        # Add speed/preset labels on points
        for _, row in encoder_data.iterrows():
            # Position label to avoid overlap
            x_offset = (
                0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0])
                if row["avg_bd_rate"] >= 0
                else -0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0])
            )
            y_offset = (
                0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                if row["avg_encoding_time"]
                >= np.median(encoder_data["avg_encoding_time"])
                else -0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            )

            ax.annotate(
                f"{int(row['speed'])}",
                (row["avg_encoding_time"], row["avg_bd_rate"]),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=color,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    alpha=0.9,
                    edgecolor=color,
                    linewidth=1,
                ),
            )

    # Customize the plot
    ax.set_xlabel("Avg Time (s)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Avg BD Rate (%)", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.4, linestyle="--")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=12)

    # Add horizontal line at y=0 for reference
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.8, zorder=1, linewidth=1)

    # Use log scale on x-axis (like in your example)
    ax.set_xscale("log")

    # Improve tick formatting
    ax.tick_params(axis="both", which="major", labelsize=12)

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")

    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
