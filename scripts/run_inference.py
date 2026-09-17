"""
Runs a trained WildfireModel checkpoint on a single, unlabeled fire scene and
outputs a per-tile fire-probability grid at the model's single 30-day
horizon (1 month head). See devlog as to why the 3-horizon matrix was dropped. 

Requirements:
    Scene layout expected under --scene-dir (mirrors build_tile_cache.py's
    per-scene temp layout, just not zipped/downloaded from S3):

    <scene-dir>/s1/<name>.SAFE/...          (extracted Sentinel-1 SAFE pre-fire product)
    <scene-dir>/s2/<name>.SAFE/...          (extracted Sentinel-2 SAFE pre-fire product)
    <scene-dir>/era5/<name>_YYYYMMDD.grib   (single ERA5 grib, antecedent window)

    The ERA5 grib filename must end in an 8-digit date (same convention as the
    training pipeline), as that date is used as the forecast cutoff, i.e. "predict
    fire risk over the ~30 days following this date," and everything in the grib
    at or after that date is ignored.

    Usage:
        python scripts/run_inference.py \\
            --checkpoint checkpoints/20260914_120000/best_model.pt \\
            --scene-dir /path/to/new_scene \\
            --cache-dir tile_cache \\
            --output-csv risk_grid.csv
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "scripts"))
sys.path.append(os.path.join(REPO_ROOT, "training"))

from data.loaders import load_era5_vars
from data.aggregator import aggregate
from model import WildfireModel, S1_KEYS, S2_KEYS, ERA5_KEYS, SPATIAL_KEYS, TEMPORAL_KEYS
from model.dataset import flatten_stats
from build_tile_cache import load_s1_pre, load_s2_pre, era5_cutoff_from_key, glob_one
from train import build_datasets


def find_scene_inputs(scene_dir):
    s1_safe_dirs = glob.glob(os.path.join(scene_dir, "s1", "*.SAFE"))
    s2_safe_dirs = glob.glob(os.path.join(scene_dir, "s2", "*.SAFE"))
    era5_gribs = glob.glob(os.path.join(scene_dir, "era5", "*.grib"))

    if len(s1_safe_dirs) != 1:
        raise ValueError(f"Expected exactly one .SAFE dir under {scene_dir}/s1, found {s1_safe_dirs}")
    if len(s2_safe_dirs) != 1:
        raise ValueError(f"Expected exactly one .SAFE dir under {scene_dir}/s2, found {s2_safe_dirs}")
    if len(era5_gribs) != 1:
        raise ValueError(f"Expected exactly one .grib file under {scene_dir}/era5, found {era5_gribs}")

    return s1_safe_dirs[0], s2_safe_dirs[0], era5_gribs[0]


def load_scene_tiles(scene_dir, dem_output_dir):
    """
    Runs a new, unlabeled scene through the same S1/S2/ERA5 loading + zonal
    aggregation pipeline used in build_tile_cache.py, then returns the
    sorted era5 lat/lon arrays needed to map each tile (i, j) back to a real coord.
    """
    s1_safe_dir, s2_safe_dir, era5_grib = find_scene_inputs(scene_dir)

    print(f"Loading Sentinel-1 from {s1_safe_dir} ...")
    s1_data = load_s1_pre(s1_safe_dir, dem_output_dir)

    print(f"Loading Sentinel-2 from {s2_safe_dir} ...")
    s2_data = load_s2_pre(s2_safe_dir)

    print(f"Loading ERA5 from {era5_grib} ...")
    cutoff_datetime = era5_cutoff_from_key(era5_grib)
    era5_data = load_era5_vars(era5_grib, cutoff_datetime=cutoff_datetime)

    # same sort aggregate() applies internally 
    era5_lats = np.sort(era5_data["u10"]["latitude"].values)
    era5_longs = np.sort(era5_data["u10"]["longitude"].values)

    print("Aggregating into tiles...")
    tiles = aggregate(s1_data, s2_data, era5_data)
    print(f"Got {len(tiles)} tiles, forecasting from cutoff date {cutoff_datetime.date()}.")

    return tiles, era5_lats, era5_longs, cutoff_datetime


def apply_embargo(tiles, embargo_days):
    if embargo_days <= 0:
        return tiles
    truncated = {}
    for key, tile in tiles.items():
        tile = dict(tile)
        if tile.get("era5_stats") is not None:
            tile["era5_stats"] = {
                k: v[: len(v) - embargo_days] for k, v in tile["era5_stats"].items()
            }
        truncated[key] = tile
    return truncated


def build_input_tensors(tile, statistic_means, era5_seq_len, spatial_mean, spatial_std,
                         temporal_mean, temporal_std):
    """
    Builds input tensors by performing mean imputation, flattening, then z-score normalizing, 
    similar to model.dataset.py's __getitem__ function.
    """
    s1_stats = dict(tile["s1_stats"]) if tile["s1_stats"] is not None else dict.fromkeys(S1_KEYS)
    for key in S1_KEYS:
        if s1_stats[key] is None or (isinstance(s1_stats[key], float) and not np.isfinite(s1_stats[key])):
            s1_stats[key] = statistic_means[f"mean_{key}"]

    s2_stats = dict(tile["s2_stats"]) if tile["s2_stats"] is not None else dict.fromkeys(S2_KEYS)
    for key in S2_KEYS:
        if s2_stats[key] is None or (isinstance(s2_stats[key], float) and not np.isfinite(s2_stats[key])):
            s2_stats[key] = statistic_means[f"mean_{key}"]

    era5_means = np.array([statistic_means[f"mean_{k}"] for k in ERA5_KEYS]).reshape(-1, 1)
    if tile["era5_stats"] is None:
        era5_matrix = np.repeat(era5_means, era5_seq_len, axis=1)
    else:
        era5_matrix = np.array([tile["era5_stats"][k] for k in ERA5_KEYS], dtype=np.float64)
        era5_matrix = np.where(np.isfinite(era5_matrix), era5_matrix, era5_means)

    s1_flat = flatten_stats(s1_stats)
    s2_flat = flatten_stats(s2_stats)
    x_spatial = torch.tensor(np.concatenate([list(s1_flat.values()), list(s2_flat.values())])).float()
    x_temporal = torch.tensor(era5_matrix).float().transpose(0, 1)

    x_spatial = (x_spatial - spatial_mean) / spatial_std
    x_temporal = (x_temporal - temporal_mean) / temporal_std
    return x_spatial.unsqueeze(0), x_temporal.unsqueeze(0)  # add batch dim


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to a train.py checkpoint (.pt).")
    parser.add_argument("--scene-dir", required=True,
                         help="Directory holding the new scene's s1/, s2/, era5/ inputs (see module docstring).")
    parser.add_argument("--cache-dir", default="tile_cache",
                         help="Only needed as a fallback for older checkpoints that don't have "
                              "normalization stats saved -- must be the same --cache-dir used to "
                              "train the checkpoint (see module docstring caveat).")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dem-dir", default="/tmp/wildfire_inference_dem")
    parser.add_argument("--output-csv", default=None,
                         help="Optional path to write the per-tile risk grid to as CSV.")
    return parser


def main():
    args = build_arg_parser().parse_args()

    print(f"Loading checkpoint from {args.checkpoint} ...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    embargo_days = checkpoint.get("era5_embargo_days", 0)
    if embargo_days:
        print(f"Checkpoint was trained with a {embargo_days}-day ERA5 embargo -- applying the same "
              f"truncation to this scene's tiles for consistency.")

    model = WildfireModel(
        checkpoint["spatial_input_dim"], checkpoint["temporal_input_dim"], checkpoint["embedding_dim"],
        checkpoint["n_layers"], checkpoint["n_head"], temporal_hidden_dim=checkpoint["temporal_hidden_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if "spatial_mean" in checkpoint:
        print("Using normalization stats saved in the checkpoint.")
        spatial_mean, spatial_std = checkpoint["spatial_mean"], checkpoint["spatial_std"]
        temporal_mean, temporal_std = checkpoint["temporal_mean"], checkpoint["temporal_std"]
        statistic_means = checkpoint["statistic_means"]
        era5_seq_len = checkpoint["era5_seq_len"]
    else:
        print(f"Checkpoint has no saved normalization stats (older run) -- re-deriving from "
              f"{args.cache_dir} (val_frac={args.val_frac}, test_frac={args.test_frac}, "
              f"seed={args.seed}). These MUST match the run that produced this checkpoint.")
        data = build_datasets(args.cache_dir, args.val_frac, args.test_frac, args.seed)
        train_ds = data["train_ds"]
        statistic_means = train_ds.inner.statistic_means
        spatial_mean, spatial_std = train_ds.spatial_mean, train_ds.spatial_std
        temporal_mean, temporal_std = train_ds.temporal_mean, train_ds.temporal_std
        era5_seq_len = train_ds.inner.era5_seq_len

    tiles, era5_lats, era5_longs, cutoff_datetime = load_scene_tiles(args.scene_dir, args.dem_dir)
    tiles = apply_embargo(tiles, embargo_days)

    rows = []
    with torch.no_grad():
        for (i, j), tile in tiles.items():
            x_spatial, x_temporal = build_input_tensors(
                tile, statistic_means, era5_seq_len, spatial_mean, spatial_std, temporal_mean, temporal_std)
            head1, _, _, _ = model(x_spatial, x_temporal)
            rows.append({
                "i": i, "j": j,
                "lat": float(era5_lats[i]), "lon": float(era5_longs[j]),
                "fire_probability_30d": float(head1.item()),
            })

    grid = pd.DataFrame(rows).sort_values(["i", "j"]).reset_index(drop=True)
    print(f"\nForecast cutoff: {cutoff_datetime.date()} (~30-day horizon)")
    print(f"{len(grid)} tiles scored. Risk summary:")
    print(f"  mean {grid['fire_probability_30d'].mean():.3f}  "
          f"max {grid['fire_probability_30d'].max():.3f}  "
          f"min {grid['fire_probability_30d'].min():.3f}")
    print(grid.to_string(index=False))

    if args.output_csv:
        grid.to_csv(args.output_csv, index=False)
        print(f"\nSaved per-tile risk grid to {args.output_csv}")


if __name__ == "__main__":
    main()
