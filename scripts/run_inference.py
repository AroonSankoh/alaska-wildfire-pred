"""
Runs a trained WildfireModel checkpoint on a single, unlabeled fire scene and
outputs a per-tile fire-probability grid at the model's single 30-day
horizon (1 month head). See devlog as to why the 3-horizon matrix was dropped. 

Requirements:
    Scene layout expected at --scene-dir: 

    <scene-dir>/S1_..._<date>_pre.SAFE    (zip archive, or an already-extracted dir)
    <scene-dir>/S1_..._<date>_post.SAFE   (zip archive, ignored -- see below)
    <scene-dir>/S2_..._<date>_pre.SAFE    (zip archive, or an already-extracted dir)
    <scene-dir>/S2_..._<date>_post.SAFE   (zip archive, ignored -- see below)
    <scene-dir>/ERA5_..._YYYYMMDD.grib
    <scene-dir>/metadata.json

    This matches the dataset's actual per-scene folder as published (e.g.
    one of the fires/{state}/{scene}/ or controls/{state}/{scene}/ folders),
    with all files sitting flat alongside eachother. Despite the ".SAFE"
    suffix, each *_pre.SAFE/*_post.SAFE entry is actually a zip archive (same
    as the original S3 fires/controls pipeline) wrapping one nested,
    differently-named *.SAFE directory -- this script extracts it to
    --extract-dir automatically (mirrors build_tile_cache.py's
    download_and_extract()). An already-extracted directory works too.

    Only the *_pre.SAFE products are used for inference (matching how the
    model was trained -- see build_tile_cache.py/load_s1_pre/load_s2_pre),
    the *_post.SAFE ones are ignored since there's no "post" for a forecast
    that hasn't happened yet. Picked out by filename suffix, so this works
    directly on a fire or control folder pulled straight from the dataset --
    no repackaging needed.

    The ERA5 grib filename must end in an 8-digit date (same convention as the
    training pipeline), as that date is used as the forecast cutoff, i.e. "predict
    fire risk over the ~30 days following this date," and everything in the grib
    at or after that date is ignored.

Output:
    Per-tile risk grid is saved by default to
    inference/{run_id}_{checkpoint_stem}/{scene_id}_risk_grid.csv, e.g.
    inference/20260914_140410_best_model/control_AK_65N141W_20190805_risk_grid.csv
    This mirrors the checkpoints/{run_id}/ layout so results stay tied to
    the exact checkpoint that produced them. Override with --output-csv.

Usage:
    python scripts/run_inference.py \
        --checkpoint checkpoints/20260914_120000/best_model.pt \
        --scene-dir /path/to/new_scene \
        --cache-dir tile_cache
"""

import argparse
import glob
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "scripts"))
sys.path.append(os.path.join(REPO_ROOT, "training"))

from data.loaders import load_era5_vars
from data.aggregator import aggregate
from model import WildfireModel
from model.dataset import dataset as TileDataset
from build_tile_cache import load_s1_pre, load_s2_pre, era5_cutoff_from_key
from train import build_datasets


def ensure_extracted_safe_dir(path, extract_root, label):
    """
    Extracts actual SAFE dirs from zip files that wrap the *_pre.SAFE/*_post.SAFE
    entries in a similar fashion as in build_tile_cache.py's download_and_extract(), 
    if the path isn't already a directory.
    """
    if os.path.isdir(path):
        nested = glob.glob(os.path.join(path, "*.SAFE"))
        if len(nested) > 1:
            raise ValueError(f"Expected at most one nested .SAFE dir under {path}, found {nested}")
        return nested[0] if nested else path

    extract_dir = os.path.join(extract_root, os.path.splitext(os.path.basename(path))[0])
    if not os.path.isdir(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)
        print(f"  {label} is a zip archive -- extracting to {extract_dir} ...")
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(extract_dir)

    safe_dirs = glob.glob(os.path.join(extract_dir, "*.SAFE"))
    if len(safe_dirs) != 1:
        raise ValueError(f"Expected exactly one .SAFE dir after extracting {path}, found {safe_dirs}")
    return safe_dirs[0]


def find_scene_inputs(scene_dir, extract_root):
    """
    Picks the *_pre.SAFE products and the single ERA5 grib out of a scene folder.
    """
    s1_outer = glob.glob(os.path.join(scene_dir, "S1_*_pre.SAFE"))
    s2_outer = glob.glob(os.path.join(scene_dir, "S2_*_pre.SAFE"))
    era5_gribs = glob.glob(os.path.join(scene_dir, "*.grib"))

    if len(s1_outer) != 1:
        raise ValueError(f"Expected exactly one S1_*_pre.SAFE entry under {scene_dir}, found {s1_outer}")
    if len(s2_outer) != 1:
        raise ValueError(f"Expected exactly one S2_*_pre.SAFE entry under {scene_dir}, found {s2_outer}")
    if len(era5_gribs) != 1:
        raise ValueError(f"Expected exactly one .grib file under {scene_dir}, found {era5_gribs}")

    s1_safe_dir = ensure_extracted_safe_dir(s1_outer[0], extract_root, "S1 pre-scene")
    s2_safe_dir = ensure_extracted_safe_dir(s2_outer[0], extract_root, "S2 pre-scene")
    return s1_safe_dir, s2_safe_dir, era5_gribs[0]


def load_scene_tiles(scene_dir, dem_output_dir, extract_root):
    """
    Runs a new, unlabeled scene through the same S1/S2/ERA5 loading + zonal
    aggregation pipeline used in build_tile_cache.py, then returns the
    sorted era5 lat/lon arrays needed to map each tile (i, j) back to a real coord.
    """
    s1_safe_dir, s2_safe_dir, era5_grib = find_scene_inputs(scene_dir, extract_root)

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


def build_tile_dataset(tiles, statistic_means, era5_seq_len):
    """
    Wraps a new scene's tiles in the real model.dataset.dataset class so
    inference uses the exact same mean-imputation + flattening logic
    training uses (__getitem__).
    """
    ds = TileDataset.__new__(TileDataset)
    ds.data_list = list(tiles.items())
    ds.statistic_means = statistic_means
    ds.era5_seq_len = era5_seq_len
    return ds


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to a train.py checkpoint (.pt).")
    parser.add_argument("--scene-dir", required=True,
                         help="A single scene folder exactly as published in the dataset (e.g. a "
                              "fires/{state}/{scene}/ or controls/{state}/{scene}/ folder) -- see "
                              "module docstring for the expected layout.")
    parser.add_argument("--cache-dir", default="tile_cache",
                         help="Only needed as a fallback for older checkpoints that don't have "
                              "normalization stats saved -- must be the same --cache-dir used to "
                              "train the checkpoint (see module docstring caveat).")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dem-dir", default="/tmp/wildfire_inference_dem")
    parser.add_argument("--extract-dir", default="/tmp/wildfire_inference_extract",
                         help="Where to unzip the scene's *_pre.SAFE archives before loading.")
    parser.add_argument("--inference-dir", default="inference",
                         help="Root output dir; results land under "
                              "<inference-dir>/<run_id>_<checkpoint_stem>/<scene_id>_risk_grid.csv "
                              "unless --output-csv overrides the path.")
    parser.add_argument("--output-csv", default=None,
                         help="Optional explicit path to write the per-tile risk grid to, "
                              "overriding the default inference/ layout.")
    return parser


def default_output_path(inference_dir, checkpoint_path, scene_dir):
    checkpoint_stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    run_id = os.path.basename(os.path.dirname(os.path.abspath(checkpoint_path)))
    checkpoint_id = f"{run_id}_{checkpoint_stem}" if run_id else checkpoint_stem
    scene_id = os.path.basename(os.path.normpath(scene_dir))
    return os.path.join(inference_dir, checkpoint_id, f"{scene_id}_risk_grid.csv"), scene_id


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
        print(f"Checkpoint has no saved normalization stats (older run), so re-deriving from "
              f"{args.cache_dir} (val_frac={args.val_frac}, test_frac={args.test_frac}, "
              f"seed={args.seed}). These are required to match the run that produced this checkpoint.")
        data = build_datasets(args.cache_dir, args.val_frac, args.test_frac, args.seed)
        train_ds = data["train_ds"]
        statistic_means = train_ds.inner.statistic_means
        spatial_mean, spatial_std = train_ds.spatial_mean, train_ds.spatial_std
        temporal_mean, temporal_std = train_ds.temporal_mean, train_ds.temporal_std
        era5_seq_len = train_ds.inner.era5_seq_len

    metadata_path = os.path.join(args.scene_dir, "metadata.json")
    scene_meta = {}
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            scene_meta = json.load(f)

    tiles, era5_lats, era5_longs, cutoff_datetime = load_scene_tiles(
        args.scene_dir, args.dem_dir, args.extract_dir)
    tiles = apply_embargo(tiles, embargo_days)

    tile_ds = build_tile_dataset(tiles, statistic_means, era5_seq_len)
    tile_keys = [key for key, _ in tile_ds.data_list]  # same order dataset.__getitem__ indexes into

    # single forward pass over every tile in the scene
    loader = torch.utils.data.DataLoader(tile_ds, batch_size=len(tile_ds), shuffle=False)
    x_spatial_batch, x_temporal_batch = next(iter(loader))
    x_spatial_batch = (x_spatial_batch - spatial_mean) / spatial_std
    x_temporal_batch = (x_temporal_batch - temporal_mean) / temporal_std

    with torch.no_grad():
        head1, _, _, _ = model(x_spatial_batch, x_temporal_batch)
    probabilities = head1.squeeze(1).tolist()

    rows = [
        {"i": i, "j": j, "lat": float(era5_lats[i]), "lon": float(era5_longs[j]), "fire_probability_30d": prob}
        for (i, j), prob in zip(tile_keys, probabilities)
    ]
    grid = pd.DataFrame(rows).sort_values(["i", "j"]).reset_index(drop=True)
    scene_verdict = float(grid["fire_probability_30d"].mean())
    n_high_risk = int((grid["fire_probability_30d"] > 0.5).sum())

    output_csv, scene_id = default_output_path(args.inference_dir, args.checkpoint, args.scene_dir)
    if args.output_csv:
        output_csv = args.output_csv

    print(f"\nScene: {scene_id}"
          + (f" ({scene_meta.get('state')}, {scene_meta.get('fire_name') or scene_meta.get('control_id')})"
             if scene_meta else ""))
    print(f"Forecast cutoff: {cutoff_datetime.date()} (~30-day horizon)")
    print(f"{len(grid)} tiles scored -- {n_high_risk} ({n_high_risk / len(grid):.0%}) above 0.5 probability")
    print(f"Per-tile stats: mean {grid['fire_probability_30d'].mean():.3f}  "
          f"max {grid['fire_probability_30d'].max():.3f}  "
          f"min {grid['fire_probability_30d'].min():.3f}")
    print(grid.to_string(index=False))
    print(f"\nVERDICT -- aggregated (mean-across-tiles) fire probability for this scene "
          f"over the next ~30 days: {scene_verdict:.3f}")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    grid.to_csv(output_csv, index=False)
    print(f"\nSaved per-tile risk grid to {output_csv}")


if __name__ == "__main__":
    main()
