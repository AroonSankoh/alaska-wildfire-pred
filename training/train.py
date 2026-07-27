"""
Trains WildfireModel off the cached tile pickles produced by
build_tile_cache.py. Never touches S3/SAFE zips -- pure local tensor work,
which is why this is expected to run fine on CPU (no CUDA needed).

Uses a real torch.utils.data.DataLoader with batch collation -- this
assumes model/architecture.py has been updated to accept a leading batch
dimension on x_spatial/x_temporal (WildfireModel.forward and
TransformerEncoder.forward), rather than the single-tile-at-a-time
gradient-accumulation workaround this script used previously.

Labels: fire tiles get label 1.0, control tiles get label 0.0 (uniformly
across all three 1mo/3mo/6mo heads, per your call on this). If you later get
real per-horizon ground truth, swap the target-building logic in run_epoch().

Usage:
    python train.py --cache-dir tile_cache --epochs 20 --batch-size 32
"""

import argparse
import glob
import os
import pickle
import random
import sys

import boto3
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

from model import dataset as WildfireDataset
from model import WildfireModel
from model.augmentation import TileAugmenter

BUCKET_NAME = "wildfire-scenes-s3-202802195212-eu-central-1-an"

S1_KEYS = ["vh_band_mean", "vh_band_std", "vv_band_mean", "vv_band_std"]
S2_KEYS = ["red_mean", "red_std", "green_mean", "green_std", "nir_mean", "nir_std",
           "swir_mean", "swir_std", "ndvi_mean", "ndvi_std", "nbr_mean", "nbr_std"]
ERA5_KEYS = ["u10", "v10", "d2m", "t2m", "tp"]
SPATIAL_KEYS = S1_KEYS + S2_KEYS
TEMPORAL_KEYS = ERA5_KEYS


# --------------------------------------------------------------------------
# Loading cached records + grouping fires with their matched controls
# --------------------------------------------------------------------------

def load_records(cache_dir):
    # tile_cache now mirrors the S3 directory structure (fires/{state}/... , controls/{state}/{fire_control}/...)
    # rather than a flat directory, so this has to walk recursively rather than glob the top level only
    records = []
    for path in sorted(glob.glob(os.path.join(cache_dir, "**", "*.pkl"), recursive=True)):
        with open(path, "rb") as f:
            records.append(pickle.load(f))
    return records


def group_by_fire(records):
    """Groups each fire with its matched controls so a train/val/test split
    never puts a fire in one split and its own controls in another."""
    fires = [r for r in records if r["kind"] == "fire"]
    controls = [r for r in records if r["kind"] == "control"]

    groups = []
    for fire in fires:
        matched = [c for c in controls if c["matched_fire_event_id"] == fire["event_id"]]
        groups.append([fire] + matched)

    matched_ids = {c["scene_id"] for g in groups for c in g[1:]}
    orphan_controls = [c for c in controls if c["scene_id"] not in matched_ids]
    if orphan_controls:
        print(f"WARNING: {len(orphan_controls)} controls have no matching fire in this cache "
              f"and will be excluded from the split: {[c['scene_id'] for c in orphan_controls]}")

    return groups


def split_groups(groups, val_frac, test_frac, seed):
    rng = random.Random(seed)
    groups = groups[:]
    rng.shuffle(groups)

    n = len(groups)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))

    test_groups = groups[:n_test]
    val_groups = groups[n_test:n_test + n_val]
    train_groups = groups[n_test + n_val:]

    flatten = lambda gs: [r for g in gs for r in g]
    return flatten(train_groups), flatten(val_groups), flatten(test_groups)


# --------------------------------------------------------------------------
# Dataset wrapper: merges tiles across scenes, tracks per-tile labels,
# optionally applies TileAugmenter (train split only)
# --------------------------------------------------------------------------

def merge_tiles(records):
    merged_tiles = {}
    labels = []
    for record in records:
        for tile_key, tile in record["tiles"].items():
            merged_key = f"{record['scene_id']}__{tile_key}"
            merged_tiles[merged_key] = tile
            labels.append(record["label"])
    return merged_tiles, labels


class LabeledTileDataset(torch.utils.data.Dataset):
    def __init__(self, tiles, labels, augmenter=None):
        self.inner = WildfireDataset(tiles)
        self.labels = labels
        self.augmenter = augmenter
        assert len(self.inner) == len(self.labels), \
            f"tile/label count mismatch: {len(self.inner)} vs {len(self.labels)}"

        if augmenter is not None:
            self.impute_spatial = torch.tensor(
                [self.inner.statistic_means[f"mean_{k}"] for k in SPATIAL_KEYS]).float()
            self.impute_temporal = torch.tensor(
                [self.inner.statistic_means[f"mean_{k}"] for k in TEMPORAL_KEYS]).float()

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        x_spatial, x_temporal = self.inner[idx]
        if self.augmenter is not None:
            x_spatial, x_temporal = self.augmenter.jitter(x_spatial, x_temporal, SPATIAL_KEYS, TEMPORAL_KEYS)
            x_spatial, x_temporal = self.augmenter.dropout(
                x_spatial, x_temporal, self.impute_spatial, self.impute_temporal)
        y = torch.tensor(self.labels[idx]).float()
        return x_spatial, x_temporal, y


def _is_missing(value):
    # mirrors model/dataset.py's _is_missing -- a tile can carry an individual NaN (e.g. a
    # single-pixel tile's std, undefined under pandas' ddof=1) without its whole stats dict
    # being None, and an unfiltered NaN poisons np.std() into NaN for the whole feature.
    # Also catches Inf (e.g. a divide-by-zero further upstream), which would otherwise
    # poison np.std() into Inf instead.
    return value is None or (isinstance(value, float) and not np.isfinite(value))


def build_feature_stds(train_tiles, train_ds):
    """Dataset-wide (train-split-only, to avoid val/test leakage) std per
    feature, same pattern as train_ds.statistic_means but with np.std."""
    stds = {}
    for key in S1_KEYS:
        vals = [t["s1_stats"][key] for t in train_tiles.values()
                if t["s1_stats"] is not None and not _is_missing(t["s1_stats"][key])]
        stds[key] = float(np.std(vals)) if len(vals) > 1 else 1.0
    for key in S2_KEYS:
        vals = [t["s2_stats"][key] for t in train_tiles.values()
                if t["s2_stats"] is not None and not _is_missing(t["s2_stats"][key])]
        stds[key] = float(np.std(vals)) if len(vals) > 1 else 1.0
    for key in ERA5_KEYS:
        vals = [t["era5_stats"][key] for t in train_tiles.values()
                if t["era5_stats"] is not None and not _is_missing(t["era5_stats"][key])]
        stds[key] = float(np.std(vals)) if len(vals) > 1 else 1.0
    return stds


# --------------------------------------------------------------------------
# Training loop -- real batched forward passes via DataLoader
# --------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, loss_fn, train, device):
    model.train() if train else model.eval()

    total_loss = 0.0
    n_correct = 0
    n_tiles = 0

    with torch.set_grad_enabled(train):
        for x_spatial, x_temporal, y in loader:
            x_spatial, x_temporal, y = x_spatial.to(device), x_temporal.to(device), y.to(device)
            batch_size = x_spatial.shape[0]

            head1, head2, head3, _ = model(x_spatial, x_temporal)
            pred = torch.cat([head1, head2, head3], dim=1)          # (batch, 3)
            target = y.unsqueeze(1).expand(-1, 3)                    # same fire/control label applied to all 3 horizons
            loss = loss_fn(pred, target)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * batch_size
            n_correct += ((pred > 0.5).float() == target).sum().item() / 3.0
            n_tiles += batch_size

    return total_loss / n_tiles, n_correct / n_tiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="tile_cache")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--s3-model-prefix", default="models/")
    parser.add_argument("--upload-to-s3", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=1)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    records = load_records(args.cache_dir)
    print(f"Loaded {len(records)} cached scenes ({sum(1 for r in records if r['kind']=='fire')} fires, "
          f"{sum(1 for r in records if r['kind']=='control')} controls)")

    groups = group_by_fire(records)
    train_records, val_records, test_records = split_groups(groups, args.val_frac, args.test_frac, args.seed)
    print(f"Split: {len(train_records)} train scenes, {len(val_records)} val scenes, {len(test_records)} test scenes")

    train_tiles, train_labels = merge_tiles(train_records)
    val_tiles, val_labels = merge_tiles(val_records)
    test_tiles, test_labels = merge_tiles(test_records)

    # feature_stds computed on TRAIN split only, to avoid val/test leakage into augmentation scale
    train_ds_unaugmented = WildfireDataset(train_tiles)
    feature_stds = build_feature_stds(train_tiles, train_ds_unaugmented)
    augmenter = TileAugmenter(feature_stds, seed=args.seed)

    train_ds = LabeledTileDataset(train_tiles, train_labels, augmenter=augmenter)
    val_ds = LabeledTileDataset(val_tiles, val_labels, augmenter=None)
    test_ds = LabeledTileDataset(test_tiles, test_labels, augmenter=None)
    print(f"Tiles: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test")

    x_spatial0, x_temporal0, _ = train_ds[0]
    model = WildfireModel(x_spatial0.shape[0], x_temporal0.shape[0],
                           args.embedding_dim, args.n_layers, args.n_head).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCELoss()

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    best_val_loss = float("inf")
    best_path = os.path.join(args.output_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, loss_fn, train=True, device=device)
        val_loss, val_acc = run_epoch(model, val_loader, optimizer, loss_fn, train=False, device=device)
        print(f"epoch {epoch:3d} | train loss {train_loss:.4f} acc {train_acc:.3f} "
              f"| val loss {val_loss:.4f} acc {val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state_dict": model.state_dict(),
                        "spatial_input_dim": x_spatial0.shape[0],
                        "temporal_input_dim": x_temporal0.shape[0],
                        "embedding_dim": args.embedding_dim,
                        "n_layers": args.n_layers,
                        "n_head": args.n_head,
                        "epoch": epoch,
                        "val_loss": val_loss}, best_path)
            print(f"  -> new best (val loss {val_loss:.4f}), saved to {best_path}")

    test_loss, test_acc = run_epoch(model, test_loader, optimizer, loss_fn, train=False, device=device)
    print(f"\nFinal test | loss {test_loss:.4f} acc {test_acc:.3f}")

    final_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save({"model_state_dict": model.state_dict(),
                "spatial_input_dim": x_spatial0.shape[0],
                "temporal_input_dim": x_temporal0.shape[0],
                "embedding_dim": args.embedding_dim,
                "n_layers": args.n_layers,
                "n_head": args.n_head,
                "test_loss": test_loss,
                "test_acc": test_acc}, final_path)

    if args.upload_to_s3:
        s3 = boto3.client("s3")
        for local_path in (best_path, final_path):
            s3.upload_file(local_path, BUCKET_NAME, args.s3_model_prefix + os.path.basename(local_path))
        print(f"Uploaded checkpoints to s3://{BUCKET_NAME}/{args.s3_model_prefix}")


if __name__ == "__main__":
    main()
