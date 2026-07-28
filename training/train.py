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
from datetime import datetime

import boto3
import matplotlib
matplotlib.use("Agg")  # headless -- EC2 has no display to render to
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

from model import dataset as WildfireDataset
from model import WildfireModel
from model import LabeledTileDataset
from model import S1_KEYS, S2_KEYS, ERA5_KEYS, SPATIAL_KEYS, TEMPORAL_KEYS
from model import is_missing_value
from model.augmentation import TileAugmenter

BUCKET_NAME = "wildfire-scenes-s3-202802195212-eu-central-1-an"

def load_records(cache_dir):
    records = []
    for path in sorted(glob.glob(os.path.join(cache_dir, "**", "*.pkl"), recursive=True)):
        with open(path, "rb") as f:
            records.append(pickle.load(f))
    return records


def group_by_fire(records):
    """
    Groups each fire with its matched controls so a train/val/test split
    never puts a fire in one split and its matching controls in another.
    """
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

def merge_tiles(records):
    """
    Merges tiles across scenes and trackes per-tile labels
    """
    merged_tiles = {}
    labels = []
    for record in records:
        for tile_key, tile in record["tiles"].items():
            merged_key = f"{record['scene_id']}__{tile_key}"
            merged_tiles[merged_key] = tile
            labels.append(record["label"])
    return merged_tiles, labels


def _safe_std(vals):
    # falls back to 1.0 both when there's too little data to compute a real std 
    if len(vals) <= 1:
        return 1.0
    std_val = float(np.std(vals))
    return std_val if std_val > 0 else 1.0


def build_feature_stds(train_tiles):
    """
    Dataset-wide (train-split-only, to avoid val/test leakage) std per
    feature, same pattern as train_ds.statistic_means but with np.std.
    """
    stds = {}
    for key in S1_KEYS:
        vals = [t["s1_stats"][key] for t in train_tiles.values()
                if t["s1_stats"] is not None and not is_missing_value(t["s1_stats"][key])]
        stds[key] = _safe_std(vals)
    for key in S2_KEYS:
        vals = [t["s2_stats"][key] for t in train_tiles.values()
                if t["s2_stats"] is not None and not is_missing_value(t["s2_stats"][key])]
        stds[key] = _safe_std(vals)
    for key in ERA5_KEYS:
        vals = [t["era5_stats"][key] for t in train_tiles.values()
                if t["era5_stats"] is not None and not is_missing_value(t["era5_stats"][key])]
        stds[key] = _safe_std(vals)
    return stds


def run_epoch(model, loader, optimizer, pos_weight, train, device):
    model.train() if train else model.eval()

    total_loss = 0.0
    n_correct = 0
    n_tiles = 0
    # confusion-matrix counts, accumulated across all 3 horizon heads
    n_tp = n_fp = n_fn = n_tn = 0.0

    with torch.set_grad_enabled(train):
        for x_spatial, x_temporal, y in loader:
            x_spatial, x_temporal, y = x_spatial.to(device), x_temporal.to(device), y.to(device)
            batch_size = x_spatial.shape[0]

            head1, head2, head3, _ = model(x_spatial, x_temporal)
            pred = torch.cat([head1, head2, head3], dim=1)          # (batch, 3)
            target = y.unsqueeze(1).expand(-1, 3)                   # same fire/control label applied to all 3 horizons

            # class weighting: prevents model from learning lazy logic like "since 3/4 of samples are controls, just predict control for 75% accuracy"
            weight = torch.where(target > 0.5, pos_weight, torch.ones_like(target))
            loss = F.binary_cross_entropy(pred, target, weight=weight)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # confusion-matrix statistics
            total_loss += loss.item() * batch_size
            pred_positive = pred > 0.5
            target_positive = target > 0.5
            n_correct += (pred_positive == target_positive).sum().item() / 3.0
            n_tp += (pred_positive & target_positive).sum().item()
            n_fp += (pred_positive & ~target_positive).sum().item()
            n_fn += (~pred_positive & target_positive).sum().item()
            n_tn += (~pred_positive & ~target_positive).sum().item()
            n_tiles += batch_size

    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0.0
    recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = n_tn / (n_tn + n_fp) if (n_tn + n_fp) > 0 else 0.0
    balanced_acc = (recall + specificity) / 2

    return {
        "loss": total_loss / n_tiles,
        "acc": n_correct / n_tiles,
        "balanced_acc": balanced_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def plot_loss_curve(train_losses, val_losses, save_path):
    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label="train loss")
    plt.plot(epochs, val_losses, label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


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

    # each run gets its own timestamped dir under --output-dir, holding the best/final
    # checkpoints and the loss-curve plot together as one package
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Run directory: {run_dir}")

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
    feature_stds = build_feature_stds(train_tiles)
    augmenter = TileAugmenter(feature_stds, seed=args.seed)

    # normalization stats, which are also train-split-only, applied identically to train/val/test
    spatial_mean = torch.tensor([train_ds_unaugmented.statistic_means[f"mean_{k}"] for k in SPATIAL_KEYS]).float()
    spatial_std = torch.tensor([feature_stds[k] for k in SPATIAL_KEYS]).float()
    temporal_mean = torch.tensor([train_ds_unaugmented.statistic_means[f"mean_{k}"] for k in TEMPORAL_KEYS]).float()
    temporal_std = torch.tensor([feature_stds[k] for k in TEMPORAL_KEYS]).float()

    train_ds = LabeledTileDataset(train_tiles, train_labels, spatial_mean, spatial_std,
                                   temporal_mean, temporal_std, augmenter=augmenter)
    val_ds = LabeledTileDataset(val_tiles, val_labels, spatial_mean, spatial_std,
                                 temporal_mean, temporal_std, augmenter=None)
    test_ds = LabeledTileDataset(test_tiles, test_labels, spatial_mean, spatial_std,
                                  temporal_mean, temporal_std, augmenter=None)
    print(f"Tiles: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test")

    # class weighting for BCE: weight the minority (fire) class by n_neg/n_pos on the train split
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight_value = (n_neg / n_pos) if n_pos > 0 else 1.0
    pos_weight = torch.tensor(pos_weight_value).float().to(device)
    print(f"Class weighting: {int(n_pos)} fire tiles, {int(n_neg)} control tiles, pos_weight={pos_weight_value:.3f}")

    x_spatial0, x_temporal0, _ = train_ds[0]
    model = WildfireModel(x_spatial0.shape[0], x_temporal0.shape[0],
                           args.embedding_dim, args.n_layers, args.n_head).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    best_val_loss = float("inf")
    best_path = os.path.join(run_dir, "best_model.pt")
    train_losses, val_losses = [], []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, pos_weight, train=True, device=device)
        val_metrics = run_epoch(model, val_loader, optimizer, pos_weight, train=False, device=device)
        print(f"epoch {epoch:3d} | train loss {train_metrics['loss']:.4f} acc {train_metrics['acc']:.3f} "
              f"| val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.3f} "
              f"bal_acc {val_metrics['balanced_acc']:.3f} "
              f"precision {val_metrics['precision']:.3f} recall {val_metrics['recall']:.3f} "
              f"f1 {val_metrics['f1']:.3f}")
        train_losses.append(train_metrics["loss"])
        val_losses.append(val_metrics["loss"])
        val_loss = val_metrics["loss"]

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

    # evaluate on the best-val-loss checkpoint, not whatever model holds after the last epoch
    best_checkpoint = torch.load(best_path, map_location=device)
    best_model = WildfireModel(x_spatial0.shape[0], x_temporal0.shape[0],
                                args.embedding_dim, args.n_layers, args.n_head).to(device)
    best_model.load_state_dict(best_checkpoint["model_state_dict"])
    print(f"Loaded best checkpoint (epoch {best_checkpoint['epoch']}, "
          f"val loss {best_checkpoint['val_loss']:.4f}) for final test evaluation")

    test_metrics = run_epoch(best_model, test_loader, optimizer, pos_weight, train=False, device=device)
    test_loss, test_acc = test_metrics["loss"], test_metrics["acc"]
    print(f"\nFinal test | loss {test_metrics['loss']:.4f} acc {test_metrics['acc']:.3f} "
          f"bal_acc {test_metrics['balanced_acc']:.3f} precision {test_metrics['precision']:.3f} "
          f"recall {test_metrics['recall']:.3f} f1 {test_metrics['f1']:.3f}")

    final_path = os.path.join(run_dir, "final_model.pt")
    torch.save({"model_state_dict": model.state_dict(),
                "spatial_input_dim": x_spatial0.shape[0],
                "temporal_input_dim": x_temporal0.shape[0],
                "embedding_dim": args.embedding_dim,
                "n_layers": args.n_layers,
                "n_head": args.n_head,
                "test_loss": test_loss,
                "test_acc": test_acc}, final_path)

    loss_curve_path = os.path.join(run_dir, "loss_curve.png")
    plot_loss_curve(train_losses, val_losses, loss_curve_path)
    print(f"Saved loss curve to {loss_curve_path}")

    if args.upload_to_s3:
        s3 = boto3.client("s3")
        run_prefix = f"{args.s3_model_prefix}{run_id}/"
        for local_path in (best_path, final_path, loss_curve_path):
            s3.upload_file(local_path, BUCKET_NAME, run_prefix + os.path.basename(local_path))
        print(f"Uploaded run artifacts to s3://{BUCKET_NAME}/{run_prefix}")


if __name__ == "__main__":
    main()
