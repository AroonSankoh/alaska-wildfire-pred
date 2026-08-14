"""
Random-search hyperparameter sweep over WildfireModel + training config.

Builds the train/val datasets ONCE via training/train.py's build_datasets()
(same split, normalization, and class weighting every trial gets) so a
20-trial sweep doesn't re-read tile_cache/ from disk 20 times. Each trial
trains a fresh model for --trial-epochs epochs (short, since this is a
search over configs, not a final training run) and is scored on val
balanced accuracy -- same metric main()'s checkpoint selection now uses,
for consistency.

Does NOT automatically retrain the winning config at full length --  it
prints a ready-to-run train.py command for that instead, so you can decide
how many epochs to actually commit to before kicking off a longer run.

Usage:
    python scripts/hyperparameter_sweep.py --cache-dir tile_cache --n-trials 20 --trial-epochs 8
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "training"))

from model import WildfireModel
from train import build_datasets, train_model

EMBEDDING_DIM_CHOICES = [16, 32, 64, 128]
TEMPORAL_HIDDEN_DIM_CHOICES = [16, 32, 64]
N_HEAD_CHOICES = [1, 2, 4, 8]
BATCH_SIZE_CHOICES = [32, 64, 128, 256]
N_LAYERS_CHOICES = [1, 2, 3, 4]
WEIGHT_DECAY_CHOICES = [0.0, 1e-5, 1e-4, 1e-3]
LR_LOG_RANGE = (-4, -2)  # log10 range -> [1e-4, 1e-2]


def sample_config(rng):
    return {
        "lr": 10 ** rng.uniform(*LR_LOG_RANGE),
        "batch_size": rng.choice(BATCH_SIZE_CHOICES),
        "embedding_dim": rng.choice(EMBEDDING_DIM_CHOICES),
        "temporal_hidden_dim": rng.choice(TEMPORAL_HIDDEN_DIM_CHOICES),
        "n_layers": rng.choice(N_LAYERS_CHOICES),
        "n_head": rng.choice(N_HEAD_CHOICES),
        "weight_decay": rng.choice(WEIGHT_DECAY_CHOICES),
    }


def run_trial(config, data, pos_weight, epochs, device):
    x_spatial0, x_temporal0, _ = data["train_ds"][0]
    model = WildfireModel(
        x_spatial0.shape[0], x_temporal0.shape[-1], config["embedding_dim"], config["n_layers"],
        config["n_head"], temporal_hidden_dim=config["temporal_hidden_dim"],
    ).to(device)

    _, best_val_metrics, _, _ = train_model(
        model, data["train_ds"], data["val_ds"], pos_weight,
        epochs=epochs, batch_size=config["batch_size"], lr=config["lr"], num_workers=0,
        device=device, weight_decay=config["weight_decay"], verbose=False,
    )
    return best_val_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="tile_cache")
    parser.add_argument("--output-dir", default="sweeps")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--trial-epochs", type=int, default=8)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42, help="Fixed train/val/test split seed, shared across every trial.")
    parser.add_argument("--sweep-seed", type=int, default=0, help="Seed for the random-search sampling itself.")
    args = parser.parse_args()

    # each sweep gets its own timestamped dir under --output-dir, same pattern as train.py's run_dir
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Run directory: {run_dir}")

    # built once and reused across every trial, the split/normalization/pos_weight stay fixed, 
    # with only the model architecture and training hyperparameters vary trial to trial
    data = build_datasets(args.cache_dir, args.val_frac, args.test_frac, args.seed)
    pos_weight = torch.tensor(data["pos_weight_value"]).float().to(device)
    print(f"Loaded {data['n_records']} cached scenes ({data['n_fire']} fires, {data['n_control']} controls)")
    print(f"Tiles: {len(data['train_ds'])} train, {len(data['val_ds'])} val")
    print(f"pos_weight={data['pos_weight_value']:.3f}\n")

    rng = random.Random(args.sweep_seed)
    results = []

    for trial in range(1, args.n_trials + 1):
        config = sample_config(rng)
        start = time.time()
        try:
            best_val_metrics = run_trial(config, data, pos_weight, args.trial_epochs, device)
            elapsed = time.time() - start
            print(f"trial {trial:3d}/{args.n_trials} | {config} "
                  f"-> val bal_acc {best_val_metrics['balanced_acc']:.4f} "
                  f"(f1 {best_val_metrics['f1']:.4f}, loss {best_val_metrics['loss']:.4f}) "
                  f"[{elapsed:.0f}s]")
            results.append({"config": config, "val_metrics": best_val_metrics, "elapsed_sec": elapsed})
        except Exception as e:
            print(f"trial {trial:3d}/{args.n_trials} | {config} -> FAILED: {e}")
            results.append({"config": config, "error": str(e)})

    results_path = os.path.join(run_dir, "sweep_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full results to {results_path}")

    successful = [r for r in results if "val_metrics" in r]
    if not successful:
        print("No trial completed successfully -- check the FAILED lines above.")
        return

    successful.sort(key=lambda r: r["val_metrics"]["balanced_acc"], reverse=True)
    print("\nTop 5 configs by val balanced accuracy:")
    for r in successful[:5]:
        print(f"  bal_acc {r['val_metrics']['balanced_acc']:.4f} | {r['config']}")

    best = successful[0]["config"]
    print(f"\nBest config: {best}")
    print("Rerun with train.py at full length to confirm, e.g.:")
    print(f"  python training/train.py --cache-dir {args.cache_dir} --epochs 30 "
          f"--batch-size {best['batch_size']} --lr {best['lr']:.6f} "
          f"--embedding-dim {best['embedding_dim']} --temporal-hidden-dim {best['temporal_hidden_dim']} "
          f"--n-layers {best['n_layers']} --n-head {best['n_head']} --weight-decay {best['weight_decay']}")


if __name__ == "__main__":
    main()
