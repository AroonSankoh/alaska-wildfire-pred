"""
Patches era5_stats in place across an existing tile_cache/ built by build_tile_cache.py,
switching from a single mean-per-variable to a fixed 30-day daily sequence per variable
(see zonal_aggregator.py's _resample_daily_last_n). s1_stats/s2_stats are left untouched,
and only the ERA5 grib (small) is re-downloaded per scene -- no SAFE zips, RTC, or DEM work.

Tile (i, j) keys are reused as-is from the existing cache, so this assumes the ERA5 grid
(lat/lon bins) for a given scene hasn't changed since the original build_tile_cache.py run.

Usage:
    python scripts/rebuild_era5_stats.py --cache-dir tile_cache --limit 3   # smoke-test
    python scripts/rebuild_era5_stats.py --cache-dir tile_cache            # full run
    python scripts/rebuild_era5_stats.py --cache-dir tile_cache --skip-existing
"""

import argparse
import json
import os
import pickle
import shutil
import sys
import time
import traceback

import boto3

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)
from data.loaders import load_era5_vars
from data.aggregator.zonal_aggregator import _resample_daily_last_n, N_ERA5_DAYS
from scripts.build_tile_cache import (
    BUCKET_NAME, FIRES_PREFIX, CONTROLS_PREFIX, discover_scenes,
    download_grib, era5_cutoff_from_key, cache_path_for, with_retries,
)

ERA5_VARS = ('u10', 'v10', 'd2m', 't2m', 'tp')
s3 = boto3.client("s3")


def rebuild_scene(metadata_key, scene_prefix, cache_dir, tmp_dir):
    """
    Re-downloads one scene's ERA5 grib, resamples to the new fixed-length daily
    sequence, and overwrites era5_stats in the existing cached pickle in place.
    """
    cache_path = cache_path_for(cache_dir, scene_prefix)
    if not os.path.exists(cache_path):
        return "skipped (not in cache)"

    with open(cache_path, "rb") as f:
        record = pickle.load(f)

    resp = s3.get_object(Bucket=BUCKET_NAME, Key=metadata_key)
    meta = json.loads(resp["Body"].read())
    era5_key = scene_prefix + meta["contents"]["ERA5"]
    cutoff_datetime = era5_cutoff_from_key(era5_key)

    scene_tmp = os.path.join(tmp_dir, scene_prefix.rstrip("/").replace("/", "_"))
    try:
        era5_local = download_grib(era5_key, scene_tmp)
        era5_data = load_era5_vars(era5_local, cutoff_datetime=cutoff_datetime)
        era5_sorted = {k: v.sortby('latitude') for k, v in era5_data.items()}
        era5_daily = {k: _resample_daily_last_n(v) for k, v in era5_sorted.items()}

        n_patched = 0
        for (i, j), tile in record["tiles"].items():
            tile["era5_stats"] = {
                var: era5_daily[var].isel(latitude=i, longitude=j).values.astype(float).tolist()
                for var in ERA5_VARS
            }
            n_patched += 1

        # write to a temp file then rename, so a crash mid-write can't corrupt the cache
        tmp_path = cache_path + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(record, f)
        os.replace(tmp_path, cache_path)
        return f"patched {n_patched} tiles"
    finally:
        if os.path.isdir(scene_tmp):
            shutil.rmtree(scene_tmp, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="tile_cache")
    parser.add_argument("--tmp-dir", default="/tmp/wildfire_era5_rebuild")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N scenes (per kind), for smoke-testing.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip scenes whose cached era5_stats already look like sequences (resume after interruption).")
    args = parser.parse_args()

    os.makedirs(args.tmp_dir, exist_ok=True)
    failures_log_path = os.path.join(args.cache_dir, "rebuild_era5_failures.log")

    n_done = n_skipped = n_failed = 0
    for kind in ("fire", "control"):
        scenes = discover_scenes(kind)
        if args.limit:
            scenes = scenes[: args.limit]
        print(f"\n=== {kind}: {len(scenes)} scenes ===")

        for metadata_key, scene_prefix in scenes:
            scene_id = scene_prefix.rstrip("/").rsplit("/", 1)[-1]
            cache_path = cache_path_for(args.cache_dir, scene_prefix)

            if args.skip_existing and os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    record = pickle.load(f)
                sample_tile = next(iter(record["tiles"].values()), None)
                already_sequence = (
                    sample_tile is not None
                    and isinstance(sample_tile["era5_stats"]["u10"], list)
                    and len(sample_tile["era5_stats"]["u10"]) == N_ERA5_DAYS
                )
                if already_sequence:
                    print(f"  [skip] {scene_id} (already rebuilt)")
                    n_skipped += 1
                    continue

            t0 = time.time()
            try:
                status = with_retries(rebuild_scene, metadata_key, scene_prefix,
                                       args.cache_dir, args.tmp_dir, label=f"rebuild({scene_id})")
                print(f"  [done] {scene_id} -> {status} ({time.time() - t0:.1f}s)")
                n_done += 1
            except Exception as e:
                error = f"{scene_id}: {e}\n{traceback.format_exc()}"
                print(f"  [FAILED] {error.splitlines()[0]}")
                with open(failures_log_path, "a") as f:
                    f.write(error + "\n\n")
                n_failed += 1

    print(f"\n{n_done} scenes patched, {n_skipped} skipped, {n_failed} failed.")
    if n_failed:
        print(f"Failure details in {failures_log_path}")


if __name__ == "__main__":
    main()
