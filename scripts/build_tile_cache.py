"""
One-time preprocessing pass: walks fires/ and controls/ in S3, downloads +
unzips each scene's S1_pre / S2_pre / ERA5 SAFE/grib product, runs them
through the existing loaders + aggregate(), and caches the resulting per-fire 
`tiles` dict to disk (and optionally back up to S3) so train.py never only 
has to touch a zipped SAFE file once.

Only *_pre data is used (matches aggregate()'s signature, which only takes
one S1/S2/ERA5 set) -- this model predicts *future* risk from pre-fire
conditions, so S1_post/S2_post are never needed as model input.

ASSUMPTIONS -- please confirm/adjust before running against the live bucket:
  1. ERA5 content values are NOT zipped (exists() treats them as exact
     .grib keys) -- downloaded directly, no extraction needed.
  2. Inside each unzipped SAFE product, band files are located via glob
     patterns (since exact filenames embed the scene ID and differ per
     scene) rather than hardcoded paths like test_pipeline.ipynb uses.
     Patterns are based on the DOME_fire example folder structure. If any
     fire's internal SAFE layout differs, the glob for that band will find
     zero/multiple matches and the scene will be logged as FAILED rather
     than silently grab the wrong file.

Usage:
    python build_tile_cache.py --limit 3               # smoke test on 3 scenes
    python build_tile_cache.py                          # full run
    python build_tile_cache.py --skip-existing           # resume after interruption
"""

import argparse
import glob
import json
import os
import pickle
import shutil
import sys
import time
import traceback
import zipfile
import rasterio

import boto3

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)
from data.loaders import load_era5_vars, load_sentinel1_bands, load_sentinel2_bands
from data.aggregator import aggregate

BUCKET_NAME = "wildfire-scenes-s3-202802195212-eu-central-1-an"
FIRES_PREFIX = "fires"
CONTROLS_PREFIX = "controls"

s3 = boto3.client("s3")


# --------------------------------------------------------------------------
# S3 discovery / download helpers
# --------------------------------------------------------------------------

def list_keys_under(prefix):
    """
    List keys under a prefix within the S3 bucket."
    """
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def discover_scenes(kind):
    """
    Returns list of (metadata_key, scene_prefix) for either a fire or control.
    """
    top_prefix = FIRES_PREFIX if kind == "fire" else CONTROLS_PREFIX
    meta_keys = [k for k in list_keys_under(f"{top_prefix}/") if k.endswith("metadata.json")]
    return [(k, k.rsplit("/", 1)[0] + "/") for k in meta_keys]


def resolve_zip_key(scene_prefix, contents_value):
    """
    Returns the single .zip object key under the scene_prefix + contents_value + '/'.
    """
    exact_key = scene_prefix + contents_value
    try:
        s3.head_object(Bucket=BUCKET_NAME, Key=exact_key)
        return exact_key
    except s3.exceptions.ClientError:
        pass

    folder_prefix = exact_key + "/"
    resp = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=folder_prefix)
    all_keys = [o["Key"] for o in resp.get("Contents", [])]
    zip_keys = [k for k in all_keys if k.lower().endswith(".zip")]
    if len(zip_keys) == 1:
        return zip_keys[0]
    
    # raise if SAFE folder cannot be resolved as either flat or nested once
    raise ValueError(
        f"Could not resolve a single object for {exact_key} -- not found as an "
        f"exact key, and folder search under {folder_prefix} found {len(all_keys)} "
        f"objects: {all_keys}"
    )

def download_and_extract(zip_key, extract_dir):
    """
    Download and extract the contents of the zip key representing the SAFE file.
    """
    os.makedirs(extract_dir, exist_ok=True)
    local_zip = os.path.join(extract_dir, os.path.basename(zip_key))
    s3.download_file(BUCKET_NAME, zip_key, local_zip)
    with zipfile.ZipFile(local_zip, "r") as zf:
        zf.extractall(extract_dir)
    os.remove(local_zip)

    # extracted contents typically land one level deep in a *.SAFE folder
    safe_dirs = glob.glob(os.path.join(extract_dir, "*.SAFE"))
    if len(safe_dirs) != 1:
        raise ValueError(f"Expected exactly one SAFE dir after extracting {zip_key}, found {safe_dirs}")
    return safe_dirs[0]


def download_grib(grib_key, dest_dir):
    """
    Download the ERA5 grib file.
    """
    os.makedirs(dest_dir, exist_ok=True)
    local_path = os.path.join(dest_dir, os.path.basename(grib_key))
    s3.download_file(BUCKET_NAME, grib_key, local_path)
    return local_path


def glob_one(pattern, label):
    matches = glob.glob(pattern)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one match for {label} ({pattern}), found {len(matches)}: {matches}")
    return matches[0]


# --------------------------------------------------------------------------
# Per-scene band loading (mirrors test_pipeline.ipynb, but path-discovered)
# --------------------------------------------------------------------------

def load_s1_pre(safe_dir, dem_output_dir):
    """
    Retrieve all Sentinel-1 files and bands necessary for fire detection analysis.
    """
    vh_band = glob_one(os.path.join(safe_dir, "measurement", "*-vh-*.tiff"), "S1 VH band")
    vv_band = glob_one(os.path.join(safe_dir, "measurement", "*-vv-*.tiff"), "S1 VV band")
    vh_cal = glob_one(os.path.join(safe_dir, "annotation", "calibration", "calibration-*-vh-*.xml"), "S1 VH calibration")
    vv_cal = glob_one(os.path.join(safe_dir, "annotation", "calibration", "calibration-*-vv-*.xml"), "S1 VV calibration")
    annotation_xml = glob_one(os.path.join(safe_dir, "annotation", "*-vh-*.xml"), "S1 annotation")
    return load_sentinel1_bands(vh_band, vv_band, vh_cal, vv_cal, annotation_xml, dem_output_dir, downsample_factor=10)


def load_s2_pre(safe_dir, downsample_factor=10):
    """
    Retrieve all Sentinel-2 bands necessary for fire detection analysis.
    """
    granule_dir = glob_one(os.path.join(safe_dir, "GRANULE", "*"), "S2 GRANULE")
    red = glob_one(os.path.join(granule_dir, "IMG_DATA", "R10m", "*_B04_10m.jp2"), "S2 red (B04)")
    green = glob_one(os.path.join(granule_dir, "IMG_DATA", "R10m", "*_B03_10m.jp2"), "S2 green (B03)")
    nir = glob_one(os.path.join(granule_dir, "IMG_DATA", "R10m", "*_B08_10m.jp2"), "S2 nir (B08)")
    swir = glob_one(os.path.join(granule_dir, "IMG_DATA", "R20m", "*_B11_20m.jp2"), "S2 swir (B11)")
    scl = glob_one(os.path.join(granule_dir, "IMG_DATA", "R20m", "*_SCL_20m.jp2"), "S2 SCL")

    # peek the native resolution to compute the downsampled target shap
    with rasterio.open(nir) as ds:
        native_height, native_width = ds.height, ds.width
    target_shape = (native_height // downsample_factor, native_width // downsample_factor)

    s2_data = load_sentinel2_bands(red, green, nir, swir, scl, target_shape=target_shape)

    bands, nir_transform, nir_shape, s2_crs = s2_data
    return bands, nir_transform, nir_shape, s2_crs


# --------------------------------------------------------------------------
# Main per-scene pipeline
# --------------------------------------------------------------------------

def _log(scene_id, msg, t0):
    print(f"    [{scene_id}] {msg} ({time.time() - t0:.1f}s elapsed)", flush=True)


def process_scene(metadata_key, scene_prefix, kind, tmp_root):
    t0 = time.time()
    resp = s3.get_object(Bucket=BUCKET_NAME, Key=metadata_key)
    meta = json.loads(resp["Body"].read())
    contents = meta["contents"]

    scene_id = meta.get("control_id") or meta["fire_name"]
    scene_tmp = os.path.join(tmp_root, scene_id.replace("/", "_"))
    os.makedirs(scene_tmp, exist_ok=True)

    try:
        _log(scene_id, "resolving S3 keys...", t0)
        s1_zip_key = resolve_zip_key(scene_prefix, contents["S1_pre"])
        s2_zip_key = resolve_zip_key(scene_prefix, contents["S2_pre"])
        era5_key = scene_prefix + contents["ERA5"]

        _log(scene_id, "downloading + extracting S1_pre...", t0)
        s1_safe_dir = download_and_extract(s1_zip_key, os.path.join(scene_tmp, "s1"))

        _log(scene_id, "downloading + extracting S2_pre...", t0)
        s2_safe_dir = download_and_extract(s2_zip_key, os.path.join(scene_tmp, "s2"))

        _log(scene_id, "downloading ERA5 grib...", t0)
        era5_local = download_grib(era5_key, os.path.join(scene_tmp, "era5"))

        dem_dir = os.path.join(scene_tmp, "dem")
        _log(scene_id, "loading S1 bands (calibration + RTC, includes DEM download)...", t0)
        s1_data = load_s1_pre(s1_safe_dir, dem_dir)

        _log(scene_id, "loading S2 bands...", t0)
        s2_data = load_s2_pre(s2_safe_dir)

        _log(scene_id, "loading ERA5 vars...", t0)
        era5_data = load_era5_vars(era5_local)

        _log(scene_id, "aggregating into tiles...", t0)
        tiles = aggregate(s1_data, s2_data, era5_data)
        _log(scene_id, f"done -- {len(tiles)} tiles", t0)

        label = 1.0 if kind == "fire" else 0.0
        record = {
            "scene_id": scene_id,
            "kind": kind,
            "label": label,
            "fire_name": meta.get("fire_name"),
            "event_id": meta.get("event_id"),
            "matched_fire_event_id": meta.get("matched_fire_event_id"),
            "tiles": tiles,
        }
        return record, None
    except Exception as e:
        return None, f"{scene_id}: {e}\n{traceback.format_exc()}"
    finally:
        shutil.rmtree(scene_tmp, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="tile_cache")
    parser.add_argument("--tmp-dir", default="/tmp/wildfire_extract")
    parser.add_argument("--s3-cache-prefix", default="cache/tiles/")
    parser.add_argument("--upload-to-s3", action="store_true", help="Also upload each cached pickle back to S3.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N scenes (per kind), for smoke-testing.")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)

    failures = []
    processed = 0

    for kind in ("fire", "control"):
        scenes = discover_scenes(kind)
        if args.limit:
            scenes = scenes[: args.limit]
        print(f"\n=== {kind}: {len(scenes)} scenes ===")

        for metadata_key, scene_prefix in scenes:
            scene_id = scene_prefix.rstrip("/").rsplit("/", 1)[-1]
            out_path = os.path.join(args.cache_dir, f"{scene_id}.pkl")

            if args.skip_existing and os.path.exists(out_path):
                print(f"  [skip] {scene_id} (already cached)")
                continue
            print(f"  [processing] {scene_id} ...")
            record, error = process_scene(metadata_key, scene_prefix, kind, args.tmp_dir)

            if error:
                print(f"  [FAILED] {error.splitlines()[0]}")
                failures.append(error)
                continue
            with open(out_path, "wb") as f:
                pickle.dump(record, f)

            if args.upload_to_s3:
                s3.upload_file(out_path, BUCKET_NAME, args.s3_cache_prefix + f"{scene_id}.pkl")

            processed += 1
            print(f"  [done] {scene_id} -> {len(record['tiles'])} tiles")

    print(f"\n{processed} scenes cached, {len(failures)} failed.")
    if failures:
        with open(os.path.join(args.cache_dir, "cache_failures.log"), "w") as f:
            f.write("\n\n".join(failures))
        print(f"Failure details written to {os.path.join(args.cache_dir, 'cache_failures.log')}")


if __name__ == "__main__":
    main()
