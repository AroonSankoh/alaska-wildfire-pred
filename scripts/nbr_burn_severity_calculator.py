"""
Computes dNBR (delta Normalized Burn Ratio) burn severity for a single scene
from its pre/post Sentinel-2 pair, and classifies it using that exact fire's own 
MTBS-calibrated dNBR thresholds and validates the result against MTBS's reported
burned acreage. 

Requirements:
    --scene-dir: a fire (or control) scene folder exactly as published,
    needing at minimum S2_*_pre.SAFE, S2_*_post.SAFE (zip archives or an
    actual directory), and a metadata.json.

    --mtbs-shapefile: path to the MTBS burn area boundary shapefile (default:
    MTBS/S_USA.MTBS_BURN_AREA_BOUNDARY/S_USA.MTBS_BURN_AREA_BOUNDARY.shp).

Output:
    Per-tile CSV (same ERA5 tile grid as everywhere else in the repo) to
    burn_severity/{scene_id}_severity_grid.csv by default. CSV columns 
    include mean corrected, dNBR, dominant severity class, and burned-pixel 
    fraction per tile. Console summary includes the whole-scene severity class 
    breakdown and, when an MTBS match was found, the perimeter-masked acreage 
    comparison against MTBS's officially reported ACRES.

Usage:
    python scripts/nbr_burn_severity_calculator.py --scene-dir path/to/fire/scene 
"""

import argparse
import glob
import json
import os
import sys
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "scripts"))

from data.aggregator import vectorize, bin_data
from data.loaders import load_era5_vars
from data.loaders.sentinel2_preprocessing import load_sentinel2_bands
from build_tile_cache import glob_one, era5_cutoff_from_key

ACRES_PER_SQM = 1.0 / 4046.8564224
DEFAULT_MTBS_SHAPEFILE = os.path.join(
    REPO_ROOT, "MTBS", "S_USA.MTBS_BURN_AREA_BOUNDARY", "S_USA.MTBS_BURN_AREA_BOUNDARY.shp")

# fallback bins (scaled dNBR*1000, USGS/literature convention) for scenes with no MTBS match
GENERIC_THRESHOLDS = {"NODATA_THR": -970, "GREENNESS_": -150, "LOW_THRESH": 100,
                       "MODERATE_T": 270, "HIGH_THRES": 660, "DNBR_OFFST": 0}

SEVERITY_CLASSES = ["no_data", "high_regrowth", "unburned", "low", "moderate", "high"]


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


def find_s2_pre_post(scene_dir, extract_root):
    pre_outer = glob.glob(os.path.join(scene_dir, "S2_*_pre.SAFE"))
    post_outer = glob.glob(os.path.join(scene_dir, "S2_*_post.SAFE"))
    if len(pre_outer) != 1:
        raise ValueError(f"Expected exactly one S2_*_pre.SAFE entry under {scene_dir}, found {pre_outer}")
    if len(post_outer) != 1:
        raise ValueError(f"Expected exactly one S2_*_post.SAFE entry under {scene_dir}, found {post_outer}")
    pre_safe_dir = ensure_extracted_safe_dir(pre_outer[0], extract_root, "S2 pre-scene")
    post_safe_dir = ensure_extracted_safe_dir(post_outer[0], extract_root, "S2 post-scene")
    return pre_safe_dir, post_safe_dir


def load_s2_scene(safe_dir, target_shape=None, downsample_factor=10):
    """
    Loads NBR, along with other sentinel2 bands, for one S2 scene.
    """
    granule_dir = glob_one(os.path.join(safe_dir, "GRANULE", "*"), "S2 GRANULE")
    red = glob_one(os.path.join(granule_dir, "IMG_DATA", "R10m", "*_B04_10m.jp2"), "S2 red (B04)")
    green = glob_one(os.path.join(granule_dir, "IMG_DATA", "R10m", "*_B03_10m.jp2"), "S2 green (B03)")
    nir = glob_one(os.path.join(granule_dir, "IMG_DATA", "R10m", "*_B08_10m.jp2"), "S2 nir (B08)")
    swir = glob_one(os.path.join(granule_dir, "IMG_DATA", "R20m", "*_B11_20m.jp2"), "S2 swir (B11)")
    scl = glob_one(os.path.join(granule_dir, "IMG_DATA", "R20m", "*_SCL_20m.jp2"), "S2 SCL")

    if target_shape is None:
        with rasterio.open(nir) as ds:
            native_height, native_width = ds.height, ds.width
        target_shape = (native_height // downsample_factor, native_width // downsample_factor)

    bands, nir_transform, nir_shape, s2_crs = load_sentinel2_bands(
        red, green, nir, swir, scl, target_shape=target_shape)
    return bands["indices"]["nbr"], nir_transform, nir_shape, s2_crs


def find_mtbs_record(mtbs_gdf, scene_meta):
    """
    Attempto to locate the matching MTBS event.
    """
    event_id = scene_meta.get("event_id")
    mtbs_post_id = scene_meta.get("mtbs_post_id")

    match = mtbs_gdf[mtbs_gdf["FIRE_ID"] == event_id] if event_id else mtbs_gdf.iloc[0:0]
    if match.empty and mtbs_post_id:
        match = mtbs_gdf[mtbs_gdf["POST_ID"] == mtbs_post_id]
    if match.empty:
        return None
    return match.iloc[0]


def classify_severity(corrected_dnbr, thresholds):
    """
    Set burn severity thresholds.
    """
    t = thresholds
    conditions = [
        corrected_dnbr <= t["NODATA_THR"],
        corrected_dnbr <= t["GREENNESS_"],
        corrected_dnbr <= t["LOW_THRESH"],
        corrected_dnbr <= t["MODERATE_T"],
        corrected_dnbr <= t["HIGH_THRES"],
    ]
    choices = SEVERITY_CLASSES[:5]  # no_data, high_regrowth, unburned, low, moderate
    return np.select(conditions, choices, default="high")


def rasterize_perimeter(geometry, out_shape, transform, dst_crs, src_crs):
    reprojected = gpd.GeoSeries([geometry], crs=src_crs).to_crs(dst_crs)
    mask = rasterio.features.rasterize(
        [(reprojected.iloc[0], 1)], out_shape=out_shape, transform=transform, fill=0, dtype="uint8")
    return mask.astype(bool)


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", required=True, help="A single fire (or control) scene folder.")
    parser.add_argument("--mtbs-shapefile", default=DEFAULT_MTBS_SHAPEFILE)
    parser.add_argument("--extract-dir", default="/tmp/wildfire_nbr_extract",
                         help="Where to unzip the scene's S2 *_pre.SAFE/*_post.SAFE archives.")
    parser.add_argument("--output-dir", default="burn_severity",
                         help="Per-tile CSV lands at <output-dir>/<scene_id>_severity_grid.csv.")
    parser.add_argument("--output-csv", default=None, help="Optional explicit output path override.")
    return parser


def main():
    args = build_arg_parser().parse_args()
    scene_id = os.path.basename(os.path.normpath(args.scene_dir))

    metadata_path = os.path.join(args.scene_dir, "metadata.json")
    with open(metadata_path) as f:
        scene_meta = json.load(f)

    print(f"Scene: {scene_id} ({scene_meta.get('state')})")
    pre_safe_dir, post_safe_dir = find_s2_pre_post(args.scene_dir, args.extract_dir)

    print(f"Loading pre-fire Sentinel-2 from {pre_safe_dir} ...")
    nbr_pre, pre_transform, pre_shape, pre_crs = load_s2_scene(pre_safe_dir)

    print(f"Loading post-fire Sentinel-2 from {post_safe_dir} ...")
    nbr_post, post_transform, post_shape, post_crs = load_s2_scene(post_safe_dir, target_shape=pre_shape)

    if pre_crs != post_crs or pre_transform != post_transform:
        raise ValueError(
            "Pre and post Sentinel-2 scenes are not pixel-aligned (different CRS/transform) -- "
            f"pre: {pre_crs}, {pre_transform}; post: {post_crs}, {post_transform}. "
            "Can't compute a direct per-pixel dNBR without reprojecting one onto the other first."
        )

    # MTBS convention: dNBR scaled by 1000, offset-corrected per that fire's own calibration
    dnbr_scaled = (nbr_pre - nbr_post) * 1000.0

    print(f"Loading MTBS shapefile from {args.mtbs_shapefile} ...")
    mtbs_gdf = gpd.read_file(args.mtbs_shapefile)
    mtbs_record = find_mtbs_record(mtbs_gdf, scene_meta)

    if mtbs_record is not None:
        print(f"Matched MTBS record: FIRE_ID={mtbs_record['FIRE_ID']} ({mtbs_record['FIRE_NAME']}), "
              f"official ACRES={mtbs_record['ACRES']}")
        thresholds = {k: mtbs_record[k] for k in
                      ("NODATA_THR", "GREENNESS_", "LOW_THRESH", "MODERATE_T", "HIGH_THRES", "DNBR_OFFST")}
    else:
        print("No MTBS record matched this scene (event_id/mtbs_post_id not found in the shapefile) -- "
              "falling back to generic literature dNBR thresholds. No acreage/perimeter validation "
              "will be run.")
        thresholds = GENERIC_THRESHOLDS

    corrected_dnbr = dnbr_scaled - thresholds["DNBR_OFFST"]
    severity = classify_severity(corrected_dnbr, thresholds)

    if mtbs_record is not None:
        perimeter_mask = rasterize_perimeter(
            mtbs_record.geometry, pre_shape, pre_transform, dst_crs=pre_crs, src_crs=mtbs_gdf.crs)
        pixel_area_sqm = abs(pre_transform.a * pre_transform.e)
        burned_classes = np.isin(severity, ["low", "moderate", "high"])
        computed_acres = float(np.sum(burned_classes & perimeter_mask)) * pixel_area_sqm * ACRES_PER_SQM
        official_acres = float(mtbs_record["ACRES"])
        print(f"\nPerimeter-masked burned acreage (low+moderate+high severity, inside MTBS boundary):")
        print(f"  computed: {computed_acres:,.1f} acres   official (MTBS): {official_acres:,.1f} acres   "
              f"ratio: {computed_acres / official_acres:.2f}")
    else:
        perimeter_mask = np.ones(pre_shape, dtype=bool)  # whole scene

    print("\nWhole-scene severity class breakdown:")
    for cls in SEVERITY_CLASSES:
        n = int(np.sum(severity == cls))
        print(f"  {cls:15s} {n:8d} pixels ({n / severity.size:.1%})")

    # per-tile summary on the same ERA5 grid used everywhere else 
    era5_glob = glob.glob(os.path.join(args.scene_dir, "*.grib"))
    if len(era5_glob) == 1:
        cutoff = era5_cutoff_from_key(era5_glob[0])
        era5_data = load_era5_vars(era5_glob[0], cutoff_datetime=cutoff)
        era5_lats = np.sort(era5_data["u10"]["latitude"].values)
        era5_longs = np.sort(era5_data["u10"]["longitude"].values)

        pixel_longs, pixel_lats = vectorize(pre_transform, pre_shape, pre_crs)
        tile_i, tile_j = bin_data(pixel_lats.ravel(), pixel_longs.ravel(), era5_lats, era5_longs)

        df = pd.DataFrame({
            "i": tile_i, "j": tile_j,
            "corrected_dnbr": corrected_dnbr.ravel(),
            "severity": severity.ravel(),
            "in_perimeter": perimeter_mask.ravel(),
        })
        df = df[(df["i"] >= 0) & (df["i"] < len(era5_lats)) & (df["j"] >= 0) & (df["j"] < len(era5_longs))]

        def summarize_tile(group):
            burned = group["severity"].isin(["low", "moderate", "high"])
            return pd.Series({
                "lat": era5_lats[group.name[0]], "lon": era5_longs[group.name[1]],
                "mean_corrected_dnbr": group["corrected_dnbr"].mean(),
                "dominant_severity": group["severity"].mode().iloc[0],
                "burned_fraction": burned.mean(),
            })

        tile_grid = df.groupby(["i", "j"]).apply(summarize_tile).reset_index()

        output_csv = args.output_csv or os.path.join(args.output_dir, f"{scene_id}_severity_grid.csv")
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        tile_grid.to_csv(output_csv, index=False)
        print(f"\nSaved per-tile severity grid ({len(tile_grid)} tiles) to {output_csv}")
    else:
        print(f"\nSkipping per-tile output -- expected exactly one .grib file under {args.scene_dir}, "
              f"found {era5_glob}.")


if __name__ == "__main__":
    main()