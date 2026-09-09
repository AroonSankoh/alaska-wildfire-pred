"""
This script computes FWI (Fire Weather Index) for a single ERA5 scene. It uses a local 
ERA5 .grib file (see README for source) to calculate FWI and related components, then prints 
them to the command line and optionally saves the result to the CSV file. It also provides 
reference values to help put calculated indices into context. 


Data requirements: 
     >120 days of hourly data (DC/DMC drought codes are recursive with a long memory)
     5 ERA5 variables (2m temp, 2m dewpoint, 10m u/v wind components, total precipitation)

Usage:
    python scripts/fwi_calculator.py path/to/scene_folder/
    python scripts/fwi_calculator.py path/to/ERA5_CA_34N117W_20200731.grib
    python scripts/fwi_calculator.py path/to/scene_folder/ --output-csv results.csv
"""

import argparse
import csv
import glob
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)
from data.loaders.era5_preprocessing import load_era5_vars, calculate_fwi

REQUIRED_VARIABLES = ["t2m", "d2m", "u10", "v10", "tp"]
MIN_REQUIRED_DAYS = 120
MIN_REQUIRED_HOURLY_TIMESTEPS = MIN_REQUIRED_DAYS * 24

# approximate reference values, calculations based on real-world values should fall well within these ranges
VALUE_RANGES = {
    "ffmc": (0, 101),
    "dmc": (0, 500),
    "dc": (0, 1000),
    "isi": (0, 60),
    "bui": (0, 300),
    "fwi": (0, 100),
}


def resolve_grib_path(scene_path):
    """
    Accepts either a direct path to a .grib file, or a directory containing exactly one.
    """
    if os.path.isfile(scene_path):
        return scene_path

    if os.path.isdir(scene_path):
        gribs = sorted(glob.glob(os.path.join(scene_path, "*.grib")))
        if len(gribs) == 1:
            return gribs[0]
        if len(gribs) == 0:
            raise ValueError(f"No .grib file found in directory: {scene_path}")
        raise ValueError(
            f"Found {len(gribs)} .grib files in {scene_path}, ambiguous which to use: {gribs}. "
            "Pass the exact .grib file path instead of the directory."
        )

    raise ValueError(f"Path does not exist: {scene_path}")


def validate_variables(variables):
    """
    Checks that all variables exist within the .grib and that there are at least 
    MIN_REQUIRED_DAYS days of each. Throws if either check fails.
    """
    missing = [v for v in REQUIRED_VARIABLES if v not in variables]
    if missing:
        raise ValueError(
            f"Missing required ERA5 variable(s): {missing}. calculate_fwi() needs all of "
            f"{REQUIRED_VARIABLES} (2m temperature, 2m dewpoint, 10m u/v wind components, "
            "total precipitation) to run."
        )

    time_dim = "valid_time" if "valid_time" in variables["t2m"].dims else "time"
    n_timesteps = variables["t2m"].sizes[time_dim]
    if n_timesteps < MIN_REQUIRED_HOURLY_TIMESTEPS:
        n_days = n_timesteps / 24
        raise ValueError(
            f"Only {n_timesteps} hourly timesteps (~{n_days:.1f} days) of data. FWI requires "
            f"at least {MIN_REQUIRED_DAYS} days ({MIN_REQUIRED_HOURLY_TIMESTEPS} hourly "
            "timesteps) for the DC/DMC drought codes to spin up properly. Provide a longer "
            "ERA5 window."
        )

    return n_timesteps


def write_result_csv(result, scene_path, output_csv):
    """
    Appends this run's result as one row to output_csv, writing the header only if the file
    doesn't already exist yet; repeated single-scene runs can accumulate into a single file.
    """
    file_exists = os.path.exists(output_csv)
    fieldnames = ["scene_path", "date", "ffmc", "dmc", "dc", "isi", "bui", "fwi"]
    with open(output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "scene_path": scene_path,
            "date": result["date"],
            "ffmc": result["ffmc"],
            "dmc": result["dmc"],
            "dc": result["dc"],
            "isi": result["isi"],
            "bui": result["bui"],
            "fwi": result["fwi"],
        })


def main():
    parser = argparse.ArgumentParser(description="Compute FWI (Fire Weather Index) for a single ERA5 scene.")
    parser.add_argument("scene_path", help="Path to a .grib file, or a directory containing exactly one.")
    parser.add_argument("--output-csv", default=None,
                         help="If given, appends the result as a row to this CSV file "
                              "(creating it with a header first, if it doesn't already exist).")
    args = parser.parse_args()

    grib_path = resolve_grib_path(args.scene_path)
    print(f"Loading {grib_path}")

    variables = load_era5_vars(grib_path)
    n_timesteps = validate_variables(variables)
    print(f"Loaded {n_timesteps} hourly timesteps (~{n_timesteps / 24:.1f} days)")

    result = calculate_fwi(variables)

    print("\nFinal day computed:")
    print(f"  date : {result['date']}")
    print(f"  FFMC : {result['ffmc']:.2f}")
    print(f"  DMC  : {result['dmc']:.2f}")
    print(f"  DC   : {result['dc']:.2f}")
    print(f"  ISI  : {result['isi']:.2f}")
    print(f"  BUI  : {result['bui']:.2f}")
    print(f"  FWI  : {result['fwi']:.2f}")

    print("\nApproximate value ranges of each index):")
    for key, (low, high) in VALUE_RANGES.items():
        value = result[key]
        flag = "OK" if low <= value <= high else "OUT OF RANGE"
        print(f"  {key.upper():5s} = {value:8.2f}  expected [{low}, {high}]  -> {flag}")

    if args.output_csv:
        write_result_csv(result, args.scene_path, args.output_csv)
        print(f"\nResult appended to {args.output_csv}")


if __name__ == "__main__":
    main()
