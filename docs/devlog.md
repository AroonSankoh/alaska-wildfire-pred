# Wildfire Prediction — Dev Log

A running record of bugs, design decisions, and dead ends encountered while building the
wildfire prediction pipeline (S3-backed scene collection → tile caching → training). Kept
as raw material for a future writeup, updated as issues come up rather than reconstructed
from memory after the fact.

---

## Initial push - 04/24/26

Pushed the first skeleton of the project all at once: the Sentinel-1/Sentinel-2/ERA5
loaders (`data/loaders/`), the zonal aggregator (`data/aggregator/zonal_aggregator.py`),
`model/dataset.py`, and `model/architecture.py`. RTC and cross-attention fusion were still
stubs (`raise NotImplementedError`) at this point, and NaNs in tile statistics were just
zeroed out (`np.nan_to_num`) rather than properly imputed.

## Radiometric terrain correction for Sentinel-1 - 05/13/26

Fully implemented RTC, which had been a stub until now. Given a scene's Sentinel-1 bounds,
it downloads the covering Copernicus GLO-30 DEM tiles from the public AWS bucket, mosaics
and reprojects them onto the scene's grid, then computes surface slope/aspect from the DEM
gradient and derives a per-pixel surface normal vector. Per-pixel incidence angles are
parsed from the Sentinel-1 annotation XML (sparse geolocation grid points) and interpolated
to the full image grid, giving a radar look vector. The local incidence angle
(`cos_local = normal · look_vector`) is then used to normalize each pixel
(`sigma0 / cos_local`), with pixels where `cos_local` is too small (layover/shadow, where the
radar geometry breaks down) masked to NaN rather than divided through. Decibel conversion
(`10 * log10`) was also moved to happen *after* RTC instead of before, since it should apply
to the terrain-corrected value, not the raw digital number. Also added orthorectification
(`orthorectify()` in `utils/geo_utils.py`) for scenes that ship ground control points instead
of a direct affine transform.

## Aggregator rewrite + RTC bug fixes - 05/15/26

Bug-tested the RTC implementation above and found two real issues:

- The shadow/layover mask used `abs(cos_local) > threshold`, which only screens out pixels
  where the radar geometry is close to perpendicular in *either* direction. Tightened it to
  `cos_local > threshold`, so back-facing slopes (radar looking away from the surface) get
  masked the same as true shadow, not treated as valid just because the magnitude was large.
- The slope/aspect gradient (`np.gradient(dem)`) was being divided directly by the transform's
  pixel spacing in *degrees*, not real ground distance — a degree of longitude covers a very
  different physical distance depending on latitude (notably so at Alaska's high latitudes).
  Added `meters_per_degree(lat)` to convert degree spacing to meters at the scene's center
  latitude before computing real-world slope.

Also rewrote `zonal_aggregator.py`'s core tiling loop: the original version was a manual
nested loop over every ERA5 lat/lon bin with a boolean mask recomputed per bin (effectively
O(bins × pixels)). Replaced it with a vectorized `pandas` `groupby(['x', 'y']).agg(['mean',
'std'])`, aggregating all pixels into per-tile statistics in one pass — a large speedup for
full-scene tiling. While rewriting it, also fixed a latitude/longitude ordering bug in
`vectorize()` (`utils/geo_utils.py`): it now consistently returns `(longitude, latitude)`,
with every call site updated to destructure it in that order — previously the return order
didn't reliably match how callers were unpacking it.

## Mean imputation + ERA5 bounds check - 05/21/26

Implemented proper mean imputation in `model/dataset.py`, replacing the earlier
`np.nan_to_num`-to-zero placeholder. `dataset.__init__` now scans for the first tile with
complete (non-`None`) stats across all three sources and computes a per-feature dataset-wide
mean from every tile that has a real value for that feature; `__getitem__` then fills in any
tile whose `s1_stats`/`s2_stats`/`era5_stats` is `None` using those means instead of zeroing
it out, which is a meaningfully better prior for a missing SAR/optical/weather reading than
"zero."

Also added the first version of the ERA5 coordinate-range guard in the aggregator: if a
scene's vectorized Sentinel-1 or Sentinel-2 pixel bounds fall outside the ERA5 grid's
lat/long extent, raise a `ValueError` rather than silently binning pixels into the wrong (or
nonexistent) ERA5 cell. This was bare-bones at first (no actual bounds in the message) — the
version with real numeric bounds attached came later, during the July defensive-coding pass.

## Cross-attention fusion - 05/21/26

Replaced the `cross_attention_fusion` stub (previously just `raise NotImplementedError`,
with the encoders' outputs simply concatenated) with a real implementation: spatial tile
embeddings attend over the temporal (ERA5) embedding sequence via `nn.MultiheadAttention`,
with spatial embeddings as the query and temporal embeddings as keys/values — the intent
being that each tile learns which time steps in its antecedent weather window matter most.
The attention output is then concatenated with the original spatial embedding (not replacing
it) before being passed to the three prediction heads. Also fixed how the temporal encoder
reshaped its input before feeding the transformer — `X.T.unsqueeze(1)` (transpose then
unsqueeze) was replaced with `X.unsqueeze(0).unsqueeze(0)`, adding the sequence and batch
dimensions directly instead of transposing a 1-D vector.

## Sentinel-2 masking improvements - 05/22/26

Extended `apply_cloud_mask` to also exclude SCL category `0` (`no_data`) pixels, not just
cloud/cloud-shadow/cirrus — a scene edge or fragmented tile can have genuine no-data pixels
that aren't clouds but are equally unusable. More importantly, cloud masking now gets applied
to every individual raw band (red, green, NIR, SWIR) before NDVI/NBR are computed from them,
rather than only masking the final NBR index after the fact — previously a cloud-contaminated
NIR or SWIR pixel could still poison the NDVI/NBR calculation even though the *output* index
was later masked. Renamed the `"raw_bands"` dict key to `"filtered_bands"` to reflect that.

Also added `nullify_nan()` to the aggregator: a tile whose Sentinel-1 or Sentinel-2 stats
came out entirely NaN (e.g. a fully cloud-masked bin) now gets its stats dict set to `None`
instead of staying a dict full of NaNs — which matters because May 21's mean-imputation logic
only treats `None` as "missing," so a stray NaN-filled dict would otherwise have silently
averaged NaNs into the per-feature means.

## Fire collection notebook - 06/08/26

I used a notebook called `query_satellite_data.ipynb` to iterate through fire events from 
2016 to 2024 found in the Monitoring Trends in Burn Severity (MTBS) shape file - a freely 
downloadable archive of fires across the United States dating back to 1984. My notebook 
searched for and synced whole fire 'packages', which included a Sentinel-1 pre and post 
fire image, Sentinel-2 pre and post fire image, an ERA-5 grib tile (built via 
`calculate_master_era5_area`) including at least 30 days of weather variable data leading up 
the ignition date of the fire, and a metadata.json detailing the contents of the package, to 
an S3 bucket under `fires/{state}/{fire_name}_fire/`. I manually checked that multiple 
constraints on all fires were satisfied, below is a small snippet of them. 

- total fire acreage > 10k 
- temporal windows between S1/S2 pre scenes < 36 hours 
- temporal windows between S1/S2 post scenes < 36 hours 
- 95% of fire boundary boxes being visible to the naked idea 
- Ensuring matching orbital direction between S1 pre and post scene.

Hence, accumulating over 125 fire packages took almost a month. 

## control collection pipeline - 07/01/26

3 controls were sampled for each fire, with the EPA Level 3 eco-region of a fire used to 
ensure regional consistency between controls and their source fire. Overlap between 
fires and controls is prevented by combining a literal fire-box exclusion (no control inside a fire's 
actual footprint, via `fire_exclusion_zone` in `control_collection_pipeline.py`) with a 20km 
minimum-separation constraint (`too_close()`) against a continuously-growing list of every 
fire and already-placed control centroid, enforced together in `sample_control_centroid()`.

## Consistency checks between controls and fires - 07/02/26

Running the complete control collection script (`control_collection_pipeline.py`) would have 
taken around 4 days, but ended up taking close to a week and a half due do small bugs 
I discovered throughout collection which resulted in me having to partially or fully restart the 
collection pipeline. Bugs were related to small discrepancies between the quality of fires and 
controls. A few of the notable ones are listed below:

1. **Enforcing matching Sentinel-1 orbital directions between fires and controls**. The 
original pipeline not only did not enforce matching orbital directions between controls 
but also did not between fires and controls. An in-place checking script 
`confirm_matching_orbital_direction.py` was used to check which controls did not fully 
adhere to their fire's Sentinel-1 orbital direction. These were then deleted, and 
`control_collection_pipeline.py` was updated to enforce consistent orbital direction.
This ended up being nearly all controls so I decided to just fully restart collection. 

2. **Enforcing a minimum bounding box coverage**. 70% of the bounding box of all fires was 
required to be contained within each Sentinel scene and ERA-5 spatial window. A similar 
requirement was originally not imposed on controls but since this mistake was caught 
around the same time as #1, the control collection script's own coverage check 
(`calculate_spatial_coverage_percentage()` / the `target_tile_geom.intersection(...)` checks 
in `collect_controls()`) was adjusted accordingly.

3. **Enforcing a stricter valid pixel requirement**. In my fire collection 
notebook, since I manually inspected the S2 pre and post scenes myself I allowed 
Sentinel-2 images with a relatively high number of invalid pixels (up to 40% for some) 
as long as the entirety of the fire boundary box was still visible. Since I did not 
manually inspect the S2 images of each control, I decided to just enforce stricter valid 
pixel requirement (80%).

## Dataset control quality assurance checks - 07/10/26

With all 375 controls collected, I conducted a series of metadata checks to ensure no 
silent errors or data corruptions snuck through. The main issue discovered was naming 
inconsistencies between the state dir a fire/control was placed in and the actual state  
that the fire occurred in. Discovered broken logic that used a hard-coded if-else 
statement to determine the state of a fire based on it's boundary box, when the MTBS 
event ID (the `event_id` field already sitting in every fire's `metadata.json`) was a much 
easier and reliable ground truth. After iterating through all controls 
and their associated fires, we determined a sample of fires/controls that needed to be 
either edited or requeried, and have listed their fixes below: 

1. **Fires with mismatched state indices were flagged**, and not only were they moved to their 
correct `state` folder but all associated SAFE, grib files were renamed and the scene's 
metadata.json was edited to reflect the new contents of each scene. Over 30 fires of the original 
125 ended up flagged, which resulted in 120 total fixes (30 fires + 90 controls).

2. Each fire's control underwent the same fixes. With the additional caveat of the control 
id (and therefore the folder name) also being edited to reflect the actual state of the fire.

3. The size of each SAFE and grib were checked to ensure that no data loss occurred during 
download time. The standard size of a SAFE (850 MB - 1.3 GB) and a grib (2 - 30 MB) were 
used as benchmarks.

4. Finally, each fire was matched to its' 3 controls. And the SAFE folders within each were 
unzipped so as to compare the manifest (metadata file for a SAFE product) with the metadata.json.
Any discrepancies would be flagged, but none were found. 

## Tile augmenter - 07/16/26

Implemented the tile augmenter, which uses Gaussian jitter and mixup to add perturbations
to the scalar features that are extracted and processed from each data source. A transformer 
is a data-hungry architecture so adding additional augmented tiles will help model alignment.

Only a small issue with `dropout` in `augmentation.py` required fixing. Tensor.uniform 
was being called and this function does not exist in PyTorch - only the in-place uniform()
call does. Replaced with a single `torch.rand` call on a preset PyTorch generator. 

## Defensive-coding hardening pass - 07/20/26

Before scaling up to a full 125-fire / 375-control run, did a pass over every `.py` file in
`data/`, `model/`, and `scripts/` looking for missing error handling. Scoped it down based on
actual failure modes observed so far rather than adding guards everywhere reflexively — e.g.
`era5_preprocessing.py` never needed `IndexError`/`KeyError` guards because the ERA5 query
always explicitly requests the same 5 variables, so a missing-variable failure would be a
real upstream bug worth crashing loudly on, not something to paper over.

Landed fixes included: calibration LUT vector-count guards, a warning (later found to be
incomplete — see below) on zero/negative calibration values, request timeouts, VH/VV
shape-match assertions, and band-shape consistency checks in the zonal aggregator.

## DEM 404s over open ocean (57 of 63 initial failures) - 07/20/26

First full tile cache run threw a wall of `404 Client Error` on DEM tile downloads. Root cause: Copernicus
GLO-30 only has tiles over land — a purely-oceanic 1°×1° DEM tile *genuinely does not exist*,
so a 404 there is expected, not a bug. Fixed `download_dem_tiles` to skip a failed tile instead
of crashing, and only raise if *every* tile for a scene fails (meaning the whole scene is over
water). Also switched `prepare_dem`'s destination array from `np.empty` to `np.zeros` so any
skipped-tile gap defaults to sea level (0m) instead of uninitialized memory.

## ERA5 coordinate-range padding (three separate bugs, same symptom) - 07/22/26

The remaining ~6 failures were `"Sentinel-X coordinate range falls outside ERA5 coordinate
range"` — the downloaded ERA5 grib's bounding box didn't fully cover the Sentinel scene's
footprint. This took three iterations to actually close out:

1. **Root cause #1 — no padding at collection time.** `calculate_master_era5_area` (fire
   notebook) and its `_from_items` twin (control pipeline) built the ERA5 `area` request
   directly from the catalog `GeoFootprint`, with zero margin. Fixed by padding all four
   `[N, W, S, E]` bounds outward by a fixed `padding_degrees=2`.

2. **Root cause #2 — requery script used the wrong date-window logic for controls.** Wrote
   `find_era5_requeries.py` (parses `cache_failures.log` for the coordinate-range error,
   unions S1+S2 bounds per scene) and `requery_era5_failures.py` (re-downloads with padded
   bounds, re-uploads to the same S3 key) to patch the still-failing scenes without a full
   rebuild. First version used the *fire* notebook's `DateOffset`-based month window for every
   scene, including controls — caught because I asked directly whether it matched the control
   pipeline's actual `pd.date_range`-based window. It didn't. Fixed with two separate
   request-builders, `build_fire_era5_request` and `build_control_era5_request`.

3. **Root cause #3 — padding computed from only one source's footprint.** Two remaining
   scenes (`FLAT_fire`, `control_NV_38N118W_20230804`) failed again, this time on the
   Sentinel-1 side specifically. Sentinel-1 IW GRD swaths (~250km) are much wider than
   Sentinel-2 MGRS tiles (~100km×100km) — padding computed only from the S2 footprint didn't
   cover S1's wider extent. Fixed by unioning bounds across *both* sources, and across
   separate `cache_failures.log` append events for the same scene (the log is append-only by
   design, for live monitoring during a run).

## Tile cache filename collisions (18 colliding scene_ids / 39 scenes) - 07/23/26

Noticed the `.pkl` count from a full run (492 by internal reconciliation) didn't match
`ls tile_cache/*.pkl | wc -l` (471). First guess was a `cwd` issue watching the wrong
directory — wrong guess, the count really was short. Wrote `find_duplicate_scene_ids.py` to
check: 18 different `scene_id`s (derived from just `control_id`/`fire_name`, the last path
component) were shared across 39 actually-distinct S3 scenes — two different fires' controls
can land on the same rounded coordinate + date and collide on name.

Fix: `cache_path_for()` now mirrors the *full* S3 prefix (`fires/{state}/{fire}/...` /
`controls/{state}/{fire_control}/{control_id}/...`) as the cache filename, so collisions are
structurally impossible. Wrote `reorganize_tile_cache.py` to migrate the already-cached flat
files into the new nested structure, and `delete_colliding_pkls.py` to drop the ambiguous ones
that couldn't be safely attributed.

This bug had a second, nastier form: it wasn't just the filename, it was also the `scene_id`
*value stored inside the pickled record itself*. `train.py`'s `merge_tiles()` builds dict keys
as `f"{scene_id}__{tile_key}"` — with colliding scene_ids, tiles from two different scenes
silently overwrote each other in that dict while every scene's *labels* still got appended
unconditionally, producing a hard `tile/label count mismatch: 52003 vs 53652` assertion
failure. Fixed at the source (`build_tile_cache.py` now stores the full path as `scene_id` in
the record, not just the short name) and retroactively via `fix_scene_ids_in_cache.py`, a fast
in-place pickle-patching script that didn't require re-touching S3/DEM/ERA5.

## ERA5 temporal leakage (the most consequential bug, methodologically) - 07/24/26

While investigating a `cfgrib` crash (below), realized something more serious: the antecedent
ERA5 window collected for each fire could include data *during or after* the fire itself, not
just before it. The collection notebook over-fetches full calendar months rather than a tight
date range, so a fire's cached ERA5 stats could include weather from while it was actively
burning — meaning the model could theoretically learn to detect an ongoing fire from its own
weather signature, which tells you nothing about *predicting* a fire that hasn't started yet.

Confirmed with a CDS UI screenshot that this isn't fixable by requerying: the Climate Data
Store's `year`/`month`/`day` request parameters are independent filters that get
cross-multiplied server-side, not a literal date range — so there's no way to ask CDS for
"the 30 days before X" when that span crosses a month boundary, only "these year/month/day
values, in any combination." This was a known, deliberate limitation from when the original
collection pipeline was designed, not new information.

Decided against a full requery/pipeline rewrite (impractical given the CDS constraint above)
in favor of a hard cutoff filter applied at *load* time, regardless of what's actually in the
downloaded grib: `era5_cutoff_from_key()` extracts the ignition/control date from the grib's
S3 key, and `load_era5_vars(..., cutoff_datetime=...)` drops every timestep at or after that
date. Used `<` rather than `<=` deliberately — since ignition timing is date-only (no time of
day), including the ignition date itself would still risk leaking mid-fire weather.

This required a full tile-cache rebuild (~2 days), since every previously-cached fire tile's
ERA5 stats were computed before this filter existed.

## `cfgrib` hypercube fragility (found *during* the rebuild above) - 07/25/26

The rebuild hit a new failure: `TypeError: '<' not supported between instances of 'tuple' and
'datetime.datetime'`, in the newly-added cutoff filter. Two things had to go right to find the
real cause, and my first attempt at both was wrong:

- `cfgrib.open_datasets()` splits one grib file into multiple "hypercube" datasets, and which
  hypercube a given variable ends up in isn't stable across files — the original code assumed
  fixed positional indices (`datasets[0]`, `datasets[1]`). Fixed by searching every hypercube
  by variable name instead. This wasn't sufficient on its own.
- The real bug: for forecast-style data with separate `time` (init) and `step` (lead-time)
  dimensions, `.stack(valid_time=('time','step'))` produces a `MultiIndex` whose values are raw
  `(time, step)` *tuples* — not real datetimes, and not comparable to one. I initially treated
  this as already fixed by the hypercube-search change; it wasn't the same bug. Only found the
  actual cause after being pushed to look again, reasoning that since the crash lived inside
  the `if cutoff_datetime is not None:` block, it had to be something specific to the new
  cutoff logic — which it was. Fixed with `_flatten_time_step()`: explicitly compute the real
  valid datetime as `time + step`, then swap it in for the MultiIndex via
  `reset_index(...).rename(...)`.

Validated the fix with a synthetic xarray test in a sandbox before calling it safe, rather than
just re-running the multi-day pipeline again on faith.

## `train.py`: getting it running at all - 07/27/26

Once tile caching was clean, `train.py` needed its own round of fixes to work with the newer
nested `tile_cache/` structure:

- `load_records()` used a top-level (non-recursive) glob against a directory that was now
  nested — fixed with `glob.glob(..., recursive=True)` and a `**` pattern.
- Running `python training/train.py` from the repo root didn't put the repo root on
  `sys.path` (Python adds the *script's own* directory, not the caller's cwd) — so
  `from model import ...` failed with `ModuleNotFoundError`. Fixed with the same
  `REPO_ROOT`/`sys.path.append` pattern `build_tile_cache.py` already used.
- The scene_id collision bug (above) resurfaced here in its record-field form, causing the
  tile/label count mismatch assertion.

## NaN and Inf propagation into `BCELoss` - 07/27/26

Last crash before the first successful run: `RuntimeError: all elements of input should be
between 0 and 1` out of `BCELoss`, since its inputs come from `F.sigmoid(...)` and
`sigmoid(NaN)` is itself `NaN`. Root-caused in two layers:

1. **NaN**, from pandas: `zonal_aggregator.py`'s `groupby(...).agg(['mean', 'std'])` defaults
   to `ddof=1`, which is undefined (`NaN`) for any tile with exactly one contributing pixel —
   a very plausible occurrence for sparsely-covered ERA5-grid cells. The existing
   `nullify_nan()` only blanked a *whole* stats dict if *every* value in it was NaN, so a tile
   with valid means but a single NaN std slipped through untouched. `model/dataset.py`'s
   imputation logic also only checked `is None`, never `NaN`, so nothing downstream caught it
   either. First fix: switch `_safe_mean` to `np.nanmean` (plus an all-NaN guard), and add an
   explicit `_is_missing()` check (catching both `None` and `NaN`) to the imputation loops in
   both `model/dataset.py` and `training/train.py`'s `build_feature_stds()`.

2. **This didn't fully fix it** — the same crash recurred after redeploying, traced to a
   second, different source: **Inf**, not NaN. `calibrate_to_sigma()` in
   `sentinel1_preprocessing.py` divided by the calibration value squared with no zero guard;
   when that value was exactly `0`, the result was `+Inf` (not NaN, despite what the existing
   warning message claimed), and `Inf` isn't caught by `np.isnan()` anywhere in the chain — it
   sailed straight through `nullify_nan()`, both `_is_missing()` checks, the dB conversion
   (`Inf > 0` is `True`, so `log10(Inf) = Inf`), and into the model, where Inf arithmetic in
   the conv/linear layers is exactly the kind of thing that degrades into NaN by the output
   layer. Fixed at the source (explicit NaN output where the calibration value is zero,
   instead of dividing into Inf) and defensively everywhere downstream (`nullify_nan`,
   both `_is_missing` helpers) by switching the check from `np.isnan` to `not np.isfinite`,
   so Inf can't sneak past the same gap again from a different source in the future.

One good side effect of finally tracking `training/` in git properly (it wasn't, until this
point — the whole file was untracked on EC2, so none of these fixes had actually been reaching
the EC2 copy until caught and re-synced): confirmed via diff that the only difference between
the untracked EC2 copy and the real fixed version was exactly the missing `_is_missing` NaN/Inf
filtering — no other drift.

## First full training run — result and diagnosis - 07/27/26

With all of the above fixed, `train.py --cache-dir tile_cache --epochs 20 --batch-size 32` ran
end to end for the first time: 500 cached scenes (125 fires, 375 controls), 53652/10703/10349
train/val/test tiles.

```
epoch   1 | train loss 1.1871 acc 0.738 | val loss 0.5815 acc 0.744
epoch   2 | train loss 0.5675 acc 0.750 | val loss 0.5667 acc 0.744
...
epoch  20 | train loss 0.5578 acc 0.750 | val loss 0.5754 acc 0.744
Final test | loss 0.5662 acc 0.750
```

Diagnosis: this is class-imbalance collapse, not a data-quantity problem. Two tells:

- Accuracy is pinned at ~0.744-0.750 in *every single epoch*, train and val alike — almost
  exactly the dataset's control fraction (375 / 500 = 75%). A model that always predicts
  "control" gets ~75% accuracy for free on this split.
- The loss plateau (~0.56-0.58) matches the BCE loss of always predicting the constant
  base-rate probability (0.75) against a 75/25 split almost exactly (≈0.562 analytically).

Together, this says the model converged to just outputting the prior and ignoring its inputs
entirely. Two concrete, likely-fixable causes identified before touching hyperparameters:

1. **No class-imbalance handling** — plain `nn.BCELoss()`, no positive-class weighting, no
   stratified sampling, with a 3:1 control:fire ratio.
2. **No input normalization** — `build_feature_stds()` computes per-feature mean/std, but they
   are currently only used inside `TileAugmenter` (jitter noise scale, dropout imputation
   value) and never actually applied to standardize `x_spatial`/`x_temporal` before they reach
   the model. Features span wildly different scales (Kelvin-scale temperature, small-magnitude
   precipitation, negative-dB SAR, unit-scale NDVI) with no z-scoring — a well-known way to
   get exactly this kind of "gave up, predicted the mean" collapse.

Not yet implemented — next step once we return to this thread.
