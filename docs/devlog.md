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

## Class weighting + input normalization - 07/28/26

Implemented both fixes identified above.

**Class weighting**: switched from `nn.BCELoss()` to `F.binary_cross_entropy(pred, target,
weight=...)` in `run_epoch()`, with a per-batch `weight` tensor built via
`torch.where(target > 0.5, pos_weight, torch.ones_like(target))`. `pos_weight` itself
(`n_neg / n_pos`) is computed once in `main()` from the actual per-tile train labels, not
assumed from the 125/375 scene-level ratio, since different scenes can yield different tile
counts. This makes getting a fire tile wrong cost proportionally more than getting a control
tile wrong, closing off the "always predict control" shortcut. Kept `nn.BCELoss`'s sigmoid
contract in `architecture.py` unchanged (didn't switch to `BCEWithLogitsLoss`) to avoid
touching the model's output semantics elsewhere.

**Input normalization**: added `spatial_mean`/`spatial_std`/`temporal_mean`/`temporal_std`
to `LabeledTileDataset`, computed once in `main()` from `train_ds_unaugmented.statistic_means`
(mean) and `feature_stds` (std) — both already train-split-only, now reused for
normalization instead of just augmentation. Applied in `__getitem__` as a plain z-score
(`(x - mean) / std`) *after* `TileAugmenter`'s jitter/dropout, since those operate in raw
scale (their noise magnitude and imputation values are raw-scale too) — normalizing first
would have made the jitter noise scale meaningless. Same train-derived stats are reused for
val/test to avoid leaking their statistics into what the model treats as "normal."

While wiring this up, also hardened `build_feature_stds()`'s fallback: it already defaulted
to `std=1.0` when there wasn't enough data to compute a real std, but a feature that's
genuinely constant across the whole train split (real std of exactly `0`) wasn't guarded —
dividing by that during normalization would silently produce `Inf`, the same class of bug
fixed in the NaN/Inf entry above. Added `_safe_std()` to catch that case too.

Not yet re-run — next step is rerunning `train.py` and checking whether accuracy actually
moves off the ~75% floor.

## Cleanup pass on the class-weighting/normalization change - 07/28/26

A few small things caught in review of the change above:

- `build_feature_stds(train_tiles, train_ds)` never actually used `train_ds` — dropped the
  parameter and updated the one call site.
- `weight = torch.where(target == 1.0, pos_weight, torch.ones_like(target))` tripped
  SonarQube's floating-point-equality warning. `target` is only ever exactly `0.0` or `1.0`
  in practice (built straight from the label list, no arithmetic on it in between), so this
  was never actually unsafe, but switched to `target > 0.5` anyway to match the threshold
  style already used for accuracy (`pred > 0.5`) and avoid relying on that guarantee holding
  forever.
- Added per-run output directories and a loss-curve plot: each call to `train.py` now creates
  `checkpoints/{timestamp}/` (via `run_id = datetime.now().strftime(...)`) holding
  `best_model.pt`, `final_model.pt`, and a new `loss_curve.png` (`plot_loss_curve()`, using
  `matplotlib` with the `Agg` backend since EC2 has no display to render to). Runs used to
  overwrite the same `best_model.pt`/`final_model.pt` in a flat `checkpoints/` dir every time,
  so there was no way to compare or even keep more than one run's output. Also updated the
  `--upload-to-s3` path to upload under `{s3_model_prefix}{run_id}/` instead of flattening
  every run's `best_model.pt` to the same S3 key.

## Moved `LabeledTileDataset` into `model/dataset.py` - 07/28/26

`LabeledTileDataset` had been living in `training/train.py`, but it's really a data-layer
class (wraps `dataset` with labels, augmentation, and normalization) rather than
training-loop logic, so moved it into `model/dataset.py` alongside `dataset` itself —
`train.py` should stay focused on the argparse/loop/checkpointing side, and this way it's
importable for a future eval or inference script without dragging in the rest of the
training script.

While moving it, also consolidated the duplication this exposed: `train.py` had its own
copies of `S1_KEYS`/`S2_KEYS`/`ERA5_KEYS` (as module-level constants) and `dataset.py`'s
`__getitem__` had a *second*, separately hardcoded copy of the same three key lists
(`s1_keys`/`s2_keys`/`era5_keys`, lowercase, local to the method). Promoted all five key
lists (`S1_KEYS`, `S2_KEYS`, `ERA5_KEYS`, `SPATIAL_KEYS`, `TEMPORAL_KEYS`) to module-level
constants in `model/dataset.py`, and pointed both `dataset.__getitem__` and
`LabeledTileDataset` at the same set — one definition instead of two that had to be kept in
sync by hand. Also promoted `train.py`'s local `_is_missing()` helper to a shared,
non-private `is_missing_value()` in `model/dataset.py`, used by both `dataset.__getitem__`
and `train.py`'s `build_feature_stds()`. `model/__init__.py` now re-exports
`LabeledTileDataset`, the five key-list constants, and `is_missing_value`.

Purely a structural move — no behavior change, verified via `py_compile` on all three
touched files.

## First run with class weighting + normalization - 07/28/26

Reran `train.py` with both fixes live. Loss curve: train loss falls smoothly (0.97 → 0.92),
val loss stays noisy and roughly flat (0.985-1.035), no longer pinned dead flat at the old
~0.744 accuracy/~0.56 loss floor — so the class-weighted BCE broke the "always predict
control" collapse from before. First read of the plot looked like classic overfitting (train
down, val flat), but the full printout told a different story once accuracy was visible:

```
epoch   1 | train loss 0.9719 acc 0.626 | val loss 1.0235 acc 0.561
...
epoch  12 | train loss 0.9217 acc 0.640 | val loss 0.9902 acc 0.441   (best val loss so far)
...
epoch  20 | train loss 0.9164 acc 0.645 | val loss 1.0126 acc 0.524
Final test | loss 0.9333 acc 0.576
```

Train accuracy plateaus around 63-65% — itself *below* the naive 75% majority-class
baseline, and barely moving over 20 epochs. Val accuracy swings wildly (0.44-0.67) with no
relationship to val loss at all (epoch 12's best-val-loss checkpoint has one of the *worst*
val accuracies in the whole run). That combination — train not fitting well either, and
val accuracy decoupled from val loss — doesn't match overfitting (train low/confident, val
diverging upward); it looks more like plain accuracy no longer being a meaningful metric
once the loss is class-weighted (a model can trade raw accuracy for lower weighted loss by
not defaulting to the majority class), plus a still-noisy/weak signal overall. Conclusion:
not overfitting yet — the more useful next step is tracking a real classification metric
(precision/recall/F1 or balanced accuracy per class) instead of plain thresholded accuracy,
since accuracy and the training objective are no longer aligned.

Also fixed a real bug surfaced by this discussion: `main()`'s final test evaluation ran on
whatever `model` held after the *last* epoch, never on `best_model.pt` (the actual
best-val-loss checkpoint saved to disk) — so "Final test" was silently reporting the
last-epoch model's performance, not the best one. Fixed by reloading `best_model.pt`'s state
dict into a separate `best_model` instance for the final test pass, leaving `model`'s
last-epoch weights untouched so `final_model.pt` still saves something meaningfully
different from `best_model.pt`.

## Added balanced accuracy / precision / recall / F1 - 07/28/26

Following directly from the metric-mismatch conclusion above: `run_epoch()` now accumulates
TP/FP/FN/TN counts (across all 3 horizon heads, same flattening the existing accuracy count
already did) and returns balanced accuracy, precision, recall, and F1 on the fire (positive)
class alongside loss/accuracy, instead of just the two. Balanced accuracy is `(recall +
specificity) / 2` — a single number that isn't inflated by the majority class the way plain
accuracy is, so it's what I'd actually trust as a headline number given the 3:1 class
imbalance. Precision/recall/F1 on the fire class are there for a finer-grained read on the
same question (e.g. whether the model is trading recall for precision or vice versa, which a
single accuracy or balanced-accuracy number would hide).

`run_epoch()` now returns a dict instead of a `(loss, acc)` tuple to keep the growing set of
metrics from turning into an unreadable positional-tuple return; updated the three call sites
in `main()` accordingly. The per-epoch and final-test print lines now report val bal_acc /
precision / recall / f1 alongside the existing loss/acc (train metrics still print loss/acc
only, to keep the line from getting too long — bal_acc etc. are the ones that actually needed
fixing).

## Second run with balanced accuracy visible - not overfitting, just weak - 07/28/26

Rerun with the new metrics live: val balanced accuracy sits in a 0.54-0.60 band across all 20
epochs (final test: 0.595), against a 0.50 floor for random guessing. So the model is
extracting *some* real signal — meaningfully better than chance, and recall on the fire class
gets as high as 0.65-0.74 in several epochs, which the old collapsed model could never
produce — but 0.55-0.60 balanced accuracy is weak, not something to trust operationally yet.
Both train and val loss plateau early and stay flat rather than diverging, which rules out
overfitting as the explanation; this looks like the model converging to a weak local optimum
and getting stuck there.

Also noticed while reading this run: val loss and val balanced accuracy don't track each
other. Epoch 15 had the best val loss (0.9870) but a mediocre bal_acc (0.560); epoch 17 had
worse val loss (1.0290) but the best bal_acc of the whole run (0.575). Since loss and the
metric that actually matters given the 3:1 imbalance have diverged, selecting `best_model.pt`
by lowest val loss was picking a systematically different (and probably worse, by the metric
that matters) checkpoint than selecting by val balanced accuracy would.

## Checkpoint selection: val loss -> val balanced accuracy - 07/28/26

Fixed the mismatch above directly: `main()`'s checkpoint-selection loop now tracks
`best_val_balanced_acc` (`float("-inf")` init) and saves `best_model.pt` whenever a new epoch
beats it, instead of comparing on `val_loss`. The checkpoint dict now stores both
`val_balanced_acc` and `val_loss` for reference, and the "Loaded best checkpoint" print at
final-test time reports both too, so it's still possible to see what the val loss was for the
selected epoch even though it's no longer the selection criterion.

Next step: rerun once more to confirm this doesn't just relocate the same weak-signal problem
to a different epoch, then move on to the hyperparameter sweep script (also considering the
temporal transformer's undersized `d_model=5` bottleneck as something worth including, not
just standard training hyperparameters like lr/batch size/epochs).

## Rerun with fixed checkpoint selection: mechanically correct, ceiling unchanged - 07/28/26

Reran with the fix above. Confirmed epoch 12 (bal_acc 0.5663) was genuinely the highest val
balanced accuracy of the whole run — the selection logic is doing what it's supposed to.
But final test balanced accuracy (0.591) landed within noise of the previous run's (0.595),
which is expected: checkpoint selection only decides *which* epoch's weights you keep, it
can't raise a run's ceiling. Two clean runs now show the same weak-signal plateau (val
balanced accuracy stuck in a 0.54-0.60 band) regardless of which epoch gets picked, which is
the actual signal that it's time for the hyperparameter sweep rather than more single-config
runs.

## Widened temporal transformer + hyperparameter sweep script - 07/28/26

Before building the sweep, added a real architecture fix rather than just sweeping around the
existing bottleneck: `TransformerEncoder` in `model/architecture.py` had no input projection,
so its internal `d_model` was hard-tied to `temporal_input_dim` — just 5 (the raw ERA5
variables: u10, v10, d2m, t2m, tp). A 5-dimensional attention space is very little room for a
transformer to represent anything in. Added `self.input_proj = nn.Linear(input_dim,
hidden_dim)` ahead of the transformer layers, so the internal width is now a separate,
tunable `hidden_dim` (renamed the constructor's positional args accordingly). `WildfireModel`
gained a new `temporal_hidden_dim` param (default `32`) plumbed through, plus explicit
`ValueError`s if `embedding_dim` or `temporal_hidden_dim` aren't divisible by `n_head` (both
feed `nn.MultiheadAttention`/`nn.TransformerEncoderLayer`, which require that and otherwise
fail with a much less obvious error).

Worth flagging honestly, since it affects how much to expect from this fix: widening
`d_model` does NOT fix a deeper issue I noticed while making this change. `x_temporal` going
into the transformer is a single flat vector per tile (the ERA5 stats are already
mean-aggregated over the whole antecedent window in `zonal_aggregator.py`, not preserved as a
real multi-timestep sequence), and `TransformerEncoder.forward` feeds it in as
`X.unsqueeze(0)` — a sequence of length 1. Self-attention over a single token is a no-op (one
token can only attend to itself), so the "attention" in the temporal encoder isn't
contributing anything beyond the linear/FFN sublayers regardless of `d_model`. Properly
fixing that would mean preserving the ERA5 time series as actual separate sequence positions
upstream (a real pipeline change, not just a hyperparameter or a small architecture tweak) —
noted here as a candidate for later, not implemented now.

To support the sweep without re-reading `tile_cache/` from disk on every trial, refactored
`training/train.py`: pulled dataset construction out of `main()` into `build_datasets()`
(returns the three `LabeledTileDataset` splits plus scene/tile counts and `pos_weight_value`),
and pulled the epoch loop out into `train_model()` (returns the best-val-balanced-accuracy
state dict in memory via `copy.deepcopy`, rather than round-tripping through disk on every
improving epoch the way `main()` used to). `main()` now just calls both and handles
argparse/checkpoint-saving/plotting — same behavior as before, verified by re-reading the
full diff line by line since this touched nearly the whole file. Added `--weight-decay`
(default `0.0`, passed straight to `Adam`) and `--temporal-hidden-dim` (default `32`) CLI
args as part of this.

New `scripts/hyperparameter_sweep.py`: random search using `build_datasets()`/`train_model()`
directly (dataset built once, reused across every trial). Searches `lr` (log-uniform,
1e-4 to 1e-2), `batch_size`, `embedding_dim`, `temporal_hidden_dim`, `n_layers`, `n_head`, and
`weight_decay`. `embedding_dim`/`temporal_hidden_dim` candidate sets were deliberately chosen
as multiples of every candidate `n_head` value, so every sampled combination is valid by
construction — no divisibility-rejection/retry logic needed. Each trial trains a short run
(`--trial-epochs`, default 8) rather than a full one, scored on val balanced accuracy for
consistency with the checkpoint-selection metric. Writes every trial's config + result to
`sweeps/sweep_results.json`, prints the top 5 by val balanced accuracy, and prints a ready-to
-run `train.py` command for the winning config — deliberately does NOT auto-retrain the
winner at full length, so there's a chance to sanity-check the winning config before
committing a longer run to it.

Not yet run — next step is actually kicking off the sweep on EC2.

## First sweep run - sequential, CPU-only - 07/28/26

Kicked off `hyperparameter_sweep.py --n-trials 20 --trial-epochs 8` on EC2. Confirmed it's
running as designed: trial 1 (`embedding_dim=16`, `n_head=8`, `temporal_hidden_dim=32`, an
otherwise-valid divisibility combo) completed cleanly at `val bal_acc 0.5619`, in the same
~0.54-0.60 band every full run has landed in — no surprises, dataset/pos_weight counts match
prior runs exactly since the split seed is unchanged. At ~125s/trial for 8 epochs, 20 trials
is roughly 40 minutes total on CPU.

Trials run strictly sequentially (`for trial in ...: run_trial(...)`), each one training to
completion before the next starts — deliberate given this is a single CPU box with no GPU, so
"concurrent" trials would just be fighting each other for the same cores rather than actually
speeding anything up, and keeping one model in memory at a time avoids any risk of trials
interfering with each other's `torch` global state.

**Future work note**: if this ever moves to a GPU instance (or a multi-core box where trials
plausibly wouldn't just contend with each other), it'd be worth adding real concurrency — a
`--n-workers`/`--parallel` flag on the existing script (e.g. `multiprocessing` or
`concurrent.futures`, each worker pinned to its own GPU or CPU affinity), or a separate
script entirely if the concurrency model ends up being different enough (e.g. dispatching
trials as independent EC2/cloud jobs rather than in-process workers) to not be worth
shoehorning into the current sequential design. Not needed while everything runs on a single
CPU box, but worth remembering once that changes.

## Full sweep confirmation run - 08/02/26

Ran the sweep's winning config at full length (30 epochs). Result landed right where the
20-trial clustering predicted: best val bal_acc 0.5863 at epoch 20, final test bal_acc 0.580.
Confirms the ~0.52-0.60 ceiling wasn't a short-training artifact — more epochs on the winning
config didn't break through it. Next lever is the ERA5 mean-aggregation limitation flagged
during the sweep write-up, not further hyperparameter search.

## ERA5 daily-sequence pipeline change - 08/12/26

Implemented the pipeline change flagged during the sweep: `zonal_aggregator.py`'s
`aggregate()` was mean-collapsing each tile's whole ERA5 antecedent window into a single
scalar per variable before it ever reached the model, so `TransformerEncoder` was attending
over a sequence of length 1 (a no-op — self-attention over one token can only attend to
itself). Fixed by resampling each ERA5 variable to daily means and keeping a fixed
`N_ERA5_DAYS=30` window (`_resample_daily_last_n`), so `era5_stats` per tile is now
`{var: [30 daily floats]}` instead of `{var: float}`. Resampled once per variable over the
whole grid, not per tile — doing it ~500x per scene for no reason would've been drastically
slower since a tile's daily series is just a lat/lon slice of the same grid.

Picked a fixed 30-day window (truncate, not pad) over keeping the full variable-length
collected window (33-61 days depending on where in its month a fire/control landed) with
padding + an attention mask — the collection strategy already guarantees at least 30 days per
scene, so truncating avoids padding/masking complexity with no real downside.

Downstream changes to consume `(seq_len, n_vars)` instead of a flat `(n_vars,)` vector:
`model/dataset.py`'s mean-imputation now flattens across tiles and days (a NaN/Inf day gets
imputed with that variable's dataset-wide mean, not the whole sequence); `x_temporal` is built
via transpose to `(seq_len, n_vars)`. `model/augmentation.py`'s `TileAugmenter.jitter` now
samples independent noise per timestep instead of one constant offset repeated across all 30
days (a single per-variable offset would've been a much weaker augmentation); `dropout`
needed no code change since its mask already derives from `x_temporal`'s actual shape.
`training/train.py`'s `build_feature_stds` ERA5 branch flattens across days the same way.

`model/architecture.py`'s `TransformerEncoder` had no positional encoding at all (meaningless
at sequence length 1). Added a fixed sinusoidal encoding rather than a learned `nn.Embedding`
table — sinusoidal adds zero trainable parameters, which matters given the dataset is already
small enough that the sweep plateaued at ~0.52-0.60 bal_acc regardless of model size; a
learned encoding would add parameters with limited data to learn them well. Also switched
`nn.TransformerEncoderLayer` to `batch_first=True` to keep `(batch, seq_len, hidden)` shape
throughout, and widened `WildfireModel.forward`'s dimensionality assertions from `(1, 2)` to
`(2, 3)` to match the new unbatched/batched shapes.

Since `s1_stats`/`s2_stats` don't change in this pipeline change, wrote
`scripts/rebuild_era5_stats.py` to patch each cached tile's `era5_stats` field in place
rather than rerunning the full multi-day `build_tile_cache.py` (S3 downloads, RTC, cloud
masking, everything) — it only re-downloads each scene's small ERA5 grib (discarded after the
original run) and overwrites `era5_stats` using the `(i, j)` tile keys already in the cache.

Not yet run on EC2 — next step is a smoke test on a few scenes before running across the full
cache.

## Divergent git branches after parallel EC2/Mac edits - 08/13/26

Pushed a follow-up commit from the Mac (the `x_temporal.shape[-1]` fix below) and hit a
rejected push — turned out the EC2 side had independently implemented the same ERA5
sequence pipeline change and already pushed it, so `main` had diverged. Diffed `main` against
`origin/main` before touching anything: the real difference was almost entirely cosmetic
(`N_ERA5_DAYS` constant placement/wording), except origin was missing the `shape[-1]` fix
below entirely. `git pull --rebase origin main` resolved cleanly. Lesson: when running the
same Claude session across two machines on the same repo, check `git log origin/main` before
assuming a rejected push means something trivial.

## Model-construction shape bug - 08/13/26

Caught before running training again: `train.py`'s `main()` and `hyperparameter_sweep.py`'s
`run_trial()` both compute `temporal_input_dim` as `x_temporal0.shape[0]`, correct when
`x_temporal0` was a flat `(n_vars,)` vector but wrong now that it's `(seq_len, n_vars)` =
`(30, 5)` -- `shape[0]` is the sequence length, not the per-timestep feature count
`TransformerEncoder.input_proj` expects. Would have built `nn.Linear(30, hidden_dim)` and
crashed on real `(batch, 30, 5)` input. Fixed all four call sites to `x_temporal0.shape[-1]`.

## Severe training slowdown from unvectorized per-day loops - 08/13/26

First real sweep attempt after the rebase stalled for 30+ minutes with zero trials completing
(the old sweep's slowest trial finished in ~12 minutes). Root cause: two spots in the ERA5
sequence change did Python-level loops over every (day, variable) pair per tile, per epoch,
instead of vectorized numpy ops. `model/dataset.py`'s `__getitem__` ran a 150-iteration
(30 days x 5 vars) list comprehension calling `is_missing_value()` per entry, for every one of
53,652 training tiles, every epoch. `model/augmentation.py`'s `jitter()` was worse -- it made
150 individual scalar `self.rng.normal()` calls per tile per epoch instead of one vectorized
call. Across 8 epochs that's tens of millions of Python-level calls just for these two spots.

Fixed both: `dataset.py`'s imputation now builds a `(5, 30)` numpy array directly from the
per-key lists (5 Python-level iterations, not 150) and imputes via a single `np.where` against
the whole array. `augmentation.py`'s `jitter()` now samples the entire `(seq_len, n_vars)`
noise matrix in one `self.rng.normal(..., size=(seq_len, n_vars))` call instead of a nested
Python loop. Both were introduced in the original ERA5 sequence diff and only surfaced once
actually run at full dataset scale -- worth remembering that "looks correct" and "runs fast at
this scale" are different checks, especially for anything inside a per-`__getitem__` hot path.

## "Numpy is not available" after the vectorization fix - 08/13/26

All 20 sweep trials failed immediately with `Numpy is not available` after the vectorization
fix above. Root cause was the EC2 env's numpy/torch ABI mismatch flagged earlier as a harmless
warning (`Failed to initialize NumPy: _ARRAY_API not found`) -- it turned out not to be
harmless for one specific call: `torch.from_numpy()`, used in the new vectorized `jitter()`,
requires the same broken C-API hook and has no fallback, so it hard-fails. `torch.tensor()`
(used everywhere else, including the pre-existing `x_spatial` line that had always worked in
this same environment) takes a slower conversion path that doesn't depend on that hook. Fixed
by swapping `torch.from_numpy(...)` to `torch.tensor(...)` in `jitter()` -- same vectorized
rng call, just a different numpy-to-tensor conversion. `torch.from_numpy` was the only call to
that function anywhere in the repo, confirmed via grep.

## Post-rebuild model-construction bug - 08/13/26

Smoke test and full rebuild (`--skip-existing`, 494 patched + 6 skipped, 0 failed) both ran
clean, but before kicking off training again, caught a real bug in how the model gets built:
`train.py`'s `main()` and `hyperparameter_sweep.py`'s `run_trial()` both compute
`temporal_input_dim` as `x_temporal0.shape[0]`, which was correct back when `x_temporal0` was
a flat `(n_vars,)` vector (shape[0] = 5 features). Now that it's `(seq_len, n_vars)` =
`(30, 5)`, `shape[0]` is the sequence length, not the per-timestep feature count
`TransformerEncoder.input_proj` actually expects — would've built `nn.Linear(30, hidden_dim)`
and crashed the instant it saw real `(batch, 30, 5)` input. Fixed by switching all four call
sites (`train.py` main-run model, its `model_config` dict, its best-checkpoint-reload model,
and the sweep's `run_trial`) to `x_temporal0.shape[-1]` instead. Caught by re-reading the
model-construction code path specifically for the new tensor shapes before running anything,
not by an actual crash.
