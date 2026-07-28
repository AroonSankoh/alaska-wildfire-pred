import torch
import numpy as np

S1_KEYS = ['vh_band_mean', 'vh_band_std', 'vv_band_mean', 'vv_band_std']
S2_KEYS = ['red_mean', 'red_std', 'green_mean', 'green_std', 'nir_mean', 'nir_std', 'swir_mean',
           'swir_std', 'ndvi_mean', 'ndvi_std', 'nbr_mean', 'nbr_std']
ERA5_KEYS = ['u10', 'v10', 'd2m', 't2m', 'tp']
SPATIAL_KEYS = S1_KEYS + S2_KEYS
TEMPORAL_KEYS = ERA5_KEYS


def is_missing_value(value):
    return value is None or (isinstance(value, float) and not np.isfinite(value))


class dataset(torch.utils.data.Dataset):
    data_list = None

    def __init__(self, tiles):
        """
        Initializes the dataset using the tiles from the zonal aggregator.
        """
        self.data_list = list(tiles.items())
        self.statistic_means = {}

        # retrieve mean values from all source statistics for mean imputation
        valid_tile = None
        for _, tile in self.data_list:
             if tile["s1_stats"] is not None and tile["s2_stats"] is not None and tile["era5_stats"] is not None:
                  valid_tile = tile
                  break
        if valid_tile is None:
              raise ValueError("No valid (non-NaN) tiles exist to initialize the dataset with.")

        # guard before np.mean could be called on empty slices
        def _safe_mean(values, label):
            if len(values) == 0:
                raise ValueError(
                f"No non-None values found for '{label}' across the entire dataset. "
                f"This indicates an upstream bug, like a source failing across the entire dataset."
            )
            finite_values = [v for v in values if np.isfinite(v)]
            if len(finite_values) == 0:
                raise ValueError(
                    f"All values for '{label}' are NaN/Inf across the entire dataset -- "
                    f"cannot compute an imputation mean."
                )
            return float(np.mean(finite_values))

        for key in valid_tile["s1_stats"].keys():
            values = [tile["s1_stats"][key] for _, tile in self.data_list if tile["s1_stats"] is not None]
            self.statistic_means[f"mean_{key}"] = _safe_mean(values, f"s1_stats.{key}")

        for key in valid_tile["s2_stats"].keys():
            values = [tile["s2_stats"][key] for _, tile in self.data_list if tile["s2_stats"] is not None]
            self.statistic_means[f"mean_{key}"] = _safe_mean(values, f"s2_stats.{key}")

        for key in valid_tile["era5_stats"].keys():
            values = [tile["era5_stats"][key] for _, tile in self.data_list if tile["era5_stats"] is not None]
            self.statistic_means[f"mean_{key}"] = _safe_mean(values, f"era5_stats.{key}")


    def __getitem__(self, index):
        """
        Returns the item in the dataset at the specified index.
        """
        _ , tile = self.data_list[index]

        # mean imputation for Sentinel-1, Sentinel-2, and ERA5
        if tile["s1_stats"] is None:
                tile["s1_stats"] = dict.fromkeys(S1_KEYS)
        for key in S1_KEYS:
            if is_missing_value(tile["s1_stats"][key]):
                tile["s1_stats"][key] = self.statistic_means[f"mean_{key}"]

        if tile["s2_stats"] is None:
                tile["s2_stats"] = dict.fromkeys(S2_KEYS)
        for key in S2_KEYS:
            if is_missing_value(tile["s2_stats"][key]):
                tile["s2_stats"][key] = self.statistic_means[f"mean_{key}"]

        if tile["era5_stats"] is None:
                tile["era5_stats"] = dict.fromkeys(ERA5_KEYS)
        for key in ERA5_KEYS:
            if is_missing_value(tile["era5_stats"][key]):
                tile["era5_stats"][key] = self.statistic_means[f"mean_{key}"]

        # flatten and concatenate the sentinel statistics into a single vector
        s1_flattened = flatten_stats(tile["s1_stats"] or {})
        s2_flattened = flatten_stats(tile["s2_stats"] or {})

        x_spatial = torch.tensor(np.concatenate([list(s1_flattened.values()), list(s2_flattened.values())])).float()

        # stack era5 statistics into a 2D vector
        x_temporal = torch.tensor(np.array([tile["era5_stats"][k] for k in ERA5_KEYS])).float()

        return x_spatial, x_temporal

    def __len__(self):
        """
        Returns the size of the dataset.
        """
        if self.data_list is None:
            return 0
        else:
            return len(self.data_list)


def flatten_stats(dictionary, parent_key='', sep=''):
    """
    Flatten nested dictionaries outputted by data loaders.
    """
    items = []
    for k, v in dictionary.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_stats(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class LabeledTileDataset(torch.utils.data.Dataset):
    """
    Wraps dataset with per-tile labels, optional train-time augmentation (TileAugmenter),
    and z-score input normalization. 
    """
    def __init__(self, tiles, labels, spatial_mean, spatial_std, temporal_mean, temporal_std, augmenter=None):
        self.inner = dataset(tiles)
        self.labels = labels
        self.augmenter = augmenter
        assert len(self.inner) == len(self.labels), \
            f"tile/label count mismatch: {len(self.inner)} vs {len(self.labels)}"

        # input normalization stats, which are always derived from the train split
        self.spatial_mean = spatial_mean
        self.spatial_std = spatial_std
        self.temporal_mean = temporal_mean
        self.temporal_std = temporal_std

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
            # apply normalization after augmentation because jitter/dropout operate at a raw scale
            x_spatial, x_temporal = self.augmenter.jitter(x_spatial, x_temporal, SPATIAL_KEYS, TEMPORAL_KEYS)
            x_spatial, x_temporal = self.augmenter.dropout(
                x_spatial, x_temporal, self.impute_spatial, self.impute_temporal)

        # z-score normalization so all features can be read by the model with comparable scales
        x_spatial = (x_spatial - self.spatial_mean) / self.spatial_std
        x_temporal = (x_temporal - self.temporal_mean) / self.temporal_std

        y = torch.tensor(self.labels[idx]).float()
        return x_spatial, x_temporal, y
