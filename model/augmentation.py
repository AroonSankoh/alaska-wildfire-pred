import torch
import numpy as np


class TileAugmenter:
    """
    Augmentation for zonal-statistics tile vectors by appling Gaussian jitter and mixup.
    apply to scalar mean/std features.
    """

    def __init__(self, feature_stds, noise_scale=0.05, mixup_alpha=0.2, dropout_p=0.1, seed=None):
        """
        feature_stds: dict mapping feature name -> dataset-wide std,
                      used to scale jitter per-feature.
        noise_scale: fraction of each feature's std to use as jitter magnitude.
        mixup_alpha: Beta distribution param for mixup interpolation strength.
        dropout_p: probability of dropping (NaN-ing) a given stat, to be
                   caught by the existing mean-imputation logic.
        """
        if seed is None:
            raise ValueError("TileAugmenter requires an explicit seed for reproducibility.")
        self.feature_stds = feature_stds
        self.noise_scale = noise_scale
        self.mixup_alpha = mixup_alpha
        self.dropout_p = dropout_p
        self.rng = np.random.default_rng(seed)
        self.torch_gen = torch.Generator().manual_seed(seed)

    def jitter(self, x_spatial, x_temporal, spatial_keys, temporal_keys):
        """
        Add small Gaussian noise scaled by each feature's dataset std.
        """
        spatial_noise = torch.tensor([
            self.rng.normal(0, self.noise_scale * self.feature_stds.get(k, 1.0))
            for k in spatial_keys
        ]).float()
        # x_temporal is (seq_len, n_vars), so noise is sampled per timestep, not one constant per variable
        seq_len = x_temporal.shape[0]
        temporal_noise = torch.tensor([
            [self.rng.normal(0, self.noise_scale * self.feature_stds.get(k, 1.0)) for k in temporal_keys]
            for _ in range(seq_len)
        ]).float()
        return x_spatial + spatial_noise, x_temporal + temporal_noise

    def dropout(self, x_spatial, x_temporal, impute_value_spatial, impute_value_temporal):
        """
        Randomly replace a feature with its dataset mean, mimicking missing-scene
        imputation so the model doesn't overfit to always-present features.
        """
        mask_s = torch.rand(x_spatial.shape, generator=self.torch_gen) < self.dropout_p
        mask_t = torch.rand(x_temporal.shape, generator=self.torch_gen) < self.dropout_p
        x_spatial = torch.where(mask_s, impute_value_spatial, x_spatial)
        x_temporal = torch.where(mask_t, impute_value_temporal, x_temporal)
        return x_spatial, x_temporal

    def mixup(self, x_spatial_a, x_temporal_a, y_a, x_spatial_b, x_temporal_b, y_b):
        """
        Interpolate between two samples and their labels. y_a / y_b are tuples
        of (1mo, 3mo, 6mo) risk labels.
        """
        lam = self.rng.beta(self.mixup_alpha, self.mixup_alpha)
        x_spatial = lam * x_spatial_a + (1 - lam) * x_spatial_b
        x_temporal = lam * x_temporal_a + (1 - lam) * x_temporal_b
        y = tuple(lam * ya + (1 - lam) * yb for ya, yb in zip(y_a, y_b))
        return x_spatial, x_temporal, y