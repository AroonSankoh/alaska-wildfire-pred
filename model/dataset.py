import torch
import numpy as np

class dataset(torch.utils.data.Dataset):
    data_list = None

    def __init__(self, tiles):
        """
        Initializes the dataset using the tiles from the zonal aggregator.
        """
        self.data_list = list(tiles.items())
        self.statistic_means = {}

        # retrieve mean values from all source statistics for mean imputation
        _ , first_tile = self.data_list[0]
        for key in first_tile["s1_stats"].keys():
            self.statistic_means[f"mean_{key}"] = np.mean([tile["s1_stats"][key] for _, tile in self.data_list 
                                                           if tile["s1_stats"] is not None])
        for key in first_tile["s2_stats"].keys():
            self.statistic_means[f"mean_{key}"] = np.mean([tile["s2_stats"][key] for _, tile in self.data_list 
                                                           if tile["s2_stats"] is not None])
        for key in first_tile["era5_stats"].keys():
            self.statistic_means[f"mean_{key}"] = np.mean([tile["era5_stats"][key] for _, tile in self.data_list 
                                                           if tile["era5_stats"] is not None])


    def __getitem__(self, index):
        """
        Returns the item in the dataset at the specified index.
        """
        _ , tile = self.data_list[index]

        s1_keys = ['vh_band_mean', 'vh_band_std', 'vv_band_mean', 'vv_band_std']
        s2_keys = ['red_mean', 'red_std', 'green_mean', 'green_std', 'nir_mean', 'nir_std', 'swir_mean', 
                   'swir_std', 'ndvi_mean', 'ndvi_std', 'nbr_mean', 'nbr_std', 'filtered_nbr_mean', 'filtered_nbr_std']
        era5_keys = ['u10', 'v10', 'd2m', 't2m', 'tp']

        # mean imputation for Sentinel-1, Sentinel-2, and ERA5
        if tile["s1_stats"] is None:
                tile["s1_stats"] = dict.fromkeys(s1_keys)
        for key in s1_keys:     
            if tile["s1_stats"][key] is None:
                tile["s1_stats"][key] = self.statistic_means[f"mean_{key}"]       
        
        if tile["s2_stats"] is None:
                tile["s2_stats"] = dict.fromkeys(s2_keys)
        for key in s2_keys:     
            if tile["s2_stats"][key] is None:
                tile["s2_stats"][key] = self.statistic_means[f"mean_{key}"] 
        
        if tile["era5_stats"] is None:
                tile["era5_stats"] = dict.fromkeys(era5_keys)
        for key in era5_keys:     
            if tile["era5_stats"][key] is None:
                tile["era5_stats"][key] = self.statistic_means[f"mean_{key}"] 

        # flatten and concatenate the sentinel statistics into a single vector
        s1_flattened = flatten_stats(tile["s1_stats"] or {})
        s2_flattened = flatten_stats(tile["s2_stats"] or {})

        x_spatial = torch.tensor(np.concatenate([list(s1_flattened.values()), list(s2_flattened.values())])).float()

        # stack era5 statistics into a 2D vector 
        x_temporal = torch.tensor(np.array([tile["era5_stats"][k] for k in era5_keys])).float()

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
