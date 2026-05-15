import torch
import numpy as np

class dataset(torch.utils.data.Dataset):
    data_list = None

    def __init__(self, tiles):
        """
        Initializes the dataset using the tiles from the zonal aggregator.
        """
        self.data_list = list(tiles.items())

    def __getitem__(self, index):
        """
        Returns the item in the dataset at the specified index.
        """

        _ , tile = self.data_list[index]

        # flatten and concatenate the sentinel statistics into a single vector
        s1_flattened = flatten_stats(tile["s1_stats"] or {})
        s2_flattened = flatten_stats(tile["s2_stats"] or {})

        ## TODO: replace with mean imputation, replacing NaNs with zeros is not mathematically accurate
        x_spatial = torch.tensor(np.nan_to_num(np.concatenate([list(s1_flattened.values()), list(s2_flattened.values())]))).float()

        # stack era5 statistics into a 2D vector 
        keys = ['u10', 'v10', 'd2m', 't2m', 'tp']
        ## TODO: replace with mean imputation, replacing NaNs with zeros is not mathematically accurate
        x_temporal = torch.tensor(np.nan_to_num(np.array([tile["era5_stats"][k] for k in keys]))).float()

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

        


        
