# Hybrid Wildfire Detection Model

This is a stub design for a hybrid wildfire detection model that seeks to predict an area's risk of wildfire within 1, 3, and 6 month periods. The model processes and aggregates data from three sources (two satellite and one weather data), feeds this aggregated data into a hybrid convolutional-transformer deep learning model, and then output a fire risk score within the previously mentioned time horizons. All data required to perform inference for this model can be freely obtained online! 

## Data Sources 
Sentinel-1 is a satellite that uses Synthetic Aperture Radar (SAR) to image the ground at 20m resolution. You can obtain a Sentinel-1 scene through the Copernicus Open Access Hub (https://dataspace.copernicus.eu/). 

Sentinel-2 is a satellite that uses high-res optics to image the ground at 10m resolution. You can obtain a Sentinel-2 scene through the Copernicus Open Access Hub (https://dataspace.copernicus.eu/). 

ERA5 uses ECMWF reanalysis to collect periodic weather data, such as temperature, humidity, windspeed, etc. You can obtain an ERA5 data package from the Copernicus Climate Data Store (https://cds.climate.copernicus.eu/). 

## Repository Structure 
```
wildfire-pred/
├── data/
│   ├── aggregator/
│   │   └── zonal_aggregator.py
│   └── loaders/
│       ├── era5_preprocessor.py
│       ├── sentinel1_preprocessor.py
│       └── sentinel2_preprocessor.py
├── model/
│   ├── architecture.py
│   ├── augmentation.py
│   └── dataset.py
├── notebooks/
│   └── sentinel2_demo.ipynb
├── scripts/
│   └── build_tile_cache.py
├── env.yml
└── README.md
```

## Setup 
Check the env.yml for environment dependencies, conda is the recommended package manager. Run the following in your terminal once conda is installed and working: 
```bash
conda env create -f env.yml
conda activate wildfire-pred
```

## Included Data Set 
A full scene includes Sentinel-1 pre and post fire SAFE files, Sentinel-2 pre and post SAFE files, an ERA-5 grib file, and a metadata.json with details of the contents of each asset within the scene.
Each fire is paired with three controls that match the fires EPA Level III Eco-region of the fire. 125 fires and 375 control scenes collected over seven US states (Alaska, California, Idaho, Montana, Nevada, Oregon, Washington) comprise the full dataset. 

## Usage 
Once you have identified a dataset for analysis, it is highly recommended to build and cache aggregated tiles before anything else. This way it won't be necessary to repeatedly load and aggregate source data whenever performing model training 
or infererence. Investigate and edit the global variables within ```scripts/build_tile_cache.py``` to ensure the correct source data is used for tiling.  

*Add similar notes on training and inference scripts once those are complete*

## TODOs
Additions (in order of importance) yet to implement include: 
- Training and inference scripts
- Fire Weather Index calculator for ERA5 data (it'd be a very useful performance benchmark)
- Test Cases 
- Documentation