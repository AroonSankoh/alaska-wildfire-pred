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
│   └── dataset.py
├── notebooks/
│   └── sentinel2_demo.ipynb
├── scripts/
├── env.yml
└── README.md
```

## Setup 
Check the env.yml for environment dependencies, conda is the recommended package manager. Run the following in your terminal once conda is installed and working: 
```bash
conda env create -f env.yml
conda activate wildfire-pred
```

## Reproduction
Data was collected over the clear fire in Fairbanks, Alaska (LAT: 64.32, LON: -149.13), which began on June 27 2022. To obtain the same scenes, search "Fairbanks, US" in the Copernicus Browser and zoom to around 50km. Pre-fire scenes are dated May 29 2022 and post-fire scenes are dated Aug 8 2022 for Sentinel-2 and Aug 9 2022 for Sentinel-1. Remember to set the cloud cover filter "MSI" to at least 20% when searching for the correct scene, or else your data will be corrupted. There is currently a tutorial notebook that steps you through how to use the sentinel-2 loader, further tutorials are currently being developed.

## TODOs
Functions yet to implement include: 
- Fire Weather Index calculator for ERA5 data
- Radiometric Terrain Correction for S1 scenes
- Cross-attention fusion for model encoders 
- Mean imputation for NaN values within tiles
- Tutorial notebooks for loaders, model training, and hyper-parameter optimization