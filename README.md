# time-series-crop-model
A demonstration project looking at crop type segmentation using time series.

# Goal
The goals is to implement an active learning approach to systematically detect underperforming crops from the model predictions and strategically incorporate them back into the training pipeline. This process aims to refine the model’s accuracy by iteratively improving its ability to differentiate between crops, particularly in AOIs where it currently struggles due to a lack of representative training data.

The specific tasks are

* Identify underperforming crops from the provided raster dataset.
* Diagnose confusion between crop types and propose a strategy to improve model performance.
* Implement efficient geospatial processing using Rasterio, Geopandas, and parallel computation.
* Provide a structured output and recommendations on how to enhance the training pipeline.

# Getting started

## Dev environment

Ensure you have [Remote Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) installed in VS Code, then run:

```
docker-compose build devcontainer
docker-compose up -d devcontainer
```

Then use `Devcontainers: Attach to Running Container` command and attach to the `devcontainer-${USER}` container running on your machine.

## Running
Example:

```
python src/cli.py determine-poor-performance \
-p /code/data/ml_2021-08-01_2022-12-31_u0c.tif \
-l /code/data/u0c_gt_filtered_2022.gpkg \
-pc 3 \
-cc 4 \
-n 10 \
-u
```

# Next Steps
W.I.P.

# TODO
* tests run but dont seem to do anything
* analyze output to see underperformance