# time-series-crop-model
A demonstration project looking at crop type segmentation using time series.

# Goal
The goals is to implement an active learning approach to systematically detect underperforming crops from the model predictions and strategically incorporate them back into the training pipeline. This process aims to refine the model’s accuracy by iteratively improving its ability to differentiate between crops, particularly in AOIs where it currently struggles due to a lack of representative training data.

# Getting started

## Dev environment

Ensure you have [Remote Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) installed in VS Code, then run:

```
docker-compose build devcontainer
docker-compose up -d devcontainer
```

Then use `Devcontainers: Attach to Running Container` command and attach to the `devcontainer-${USER}` container running on your machine.

## Running
W.I.P.