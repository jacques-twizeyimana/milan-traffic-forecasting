# Milan mobile Internet traffic forecasting

This project studies whether short-term Internet traffic in Milan can be forecast from its recent history. We use the Telecom Italia mobile traffic dataset and compare three models:

- autoregression (AR)
- additive Holt-Winters exponential smoothing
- a small long short-term memory network (LSTM)

The task is a one-step-ahead forecast at ten-minute intervals. The test period is 16–22 December 2013, and the three areas used in the study are the areas with the highest total Internet activity across the published data.

## What we did

The data is aggregated by Square ID and timestamp. We split the time series chronologically:

- training: 1–8 December
- validation: 9–15 December
- test: 16–22 December

The validation period is used to select model settings. The test period is kept separate until the final comparison. Each model is evaluated on the same observed targets using MAE and RMSE. MAPE is also reported, excluding targets that are exactly zero.

The forecasts are generated causally: for each interval, the model predicts before the observation for that interval is used. Missing intervals remain missing in the observed data. Missing values are only forward-filled when they are needed as model inputs, and genuine zero traffic values are preserved.

## Results and generated files

The main report is [`report.pdf`](report.pdf). The repository also contains the data used to reproduce the analysis:

- `outputs/figures/` contains the exploratory analysis, forecasts, and error-window figures.
- `outputs/tables/` contains the area ranking, predictions, validation experiments, metrics, timings, and failure-window tables.
- `outputs/environment.json` records the Python environment, hardware, random seed, and timing definitions.
- `outputs/preparation.json` records the preparation run, coverage, selected areas, and resource measurements.
- `outputs/memory.json` contains the memory comparison between the baseline and optimized data-loading approaches.
- `outputs/dataset_manifest.json` records the source version and checksums.

## Reproducing the analysis

Python 3.12 is recommended. From the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -u src/prepare.py
python src/analyze.py
python -u src/experiment.py
python src/analyze.py
python src/report.py
python src/verify.py
```

The preparation step downloads approximately 20.8 GB from the official Harvard Dataverse release. Harvard requires an email address in its dataset guestbook, so set `DATAVERSE_EMAIL` to an address you are permitted to share before running the preparation script:

```bash
export DATAVERSE_EMAIL="your-email@example.com"
```

The loader processes one source file at a time, checks its size and MD5 checksum, writes compact Parquet aggregates, and removes only its own raw intermediate file. It uses bounded HTTP byte ranges to speed up each download while processing daily files sequentially. Around 4 GB of free disk space and several GB of RAM are recommended, with additional headroom for the download and processing steps.

Cached daily files are reused. To rebuild one day, remove that day's Parquet and JSON files. The complete source download is not required again when the cached files are available.

## Implementation choices

All country-level records are summed for each Square ID and timestamp using `float64`. The models use a small predefined parameter grid. This keeps the comparison manageable and makes the selection procedure explicit; it is not intended to be an exhaustive search.

The AR model works in the original units. Holt-Winters is scaled using the training standard deviation for numerical stability. LSTM inputs and targets are standardized using training-period statistics only. Predictions are converted back to the original units and clipped at zero.

The LSTM uses CPU execution, seed 42, one PyTorch thread, no batch shuffling, mean squared error loss, Adam optimization, no dropout, and no additional layers. Its final epoch count comes from validation. Timing results include the operations described in `outputs/environment.json` and should be interpreted as measurements for this machine, not as general performance guarantees.

## Limitations

The analysis uses only the traffic history. It does not include spatial covariates, weather, public events, or other external information. The selection of the top three areas is retrospective because it is required by the assignment; observations after 15 December are not used for fitting or model selection.

Missing records cannot always be distinguished from a genuine service outage. For that reason, coverage is reported and missing observations are not automatically interpreted as zero demand. MAPE can also be sensitive to small non-zero denominators, so it is considered alongside MAE and RMSE.

## Dataset attribution

The data comes from the [Telecom Italia Big Data Challenge](http://www.telecomitalia.com/tit/en/bigdatachallenge.html). The official release is available through [Harvard Dataverse](https://doi.org/10.7910/DVN/EGZHFV). The source and derived database outputs are provided under the [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/) licence, as recorded in the dataset manifest.
