# Milan mobile Internet traffic forecasting

A formative empirical comparison of autoregression, additive Holt-Winters and a small LSTM. Forecasts are one interval (10 minutes) ahead for December 16-22, 2013 in Europe/Rome. The top three areas are selected by total Internet activity over all 62 published daily files. This retrospective selection is assignment-mandated; observations after December 15 never enter parameter fitting or model selection.

## Run

Python 3.12 is recommended. From this folder:

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

The initial download transfers about 20.8 GB. The loader processes one approximately 330 MB source file at a time, verifies the official MD5 and size, writes compact Parquet aggregates and a completion audit, then deletes only its own raw intermediate. Harvard requires an email in its dataset guestbook. Set `DATAVERSE_EMAIL` to an email you authorize sharing with Harvard Dataverse before the first run. The loader uses the official guestbook POST and signed download URL; no access restriction is bypassed. Contact details remain outside tracked outputs. About 4 GB of free disk and several GB of RAM are recommended; keep additional headroom. Network speed dominates preparation time. Eight bounded HTTP byte ranges accelerate each daily download while daily files are still processed sequentially; the final whole-file checksum verifies their assembly. Cached daily files are reused; remove their matching Parquet and JSON files to rebuild a day. `outputs/dataset_manifest.json` records the exact source version and checksums. Daily aggregation reads only three columns in 250,000-row chunks. It retains only one day's intermediate groups, not the full raw dataset.

## Outputs

- `outputs/report.pdf`: complete research report, including nine forecast comparisons.
- `outputs/figures/`: labeled PNGs of EDA, predictions, and failure windows.
- `outputs/tables/`: area ranking, predictions, validation experiments, epoch history, three metric tables, timings, peak errors, failure windows, and model ranks.
- `outputs/memory.json`: identical 100,000-row sample in separate baseline/optimized processes.
- `outputs/preparation.json`: full-run preparation memory/time, coverage and selected IDs.
- `outputs/environment.json`: runtime, hardware, package versions, seed and timing definitions.
- `docs/presentation.md`: outline for an individual 7-10 minute recording.
- `docs/rubric.md`: evidence mapped to all 100 rubric points.

## Design decisions

Country-level records are summed at each Square ID/timestamp using float64. Blank Internet fields are skipped when other measured contributions exist; all-blank groups remain missing. Missing intervals remain NaN in observed outputs and are causally forward-filled only in inputs. All models score the same observed targets. Leading missing history triggers an error rather than inventing values. Genuine zeros are preserved. This conservative policy cannot distinguish an absent activity record from an outage; report coverage and avoid interpreting missing records as certain zero demand.

Use UTC for decoding Unix milliseconds and Europe/Rome for calendar splits. Training ends December 8; validation is December 9-15; test is December 16-22. Each forecast is generated before consuming the observation for that interval. Parameters remain fixed; Holt-Winters states update with new observations. Each area/model has its own selected configuration. Validation RMSE chooses the configuration; test RMSE does not tune anything. LSTM inputs and targets are standardized with training-only mean/std, then inverse transformed. AR uses original-unit lags; Holt-Winters divides by training std for numerical conditioning and reverses it for predictions. All predictions are clipped at zero.

The small predefined grid is an explicit parameter-optimization strategy allowed by `activity.tx`, not a claim of exhaustive model optimization. CPU, seed 42, one PyTorch thread, no batch shuffling, MSE loss, Adam, no dropout or extra layers. Final LSTM fits use the validation-selected epoch count. No spatial covariates or event labels are used. Persistence is a reference, not one of the three assessed models.

MAE and RMSE use all observed targets; MAPE excludes exact-zero targets and reports the number excluded. Small nonzero denominators can still dominate MAPE. Timing uses `perf_counter`: final fit includes preprocessing, inference includes the causal Python loop and HW updates, but excludes downloading/plotting/file I/O. Search training cost is reported separately; epoch validation is included in LSTM search cost. Single-run timings describe this hardware, not statistically stable latency estimates.

## Submission

This folder is ready for a GitHub repository, but is not published automatically. Keep the source documents, code, pinned dependencies, report, figures and result tables; exclude `data/` and `.venv/` using `.gitignore`. Do not commit raw data. Insert the actual GitHub URL and recorded individual-video URL into `docs/submission.json`, then regenerate the report. The author must understand and present the implementation and verify the institutional rules for declaring AI assistance.

## Dataset attribution

Data [from BigDataChallenge contest](http://www.telecomitalia.com/tit/en/bigdatachallenge.html), Telecom Italia; [official dataset](https://doi.org/10.7910/DVN/EGZHFV). The source and derived database outputs are provided under [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/), as specified in the preserved dataset manifest. The release metadata retrieved for this run is version 1.3; source data files are version 1.
