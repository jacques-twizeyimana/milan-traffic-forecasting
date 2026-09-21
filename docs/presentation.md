# Individual presentation outline (target: 8-9 minutes)

Use your own voice and explain the code and real outputs. This outline is not a recorded submission.

1. **0:00-0:45 — Problem.** Explain one-step 10-minute forecasting, demand planning, and the research question. Distinguish activity units from bytes.
2. **0:45-2:00 — Data handling.** Show prepare.py and memory.json. Explain 62 files, checksum verification, country aggregation, column selection, chunking, and measured 71.9% DataFrame-memory reduction. Explain missing versus zero.
3. **2:00-3:15 — Evidence.** Show distribution and five-area plots. Name top areas [5161, 5059, 5259]. Explain ACF lag 1 (0.982), daily/weekly dependence, and daily profile differences.
4. **3:15-4:45 — Models.** Explain AR coefficients, Holt-Winters level/trend/season, and one-layer LSTM. Show rolling() and why the target is revealed only after forecasting. Discuss the retrospective selection caveat.
5. **4:45-6:00 — Experiments.** Show experiments.csv and one epoch trace. Explain Dec 9-15 validation, bounded grid search, refitting and fixed test parameters. State one selected configuration and its actual validation improvement.
6. **6:00-7:15 — Results.** Show three metric tables, one forecast panel and timings.csv. Explain why Holt-Winters wins mean RMSE rank; compare persistence and mention any MAE/MAPE disagreement.
7. **7:15-8:15 — Failure case.** Show a failure window and peak_errors.csv. Explain a visible missed change without claiming an unverified event caused it. Discuss one-seed and one-week limitations.
8. **8:15-9:00 — Conclusion.** Summarize accuracy/cost/interpretability and a concrete next experiment. Show README commands and the repository link.

Record the video yourself. Add its real URL and the published repository URL to docs/submission.json and rerun src/report.py.
