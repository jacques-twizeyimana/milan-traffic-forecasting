"""Build the research PDF and presentation outline from actual saved results."""
import json
from pathlib import Path
from xml.sax.saxutils import escape
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,Table,TableStyle,PageBreak

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'; TAB=OUT/'tables'
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleSmall',parent=styles['Title'],fontSize=22,leading=27,spaceAfter=14))
styles['BodyText'].fontSize=10;styles['BodyText'].leading=13;styles['BodyText'].spaceAfter=6
styles['Heading1'].fontSize=16;styles['Heading1'].spaceBefore=8
styles['Heading2'].fontSize=12
styles.add(ParagraphStyle(name='CaptionSmall',parent=styles['BodyText'],fontSize=8,leading=10,textColor=colors.HexColor('#475569')))
styles.add(ParagraphStyle(name='TableBlack',parent=styles['CaptionSmall'],textColor=colors.black))
story=[]


def p(text,style='BodyText'): story.append(Paragraph(text,styles[style]))
def h(text):p(text,'Heading1')
def page():story.append(PageBreak())
def fig(name,width=475,height=None,caption=''):
    image=Image(str(OUT/'figures'/f'{name}.png'))
    image.drawHeight=height or width*image.imageHeight/image.imageWidth
    image.drawWidth=width
    story.append(image)
    if caption:p(caption,'CaptionSmall')
    story.append(Spacer(1,7))
def table(headers,rows,widths=None,style='CaptionSmall'):
    cells=[[Paragraph(escape(str(v)),styles[style]) for v in row] for row in [headers]+rows]
    t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7eef1')),
        ('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#64748b')),
        ('BOTTOMPADDING',(0,0),(-1,-1),4),('TOPPADDING',(0,0),(-1,-1),4)]))
    story.append(t);story.append(Spacer(1,9))
def footer(canvas,doc):
    canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#64748b'))
    canvas.drawString(48,28,'Milan mobile Internet activity | Formative 1 | September 2026')
    canvas.drawRightString(547,28,str(doc.page))


def main():
    required=['preparation.json','memory.json','eda.json','environment.json','tables/metrics.csv',
              'tables/timings.csv','tables/predictions.csv','tables/experiments.csv','tables/failures.csv']
    missing=[x for x in required if not (OUT/x).exists()]
    if missing:raise SystemExit('Run the real-data pipeline first. Missing: '+', '.join(missing))
    prep=json.loads((OUT/'preparation.json').read_text());mem=json.loads((OUT/'memory.json').read_text())
    eda=json.loads((OUT/'eda.json').read_text());env=json.loads((OUT/'environment.json').read_text())
    submission=json.loads((ROOT/'docs/submission.json').read_text())
    scores=pd.read_csv(TAB/'metrics.csv');timing=pd.read_csv(TAB/'timings.csv')
    order = {'AR': 0, 'Holt-Winters': 1, 'LSTM': 2, 'Persistence': 3}
    scores = scores.assign(_order=scores.model.map(order)).sort_values(['area','_order']).drop(columns='_order')
    trials=pd.read_csv(TAB/'experiments.csv');ranks=pd.read_csv(TAB/'model_ranks.csv')
    failures=pd.read_csv(TAB/'failures.csv');peaks=pd.read_csv(TAB/'peak_errors.csv')
    first=pd.read_csv(TAB/'first_two_weeks_summary.csv');pred=pd.read_csv(TAB/'predictions.csv')
    ids=prep['top_three'];winner=ranks.iloc[0].model
    assessed=scores[scores.model!='Persistence']
    p('Comparing simple sequential models<br/>for Milan mobile traffic forecasting','TitleSmall')
    p('An empirical study of 10-minute, one-step-ahead forecasts<br/>Evaluation: December 16-22, 2013','Heading2')
    p('Author: '+escape(submission['author']))
    p(f'<b>Finding.</b> {winner} has the lowest mean per-area RMSE rank among the three assessed models. '
      'This conclusion applies to three retrospectively selected high-traffic areas and one held-out week, '
      'not to all Milan areas or longer forecast horizons.')
    h('1. Introduction')
    p('Short-horizon mobile traffic forecasts can inform near-term capacity allocation and identify emerging overload. '
      'Underprediction during a traffic peak may leave too little capacity, whereas persistent overprediction wastes resources. '
      'The research question is: how do distinct sequential models compare for one-step-ahead Internet activity forecasting, '
      'and how does their performance vary across high-traffic geographical areas?')
    p('This study compares a linear autoregression (AR), additive Holt-Winters smoothing, and a compact long short-term memory '
      '(LSTM) network. The objective is an interpretable accuracy-versus-computation comparison, with persistence as a reference. '
      'Traffic is an anonymized activity measure, not directly measured throughput or bytes.')
    h('2. Related work')
    p('Barlacchi et al. [1] describe the multi-source Milan dataset and its spatial and temporal aggregation. '
      'That provenance supports the area/time aggregation used here, but does not identify causal explanations for particular peaks. '
      'The official release [2] is used rather than a preselected third-party subset.')
    p('Santos et al. [3] study LSTM and GRU on the same Milan release. They aggregate to 30-minute intervals, average within '
      'traffic-similarity clusters, and find LSTM competitive with GRU. This motivates a nonlinear recurrent comparator. '
      'Here the unit is one area at 10-minute resolution, so their numerical errors cannot be compared directly. '
      'Their repeated runs also provide stronger stochastic evidence than this single-seed formative study.')
    p('Kochetkova et al. [4] compare seasonal ARIMA and Holt-Winters on mobile traffic from Portugal, with different winners '
      'for download and upload traffic. This motivates retaining lightweight statistical methods and evaluating each area '
      'rather than assuming neural networks win. Our AR is deliberately simpler than seasonal ARIMA: its coefficients are '
      'directly interpretable, but it has no differencing or moving-average innovation terms. Holt-Winters [5] explicitly '
      'updates level, trend and seasonality; LSTM can learn nonlinear sequence dependence but requires more computation.')
    page();h('3. Dataset and data preparation')
    p(f'The official release contains {prep["files"]} daily files ({prep["source_bytes"]/1e9:.2f} GB), '
      f'covering {escape(prep["start"])} to {escape(prep["end"])}. It divides Milan into 10,000 areas. '
      'All 62 published files, including January 1, 2014, contribute to total-traffic ranking. '
      f'The selected IDs, in descending order, are {", ".join(map(str,ids))}. This full-period selection uses future '
      'information to define the study population, as the assignment requires. No future values enter model fitting or tuning.')
    p('The loader verifies every source file against the official size and MD5, reads 250,000 rows at a time, '
      'and selects only Square ID, Unix-millisecond timestamp and Internet activity. It sums country-level contributions '
      'with float64 accumulation, combines groups crossing chunk boundaries, and writes compressed daily Parquet files. '
      'Only verified, successfully aggregated raw downloads are deleted. A manifest and daily audit records support resumption.')
    p('Unix milliseconds are decoded in UTC and converted to Europe/Rome before defining calendar windows. '
      'A regular 10-minute grid preserves absent observations as missing. Blank country-level contributions are omitted '
      'when other observed contributions exist; all-blank groups stay missing. Genuine zeros stay zero. Model inputs use '
      'causal forward fill; missing targets are excluded consistently. Absence may mean no recorded activity or a collection gap; '
      'the source does not resolve this distinction for each missing point.')
    p('Missing selected-area intervals: '+escape(str(prep['missing_selected']))+'. '
      'The pipeline rejects missing initial histories and invalid IDs, negative observed activity, or off-grid timestamps.')
    reduction=100*(1-mem['optimized']['dataframe_bytes']/mem['baseline']['dataframe_bytes'])
    table(['Identical 100,000-row sample','DataFrame MiB','Peak process MiB'],[
        [mode,f'{mem[mode]["dataframe_bytes"]/2**20:.2f}',f'{mem[mode]["peak_rss_bytes"]/2**20:.2f}'] for mode in ['baseline','optimized']], [235,110,130],style='TableBlack')
    p(f'Column selection and dtypes reduce sample DataFrame memory by {reduction:.1f}%. Baseline reads all eight '
      'columns with inferred dtypes; optimized reads three columns with explicit dtypes. Each measurement runs in a '
      'fresh subprocess and peak RSS includes imports and parsing allocations, so its reduction need not match DataFrame memory. '
      f'The full preparation process peaks at {prep.get("initial_peak_rss_bytes",prep["peak_rss_bytes"])/2**20:.1f} MiB; compact daily files occupy '
      f'{prep["aggregate_bytes"]/1e9:.2f} GB. The strategy trades disk I/O and aggregation time for bounded RAM. '
      'The sample comparison is measured evidence, not a claim that the entire unoptimized dataset was loaded.')
    h('4. Exploratory analysis')
    fig('distribution',caption='Figure 1. Full-release area totals and cumulative concentration; all 10,000 areas.')
    p(f'The median area total is {eda["median_area_total"]:,.0f} activity units, while the largest is '
      f'{eda["max_area_total"]:,.0f}. The busiest 1% of areas account for {eda["top_one_percent_share"]*100:.1f}% '
      'of observed Internet activity. This concentration explains the focus on busy areas but also limits generalization '
      'to quieter regions. The distribution concerns accumulated activity, not the physical land area or number of users.')
    page();h('4.1 First two weeks across five areas')
    fig('first_two_weeks',width=460,caption='Figure 2. First 14 days from the first published observation; separate y-scales preserve each area\'s variation.')
    table(['Square ID','Mean','Std / mean','Maximum'],[
        [int(r.area),f'{r["mean"]:.1f}',f'{r.coefficient_of_variation:.2f}',f'{r["max"]:.1f}'] for _,r in first.iterrows()], [110,120,120,125])
    volatile=first.loc[first.coefficient_of_variation.idxmax()]
    p(f'Square {int(volatile.area)} has the largest relative variation in this five-area sample '
      f'(standard deviation / mean = {volatile.coefficient_of_variation:.2f}). '
      'Squares 5259 and 4159 show strong weekday daytime activity and quieter weekends. Square 5161 instead has '
      'large weekend peaks and an isolated spike above 8,000 units on November 2. Square 5059 retains a daily cycle '
      'throughout the fortnight; 4556 has smaller relative variation and more irregular intraday peaks. '
      'These differences motivate area-specific fits and testing daily/weekly history. Work/rest routines are plausible '
      'contributors, but the measurements do not identify a venue or causal event.')
    page();h('4.2 Temporal dependence and periodic structure')
    fig('autocorrelation',caption='Figure 3. Autocorrelation computed before December 9; dashed lines mark one and seven days.')
    p(f'Lag-1 correlation is {eda["acf_lag1"]:.3f}, daily-lag correlation {eda["acf_daily"]:.3f}, and '
      f'weekly-lag correlation {eda["acf_weekly"]:.3f}. Strong immediate dependence motivates short AR lags and '
      'the persistence benchmark. Daily/weekly lags test whether recurring patterns add information beyond recent history. '
      'Autocorrelation alone does not establish causality or guarantee incremental predictive value.')
    fig('daily_profiles',caption='Figure 4. Mean weekday/weekend profiles of the highest-traffic area, training period only.')
    p(f'Training-period weekday mean is {eda["weekday_mean"]:.1f}, versus {eda["weekend_mean"]:.1f} on weekends. '
      'The profile differences expose a limitation of a single daily seasonal template. A 144-step LSTM sees one day of '
      'history but has no explicit day-of-week input; only the extended AR candidate directly includes a weekly lag.')
    page();h('4.3 Decomposition and stationarity diagnostics')
    fig('decomposition',caption='Figure 5. Robust STL with a 144-interval daily period; used for training-period diagnosis, never as a forecasting input.')
    p(f'Daily seasonal strength, max(0, 1 - Var(residual)/Var(residual + season)), is {eda["seasonal_strength"]:.3f}. '
      'A changing trend or structured residual indicates that a fixed repeating daily cycle is incomplete. STL uses a '
      'two-sided smoother, so this decomposition is restricted to the exploratory training period and does not feed forecasts.')
    p(f'Augmented Dickey-Fuller p-values are {eda["adf_p"]:.3g} for the level series and '
      f'{eda["adf_difference_p"]:.3g} after first differencing (constant term, AIC lag selection). '
      'The null is a unit root; rejection is not proof of constant variance or absence of seasonal change. '
      'These diagnostics form part of the second investigation with decomposition and daily profiles, and are not used '
      'to tune against the test week.')
    h('5. Methodology')
    p('The initial training period ends December 8. December 9-15 is validation; December 16-22 is the untouched test '
      'week. Every area has 1,008 forecast targets. At target t the predictor sees only observations through t-1. '
      'The actual observation becomes available after forecasting and may inform the next prediction. This is rolling '
      'one-step evaluation, not a week-ahead recursive forecast. Models train separately for each area.')
    p('AR fits an intercept and lag coefficients by ordinary least squares: prediction(t) = intercept + sum '
      'coefficient(j) x(t-j). Candidate lags are 1-6, 1-12, and 1-12 plus 144 and 1,008. '
      'Original units are retained; there is no normalization or differencing. Training targets with missing observations '
      'are excluded. The longest candidate sacrifices its first week for lag construction.')
    p('Holt-Winters estimates additive level, additive trend, and 144 seasonal states. Prediction(t) = previous level '
      '+ phi x previous trend + seasonal state(t-144), with phi=1 for the undamped candidate. The second candidate '
      'estimates damping. Training-only standard deviation rescales fitting for numerical stability. L-BFGS-B is allowed '
      '2,000 iterations and 100,000 function evaluations after the default evaluation limit caused convergence warnings; '
      'all accepted fits must converge. This numerical correction uses optimizer status, not test errors. Smoothing parameters '
      'and initial states minimize training squared error. At evaluation, parameters stay fixed and states update only '
      'after a new observation arrives. Library equivalence tests verify the recurrence.')
    page();h('5.1 LSTM and bounded parameter search')
    p('LSTM consumes a sequence shaped [batch, length, 1]. A single recurrent layer feeds its last output into one '
      'linear neuron. It compares length 12 (two hours) and 144 (one day), each with 16 or 32 hidden units. '
      'Training-only mean and standard deviation normalize inputs and targets; predictions are inverse transformed. '
      'The network is stateless between windows. It uses Adam at 0.001, MSE loss, chronological batches of 64, '
      'seed 42, CPU execution and one PyTorch thread. Training stops after five epochs without validation improvement '
      'or 30 epochs; the best checkpoint is retained.')
    p('A predefined small grid, allowed by the activity instructions, provides systematic experimentation: three AR, '
      'two Holt-Winters and four LSTM configurations per area, 27 experiments in total. Validation RMSE chooses each '
      'area/model configuration. Final models are refit through December 15, including a newly fitted scaler, and LSTM '
      'uses the selected epoch count. Training configuration is frozen before test evaluation. All models clip negative '
      'predictions to zero. Complete trials and epoch traces are exported rather than selectively reporting wins.')
    table(['Area','Model','Selected configuration','Epochs'],[
        [r.area,r.model, r.config,int(r.epochs)] for _,r in assessed.iterrows()], [40,75,310,50])
    for (area,model),g in trials.groupby(['area','model']):
        best=g.loc[g.RMSE.idxmin()];worst=g.RMSE.max()
        p(f'{area}, {model}: selected validation RMSE {best.RMSE:.2f}, compared with {worst:.2f} for the '
          f'least accurate candidate ({100*(worst-best.RMSE)/worst:.1f}% reduction). '
          '','CaptionSmall')
    p('MAE = mean absolute error; RMSE = square root of mean squared error; MAPE = 100 times mean absolute '
      'relative error over nonzero observed targets. Missing targets are excluded identically. RMSE emphasizes large '
      'misses relevant to overload, while MAPE may overemphasize low traffic. Overall selection uses mean within-area '
      'RMSE rank, giving each area equal weight rather than letting its traffic scale determine the result.')
    page();h('6. Results and discussion')
    for area in ids:
        p(f'Square {area}','Heading2')
        sub=scores[scores.area==area]
        table(['Model','MAE','MAPE (%)','RMSE','Scored'],[
            [r.model,f'{r.MAE:.3f}',f'{r.MAPE:.3f}',f'{r.RMSE:.3f}',int(r.n_scored)] for _,r in sub.iterrows()], [125,90,90,90,80])
        best=sub[sub.model!='Persistence'].sort_values('RMSE').iloc[0]
        baseline=sub[sub.model=='Persistence'].iloc[0]
        delta=100*(baseline.RMSE-best.RMSE)/baseline.RMSE
        p(f'{best.model} leads the assessed models on RMSE ({best.RMSE:.3f}); its relative RMSE improvement '
          f'over persistence is {delta:.1f}%. Negative improvement means persistence is better. '
          f'This area excludes {int(best.n_missing)} unavailable targets and {int(best.n_zero)} zero targets from MAPE.')
    table(['Model','Mean RMSE rank','Mean MAE rank','Mean MAPE rank'],[
        [r.model,f'{r.RMSE_rank:.2f}',f'{r.MAE_rank:.2f}',f'{r.MAPE_rank:.2f}'] for _,r in ranks.iterrows()], [130,115,115,115])
    mape_leaders = ', '.join(ranks.loc[ranks.MAPE_rank == ranks.MAPE_rank.min(), 'model'])
    p(f'{winner} is the overall RMSE-rank winner. '
      f'The MAE-rank winner is {ranks.sort_values("MAE_rank").iloc[0].model}; '
      f'the best mean MAPE rank is shared by {mape_leaders}. '
      'Holt-Winters has the lowest RMSE in every area, consistent with strong daily seasonality and adaptive local states. '
      'However, AR has the lowest MAE and MAPE in 5259, while LSTM has the lowest MAPE in 5059. '
      'The neural model therefore provides no consistent accuracy advantage for this horizon, despite its added cost. '
      'This qualifies the LSTM evidence in [3]: clustered 30-minute series and individual 10-minute series are different tasks.')
    for area in ids:
        page();h(f'6.1 Forecast comparisons: Square {area}')
        for model in ['AR','Holt-Winters','LSTM']:
            fig(f'forecast_{area}_{model}',width=475)
        p('All panels share the same observed targets and forecast timestamps. Each prediction has access to earlier '
          'test-week observations, but never its own target or later values. Persistent lag at abrupt transitions and '
          'systematic smoothing of peaks should be distinguished from good tracking during stable periods.','CaptionSmall')
    page();h('6.2 Computational comparison')
    table(['Area','Model','Refit (s)','Forecast week (s)','Search fits (s)'],[
        [r.area,r.model,f'{r.training_seconds:.4f}',f'{r.prediction_seconds:.4f}',f'{r.tuning_seconds:.2f}'] for _,r in timing.iterrows()], [45,115,95,110,110])
    p('Wall-clock timing uses time.perf_counter around the final fit and around the 1,008-step prediction loop '
      'separately. Final fitting includes scaling and window construction; prediction includes per-step input construction '
      'and Holt-Winters state updates. Downloading, CSV writes and plots are excluded. Search-fit totals include LSTM '
      'epoch validation; standalone post-fit validation inference is excluded. Timings are single measured runs, '
      'not repeated latency confidence intervals. Per-forecast times are saved in timings.csv.')
    p(f'Hardware/runtime: {escape(env["platform"])}; processor {escape(env["processor"])}; '
      f'{env["cpu_count"]} logical CPUs; {env["memory_bytes"]/2**30:.1f} GiB RAM; Python {env["python"]}; '
      'CPU inference and one PyTorch thread. Full dependency versions and execution settings are in environment.json.')
    means=timing.groupby('model')[['training_seconds','prediction_seconds']].mean()
    fastest=means.training_seconds.idxmin()
    p(f'{fastest} has the shortest mean final-fitting time ({means.loc[fastest,"training_seconds"]:.4f} s). '
      f'Mean LSTM fitting takes {means.loc["LSTM","training_seconds"]:.2f} s, compared with '
      f'{means.loc["Holt-Winters","training_seconds"]:.2f} s for Holt-Winters. '
      'Inference latency should be compared with the 600-second sampling interval, while fitting and grid-search '
      'cost matter when deploying thousands of area-specific models. These measurements include Python overhead '
      'and are not optimized serving benchmarks.')
    p('Interpretability also differs: AR exposes lag coefficients; Holt-Winters separates evolving level, trend '
      'and a repeating daily template; LSTM has learned gates and hidden states without a direct physical interpretation. '
      'The comparison tests these distinct inductive assumptions, not equally sized parameter counts. Different history '
      'lengths and tuning budgets are stated limitations.')
    page();h('6.3 Failure analysis')
    for _,r in failures.iterrows():
        fig(f'failure_{r.area}',width=450)
        p(f'Square {r.area}: the worst complete six-hour window for its lowest-RMSE assessed model, {r.best_model}, '
          f'runs from {escape(r.start)} to {escape(r.end)} (36 observations; RMSE {r.window_rmse:.2f}). '
          'This post-hoc diagnostic does not alter fitted parameters.','CaptionSmall')
    page();h('6.4 Peaks and limitations')
    table(['Area','Model','Peak MAE','Peak bias'],[
        [r.area,r.model,f'{r.peak_mae:.2f}',f'{r.peak_bias:.2f}'] for _,r in peaks.iterrows()], [60,145,135,135])
    p('Peak targets are observations at or above each area\'s test-week 95th percentile. Bias is predicted minus '
      'observed, so a negative value means underprediction. This threshold is used only for retrospective diagnosis. '
      'On December 22 in 5161, all models lag the afternoon rise and remain too high during the evening decline. '
      'The 5059 and 5259 windows contain rapid intraday reversals that the models smooth and follow late. '
      'Holt-Winters underpredicts the top 5% of observations in all three areas. Its lowest overall RMSE does not '
      'imply lowest peak MAE: AR is better on peak MAE in 5161 and 5059, and LSTM in 5259. Capacity planning '
      'may therefore favor an asymmetric underprediction loss or safety margin. These are diagnosed limitations, '
      'not evidence of any identified external event.')
    p('The analysis is limited to one seed, one validation week, one test week and three high-volume areas. '
      'There are no confidence intervals across seeds or seasonal test folds. Adjacent grid areas may share traffic '
      'structure and should not be treated as independent replications. Anonymized activity is a proxy for demand, '
      'not an engineering capacity measurement. No claim is made about modern 5G traffic from this 2013 release. '
      'Forward fill can suppress unobserved changes; full-period selection is retrospective; missing source contributions '
      'cannot be fully identified. Additive daily Holt-Winters cannot explicitly represent both daily and weekly patterns.')
    h('7. Conclusion and future work')
    p(f'{winner} offers the best mean RMSE rank in this experiment. The area tables, baseline comparison and measured '
      'computational cost determine whether that advantage is practically worthwhile. Model choice must reflect both '
      'local traffic behavior and operational tolerance for missed peaks; a single aggregate error is insufficient.')
    p('The next useful experiments are repeated chronological evaluation weeks, several LSTM seeds, explicit '
      'day-of-week indicators, and an assessment of whether neighboring-area information improves peak prediction. '
      'A multi-seasonal statistical baseline and uncertainty intervals would test specific limitations before adding '
      'deeper architectures. These are future work, not implemented results.')
    h('Reproducibility and submission')
    p('README.md supplies exact commands; requirements.txt pins dependencies; preparation audits identify sources; '
      'CSV tables preserve all experiments and predictions. Tests cover aggregation, chronological alignment, metric '
      'semantics, causal prediction and Holt-Winters library equivalence. The report and figures are regenerated '
      'from these files.')
    p('GitHub: '+escape(submission['github_url'])+'<br/>Individual video: '+escape(submission['video_url']))
    page();h('References')
    references=[
        '[1] G. Barlacchi et al., "A multi-source dataset of urban life in the city of Milan and the Province of Trentino," Scientific Data, vol. 2, art. 150055, 2015. https://doi.org/10.1038/sdata.2015.55',
        '[2] Telecom Italia, "Telecommunications - SMS, Call, Internet - MI," Harvard Dataverse, V1, 2015. https://doi.org/10.7910/DVN/EGZHFV',
        '[3] G. L. Santos, P. Rosati, T. Lynn, J. Kelner, D. Sadok, and P. T. Endo, "Predicting Short-term Mobile Internet Traffic from Internet Activity using Recurrent Neural Networks," arXiv:2010.05741, 2020. https://arxiv.org/abs/2010.05741',
        '[4] I. Kochetkova, A. Kushchazli, S. Burtseva, and A. Gorshenin, "Short-Term Mobile Network Traffic Forecasting Using Seasonal ARIMA and Holt-Winters Models," Future Internet, vol. 15, no. 9, art. 290, 2023. https://doi.org/10.3390/fi15090290',
        '[5] R. J. Hyndman and G. Athanasopoulos, Forecasting: Principles and Practice, 3rd ed., OTexts, 2021, sec. 8.3. https://otexts.com/fpp3/holt-winters.html',
        '[6] pandas development team, "Scaling to large datasets," pandas documentation. https://pandas.pydata.org/docs/user_guide/scale.html',
        '[7] statsmodels developers, "ExponentialSmoothing," API documentation. https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html']
    for ref in references:p(escape(ref))
    p('Data <link href="http://www.telecomitalia.com/tit/en/bigdatachallenge.html">from BigDataChallenge contest</link>, Telecom Italia, under ODbL 1.0. Release metadata version 1.3; data files version 1. Web references consulted September 2026. The download manifest and package version record are preserved locally.','CaptionSmall')
    SimpleDocTemplate(str(OUT/'report.pdf'),pagesize=(595,842),rightMargin=48,leftMargin=48,
        topMargin=42,bottomMargin=45,title='Milan traffic forecasting: comparative sequential models',
        author=submission['author']).build(story,onFirstPage=footer,onLaterPages=footer)
    presentation=f'''# Individual presentation outline (target: 8-9 minutes)

Use your own voice and explain the code and real outputs. This outline is not a recorded submission.

1. **0:00-0:45 — Problem.** Explain one-step 10-minute forecasting, demand planning, and the research question. Distinguish activity units from bytes.
2. **0:45-2:00 — Data handling.** Show prepare.py and memory.json. Explain {prep['files']} files, checksum verification, country aggregation, column selection, chunking, and measured {reduction:.1f}% DataFrame-memory reduction. Explain missing versus zero.
3. **2:00-3:15 — Evidence.** Show distribution and five-area plots. Name top areas {ids}. Explain ACF lag 1 ({eda['acf_lag1']:.3f}), daily/weekly dependence, and daily profile differences.
4. **3:15-4:45 — Models.** Explain AR coefficients, Holt-Winters level/trend/season, and one-layer LSTM. Show rolling() and why the target is revealed only after forecasting. Discuss the retrospective selection caveat.
5. **4:45-6:00 — Experiments.** Show experiments.csv and one epoch trace. Explain Dec 9-15 validation, bounded grid search, refitting and fixed test parameters. State one selected configuration and its actual validation improvement.
6. **6:00-7:15 — Results.** Show three metric tables, one forecast panel and timings.csv. Explain why {winner} wins mean RMSE rank; compare persistence and mention any MAE/MAPE disagreement.
7. **7:15-8:15 — Failure case.** Show a failure window and peak_errors.csv. Explain a visible missed change without claiming an unverified event caused it. Discuss one-seed and one-week limitations.
8. **8:15-9:00 — Conclusion.** Summarize accuracy/cost/interpretability and a concrete next experiment. Show README commands and the repository link.

Record the video yourself. Add its real URL and the published repository URL to docs/submission.json and rerun src/report.py.
'''
    (ROOT/'docs/presentation.md').write_text(presentation)
    print('Created outputs/report.pdf and docs/presentation.md')

if __name__=='__main__':main()
