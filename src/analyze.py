"""Generate the required EDA, forecast comparisons and failure investigations."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import acf,adfuller
from statsmodels.tsa.seasonal import STL

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'; FIG=OUT/'figures'; TABLE=OUT/'tables'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'figure.dpi':110,'savefig.dpi':170})
COLORS={'AR':'#007c91','Holt-Winters':'#c06c28','LSTM':'#8156a1'}


def save(fig,name):
    fig.tight_layout(); fig.savefig(FIG/f'{name}.png',bbox_inches='tight'); plt.close(fig)


def date_axis(ax):
    locator=mdates.AutoDateLocator(minticks=4,maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator,tz='Europe/Rome'))
    ax.set_xlabel('Date / time (Europe/Rome)'); ax.set_ylabel('Internet activity units')


def eda():
    FIG.mkdir(parents=True,exist_ok=True)
    frame=pd.read_parquet(ROOT/'data/selected.parquet')
    totals=pd.read_csv(TABLE/'area_totals.csv')
    ids=totals.square_id.head(3).tolist()
    fig,axes=plt.subplots(1,2,figsize=(10,3.4))
    axes[0].hist(totals.total,bins=60,color='#007c91')
    axes[0].set(xlabel='Total Internet activity units',ylabel='Number of areas',title='Full-period distribution: 10,000 areas')
    axes[1].plot(np.arange(1,10001),totals.total.cumsum()/totals.total.sum()*100,color='#007c91')
    axes[1].set(xlabel='Areas ranked by descending traffic',ylabel='Cumulative traffic (%)',title='Concentration of Internet activity')
    save(fig,'distribution')
    first=frame.loc[(frame.index>=frame.index.min())&(frame.index<frame.index.min()+pd.Timedelta(days=14))]
    fig,axes=plt.subplots(5,1,figsize=(10,9),sharex=True)
    for ax,area in zip(axes,list(dict.fromkeys(ids+[4159,4556]))):
        ax.plot(first.index,first[area],lw=.6,color='#007c91');ax.set_title(f'Square {area}',loc='left');ax.set_ylabel('Activity units')
    date_axis(axes[-1]);save(fig,'first_two_weeks')
    train=frame[ids[0]].loc[frame.index<pd.Timestamp('2013-12-09',tz='Europe/Rome')].ffill().dropna()
    correlations=acf(train,nlags=1008,fft=True)
    fig,ax=plt.subplots(figsize=(10,3))
    ax.plot(np.arange(len(correlations))/144,correlations,color='#007c91')
    ax.axhline(0,color='grey',lw=.5)
    for day in [1,7]:ax.axvline(day,color='#c06c28',ls='--',lw=.8)
    ax.set(xlabel='Lag (days; 144 intervals/day)',ylabel='Autocorrelation',title=f'Temporal dependence: Square {ids[0]}, training period')
    save(fig,'autocorrelation')
    decomposition=STL(train,period=144,robust=True).fit()
    fig,axes=plt.subplots(3,1,figsize=(10,6),sharex=True)
    for ax,name,data in zip(axes,['Trend','Daily seasonality','Residual'],[decomposition.trend,decomposition.seasonal,decomposition.resid]):
        ax.plot(train.index,data,lw=.6,color='#007c91');ax.set_ylabel('Activity units');ax.set_title(name,loc='left')
    date_axis(axes[-1]);save(fig,'decomposition')
    profiles=pd.DataFrame({'value':train,'slot':train.index.hour*6+train.index.minute//10,'weekend':train.index.dayofweek>=5})
    daily=profiles.groupby(['weekend','slot']).value.mean().unstack(0)
    fig,ax=plt.subplots(figsize=(10,3))
    for weekend in daily.columns:ax.plot(daily.index/6,daily[weekend],label='Weekend' if weekend else 'Weekday')
    ax.legend();ax.set(xlabel='Hour of day (Europe/Rome)',ylabel='Mean Internet activity units',title=f'Daily profiles: Square {ids[0]}, training period')
    save(fig,'daily_profiles')
    summary=dict(top_three=ids,top_one_percent_share=float(totals.total.head(100).sum()/totals.total.sum()),
        median_area_total=float(totals.total.median()),max_area_total=float(totals.total.max()),
        acf_lag1=float(correlations[1]),acf_daily=float(correlations[144]),acf_weekly=float(correlations[1008]),
        adf_p=float(adfuller(train,autolag='AIC',result_object=False)[1]),adf_difference_p=float(adfuller(train.diff().dropna(),autolag='AIC',result_object=False)[1]),
        seasonal_strength=float(max(0,1-np.var(decomposition.resid)/np.var(decomposition.resid+decomposition.seasonal))),
        weekday_mean=float(train[train.index.dayofweek<5].mean()),weekend_mean=float(train[train.index.dayofweek>=5].mean()))
    stats=first.agg(['mean','std','min','max']).T
    stats['coefficient_of_variation']=stats['std']/stats['mean']
    stats.to_csv(TABLE/'first_two_weeks_summary.csv',index_label='area')
    (OUT/'eda.json').write_text(json.dumps(summary,indent=2))
    return frame,ids


def results():
    pred=pd.read_csv(TABLE/'predictions.csv')
    pred['time']=pd.to_datetime(pred.time,utc=True).dt.tz_convert('Europe/Rome')
    scores=pd.read_csv(TABLE/'metrics.csv'); failures=[]; peaks=[]
    for area,g in pred.groupby('area'):
        for model,m in g.groupby('model'):
            fig,ax=plt.subplots(figsize=(10,3))
            ax.plot(m.time,m.actual,label='Observed',color='#333333',lw=.9)
            ax.plot(m.time,m.predicted,label=model,color=COLORS[model],lw=.7,alpha=.85)
            ax.set_title(f'{model} | Square {area} | One-step forecasts, December 16-22, 2013')
            ax.legend(loc='upper right');date_axis(ax);save(fig,f'forecast_{area}_{model}')
            threshold=m.actual.quantile(.95); mask=m.actual>=threshold
            peaks.append(dict(area=area,model=model,threshold=threshold,peak_points=int(mask.sum()),
                peak_bias=float((m.predicted[mask]-m.actual[mask]).mean()),
                peak_mae=float((m.predicted[mask]-m.actual[mask]).abs().mean())))
        sub=scores[(scores.area==area)&(scores.model!='Persistence')].copy()
        sub[['model','MAE','MAPE','RMSE','n_scored','n_missing','n_zero']].to_csv(TABLE/f'metrics_{area}.csv',index=False)
        best=sub.sort_values('RMSE').iloc[0].model
        m=g[g.model==best].reset_index(drop=True)
        end=int(((m.actual-m.predicted)**2).rolling(36,min_periods=36).mean().idxmax())
        start=end-35
        failures.append(dict(area=area,best_model=best,start=str(m.time.iloc[start]),end=str(m.time.iloc[end]),
            window_rmse=float(np.sqrt(((m.actual.iloc[start:end+1]-m.predicted.iloc[start:end+1])**2).mean()))))
        fig,ax=plt.subplots(figsize=(10,3))
        ax.plot(m.time.iloc[start:end+1],m.actual.iloc[start:end+1],label='Observed',color='#333333',lw=1.5)
        for model,data in g.groupby('model'):
            data=data.reset_index(drop=True).iloc[start:end+1]
            ax.plot(data.time,data.predicted,label=model,color=COLORS[model],lw=1)
        ax.set_title(f'Square {area}: worst six-hour window for {best}');ax.legend();date_axis(ax)
        save(fig,f'failure_{area}')
    pd.DataFrame(failures).to_csv(TABLE/'failures.csv',index=False)
    pd.DataFrame(peaks).to_csv(TABLE/'peak_errors.csv',index=False)
    ranks=scores[scores.model!='Persistence'].copy()
    for metric in ['RMSE','MAE','MAPE']:ranks[f'{metric}_rank']=ranks.groupby('area')[metric].rank()
    ranks.groupby('model')[[f'{m}_rank' for m in ['RMSE','MAE','MAPE']]].mean().sort_values('RMSE_rank').to_csv(TABLE/'model_ranks.csv')

if __name__=='__main__':
    eda()
    if (TABLE/'predictions.csv').exists():results()
