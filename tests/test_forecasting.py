import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from prepare import aggregate,rank_areas
from models import AR,HW,LSTM,lag_matrix,metrics,rolling
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def test_chunk_aggregation_missing_and_zero():
    frame = pd.DataFrame({'square_id':[1,1,1,2,3],'timestamp':[0,0,0,0,0],
                          'internet':[1.,2.,np.nan,np.nan,0.]})
    full = aggregate(frame)
    chunked = pd.concat([aggregate(frame[:2]),aggregate(frame[2:])]).groupby(level=[0,1]).sum(min_count=1)
    assert_allclose(full,chunked,equal_nan=True)
    assert full.loc[(1,0)] == 3
    assert np.isnan(full.loc[(2,0)])
    assert full.loc[(3,0)] == 0


def test_ranking_and_timezone():
    assert rank_areas(pd.Series([8.,9.,9.],index=[1,3,2])).square_id.tolist()==[2,3,1]
    idx = pd.date_range('2013-12-16', '2013-12-23',freq='10min',tz='Europe/Rome',inclusive='left')
    assert len(idx)==1008
    assert str(idx[0].tz_convert('UTC'))=='2013-12-15 23:00:00+00:00'


def test_lags_and_metrics():
    assert_allclose(lag_matrix(np.arange(10),[5,6],[1,3]),[[4,2],[5,3]])
    m = metrics([0.,2.,np.nan],[1.,4.,5.])
    assert m['MAE']==1.5 and m['MAPE']==100
    assert_allclose(m['RMSE'],np.sqrt(2.5))
    assert m['n_missing']==1 and m['n_zero']==1


def test_causal_fill():
    x = pd.Series([2.,np.nan,9.])
    assert x.ffill().tolist()==[2.,2.,9.]


def series():
    t = np.arange(180)
    return 20+.02*t+3*np.sin(2*np.pi*t/12)+np.random.default_rng(42).normal(0,.1,180)


def test_ar_future_invariance():
    x=series(); model=AR([1,2,12]).fit(x[:100])
    first=rolling(model,x,100,120)
    changed=x.copy(); changed[110:]+=100
    second=rolling(model,changed,100,120)
    assert_allclose(first[:11],second[:11])


def test_hw_matches_library_fixed_parameter_filter():
    x=series()
    for damped in [False,True]:
        model=HW(damped,12).fit(x[:100])
        p=model.result.params
        expected=ExponentialSmoothing(x/model.scale,trend='add',damped_trend=damped,seasonal='add',
            seasonal_periods=12,initialization_method='known',initial_level=p['initial_level'],
            initial_trend=p['initial_trend'],initial_seasonal=p['initial_seasons']).fit(
            smoothing_level=p['smoothing_level'],smoothing_trend=p['smoothing_trend'],
            smoothing_seasonal=p['smoothing_seasonal'],damping_trend=p['damping_trend'] if damped else None,
            optimized=False).fittedvalues[100:]*model.scale
        actual=rolling(model,x,100,len(x))
        assert_allclose(actual,expected,atol=1e-8)
        a=rolling(HW(damped,12).fit(x[:100]),x,100,120)
        changed=x.copy(); changed[110:]+=100
        b=rolling(HW(damped,12).fit(x[:100]),changed,100,120)
        assert_allclose(a[:11],b[:11])


def test_lstm_scaler_and_future_invariance():
    x=series(); model=LSTM(12,4).fit(x[:100],validation=(x[100:120],x[100:120]),epochs=2)
    assert_allclose(model.mean,x[:100].mean())
    assert_allclose(model.std,x[:100].std())
    first=rolling(model,x,100,120)
    changed=x.copy();changed[110:]+=100
    assert_allclose(first[:11],rolling(model,changed,100,120)[:11])
