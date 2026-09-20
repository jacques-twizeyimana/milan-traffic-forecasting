"""End-to-end artifact smoke test. Synthetic fixtures stay in pytest's temp directory."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from pypdf import PdfReader
import analyze
import report


def test_artifacts_with_isolated_fixture(tmp_path,monkeypatch):
    for d in ['data','outputs/figures','outputs/tables','docs']:(tmp_path/d).mkdir(parents=True)
    out=tmp_path/'outputs';tab=out/'tables'
    idx=pd.date_range('2013-11-01',periods=62*144,freq='10min',tz='Europe/Rome')
    rng=np.random.default_rng(42)
    values=100+20*np.sin(np.arange(len(idx))*2*np.pi/144)+rng.normal(0,2,len(idx))
    pd.DataFrame({i:values*(1+i/10000) for i in [1,2,3,4159,4556]},index=idx).to_parquet(tmp_path/'data/selected.parquet')
    pd.DataFrame({'square_id':np.arange(1,10001),'total':np.arange(10000,0,-1)*100.}).to_csv(tab/'area_totals.csv',index=False)
    for mod in [analyze,report]:
        monkeypatch.setattr(mod,'ROOT',tmp_path);monkeypatch.setattr(mod,'OUT',out)
    monkeypatch.setattr(analyze,'FIG',out/'figures');monkeypatch.setattr(analyze,'TABLE',tab)
    monkeypatch.setattr(report,'TAB',tab);monkeypatch.setattr(report,'story',[])
    analyze.eda()
    prediction=[];metrics=[];timings=[];trials=[]
    test=idx[(idx>='2013-12-16')&(idx<'2013-12-23')]
    for area in [1,2,3]:
        for name in ['AR','Holt-Winters','LSTM']:
            actual=np.arange(len(test))%144+100.
            prediction.append(pd.DataFrame({'time':test,'area':area,'model':name,'actual':actual,'predicted':actual+1}))
            metrics.append(dict(area=area,model=name,MAE=1.,MAPE=1.,RMSE=1.,n_scored=1008,n_missing=0,n_zero=0,config='{}',epochs=1))
            timings.append(dict(area=area,model=name,training_seconds=1.,prediction_seconds=1.,tuning_seconds=1.))
            trials.append(dict(area=area,model=name,config='{}',RMSE=1.))
        metrics.append(dict(area=area,model='Persistence',MAE=2.,MAPE=2.,RMSE=2.,n_scored=1008,n_missing=0,n_zero=0,config='{}',epochs=0))
    for name,rows in [('metrics',metrics),('timings',timings),('experiments',trials)]:pd.DataFrame(rows).to_csv(tab/f'{name}.csv',index=False)
    pd.concat(prediction).to_csv(tab/'predictions.csv',index=False)
    analyze.results()
    (out/'preparation.json').write_text(json.dumps(dict(files=62,source_bytes=20_000_000_000,top_three=[1,2,3],start=str(idx[0]),end=str(idx[-1]),missing_selected={},peak_rss_bytes=1_000_000,aggregate_bytes=1_000_000)))
    (out/'memory.json').write_text(json.dumps({m:dict(dataframe_bytes=1_000_000,peak_rss_bytes=2_000_000) for m in ['baseline','optimized']}))
    (out/'environment.json').write_text(json.dumps(dict(platform='Synthetic test fixture',processor='test',cpu_count=1,memory_bytes=1_000_000,python='3.12')))
    (tmp_path/'docs/submission.json').write_text(json.dumps(dict(author='SYNTHETIC TEST ONLY',github_url='test',video_url='test')))
    report.main()
    pdf=PdfReader(out/'report.pdf')
    assert len(pdf.pages)>=10
    assert len(list((out/'figures').glob('forecast_*.png')))==9
    assert (tmp_path/'docs/presentation.md').exists()
