"""Check completed deliverables; never pass on missing real-data results."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'


def main():
    prep=json.loads((OUT/'preparation.json').read_text())
    assert prep['files']==62
    assert len(list((ROOT/'data/daily').glob('*.json')))==62
    totals=pd.read_csv(OUT/'tables/area_totals.csv')
    assert len(totals)==10000 and totals.square_id.nunique()==10000
    pred=pd.read_csv(OUT/'tables/predictions.csv')
    assert len(pred)==9*1008 and pred.groupby(['area','model']).ngroups==9
    expected=pd.date_range('2013-12-16','2013-12-23',freq='10min',tz='Europe/Rome',inclusive='left')
    for (area,model),g in pred.groupby(['area','model']):
        assert pd.DatetimeIndex(pd.to_datetime(g.time,utc=True)).equals(expected.tz_convert('UTC'))
        assert np.isfinite(g.predicted).all() and (g.predicted>=0).all()
        assert (OUT/f'figures/forecast_{area}_{model}.png').exists()
    for area,g in pred.groupby('area'):
        observed=[v.actual.to_numpy() for _,v in g.groupby('model')]
        for other in observed[1:]:np.testing.assert_allclose(observed[0],other,equal_nan=True)
        assert (OUT/f'tables/metrics_{area}.csv').exists()
        assert (OUT/f'figures/failure_{area}.png').exists()
    metrics=pd.read_csv(OUT/'tables/metrics.csv');trials=pd.read_csv(OUT/'tables/experiments.csv')
    assert len(metrics)==12 and len(trials)==27
    assert np.isfinite(metrics[['MAE','RMSE']]).all().all()
    assert (metrics.groupby('area').n_scored.nunique()==1).all()
    parameters=json.loads((OUT/'fitted_parameters.json').read_text())
    assert all(p['converged'] for p in parameters if p['model']=='Holt-Winters')
    for _,r in metrics[metrics.model!='Persistence'].iterrows():
        best=trials[(trials.area==r.area)&(trials.model==r.model)].sort_values('RMSE').iloc[0]
        assert r.config==best.config
    reader=PdfReader(OUT/'report.pdf')
    text='\n'.join(p.extract_text() for p in reader.pages)
    for title in ['Introduction','Related work','Methodology','Results and discussion','Conclusion','References']:
        assert title in text
    result={'status':'passed','source_files':62,'forecast_combinations':9,'forecast_rows':len(pred),
            'validation_experiments':len(trials),'pdf_pages':len(reader.pages)}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
