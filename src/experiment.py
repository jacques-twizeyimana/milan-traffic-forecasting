"""Chronological model selection and held-out evaluation; all outputs are measured."""
import json
import argparse
import platform
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
from models import AR,HW,LSTM,metrics,rolling

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs'


def main(only=None):
    frame = pd.read_parquet(ROOT/'data/selected.parquet')
    ids = json.loads((OUT/'preparation.json').read_text())['top_three']
    train_end = frame.index.searchsorted(pd.Timestamp('2013-12-09',tz='Europe/Rome'))
    test_start = frame.index.searchsorted(pd.Timestamp('2013-12-16',tz='Europe/Rome'))
    test_end = frame.index.searchsorted(pd.Timestamp('2013-12-23',tz='Europe/Rome'))
    assert test_end-test_start == 1008
    configs = [('AR',{'lags':list(range(1,7))}),('AR',{'lags':list(range(1,13))}),
        ('AR',{'lags':list(range(1,13))+[144,1008]}),
        ('Holt-Winters',{'damped':False}),('Holt-Winters',{'damped':True})]
    configs += [('LSTM',{'length':l,'hidden':h}) for l in [12,144] for h in [16,32]]
    classes = {'AR':AR,'Holt-Winters':HW,'LSTM':LSTM}
    rows, predictions, timing, experiments, epochs = [],[],[],[],[]
    fitted_parameters = []
    if only:
        # Refit one model after a numerical fix without repeating unaffected runs.
        def retained(filename):
            df = pd.read_csv(OUT/'tables'/filename)
            return df[df.model != only]
        rows = retained('metrics.csv').to_dict('records')
        timing = retained('timings.csv').to_dict('records')
        experiments = retained('experiments.csv').to_dict('records')
        predictions = [retained('predictions.csv')]
        epochs = pd.read_csv(OUT/'tables/epochs.csv').to_dict('records') if only != 'LSTM' else []
        fitted_parameters = [p for p in json.loads((OUT/'fitted_parameters.json').read_text()) if p['model'] != only]
        configs = [(name, config) for name, config in configs if name == only]
        classes = {only: classes[only]}
    for area in ids:
        original = frame[area].to_numpy()
        values = frame[area].ffill().to_numpy()
        assert np.isfinite(values).all(), 'Leading missing history requires source investigation'
        area_trials = []
        for name, config in configs:
            print(f'area={area} {name} {config}',flush=True)
            model = classes[name](**config)
            started = time.perf_counter()
            kwargs = {'validation':(values[train_end:test_start],original[train_end:test_start])} if name=='LSTM' else {}
            model.fit(values[:train_end],observed=original[:train_end],**kwargs)
            elapsed = time.perf_counter()-started
            pred = rolling(model,values,train_end,test_start)
            score = metrics(original[train_end:test_start],pred)
            row = dict(area=area,model=name,config=json.dumps(config),training_seconds=elapsed,
                best_epoch=getattr(model,'best_epoch',0),**score)
            experiments.append(row); area_trials.append(row)
            for entry in getattr(model,'epoch_log',[]): epochs.append(dict(area=area,config=json.dumps(config),**entry))
            pd.DataFrame(experiments).to_csv(OUT/'tables/experiments.csv',index=False)
            pd.DataFrame(epochs).to_csv(OUT/'tables/epochs.csv',index=False)
            print(f'validation RMSE={score["RMSE"]:.3f}; fit={elapsed:.2f}s',flush=True)
        for name in classes:
            choices = [r for r in area_trials if r['model']==name]
            best = min(choices,key=lambda r:r['RMSE'])
            model = classes[name](**json.loads(best['config']))
            started = time.perf_counter()
            kwargs = {'epochs':int(best['best_epoch'])} if name=='LSTM' else {}
            model.fit(values[:test_start],observed=original[:test_start],**kwargs)
            train_seconds = time.perf_counter()-started
            started = time.perf_counter()
            pred = rolling(model,values,test_start,test_end)
            predict_seconds = time.perf_counter()-started
            details = dict(area=area,model=name,config=json.loads(best['config']))
            if name=='AR': details['coefficients_intercept_then_lags']=model.beta.tolist()
            elif name=='Holt-Winters':
                details.update(alpha=float(model.alpha),beta=float(model.beta),gamma=float(model.gamma),phi=float(model.phi),
                               converged=bool(model.result.mle_retvals.success))
            else: details.update(mean=model.mean,std=model.std,epochs=model.best_epoch,parameter_count=sum(p.numel() for p in model.net.parameters()))
            fitted_parameters.append(details)
            rows.append(dict(area=area,model=name,config=best['config'],epochs=best['best_epoch'],
                **metrics(original[test_start:test_end],pred)))
            timing.append(dict(area=area,model=name,training_seconds=train_seconds,
                prediction_seconds=predict_seconds,seconds_per_forecast=predict_seconds/len(pred),
                tuning_seconds=sum(r['training_seconds'] for r in choices)))
            predictions.append(pd.DataFrame(dict(time=frame.index[test_start:test_end],area=area,model=name,
                actual=original[test_start:test_end],predicted=pred)))
        if not only:
            rows.append(dict(area=area,model='Persistence',config='previous observation',epochs=0,
                **metrics(original[test_start:test_end],values[test_start-1:test_end-1])))
        pd.DataFrame(rows).to_csv(OUT/'tables/metrics.csv',index=False)
        pd.DataFrame(timing).to_csv(OUT/'tables/timings.csv',index=False)
        pd.concat(predictions).to_csv(OUT/'tables/predictions.csv',index=False)
    (OUT/'fitted_parameters.json').write_text(json.dumps(fitted_parameters,indent=2))
    chip = platform.processor()
    if platform.system()=='Darwin':
        chip = subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip()
    machine = dict(platform=platform.platform(),processor=chip,cpu_count=psutil.cpu_count(),
        memory_bytes=psutil.virtual_memory().total,python=platform.python_version(),torch_threads=1,
        seed=42,device='CPU',timing='perf_counter wall-clock; final fit and sequential forecast loop separately; excludes file IO and plotting',
        dependencies=subprocess.check_output([str(ROOT/'.venv/bin/python'),'-m','pip','freeze'],text=True))
    (OUT/'environment.json').write_text(json.dumps(machine,indent=2))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', choices=['AR','Holt-Winters','LSTM'], help='Refresh one model in a completed run')
    main(parser.parse_args().only)
