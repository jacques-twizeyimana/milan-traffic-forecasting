"""Three small univariate forecasting models with causal one-step prediction."""
import copy
import time
import numpy as np
import torch
from torch import nn
from statsmodels.tsa.holtwinters import ExponentialSmoothing

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    valid = np.isfinite(actual)
    assert np.isfinite(predicted[valid]).all(), 'Missing/nonfinite prediction'
    y, p = actual[valid], predicted[valid]
    assert len(y), 'No observed targets'
    error = p-y
    nonzero = y != 0
    return dict(MAE=float(np.abs(error).mean()), RMSE=float(np.sqrt(np.mean(error**2))),
                MAPE=float(np.mean(np.abs(error[nonzero]/y[nonzero]))*100) if nonzero.any() else float('nan'),
                n_scored=int(valid.sum()), n_missing=int((~valid).sum()), n_zero=int((~nonzero).sum()))


def lag_matrix(values, indices, lags):
    return np.asarray(values)[np.asarray(indices)[:,None] - np.asarray(lags)[None,:]]


class AR:
    def __init__(self, lags): self.lags = np.array(lags)
    def fit(self, values, observed=None):
        indices = np.arange(max(self.lags),len(values))
        if observed is not None: indices = indices[np.isfinite(observed[indices])]
        x = lag_matrix(values,indices,self.lags)
        self.beta = np.linalg.lstsq(np.c_[np.ones(len(x)), x],values[indices],rcond=None)[0]
        return self
    def predict(self, history):
        return float(np.r_[1.,np.asarray(history)[-self.lags]] @ self.beta)


class HW:
    def __init__(self, damped=False, period=144): self.damped, self.period = damped, period
    def fit(self, values, observed=None):
        # Scale improves numerical conditioning; fitted states return to original units.
        self.scale = max(float(np.std(values)),1.)
        self.result = ExponentialSmoothing(np.asarray(values)/self.scale, trend='add',
            damped_trend=self.damped, seasonal='add', seasonal_periods=self.period,
            initialization_method='estimated').fit(optimized=True, use_brute=False,
                minimize_kwargs={'options': {'maxiter': 2000, 'maxfun': 100000}})
        if not self.result.mle_retvals.success:
            raise RuntimeError(f'Holt-Winters did not converge: {self.result.mle_retvals.message}')
        p = self.result.params
        self.alpha, self.beta, self.gamma = p['smoothing_level'],p['smoothing_trend'],p['smoothing_seasonal']
        self.phi = p['damping_trend'] if self.damped else 1.
        self.level, self.trend = float(self.result.level[-1]),float(self.result.trend[-1])
        self.seasons = list(self.result.season[-self.period:])
        self.position = 0
        return self
    def predict(self, history=None):
        return (self.level+self.phi*self.trend+self.seasons[self.position])*self.scale
    def update(self, actual):
        actual /= self.scale
        old_level, old_trend = self.level, self.trend
        old_season = self.seasons[self.position]
        self.level = self.alpha*(actual-old_season)+(1-self.alpha)*(old_level+self.phi*old_trend)
        self.trend = self.beta*(self.level-old_level)+(1-self.beta)*self.phi*old_trend
        self.seasons[self.position] = self.gamma*(actual-old_level-self.phi*old_trend)+(1-self.gamma)*old_season
        self.position = (self.position+1)%self.period


class Network(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.recurrent = nn.LSTM(1,hidden,batch_first=True)
        self.output = nn.Linear(hidden,1)
    def forward(self,x):
        return self.output(self.recurrent(x)[0][:,-1,:]).squeeze(-1)


class LSTM:
    def __init__(self, length, hidden): self.length,self.hidden = length,hidden
    def fit(self, values, observed=None, validation=None, epochs=30):
        torch.manual_seed(42)
        self.mean, self.std = float(np.mean(values)),max(float(np.std(values)),1e-8)
        scaled = (np.asarray(values)-self.mean)/self.std
        windows = np.lib.stride_tricks.sliding_window_view(scaled,self.length+1)
        valid = np.ones(len(windows),dtype=bool) if observed is None else np.isfinite(observed[self.length:])
        x = torch.tensor(windows[valid,:-1].copy(),dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(windows[valid,-1].copy(),dtype=torch.float32)
        self.net = Network(self.hidden)
        optimizer = torch.optim.Adam(self.net.parameters(),lr=.001)
        best, stale, state = float('inf'),0,None
        self.epoch_log = []
        if validation is not None:
            val_values,val_actual = validation
            all_scaled = (np.r_[values[-self.length:],val_values]-self.mean)/self.std
            val_x = torch.tensor(np.lib.stride_tricks.sliding_window_view(all_scaled,self.length+1)[:,:-1].copy(),dtype=torch.float32).unsqueeze(-1)
        for epoch in range(1,epochs+1):
            self.net.train()
            loss_sum = 0.
            for offset in range(0,len(x),64):
                optimizer.zero_grad()
                loss = nn.functional.mse_loss(self.net(x[offset:offset+64]),y[offset:offset+64])
                loss.backward(); optimizer.step()
                loss_sum += float(loss.detach())*len(x[offset:offset+64])
            record = dict(epoch=epoch,training_mse=loss_sum/len(x))
            if validation is not None:
                self.net.eval()
                with torch.no_grad():
                    prediction = torch.cat([self.net(b) for b in val_x.split(256)]).numpy()*self.std+self.mean
                score = metrics(val_actual,np.maximum(prediction,0))['RMSE']
                record['validation_rmse'] = score
                if score < best:
                    best,stale,state,self.best_epoch = score,0,copy.deepcopy(self.net.state_dict()),epoch
                else: stale += 1
            self.epoch_log.append(record)
            if validation is not None and stale>=5: break
        if state is not None: self.net.load_state_dict(state)
        else: self.best_epoch = epochs
        self.net.eval()
        return self
    def predict(self,history):
        x = torch.tensor((np.asarray(history[-self.length:])-self.mean)/self.std,dtype=torch.float32).reshape(1,-1,1)
        with torch.no_grad(): return float(self.net(x).item()*self.std+self.mean)


def rolling(model,values,start,end):
    """Predict target t before revealing it. Parameters never refit in this loop."""
    predictions = []
    for t in range(start,end):
        predictions.append(max(0.,model.predict(values[:t])))
        if isinstance(model,HW): model.update(values[t])
    return np.array(predictions)
