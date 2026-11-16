"""기준 모델: 로지스틱 회귀 vs LightGBM, 시간 기준 분할 (2010~2019 학습, 2020~2023 평가)
실행: 01 스크립트가 만든 CSV와 같은 폴더에서  python 02_기준모델.py"""
import pandas as pd, numpy as np, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

d = pd.read_csv('서울경기_일반음식점_학습데이터.csv', parse_dates=['open', 'close'])
num = ['area','n_all','n_same','n_closed_2y','n_opened_1y','close_rate_2y','same_close_rate_2y','same_share','spot_prev','spot_fail_rate']
cat = ['type', 'gu']
logc = ['area','n_all','n_same','n_closed_2y','n_opened_1y','spot_prev']
SPLIT = '2020-01-01'
P = dict(learning_rate=0.03, num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, cat_smooth=50, verbose=-1)

def Xg(x):
    x = x[num + cat].copy()
    for c_ in cat: x[c_] = pd.Categorical(x[c_], categories=sorted(d[c_].unique()))
    return x
Xl = lambda x: x[num + cat].assign(**{c_: np.log1p(x[c_]) for c_ in logc})

def score(y, p):
    k = int(len(y) * 0.1); o = np.argsort(-p)
    return f'AUC {roc_auc_score(y, p):.3f} | 상위10% {y[o[:k]].mean():.3f} | 하위10% {y[o[-k:]].mean():.3f} | 전체 {y.mean():.3f}'

def run(data, name, detail=False):
    tr, te = data[data.open < SPLIT], data[data.open >= SPLIT]
    print(f'\n[{name}] 학습 {len(tr):,} / 평가 {len(te):,}')
    lr = make_pipeline(make_column_transformer(
            (make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), num),
            (OneHotEncoder(handle_unknown='ignore', min_frequency=50), cat)),
         LogisticRegression(max_iter=3000, C=0.5)).fit(Xl(tr), tr.closed_3y)
    print('  로지스틱 회귀:', score(te.closed_3y.values, lr.predict_proba(Xl(te))[:, 1]))
    tr1, va = tr[tr.open < '2018-01-01'], tr[tr.open >= '2018-01-01']      # 조기 종료용: 학습 구간 마지막 2년
    g = lgb.LGBMClassifier(n_estimators=3000, **P).fit(Xg(tr1), tr1.closed_3y, eval_set=[(Xg(va), va.closed_3y)],
            eval_metric='auc', callbacks=[lgb.early_stopping(100, verbose=False)])
    g = lgb.LGBMClassifier(n_estimators=g.best_iteration_, **P).fit(Xg(tr), tr.closed_3y)
    p = g.predict_proba(Xg(te))[:, 1]
    print('  LightGBM     :', score(te.closed_3y.values, p))
    if detail:
        for s_, q in te.assign(p=p).groupby('sido'): print(f'    {s_}:', score(q.closed_3y.values, q.p.values))
        imp = pd.Series(g.booster_.feature_importance('gain'), index=num + cat)
        print((imp / imp.sum()).sort_values(ascending=False).round(3).to_string())
    return g

run(d, '전체(양도 추정 포함, 폐업으로 처리)', detail=True)
run(d[d.handover == 0], '양도 추정 제외')
