"""참고용: 데모 지도에 넣은 구역별 예측값(demo_data.json)을 만든 스크립트.
개발 계획의 03(최종 모델 저장), 04(구역 사전 계산)를 만들 때 참고하세요.
- 01_전처리_피처생성.py의 정제 부분을 그대로 불러와 실행하고, 01이 만든 학습 데이터 CSV로 최종 모델을 학습합니다.
- 원본 CSV 두 개, 01 스크립트, 학습 데이터 CSV와 같은 폴더에서 실행합니다.
- 파일 경로를 이 묶음에 맞게 바꾼 뒤에는 다시 실행해 보지 않았습니다.
"""
import pandas as pd, numpy as np, lightgbm as lgb, json, base64, re, warnings; warnings.filterwarnings('ignore')
src = open('01_전처리_피처생성.py', encoding='utf-8').read().split('# ---------- 7)')[0]
exec(src)                                   # 정제된 점포 테이블 df, o, c, SNAP, HORIZON
df['jibun'] = df.id.map(first.set_index('id').jibun)
feat = pd.read_csv('서울경기_일반음식점_학습데이터.csv', parse_dates=['open', 'close'])
num = ['area','n_all','n_same','n_closed_2y','n_opened_1y','close_rate_2y','same_close_rate_2y','same_share','spot_prev','spot_fail_rate']; cat = ['type','gu']
TYPES = sorted(feat.type.unique()); GUS = sorted(feat.gu.unique())
def Xg(x):
    x = x[num+cat].copy(); x['type'] = pd.Categorical(x.type, categories=TYPES); x['gu'] = pd.Categorical(x.gu, categories=GUS); return x
P = dict(n_estimators=230, learning_rate=0.03, num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, cat_smooth=50, verbose=-1)
g = lgb.LGBMClassifier(**P).fit(Xg(feat), feat.closed_3y)       # 데모용: 학습 대상 전체로 학습
print('model trained', len(feat), flush=True)

t = int((SNAP - pd.Timestamp('1900-01-01')).days) + 1
CELL = 500
df['gx'] = (df.x // CELL).astype(int); df['gy'] = (df.y // CELL).astype(int)
act = (c >= t)
df['act'] = act
cells = df[act].groupby(['gx','gy']).size().rename('n_act').reset_index()
cells = cells[cells.n_act >= 5].reset_index(drop=True); cells['cid'] = cells.index
print('cells', len(cells), 'active stores', int(act.sum()), flush=True)
# 후보 건물: 현재 영업 중인 점포가 있는 건물, 셀당 최대 12곳
b = df[act].drop_duplicates('k')[['k','x','y','gx','gy','gu']].merge(cells[['gx','gy','cid']], on=['gx','gy'])
b = b.sample(frac=1, random_state=0); b = b[b.groupby('cid').cumcount() < 12].reset_index(drop=True)
print('candidates', len(b), flush=True)
xy = df[['x','y']].values; tree = cKDTree(xy); tcode = pd.Categorical(df.type, categories=TYPES).codes; NT = len(TYPES)
bx = b[['x','y']].values; nb_n = len(b)
A = np.zeros((nb_n, NT), np.int32); C2 = np.zeros((nb_n, NT), np.int32); O1 = np.zeros(nb_n, np.int32); SP = np.zeros(nb_n, np.int32); SF = np.zeros(nb_n, np.int32)
for s in range(0, nb_n, 2000):
    q = bx[s:s+2000]; nb = tree.query_ball_point(q, 300, workers=-1)
    ln = np.fromiter((len(a_) for a_ in nb), np.int64, len(nb)); jj = np.concatenate([np.asarray(a_, dtype=np.int64) for a_ in nb]); r = np.repeat(np.arange(len(q)), ln)
    a_m = c[jj] >= t; c_m = (c[jj] < t) & (c[jj] >= t - 730)
    A[s:s+2000] = np.bincount(r[a_m]*NT + tcode[jj][a_m], minlength=len(q)*NT).reshape(-1, NT)
    C2[s:s+2000] = np.bincount(r[c_m]*NT + tcode[jj][c_m], minlength=len(q)*NT).reshape(-1, NT)
    O1[s:s+2000] = np.bincount(r[o[jj] >= t - 365], minlength=len(q))
    sp = (c[jj] < t) & (np.abs(xy[jj] - q[r]).sum(1) < 1.0)
    SP[s:s+2000] = np.bincount(r[sp], minlength=len(q)); SF[s:s+2000] = np.bincount(r[sp & ((c[jj]-o[jj]) <= HORIZON)], minlength=len(q))
print('neighbors done', flush=True)
n_all = A.sum(1); ncl = C2.sum(1)
AREAS = [20, 33, 50, 66, 100, 165]
base = pd.DataFrame({'n_all': n_all, 'n_closed_2y': ncl, 'n_opened_1y': O1, 'close_rate_2y': ncl/np.where(n_all+ncl == 0, np.nan, n_all+ncl),
                     'spot_prev': SP, 'spot_fail_rate': SF/np.where(SP == 0, np.nan, SP), 'gu': b.gu.values})
risk = np.zeros((NT, len(AREAS), len(cells)), np.uint8); cid = b.cid.values
for ti, ty in enumerate(TYPES):
    X = base.copy(); X['type'] = ty; X['n_same'] = A[:, ti]
    X['same_share'] = A[:, ti]/np.where(n_all == 0, np.nan, n_all); X['same_close_rate_2y'] = C2[:, ti]/np.where(A[:, ti]+C2[:, ti] == 0, np.nan, A[:, ti]+C2[:, ti])
    for ai, ar in enumerate(AREAS):
        X['area'] = ar; p = g.predict_proba(Xg(X))[:, 1]
        med = pd.Series(p).groupby(cid).median().reindex(range(len(cells))).values
        risk[ti, ai] = np.clip(np.round(med*200), 0, 255)
    print(ty, risk[ti, 2].mean()/2, flush=True)
# 셀 정보
dfa = df[act].merge(cells[['gx','gy','cid']], on=['gx','gy'])
def dong(s):
    if not isinstance(s, str): return ''
    tk = s.split()[2:]
    for w in tk[:3]:
        if re.search(r'(동|읍|면|가)$', w) and not re.search(r'\d+-', w): return w
    return ''
dfa['dong'] = dfa.jibun.map(dong)
mode = lambda s_: s_.value_counts().index[0] if len(s_) else ''
info = dfa.groupby('cid').agg(gu=('gu', mode), dong=('dong', lambda s_: mode(s_[s_ != ''])), lat=('lat','mean'), lon=('lon','mean'))
tc = np.bincount(dfa.cid.values*NT + pd.Categorical(dfa.type, categories=TYPES).codes, minlength=len(cells)*NT).reshape(-1, NT)   # 셀 안 업태별 영업 점포 수
bb = base.assign(cid=cid); cm = bb.groupby('cid').agg(cr=('close_rate_2y','median'), na=('n_all','median'), sp=('spot_prev','median'))
cells = cells.join(info, on='cid').join(cm, on='cid')
gus = sorted(cells.gu.unique()); 
gc = df[act].groupby('gu').agg(x=('x','mean'), y=('y','mean'), n=('x','size')).reindex(gus)
out = dict(cell=CELL, snap=str(SNAP.date()), types=TYPES, areas=AREAS, gus=gus,
    guc=[[int(r.x), int(r.y), int(r.n)] for r in gc.itertuples()],
    gx=cells.gx.tolist(), gy=cells.gy.tolist(), n=cells.n_act.tolist(), g=[gus.index(v) for v in cells.gu], d=cells.dong.fillna('').tolist(),
    cr=[int(round(v*100)) if v == v else 0 for v in cells.cr], na=[int(v) for v in cells.na],
    lat=[round(v,4) for v in cells.lat], lon=[round(v,4) for v in cells.lon],
    tc=base64.b64encode(np.clip(tc,0,255).astype(np.uint8).tobytes()).decode(), risk=base64.b64encode(risk.tobytes()).decode())
json.dump(out, open('demo_data.json','w'), ensure_ascii=False, separators=(',',':'))
print('json MB', len(json.dumps(out, ensure_ascii=False))/1e6, 'types', TYPES)
