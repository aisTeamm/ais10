"""서울+경기 일반음식점 인허가 데이터 -> 정제, 정답(3년 내 폐업), 주변 피처 생성
실행: 원본 CSV 두 개와 같은 폴더에서  python 01_전처리_피처생성.py"""
import pandas as pd, numpy as np
from scipy.spatial import cKDTree
from pyproj import Transformer

FILES = {'서울특별시': '식품_일반음식점_서울특별시.csv', '경기도': '식품_일반음식점_경기도.csv'}
RADIUS = 300            # 주변 반경(m)
HORIZON = 365 * 3       # 정답 기준: 개업 후 3년
TEMP_DAYS = 30          # 이 일수 이내 폐업은 행사/팝업용 임시 영업으로 보고 제외
COHORT_START = '2010-01-01'
OUT = '서울경기_일반음식점_학습데이터.csv'

m = {'관리번호':'id','인허가일자':'open','영업상태명':'status','폐업일자':'close','소재지면적':'area',
     '사업장명':'name','업태구분명':'type','도로명주소':'road','지번주소':'jibun','좌표정보(X)':'x','좌표정보(Y)':'y'}
log, parts, snaps = [], [], []
for sido, f in FILES.items():
    d = pd.read_csv(f, encoding='cp949', dtype=str, usecols=list(m)).rename(columns=m)
    d['sido'] = sido; n0 = len(d)
    d['addr'] = d.road.fillna(d.jibun)
    d = d[d.addr.str.startswith(sido, na=False)]            # 다른 시도 주소 제거
    parts.append(d); log.append((f'{sido} 원본', n0))
df = pd.concat(parts, ignore_index=True).drop_duplicates('id')
for c in ['x','y','area']: df[c] = pd.to_numeric(df[c].str.strip(), errors='coerce')
df['open'] = pd.to_datetime(df['open'], errors='coerce'); df['close'] = pd.to_datetime(df['close'], errors='coerce')
SNAP = df.groupby('sido').open.max().min()                  # 데이터 기준일
log.append(('합친 뒤(주소 불일치 제거)', len(df)))

# ---------- 1) 기본 정리 ----------
df = df[df.open.notna() & ~((df.status == '폐업') & df.close.isna())];   log.append(('인허가일/폐업일 누락 제거', len(df)))
df = df[df.x.notna() & df.y.notna()];                                    log.append(('좌표 없는 행 제거', len(df)))
tf = Transformer.from_crs('EPSG:5174', 'EPSG:4326', always_xy=True)
df['lon'], df['lat'] = tf.transform(df.x.values, df.y.values)
df = df[df.lat.between(36.85, 38.35) & df.lon.between(126.3, 127.9)];    log.append(('범위 밖 좌표 제거', len(df)))
df.loc[df.area <= 0, 'area'] = np.nan                                     # 면적 0은 결측으로
df['type'] = df.type.fillna('기타')
vc = df.type.value_counts(); df.loc[df.type.isin(vc[vc < 1000].index), 'type'] = '기타'   # 드문 업태 통합
df['k'] = df.x.round().astype(np.int64) * 10**7 + df.y.round().astype(np.int64)           # 같은 자리(건물) 키
df['nm'] = df.name.fillna('').str.replace(r'\s+|\(.*?\)', '', regex=True)                  # 상호 정규화
df['gu'] = df.sido.str[:2] + ' ' + df.addr.str.split().str[1]

# ---------- 2) 재신고/명의 변경: 같은 자리 + 같은 상호로 폐업 전 30일~후 90일에 새 인허가 -> 한 점포로 연결 ----------
df = df.sort_values(['k', 'nm', 'open', 'id']).reset_index(drop=True)
same = (df.k == df.k.shift()) & (df.nm == df.nm.shift()) & (df.nm != '')
gap = (df.open - df.close.shift()).dt.days
link = same & gap.between(-30, 90)
df['chain'] = (~link).cumsum()
first = df.drop_duplicates('chain', keep='first').set_index('chain')
last = df.drop_duplicates('chain', keep='last').set_index('chain')
e = first[['id','name','nm','sido','gu','k','x','y','lat','lon','open']].copy()
e[['close','status','type','addr']] = last[['close','status','type','addr']]
e['area'] = last.area.fillna(first.area)
e['n_permits'] = df.groupby('chain').size()
log.append((f'재신고/명의 변경 연결(연결된 인허가 {int(link.sum()):,}건)', len(e)))
df = e.reset_index(drop=True)

# ---------- 3) 임시 영업 제거 ----------
life = (df.close - df.open).dt.days
df = df[~(life <= TEMP_DAYS)].reset_index(drop=True);                    log.append((f'{TEMP_DAYS}일 이내 폐업(임시 영업) 제거', len(df)))
closed = df.status == '폐업'

# ---------- 4) 이전: 같은 구에서 같은 상호가 다른 자리에 전후 90일 내 개업 (흔한 상호는 제외) ----------
nc = df.nm.map(df.nm.value_counts())
cand = df[closed & (df.nm.str.len() >= 2) & (nc <= 10)][['id','gu','nm','k','close']]
j = cand.merge(df[['id','gu','nm','k','open']].rename(columns={'id':'nid','k':'nk','open':'nopen'}), on=['gu','nm'])
j = j[(j.id != j.nid) & (j.k != j.nk) & ((j.nopen - j.close).dt.days.between(-90, 90))]
df['moved'] = df.id.isin(j.id).astype(int)

# ---------- 5) 양도 추정: 같은 자리 + 같은 면적 + 다른 상호로 폐업 7일 전~60일 후 새 인허가 ----------
df['ak'] = (df.area * 2).round()
a = df[closed & df.area.notna()][['id','k','ak','nm','close']]
j = a.merge(df[df.area.notna()][['id','k','ak','nm','open']].rename(columns={'id':'nid','nm':'nnm','open':'nopen'}), on=['k','ak'])
j = j[(j.id != j.nid) & (j.nm != j.nnm) & ((j.nopen - j.close).dt.days.between(-7, 60))]
df['handover'] = df.id.isin(j.id).astype(int)

# ---------- 6) 정답 ----------
o = (df.open - pd.Timestamp('1900-01-01')).dt.days.values
c = (df.close - pd.Timestamp('1900-01-01')).dt.days.fillna(10**7).values.astype(np.int64)
df['closed_3y'] = ((c - o) <= HORIZON).astype(int)
END = SNAP - pd.Timedelta(days=HORIZON)
coh = (df.open >= COHORT_START) & (df.open <= END) & (df.moved == 0)      # 3년이 안 지난 점포와 이전 점포는 제외
log.append((f'학습 대상({COHORT_START}~{END.date()} 개업, 이전 제외)', int(coh.sum())))

# ---------- 7) 주변 피처: 개업 시점 이전 정보만 사용, 이웃 후보는 과거 점포 전체 ----------
xy = df[['x','y']].values                    # TM 좌표는 미터 단위라 그대로 거리 계산
tree = cKDTree(xy)
tcode = df.type.astype('category').cat.codes.values
idx = np.where(coh)[0]
names = ['n_all','n_same','n_closed_2y','n_same_closed_2y','n_opened_1y','spot_prev','spot_prev_closed_3y']
F = {k_: np.zeros(len(idx), dtype=np.int32) for k_ in names}
CH = 2000
for s in range(0, len(idx), CH):
    ii = idx[s:s+CH]
    nb = tree.query_ball_point(xy[ii], RADIUS, workers=-1)
    ln = np.fromiter((len(a_) for a_ in nb), np.int64, len(nb))
    jj = np.concatenate([np.asarray(a_, dtype=np.int64) for a_ in nb])
    r = np.repeat(np.arange(len(ii)), ln); i = ii[r]; t = o[i]
    before = o[jj] < t                         # 나보다 먼저 연 점포만
    active = before & (c[jj] >= t)             # 내 개업 시점에 영업 중
    same = tcode[jj] == tcode[i]
    cl2 = before & (c[jj] < t) & (c[jj] >= t - 730)
    spot = before & (c[jj] < t) & (np.abs(xy[jj] - xy[i]).sum(1) < 1.0)    # 같은 자리에서 먼저 폐업한 점포
    cnt = lambda mask: np.bincount(r[mask], minlength=len(ii))
    F['n_all'][s:s+CH] = cnt(active);            F['n_same'][s:s+CH] = cnt(active & same)
    F['n_closed_2y'][s:s+CH] = cnt(cl2);         F['n_same_closed_2y'][s:s+CH] = cnt(cl2 & same)
    F['n_opened_1y'][s:s+CH] = cnt(before & (o[jj] >= t - 365))
    F['spot_prev'][s:s+CH] = cnt(spot);          F['spot_prev_closed_3y'][s:s+CH] = cnt(spot & ((c[jj] - o[jj]) <= HORIZON))
    if s % 40000 == 0: print('피처 계산', s, '/', len(idx), flush=True)

out = df.loc[idx, ['id','name','type','sido','gu','addr','area','open','close','status','lat','lon','n_permits','handover','closed_3y']].reset_index(drop=True)
for k_, v in F.items(): out[k_] = v
nz = lambda s_: s_.replace(0, np.nan)
out['close_rate_2y'] = (out.n_closed_2y / nz(out.n_all + out.n_closed_2y)).round(4)            # 최근 2년 주변 폐업률
out['same_close_rate_2y'] = (out.n_same_closed_2y / nz(out.n_same + out.n_same_closed_2y)).round(4)
out['same_share'] = (out.n_same / nz(out.n_all)).round(4)
out['spot_fail_rate'] = (out.spot_prev_closed_3y / nz(out.spot_prev)).round(4)
out['lat'] = out.lat.round(6); out['lon'] = out.lon.round(6)
out.to_csv(OUT, index=False, encoding='utf-8-sig')
print(pd.DataFrame(log, columns=['단계', '남은 행']).to_string(index=False))
print('기준일', SNAP.date(), '| 이전 표시', int(df.moved.sum()), '| 학습 대상 중 양도 추정', int(out.handover.sum()))
print(out.groupby('sido').closed_3y.agg(['size', 'mean']).round(3))
print('양도 추정 비중(3년 내 폐업 중)', out[out.closed_3y == 1].handover.mean().round(3))
