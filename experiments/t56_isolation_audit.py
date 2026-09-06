"""t56_isolation_audit.py -- 隔离审计: 被排除的列换成任意值, 结果是否一字不差.

把 delta, gamma, vega, theta 和中间价全部替换成随机数, 重新走一遍完整流程:
第一步增广、第二步分块、第三步预算、拟合与预测. 如果准入集合真的与这些列无关,
那么增广矩阵、块划分、候选预算和逐行预测都应当逐位相同.

单线程运行: 森林的多线程归约不是逐位可重复的(同一份数据两次之间相差 7e-16),
这是浮点求和次序的问题, 与准入无关.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from t4_clean_sweep import load, BASE
from dynET import augment_basic, block_labels, block_count

SEED, NT = 42, 300
EXCLUDED = ['delta', 'gamma', 'vega', 'theta', 'mid']


def run(d):
    tr = d.iloc[:20000]; te = d.iloc[20000:26000]
    A, Ate = augment_basic(tr[BASE].values), augment_basic(te[BASE].values)
    lab = block_labels(A); M = int(lab.max()) + 1
    p = ExtraTreesRegressor(n_estimators=NT, max_features=M / A.shape[1],
                            bootstrap=False, min_samples_leaf=1, n_jobs=1,
                            random_state=SEED).fit(A, tr['IV'].values).predict(Ate)
    return A, lab, M, p


def main():
    d = load().reset_index(drop=True)
    rng = np.random.default_rng(0)
    for c in EXCLUDED:
        d[c] = rng.normal(size=len(d))          # 先给一组任意值
    A1, l1, M1, p1 = run(d)
    for c in EXCLUDED:
        d[c] = rng.uniform(-1e6, 1e6, size=len(d))   # 再换一组完全不同的
    A2, l2, M2, p2 = run(d)

    same = dict(
        enriched_matrix=bool(np.array_equal(A1, A2)),
        block_labels=bool(np.array_equal(l1, l2)),
        block_count=(M1 == M2),
        predictions=bool(np.array_equal(p1, p2)),
    )
    print('被排除的列:', ', '.join(EXCLUDED))
    print(f'训练 {20000} 行, 测试 {len(p1)} 行, M = {M1}/{A1.shape[1]}')
    for k, v in same.items():
        print(f'  {k:18s} {"逐位相同" if v else "不同"}')
    print('\n全部相同:', all(same.values()))
    pd.Series(same).to_csv(Path(__file__).resolve().parents[1] /
                           'results' / 'isolation_audit.csv', header=False)


if __name__ == '__main__':
    main()
