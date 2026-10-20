"""聚合论文 表4-5「整体性能对比」。

把分散在四处的实验产物合并成一张可直接贴进论文的 markdown 表：

    KG-HybridRec(本文)  checkpoints/abl_m3/evaluation_results.json
    KGAT(骨架基线)      checkpoints/abl_m0/evaluation_results.json
    LightGCN / BPR-MF   experiments/results/purchase_dense.json
    RippleNet / KGCL    experiments/results/purchase_dense.json
    MKR / KGIN / CKAN   experiments/results/kg_baselines.json

每列最优值加粗（回应评语「将最好的性能字表加粗或者下划线显示」）。

用法：
    PYTHONUTF8=1 /c/Python311/python.exe experiments/aggregate_table45.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

METRICS = ['hr@10', 'ndcg@10', 'precision@10', 'hr@20', 'ndcg@20', 'precision@20']
HEADERS = ['HR@10', 'NDCG@10', 'P@10', 'HR@20', 'NDCG@20', 'P@20']

# 表4-5 行顺序：主模型 → 同族骨架 → 最强图基线 → 传统 → 纯 KG 方法 → 新增 KG 方法
ORDER = [
    ('KG-HybridRec(本文)', 'abl_m3'),
    ('KGAT', 'abl_m0'),
    ('LightGCN', 'baselines:lightgcn'),
    ('BPR-MF', 'baselines:bprmf'),
    ('RippleNet', 'baselines:ripplen'),
    ('KGCL', 'baselines:kgcl'),
    ('MKR', 'kg:mkr'),
    ('KGIN', 'kg:kgin'),
    ('CKAN', 'kg:ckan'),
]

# 最终版 docx 表4-5 已写入的数值，用于核对产物是否漂移
DOCX = {
    'KG-HybridRec(本文)': [0.5906, 0.4034, 0.0591, 0.7044, 0.4314, 0.0352],
    'KGAT': [0.5841, 0.3990, 0.0584, 0.6929, 0.4266, 0.0346],
    'LightGCN': [0.5900, 0.4609, 0.0590, 0.6642, 0.4797, 0.0332],
    'BPR-MF': [0.5201, 0.3909, 0.0520, 0.6053, 0.4124, 0.0303],
    'RippleNet': [0.5153, 0.3396, 0.0515, 0.6309, 0.3688, 0.0315],
    'KGCL': [0.5151, 0.3896, 0.0515, 0.5974, 0.4103, 0.0299],
}


def _load(path):
    p = ROOT / path
    if not p.exists():
        return None
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def collect():
    base = (_load('experiments/results/purchase_dense.json') or {}).get('results', {})
    kg = (_load('experiments/results/kg_baselines.json') or {}).get('results', {})

    rows = []
    for name, src in ORDER:
        if src.startswith('abl_'):
            d = _load(f'checkpoints/{src}/evaluation_results.json')
            vals = [round(d[m], 4) for m in METRICS] if d else None
        elif src.startswith('baselines:'):
            key = src.split(':', 1)[1]
            row = base.get(key)
            vals = [round(row[m], 4) for m in METRICS] if row else None
        else:
            key = src.split(':', 1)[1]
            row = kg.get(key)
            vals = [round(row[m], 4) for m in METRICS] if row else None
        rows.append((name, vals))
    return rows


def render(rows):
    present = [(n, v) for n, v in rows if v]
    best = [max(v[i] for _, v in present) for i in range(len(METRICS))]

    out = ['表4-5 整体性能对比', 'Tab.4-5 Overall Performance Comparison', '',
           '| 算法 | ' + ' | '.join(HEADERS) + ' |',
           '|' + '---|' * (len(HEADERS) + 1)]
    for name, vals in rows:
        if not vals:
            out.append(f'| {name} | ' + ' | '.join(['—'] * len(HEADERS)) + ' |')
            continue
        cells = [f'**{v:.4f}**' if abs(v - best[i]) < 1e-9 else f'{v:.4f}'
                 for i, v in enumerate(vals)]
        out.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(out)


def drift_check(rows):
    """核对产物数值与最终版 docx 表4-5 是否一致。"""
    msgs = []
    for name, vals in rows:
        ref = DOCX.get(name)
        if not ref or not vals:
            continue
        diffs = [f'{HEADERS[i]} {vals[i]:.4f}≠{ref[i]:.4f}'
                 for i in range(len(METRICS)) if abs(vals[i] - ref[i]) > 5e-5]
        if diffs:
            msgs.append(f'  {name}: ' + '; '.join(diffs))
    return msgs


def main():
    rows = collect()
    table = render(rows)
    print(table)
    print()

    missing = [n for n, v in rows if not v]
    if missing:
        print(f'⚠ 缺少实验产物，以下行留空: {", ".join(missing)}')
        if any(n in missing for n in ('MKR', 'KGIN', 'CKAN')):
            print('  → 先跑: python -m recommender.baselines_kg --models mkr kgin ckan --epochs 30')
        print()

    drift = drift_check(rows)
    if drift:
        print('⚠ 以下行与最终版 docx 表4-5 数值不一致（产物为准，需确认论文是否同步更新）:')
        print('\n'.join(drift))
    else:
        print('✓ 已有各行与最终版 docx 表4-5 完全一致')

    out = ROOT / 'experiments/results/table45.md'
    out.write_text(table + '\n', encoding='utf-8')
    print(f'\nsaved -> {out}')
    return 1 if missing else 0


if __name__ == '__main__':
    sys.exit(main())