"""汇总 view_sparse 数据上消融实验结果成表.

读取 checkpoints/vs_abl_{m0,m1,m2,m3}/evaluation_results.json,
计算 vs M0 与单项贡献, 打印 Markdown 消融表, 并写 experiments/results/ablation_viewsparse_summary.json.

与 aggregate_ablation_real.py 同结构, 仅 tag 前缀 vs_abl_ 与 main-table reference 不同.

用法:
    PYTHONUTF8=1 python experiments/aggregate_ablation_viewsparse.py
"""
import json
from pathlib import Path

# (显示名, tag, A, B, C, 单项贡献说明) - 与 purchase 完全一致
CONFIGS = [
    ('M0',            'vs_abl_m0', 0, 0, 0, '基线'),
    ('M0+A',          'vs_abl_m1', 1, 0, 0, '+A 自适应门控'),
    ('M0+A+B',        'vs_abl_m2', 1, 1, 0, '+B 图对比学习'),
    ('M0+A+B+C (M3)', 'vs_abl_m3', 1, 1, 1, '+C 注意力可解释'),
]

BASE = Path('checkpoints')
OUT_JSON = Path('experiments/results/ablation_viewsparse_summary.json')


def load_metrics(tag):
    p = BASE / tag / 'evaluation_results.json'
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding='utf-8'))


def main():
    rows = []
    m0_hr = None
    for label, tag, a, b, c, note in CONFIGS:
        m = load_metrics(tag)
        if m is None:
            rows.append(dict(label=label, tag=tag, A=a, B=b, C=c,
                             hr10=None, ndcg10=None, vs_m0=None, contrib_delta=None,
                             note=note, status='MISSING'))
            continue
        hr = m.get('hr@10'); ndcg = m.get('ndcg@10')
        if label == 'M0':
            m0_hr = hr
        rows.append(dict(label=label, tag=tag, A=a, B=b, C=c,
                         hr10=hr, ndcg10=ndcg, vs_m0=None, contrib_delta=None,
                         note=note, status='OK'))

    # vs M0 + 单项贡献 (当前 - 上一个配置)
    prev_hr = None
    for r in rows:
        if r['hr10'] is None:
            prev_hr = None  # 缺失则断链
            continue
        r['vs_m0'] = round(r['hr10'] - m0_hr, 4) if (r['label'] != 'M0' and m0_hr is not None) else 0.0
        r['contrib_delta'] = round(r['hr10'] - prev_hr, 4) if prev_hr is not None else None
        prev_hr = r['hr10']

    # Markdown 表
    lines = []
    lines.append('# view_sparse 场景消融结果 (20 epoch, sampled-100, seed=42)\n')
    lines.append('| 配置 | A | B | C | HR@10 | NDCG@10 | vs M0 | 单项贡献ΔHR |')
    lines.append('|---|---|---|---|---|---|---|---|')
    for r in rows:
        hr = f"{r['hr10']:.4f}" if r['hr10'] is not None else '?'
        nd = f"{r['ndcg10']:.4f}" if r['ndcg10'] is not None else '?'
        vs = f"{r['vs_m0']:+.4f}" if r.get('vs_m0') is not None else '-'
        cd = r.get('contrib_delta')
        cd_s = f"{cd:+.4f}" if cd is not None else '-'
        a_s = '✓' if r['A'] else '✗'
        b_s = '✓' if r['B'] else '✗'
        c_s = '✓' if r['C'] else '✗'
        st = '' if r['status'] == 'OK' else f" ({r['status']})"
        lines.append(f"| {r['label']}{st} | {a_s} | {b_s} | {c_s} | {hr} | {nd} | {vs} | {cd_s} |")
    md = '\n'.join(lines)
    print(md)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        'protocol': 'leave-one-out + sampled-100, 20 epoch, seed=42',
        'data': 'data/amazon_fashion_view_sparse.json',
        'configs': rows,
        'm3_hr10_ref_main_table': 0.5378,
        'note': 'view_sparse 场景 4 配置消融, 与 purchase 表同协议同超参横向对照. '
                'C (attention_explain) 只构造模块, 不参与 forward/predict_score, '
                '故 M0+A+B 与 M0+A+B+C(M3) 的 HR 相同 (设计如此, C 的价值在可解释性, 不在精度).',
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n汇总 JSON: {OUT_JSON}')


if __name__ == '__main__':
    main()
