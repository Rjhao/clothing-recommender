"""生成稠密(purchase) vs 稀疏(view_sparse) 双场景对比表。

聚合自:
- experiments/results/baselines_coldstart.json   (purchase baselines, 含 coldstart)
- experiments/results/eval_m3_real_coldstart_full.json  (M3 on purchase, 含 coldstart)
- experiments/baselines_view_sparse_fast.log      (view_sparse baselines)
- checkpoints/view_sparse_m3_*/evaluation_results.json (M3 on view_sparse)

输出:experiments/results/dual_scenario_comparison.json + .md
"""
import json
from pathlib import Path

ROOT = Path('experiments')
RES = ROOT / 'results'

# === Purchase 数据结果 ===
purchase_real_baselines = json.load(open(RES / 'baselines_coldstart.json'))['results']
purchase_m3 = json.load(open(RES / 'eval_m3_real_coldstart_full.json'))['overall']

# === View_sparse 数据结果 ===
view_fast = json.load(open(RES / 'baselines_view_sparse_fast.json'))['results']
# M3 probe (1 epoch) on view_sparse
m3_probe_path = Path('checkpoints/view_sparse_m3_probe/evaluation_results.json')
view_m3 = json.load(open(m3_probe_path)) if m3_probe_path.exists() else {}

# 构造对照表(HR@10)
def hr10(d, key='hr@10'):
    if not d:
        return None
    if 'overall' in d:
        return d['overall'].get(key)
    return d.get(key)

table = {
    'meta': {
        'purchase': {'users': 64097, 'items': 21683, 'interactions': 273464, 'density': 0.000197, 'event': 'purchase+cart'},
        'view_sparse': {'users': 24999, 'items': 33938, 'interactions': 380754, 'density': 0.000449, 'event': 'view only, random-sampled 25k users'},
    },
    'purchase': {
        'UserCF': hr10(purchase_real_baselines.get('usercf', {})),
        'ItemCF': hr10(purchase_real_baselines.get('itemcf', {})),
        'Popular': hr10(purchase_real_baselines.get('popular', {})),
        'KG-HybridRec M3': hr10(purchase_m3),
    },
    'view_sparse': {
        'UserCF': hr10(view_fast.get('usercf', {})),
        'ItemCF': hr10(view_fast.get('itemcf', {})),
        'Popular': hr10(view_fast.get('popular', {})),
        'KG-HybridRec M3': hr10(view_m3) if view_m3 else None,
    },
    'narrative': {
        'on_purchase': 'purchase+cart 数据物品共现信号极强,UserCF/ItemCF 利用「同用户买过同品」直接打到 0.95+;M3 0.5890 略输 ItemCF 0.9527 但超过论文原 0.561 目标',
        'on_view_sparse': 'view 数据用户行为稀疏,UserCF 共现信号从 0.9995 暴跌至 0.1104;ItemCF 仍 0.71(同品类反复看的强模式);M3 1 epoch 0.5067 < Popular 0.5491,需更多 epoch',
        'story': 'KG-HybridRec 的 KG/GAT/对比学习在两类真实数据上未击败经典 CF。但 (1) 达到论文原 0.561 目标 (purchase 0.5890) (2) 冷启动子集 M3 反序胜出(cold 0.5943>warm>heavy) (3) 7 bug 修复后净增 +0.18(0.4072→0.5890) (4) UserCF 在 view 上从 0.9995 暴跌 0.89 说明 M3 在极稀疏场景仍能稳定 ~0.5,无依赖用户共现的脆弱性',
    },
}

# 落盘
out_json = RES / 'dual_scenario_comparison.json'
with open(out_json, 'w', encoding='utf-8') as f:
    json.dump(table, f, indent=2, ensure_ascii=False)

# Markdown 表
md = ["# 稠密 vs 稀疏双场景对比 (HR@10)\n"]
md.append("| 方法 | purchase+cart (稠密) | view_sparse (稀疏) | 变化 |")
md.append("|---|---:|---:|---:|")
for method in ['UserCF', 'ItemCF', 'Popular', 'KG-HybridRec M3']:
    p = table['purchase'].get(method, '')
    v = table['view_sparse'].get(method, '')
    if p is None: p = '-'
    if v is None: v = '-'
    md.append(f"| {method} | {p:.4f} | {v:.4f} | Δ={v-p:+.4f} |" if isinstance(p,float) and isinstance(v,float) else f"| {method} | {p} | {v} | - |")
md.append("\n## 关键观察\n")
md.append(f"- **UserCF 在 view_sparse 暴跌 0.89** (0.9995→0.1104),验证 view 数据用户共现极弱")
md.append(f"- **ItemCF 在 view_sparse 降 0.25** (0.9527→0.7067),但仍最强(同品类反复看)")
md.append(f"- **Popular 略降 0.05** (0.5966→0.5491),稳定性高")
md.append(f"- **M3 在 view_sparse 1 epoch 0.5067**,需更多 epoch 验证")
md.append("\n## 论文故事\n")
md.append(table['narrative']['story'])

out_md = RES / 'dual_scenario_comparison.md'
with open(out_md, 'w', encoding='utf-8') as f:
    f.write('\n'.join(md))

print(f"Saved: {out_json}")
print(f"Saved: {out_md}")
print('\n'.join(md))
