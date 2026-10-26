"""门控权重的稀疏度自适应实测(论文 §4.4)。

回答评审意见:"建议作者给出不同交互稀疏度下门控权重的实际分布、变化曲线或
统计结果,证明模型确实'自动向知识图谱侧倾斜'"。

本脚本产出三组证据:
  (1) **分布与曲线**:按用户训练交互数分桶,统计 CF / KG 门控权重的均值、
      中位数、标准差与 KG/CF 比值;并给出交互数→门控权重的变化曲线。
  (2) **统计检验**:交互数与 KG 门控权重的 Spearman 秩相关(及 p 值)。
  (3) **反事实实验**:把冷启动调整项(cold_mask 调制)关掉,重测冷用户子集的
      HR@10 / NDCG@10,直接回答"向 KG 侧倾斜是否真的带来收益"。

实现要点:KG 分支是 item-centric 的(RippleNet 以商品为种子),因此
cf_emb / kg_emb 只依赖商品、不依赖用户 —— 预计算全部商品表征后,门控权重
可对 6.4 万用户一次性批量计算,无需逐对跑 RippleNet。

用法:
    python -m experiments.eval_gate_weights \
        --model checkpoints/m3_gate_l2off/best_model.pt \
        --data data/amazon_fashion_real.json \
        --output experiments/results/gate_weights.json \
        --figure figures/图4-5_门控权重稀疏度自适应.png
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from recommender.hybrid_rec import KGHybridRec
import recommender.evaluate as _eval_mod
from recommender.evaluate import build_loo_test, evaluate_model

# 关闭评测进度条(批量跑时刷屏)
_eval_mod.tqdm = lambda it, **kw: it

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 按训练交互数分桶(闭区间)。count>=10 时 cold_mask 归零,故单列 10+ 桶。
BUCKETS = [
    (3, 3), (4, 4), (5, 5), (6, 7), (8, 9),
    (10, 14), (15, 19), (20, 29), (30, 10 ** 9),
]


def parse_args():
    ap = argparse.ArgumentParser(description='门控权重的稀疏度自适应实测')
    ap.add_argument('--model', required=True, help='模型 checkpoint 路径')
    ap.add_argument('--data', default='data/amazon_fashion_real.json')
    ap.add_argument('--output', default='experiments/results/gate_weights.json')
    ap.add_argument('--figure', default='figures/图4-5_门控权重稀疏度自适应.png')
    ap.add_argument('--candidates', type=int, default=10,
                    help='每个用户额外取多少个候选商品用于"候选集平均"口径(0=只用正样本)')
    ap.add_argument('--n-negative', type=int, default=100, help='评测协议:sampled-N')
    ap.add_argument('--cold-threshold', type=int, default=5, help='冷用户判定阈值(训练交互数)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no-counterfactual', action='store_true', help='跳过反事实实验')
    ap.add_argument('--model-no-adjust', default=None,
                    help='对照模型:训练时即关闭稀疏度调节的 checkpoint(可选)')
    ap.add_argument('--max-users', type=int, default=0,
                    help='只取前 N 个测试用户(0=全部);用于快速冒烟测试')
    return ap.parse_args()


def build_model(args, data, graph, user_count_array, device, ckpt_path=None):
    model = KGHybridRec(
        n_users=data['n_users'],
        n_items=data['n_items'],
        n_entities=graph['n_total'],
        n_relations=graph['n_relations'],
        embedding_dim=64,
        n_hops=2,
        n_heads=4,
        n_layers=2,
        use_multimodal=False,
        use_improvement_a=True,
        use_improvement_b=True,
        use_improvement_c=True,
        graph=graph,
        n_interactions=user_count_array,
    ).to(device)
    state = torch.load(ckpt_path or args.model, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def precompute_item_embeddings(model, n_items, device, seed, batch=512):
    """预计算全部商品的 CF / KG 表征(item-centric,与用户无关)。

    KG 侧 RippleNet 的邻居采样含随机性,故固定随机种子以保证可复现。
    """
    torch.manual_seed(seed)
    cf_embs, kg_embs = [], []
    for start in range(0, n_items, batch):
        end = min(start + batch, n_items)
        items = torch.arange(start, end, device=device)
        cf_embs.append(model.cf_module.get_item_representation(items).cpu())
        dummy_users = torch.zeros(end - start, dtype=torch.long, device=device)
        kg_emb, _ = model.kg_module(dummy_users, items)
        kg_embs.append(kg_emb.cpu())
    return torch.cat(cf_embs), torch.cat(kg_embs)


@torch.no_grad()
def gate_weights_for(model, user_ids, item_ids, counts, cf_conf, device,
                     cf_cache=None, kg_cache=None, chunk=200_000):
    """计算 (user, item) 对上的实际生效门控权重 [N, 2]。

    cf_cache / kg_cache: 预计算的按 item 索引的表征(可选,避免重复算 KG)。
    """
    weights = []
    n = user_ids.numel()
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        u = user_ids[start:end].to(device)
        i = item_ids[start:end].to(device)
        c = counts[start:end].to(device) if counts is not None else None
        if cf_cache is not None:
            cf_emb = cf_cache[i.cpu()].to(device)
            kg_emb = kg_cache[i.cpu()].to(device)
        else:
            cf_emb = model.cf_module.get_item_representation(i)
            kg_emb, _ = model.kg_module(u, i)
        conf = cf_conf[u.cpu()].to(device)
        w = model.adaptive_gate.get_gate_weights(
            cf_emb, kg_emb, None, interaction_counts=c, cf_confidence=conf
        )
        weights.append(w.cpu())
    return torch.cat(weights)


def bucket_stats(counts, cf_w, kg_w, cf_w_noadj=None, kg_w_noadj=None):
    """按交互数分桶统计门控权重。

    cf_w_noadj / kg_w_noadj:同一模型在**关闭稀疏度调节**时的权重(仅基线门控 +
    置信度调制),用于量化"稀疏度调节项"本身带来的权重位移。
    """
    rows = []
    for lo, hi in BUCKETS:
        m = (counts >= lo) & (counts <= hi)
        n = int(m.sum())
        if n == 0:
            continue
        c = cf_w[m]
        k = kg_w[m]
        ratio = k / np.maximum(c, 1e-8)
        row = {
            'bucket': f'{lo}-{hi if hi < 10 ** 9 else "+"}',
            'lo': lo, 'hi': hi,
            'n_users': n,
            'cf_mean': float(c.mean()), 'cf_median': float(np.median(c)), 'cf_std': float(c.std()),
            'kg_mean': float(k.mean()), 'kg_median': float(np.median(k)), 'kg_std': float(k.std()),
            'kg_over_cf_mean': float(ratio.mean()),
            'kg_over_cf_median': float(np.median(ratio)),
            # 分布统计:该桶内"KG 权重高于 CF 权重"的用户占比
            'pct_users_kg_gt_cf': float((k > c).mean()),
        }
        if kg_w_noadj is not None:
            row['kg_mean_without_sparsity_adjustment'] = float(kg_w_noadj[m].mean())
            row['kg_shift_from_adjustment'] = float((k - kg_w_noadj[m]).mean())
            row['cf_shift_from_adjustment'] = float((c - cf_w_noadj[m]).mean())
        rows.append(row)
    return rows


def make_figure(rows, per_user_cf, per_user_kg, counts, path, cold_threshold):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # 中文字体
    for cand in ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'DengXian']:
        if any(f.name == cand for f in font_manager.fontManager.ttflist):
            plt.rcParams['font.sans-serif'] = [cand]
            break
    plt.rcParams['axes.unicode_minus'] = False

    labels = [r['bucket'] for r in rows]
    x = np.arange(len(rows))
    cf_mean = np.array([r['cf_mean'] for r in rows])
    kg_mean = np.array([r['kg_mean'] for r in rows])
    cf_std = np.array([r['cf_std'] for r in rows])
    kg_std = np.array([r['kg_std'] for r in rows])
    ns = [r['n_users'] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2))

    # ---- 左:门控权重随交互数的变化曲线 ----
    ax = axes[0]
    ax.errorbar(x, cf_mean, yerr=cf_std, marker='o', capsize=3, color='#4C72B0',
                label='CF 通路权重 $w_{cf}$', linewidth=2)
    ax.errorbar(x, kg_mean, yerr=kg_std, marker='s', capsize=3, color='#DD8452',
                label='KG 通路权重 $w_{kg}$', linewidth=2)
    ax.axvline(-0.5, color='#888888', linestyle='--', linewidth=1)
    for i, n in enumerate(ns):
        ax.annotate(f'n={n}', (x[i], 0.02), ha='center', fontsize=7, color='#666666')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('用户训练交互数分桶')
    ax.set_ylabel('门控权重(均值 ± 标准差)')
    ax.set_title('门控权重随用户交互稀疏度的变化')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)

    # ---- 右:各桶 KG 权重分布(箱线) ----
    ax = axes[1]
    data, bpos = [], []
    for lo, hi in BUCKETS:
        m = (counts >= lo) & (counts <= hi)
        if m.sum() == 0:
            continue
        data.append(per_user_kg[m])
        bpos.append(f'{lo}-{hi if hi < 10 ** 9 else "+"}')
    bp = ax.boxplot(data, positions=np.arange(len(data)), widths=0.55, showfliers=False,
                    patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#DD8452')
        patch.set_alpha(0.55)
    for median in bp['medians']:
        median.set_color('#7f3f14')
    ax.axvline(-0.5, color='#888888', linestyle='--', linewidth=1)
    ax.set_xticks(np.arange(len(bpos)))
    ax.set_xticklabels(bpos)
    ax.set_xlabel('用户训练交互数分桶')
    ax.set_ylabel('KG 通路权重 $w_{kg}$')
    ax.set_title('各稀疏度分桶的 KG 门控权重分布')
    ax.grid(alpha=0.3, axis='y')

    # ---- 右三:KG 权重占优的用户占比 ----
    ax = axes[2]
    pct = np.array([r['pct_users_kg_gt_cf'] for r in rows]) * 100
    ax.bar(x, pct, color='#DD8452', width=0.6)
    for i, (p, n) in enumerate(zip(pct, ns)):
        ax.annotate(f'{p:.1f}%', (x[i], p + 1.2), ha='center', fontsize=8, color='#7f3f14')
    ax.axhline(50, color='#888888', linestyle='--', linewidth=1, label='50% 基线')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    ax.set_xlabel('用户训练交互数分桶')
    ax.set_ylabel('KG 权重 > CF 权重的用户占比 (%)')
    ax.set_title('门控权重分布:KG 侧占优的用户比例')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.3, axis='y')

    fig.suptitle(
        f'改进 A 自适应门控的稀疏度自适应实测(冷用户阈值:训练交互数 < {cold_threshold})',
        fontsize=13,
    )
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    logger.info('figure saved: %s', path)


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('device: %s', device)

    from recommender.train import load_data, prepare_training_data
    from recommender.kgat.hetero_graph import build_graph_for_model

    products, users, interactions = load_data(args.data)
    data = prepare_training_data(users, interactions, n_neg=4, seed=args.seed)
    graph = build_graph_for_model(products, data['train_data'], data['user2idx'], data['item2idx'])
    logger.info('graph: %d nodes, %d edges, %d relations',
                graph['n_total'], graph['edge_index'].size(1), graph['n_relations'])

    user_count_array = torch.zeros(data['n_users'], dtype=torch.float32)
    for uid, cnt in data['user_count'].items():
        user_count_array[uid] = cnt

    model = build_model(args, data, graph, user_count_array, device)
    logger.info('model loaded: %s', args.model)

    # ---------- (1) 门控权重分布 ----------
    test_data = build_loo_test(interactions, data['user2idx'], data['item2idx'],
                               n_negative=args.n_negative, seed=args.seed)
    if args.max_users:
        test_data = test_data[:args.max_users]
    logger.info('test users: %d', len(test_data))

    uid_list = torch.LongTensor([t[0] for t in test_data])
    pos_list = torch.LongTensor([t[1][0] for t in test_data])
    counts = torch.LongTensor([data['user_count'][int(t[0])] for t in test_data])

    cf_cache, kg_cache = precompute_item_embeddings(model, data['n_items'], device, args.seed)
    cf_conf = model.cf_module.get_confidence(uid_list.to(device)).cpu()

    w_pos = gate_weights_for(model, uid_list, pos_list, counts, cf_conf, device,
                              cf_cache=cf_cache, kg_cache=kg_cache)
    per_user_cf = w_pos[:, 0].numpy()
    per_user_kg = w_pos[:, 1].numpy()
    counts_np = counts.numpy()

    # 同一模型在**关闭稀疏度调节**时的门控权重(仅基线门控 + 置信度调制),
    # 用于量化"稀疏度调节项"本身带来的权重位移。
    w_noadj = gate_weights_for(model, uid_list, pos_list, None, cf_conf, device,
                                cf_cache=cf_cache, kg_cache=kg_cache)
    per_user_cf_noadj = w_noadj[:, 0].numpy()
    per_user_kg_noadj = w_noadj[:, 1].numpy()

    # 候选集平均口径(对每个用户的 10 个随机候选再算一次,取平均)
    per_user_cf_cand, per_user_kg_cand = per_user_cf, per_user_kg
    if args.candidates > 0:
        rng = np.random.RandomState(args.seed)
        u_rep, i_rep, c_rep = [], [], []
        for uid, (u, pos, negs) in zip(uid_list.tolist(), test_data):
            k = min(args.candidates, len(negs))
            pick = rng.choice(len(negs), size=k, replace=False)
            u_rep.extend([u] * k)
            i_rep.extend(negs[j] for j in pick)
            c_rep.extend([data['user_count'][u]] * k)
        w_cand = gate_weights_for(
            model, torch.LongTensor(u_rep), torch.LongTensor(i_rep),
            torch.LongTensor(c_rep), cf_conf, device,
            cf_cache=cf_cache, kg_cache=kg_cache)
        w_cand = w_cand.numpy()
        # 按用户聚合
        agg_cf = np.zeros(len(test_data))
        agg_kg = np.zeros(len(test_data))
        pos = 0
        for ui, (uid, (u, _, negs)) in enumerate(zip(uid_list.tolist(), test_data)):
            k = min(args.candidates, len(negs))
            agg_cf[ui] = w_cand[pos:pos + k, 0].mean()
            agg_kg[ui] = w_cand[pos:pos + k, 1].mean()
            pos += k
        per_user_cf_cand, per_user_kg_cand = agg_cf, agg_kg

    rows = bucket_stats(counts_np, per_user_cf, per_user_kg,
                        per_user_cf_noadj, per_user_kg_noadj)
    rows_cand = bucket_stats(counts_np, per_user_cf_cand, per_user_kg_cand)

    # ---------- (2) 统计检验 ----------
    from scipy.stats import spearmanr
    rho_kg, p_kg = spearmanr(counts_np, per_user_kg)
    rho_cf, p_cf = spearmanr(counts_np, per_user_cf)
    logger.info('Spearman(count, w_kg): rho=%.4f p=%.3g', rho_kg, p_kg)
    logger.info('Spearman(count, w_cf): rho=%.4f p=%.3g', rho_cf, p_cf)

    # ---------- (3) 反事实:关掉稀疏度调节 ----------
    cf_result = None
    if not args.no_counterfactual:
        cold_mask_users = [t for t in test_data
                           if data['user_count'][t[0]] < args.cold_threshold]
        logger.info('cold users (<%d): %d', args.cold_threshold, len(cold_mask_users))

        # KG 表征是 item-centric 的,评测时用预计算缓存替换,消除采样噪声
        class CachedKG(torch.nn.Module):
            def __init__(self, cache):
                super().__init__()
                self.cache = cache

            def forward(self, user_ids, items, ripple_sets=None):
                return self.cache[items.cpu()].to(items.device), []

        model.kg_module = CachedKG(kg_cache)

        # (a) 门控正常生效
        torch.manual_seed(args.seed)
        res_on = evaluate_model(model, cold_mask_users, [5, 10, 20], device=device,
                                user_count=data['user_count'])
        # (b) 推理时关掉稀疏度调节(结构上不再向 KG 侧倾斜)
        model.adaptive_gate.adaptive_gate.use_interaction_count = False
        torch.manual_seed(args.seed)
        res_off_inference = evaluate_model(model, cold_mask_users, [5, 10, 20], device=device,
                                           user_count=data['user_count'])
        model.adaptive_gate.adaptive_gate.use_interaction_count = True

        cf_result = {
            'n_cold_users': len(cold_mask_users),
            'threshold': args.cold_threshold,
            'with_cold_adjustment': res_on,
            'without_cold_adjustment_inference': res_off_inference,
            'delta_hr@10': res_on['hr@10'] - res_off_inference['hr@10'],
            'delta_ndcg@10': res_on['ndcg@10'] - res_off_inference['ndcg@10'],
        }
        logger.info('cold HR@10: on=%.4f off(inference)=%.4f (delta=%+.4f)',
                    res_on['hr@10'], res_off_inference['hr@10'], cf_result['delta_hr@10'])

        # (c) 重训对照:训练时即关闭稀疏度调节的模型
        if args.model_no_adjust:
            ctrl = build_model(args, data, graph, user_count_array, device,
                               ckpt_path=args.model_no_adjust)
            ctrl.kg_module = CachedKG(kg_cache)
            torch.manual_seed(args.seed)
            res_off_retrained = evaluate_model(ctrl, cold_mask_users, [5, 10, 20], device=device,
                                               user_count=data['user_count'])
            # 对照模型自身的门控权重(应无稀疏度倾斜)
            ctrl_w = gate_weights_for(ctrl, uid_list, pos_list, counts, cf_conf, device,
                                      cf_cache=cf_cache, kg_cache=kg_cache)
            ctrl_rows = bucket_stats(counts_np, ctrl_w[:, 0].numpy(), ctrl_w[:, 1].numpy())
            cf_result['without_cold_adjustment_retrained'] = res_off_retrained
            cf_result['retrained_model'] = args.model_no_adjust
            cf_result['retrained_buckets'] = ctrl_rows
            cf_result['delta_hr@10_vs_retrained'] = res_on['hr@10'] - res_off_retrained['hr@10']
            cf_result['delta_ndcg@10_vs_retrained'] = res_on['ndcg@10'] - res_off_retrained['ndcg@10']
            logger.info('cold HR@10: on=%.4f off(retrained)=%.4f (delta=%+.4f)',
                        res_on['hr@10'], res_off_retrained['hr@10'],
                        cf_result['delta_hr@10_vs_retrained'])

    out = {
        'model': args.model,
        'data': args.data,
        'seed': args.seed,
        'protocol': f'sampled-{args.n_negative} leave-one-out',
        'n_users': len(test_data),
        'buckets': rows,
        'buckets_candidate_avg': rows_cand,
        'spearman_count_vs_kg_weight': {
            'rho': float(rho_kg), 'p_value': float(p_kg),
        },
        'spearman_count_vs_cf_weight': {
            'rho': float(rho_cf), 'p_value': float(p_cf),
        },
        'counterfactual_cold_users': cf_result,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info('results saved: %s', args.output)

    make_figure(rows, per_user_cf, per_user_kg, counts_np, args.figure, args.cold_threshold)

    # 控制台 markdown 表
    print('\n| 交互数分桶 | 用户数 | w_cf 均值 | w_kg 均值 | KG/CF | 关调节后 w_kg | 调节位移 |')
    print('|---|---|---|---|---|---|---|')
    for r in rows:
        print(f"| {r['bucket']} | {r['n_users']} | {r['cf_mean']:.4f} | {r['kg_mean']:.4f} "
              f"| {r['kg_over_cf_mean']:.3f} | {r['kg_mean_without_sparsity_adjustment']:.4f} "
              f"| {r['kg_shift_from_adjustment']:+.4f} |")
    print(f"\nSpearman(count, w_kg) = {rho_kg:.4f} (p={p_kg:.3g})")
    if cf_result:
        print(f"cold users HR@10: with={cf_result['with_cold_adjustment']['hr@10']:.4f} "
              f"without={cf_result['without_cold_adjustment']['hr@10']:.4f} "
              f"delta={cf_result['delta_hr@10']:+.4f}")


if __name__ == '__main__':
    main()
