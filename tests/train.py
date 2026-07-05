# -*- coding: utf-8 -*-
"""
训练 + 评估 + 消融。支持两种数据来源：
  --source synthetic   合成数据（用于冒烟自测，无需任何外部依赖）
  --source neo4j       从你的 Neo4j 真实数据训练（需 data_neo4j.py 跑通）

消融开关：--no_cagf  --no_csl  --no_mm  --no_gat
对应论文表 4-19 / 4-20 / 4-21 的各变体。
评估指标：HR@K, NDCG@K, Recall@K, Coverage；并单独输出新用户/新商品冷启动子集。
"""
import argparse, numpy as np, torch
from model import KGHybridRec, bpr_loss


# ---------------- 评估 ----------------
def evaluate(model, data, K=10, cold_users=None, cold_items=None):
    model.eval()
    with torch.no_grad():
        zu, hi, _ = model.encode(data['edge_index'], data['item_ent'],
                                 data['img'], data['txt'], data['inter_cnt'])
        eu = data['eval_users']; chunks = []
        for s in range(0, len(eu), 256):                 # 分块打分，避免大数据 OOM
            ub = zu[eu[s:s+256]]
            sc = model.predict(torch.cat([
                ub.unsqueeze(1).expand(-1, model.NI, -1),
                hi.unsqueeze(0).expand(ub.size(0), -1, -1)], -1)).squeeze(-1)
            chunks.append(sc.cpu())
        scores = torch.cat(chunks, 0).numpy()
    hr=ndcg=rec=0.0; covered=set(); n=0
    sub_hr={}; sub_n={}
    for r, u in enumerate(data['eval_users'].tolist()):
        gt = data['test_pos'].get(u, set())
        if not gt: continue
        s = scores[r].copy()
        for it in data['train_pos'].get(u, set()):
            s[it] = -1e9
        topk = np.argsort(-s)[:K]
        covered.update(topk.tolist())
        hit = len(set(topk.tolist()) & gt)
        hr += 1.0 if hit>0 else 0.0
        rec += hit/len(gt)
        dcg = sum(1.0/np.log2(i+2) for i,it in enumerate(topk) if it in gt)
        idcg = sum(1.0/np.log2(i+2) for i in range(min(len(gt),K)))
        ndcg += dcg/idcg if idcg>0 else 0.0
        n += 1
        for name,cset in [('cold_user',cold_users),('cold_item',cold_items)]:
            flag = (u in cset) if name=='cold_user' and cset else False
            if name=='cold_item' and cset: flag = len(gt & cset)>0
            if flag:
                sub_hr[name]=sub_hr.get(name,0)+(1.0 if hit>0 else 0.0); sub_n[name]=sub_n.get(name,0)+1
    out = {'HR':hr/n, 'NDCG':ndcg/n, 'Recall':rec/n, 'Coverage':len(covered)/model.NI}
    for name in sub_hr: out[f'{name}_HR'] = sub_hr[name]/max(sub_n[name],1)
    return out


# ---------------- 合成数据（冒烟自测）----------------
def make_synthetic(NU=200, NI=300, NE=80, NR=6, seed=0):
    rng = np.random.default_rng(seed)
    train_pos, test_pos = {}, {}
    rows=[]
    for u in range(NU):
        k = rng.integers(1, 8)
        its = rng.choice(NI, size=k, replace=False)
        tr = set(its[:max(1,k//2)].tolist()); te=set(its[max(1,k//2):].tolist())
        train_pos[u]=tr; test_pos[u]=te
        for i in tr: rows.append((u,i))
    # 异构图边：用户-商品交互(双向) + 商品-实体(双向)
    NU_, NI_ = NU, NI
    ent_base = NU+NI
    E=[]
    for u,i in rows:
        E.append((u, NU+i)); E.append((NU+i, u))
    item_ent = list(range(NI))   # 商品自身作为实体0..NI-1（在 TransR.ent 中）
    # 额外实体连接（类目/关键词）
    triples=[]
    for i in range(NI):
        for _ in range(2):
            ent = int(rng.integers(0, NE)); rel=int(rng.integers(0,NR))
            node = ent_base + ent
            E.append((NU+i, node)); E.append((node, NU+i))
            triples.append((i, rel, NI+ent))   # 头=商品实体i, 尾=外部实体
    edge_index = torch.tensor(E, dtype=torch.long).t().contiguous()
    inter_cnt = torch.zeros(NI, dtype=torch.long)
    for _,i in rows: inter_cnt[i]+=1
    data = {
        'edge_index': edge_index,
        'item_ent': torch.arange(NI),
        'img': torch.randn(NI,128), 'txt': torch.randn(NI,768),
        'inter_cnt': inter_cnt,
        'train_pos': train_pos, 'test_pos': test_pos,
        'eval_users': torch.arange(NU),
        'triples': torch.tensor(triples, dtype=torch.long),
        'rows': torch.tensor(rows, dtype=torch.long),
        'n_users':NU,'n_items':NI,'n_entities':NI+NE,'n_relations':NR,
    }
    cold_users = {u for u in train_pos if len(train_pos[u])<2}
    cold_items = {i for i in range(NI) if inter_cnt[i].item()<2}
    return data, cold_users, cold_items


def train(args):
    if args.source=='synthetic':
        data, cu, ci = make_synthetic()
    else:
        from data_neo4j import load_from_neo4j
        data, cu, ci = load_from_neo4j(args)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    for k in ['edge_index','item_ent','img','txt','inter_cnt','triples','rows','eval_users']:
        data[k]=data[k].to(dev)
    m = KGHybridRec(data['n_users'],data['n_items'],data['n_entities'],
                    data['n_relations'], dim=args.dim).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=args.lr)
    rows = data['rows']; NI=data['n_items']
    for ep in range(args.epochs):
        m.train(); opt.zero_grad()
        zu, hi, beta = m.encode(data['edge_index'], data['item_ent'], data['img'],
                                data['txt'], data['inter_cnt'],
                                use_mm=not args.no_mm, use_gat=not args.no_gat,
                                use_cagf=not args.no_cagf)
        u = rows[:,0]; pos = rows[:,1]
        neg = torch.randint(0, NI, (rows.size(0),), device=dev)
        s_pos = m.score_pairs(zu, hi, u, pos)
        s_neg = m.score_pairs(zu, hi, u, neg)
        loss = bpr_loss(s_pos, s_neg)
        # KG 嵌入损失
        tr = data['triples']
        if tr.numel()>0:
            t_neg = torch.randint(0, m.kg.ent.num_embeddings, (tr.size(0),), device=dev)
            loss = loss + args.lam_kg * m.kg.loss(tr[:,0], tr[:,1], tr[:,2], t_neg)
        # KG-CSL 对比自监督：必须用整张图的节点特征(N行)，且不 detach 以便梯度回传到 embedding
        if not args.no_csl:
            x_full = m.build_x(data['edge_index'])
            loss = loss + args.lam_csl * m.ssl(data['edge_index'], x_full)
        loss = loss + args.lam_l2 * sum(p.pow(2).sum() for p in m.parameters())
        loss.backward(); opt.step()
        if (ep+1)%max(1,args.epochs//5)==0:
            print(f"  epoch {ep+1}/{args.epochs}  loss={loss.item():.4f}")
    res = evaluate(m, data, K=args.K, cold_users=cu, cold_items=ci)
    tag = []
    for f in ['no_cagf','no_csl','no_mm','no_gat']:
        if getattr(args,f): tag.append(f)
    print("变体:", ",".join(tag) if tag else "完整模型 A1")
    print("结果:", {k:round(v,4) for k,v in res.items()})
    return res


if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='synthetic', choices=['synthetic','neo4j'])
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--dim', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--K', type=int, default=10)
    ap.add_argument('--lam_kg', type=float, default=0.1)
    ap.add_argument('--lam_csl', type=float, default=0.1)
    ap.add_argument('--lam_l2', type=float, default=1e-5)
    ap.add_argument('--no_cagf', action='store_true')
    ap.add_argument('--no_csl', action='store_true')
    ap.add_argument('--no_mm', action='store_true')
    ap.add_argument('--no_gat', action='store_true')
    # neo4j 选项
    ap.add_argument('--neo4j_uri', default='bolt://localhost:7687')
    ap.add_argument('--neo4j_user', default='neo4j')
    ap.add_argument('--neo4j_pwd', default='password')
    ap.add_argument('--img_dir', default='', help='商品图片目录(供CLIP);留空则用文本/随机回退')
    train(ap.parse_args())
