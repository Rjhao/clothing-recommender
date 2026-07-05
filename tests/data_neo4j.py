# -*- coding: utf-8 -*-
"""
从你的 Neo4j 真实数据构建训练所需张量（复用你 kg_hybrid_complete.py 的查询思路）。
输出与 train.make_synthetic 完全一致的 data 字典。

依赖：neo4j（或把 _run_query 换成你项目里的 Neo4jClient.graph.run）。
特征：
  - 文本(BERT)  : 用商品 name/描述/关键词，bert-base-chinese 取 [CLS]，768维
  - 图像(CLIP)  : 用 Image 节点 image_path 指向的图片，CLIP-ViT，投影到128维
  任一不可用时自动回退为随机/零向量，并打印告警（保证流程能跑，但多模态此时无效）。
"""
import numpy as np, torch, os


def _run_query(args, cypher, **params):
    from neo4j import GraphDatabase
    drv = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_pwd))
    with drv.session() as s:
        return [r.data() for r in s.run(cypher, **params)]
    # 若用你现有的 py2neo 客户端，可替换为：
    #   from database.neo4j_client import Neo4jClient
    #   return Neo4jClient().graph.run(cypher, **params).data()


REL_TYPES = ['BELONGS_TO','HAS_KEYWORD','ALTERNATIVE','COORDINATES_WITH','HAS_IMAGE','MENTIONS']


def load_from_neo4j(args, min_purchases=4, seed=42):
    rng = np.random.default_rng(seed)
    # 1) 交互
    inter = _run_query(args, """
        MATCH (u:User)-[:PURCHASED]->(p:Product)
        RETURN u.id AS uid, p.id AS pid""")
    user_items = {}
    for r in inter:
        user_items.setdefault(r['uid'], []).append(r['pid'])
    user_items = {u:v for u,v in user_items.items() if len(v)>=min_purchases}
    users = sorted(user_items); uidx = {u:i for i,u in enumerate(users)}
    items = sorted({p for v in user_items.values() for p in v})
    iidx = {p:i for i,p in enumerate(items)}
    NU, NI = len(users), len(items)

    # 2) 训练/测试划分（50/50，与原实验一致）
    train_pos, test_pos, rows = {}, {}, []
    for u in users:
        its=[iidx[p] for p in user_items[u] if p in iidx]
        rng.shuffle(its); sp=max(1,len(its)//2)
        tr=set(its[:sp]); te=set(its[sp:])
        train_pos[uidx[u]]=tr; test_pos[uidx[u]]=te
        for i in tr: rows.append((uidx[u], i))

    # 3) KG 三元组 + 异构图边
    ent_map={}; triples=[]; E=[]
    def ent_id(name):
        if name not in ent_map: ent_map[name]=len(ent_map)
        return ent_map[name]
    # 商品本身占据实体 0..NI-1
    for p,i in iidx.items(): ent_map[f'P::{p}']=i
    for rel in REL_TYPES:
        try:
            recs=_run_query(args, f"""
                MATCH (p:Product)-[:{rel}]->(o) RETURN p.id AS pid, o.id AS oid""")
        except Exception:
            recs=[]
        rid=REL_TYPES.index(rel)
        for r in recs:
            if r['pid'] not in iidx: continue
            h=iidx[r['pid']]; t=ent_id(f'{rel}::{r["oid"]}')
            triples.append((h, rid, t))
    n_ent=len(ent_map)
    ent_base=NU+NI
    for u,i in rows: E.append((u, NU+i)); E.append((NU+i, u))
    for h,rid,t in triples:        # 统一编号：实体 eid -> 图节点 NU+eid
        node = NU + t
        E.append((NU+h, node)); E.append((node, NU+h))

    edge_index=torch.tensor(E,dtype=torch.long).t().contiguous()
    inter_cnt=torch.zeros(NI,dtype=torch.long)
    for _,i in rows: inter_cnt[i]+=1

    # 4) 多模态特征
    txt = _bert_features(args, items, iidx)
    img = _clip_features(args, items, iidx)

    data = {
        'edge_index':edge_index, 'item_ent':torch.arange(NI),
        'img':img, 'txt':txt, 'inter_cnt':inter_cnt,
        'train_pos':train_pos, 'test_pos':test_pos,
        'eval_users':torch.arange(NU),
        'triples':torch.tensor(triples,dtype=torch.long) if triples else torch.zeros((0,3),dtype=torch.long),
        'rows':torch.tensor(rows,dtype=torch.long),
        'n_users':NU,'n_items':NI,'n_entities':max(n_ent,NI),'n_relations':len(REL_TYPES),
    }
    cold_users={u for u in train_pos if len(train_pos[u])<2}
    cold_items={i for i in range(NI) if inter_cnt[i].item()<2}
    print(f"[neo4j] users={NU} items={NI} edges={edge_index.size(1)} triples={len(triples)} ent={n_ent}")
    return data, cold_users, cold_items


def _bert_features(args, items, iidx, dim=768):
    try:
        from transformers import AutoTokenizer, AutoModel
        texts=[_item_text(args, p) for p in items]
        tok=AutoTokenizer.from_pretrained('bert-base-chinese')
        mdl=AutoModel.from_pretrained('bert-base-chinese').eval()
        out=torch.zeros(len(items),dim)
        with torch.no_grad():
            for s in range(0,len(texts),32):
                b=tok(texts[s:s+32],padding=True,truncation=True,max_length=64,return_tensors='pt')
                out[s:s+32]=mdl(**b).last_hidden_state[:,0]
        return out
    except Exception as e:
        print(f"[告警] BERT 不可用({e})，文本特征回退为随机。多模态文本分支此时无效。")
        return torch.randn(len(items),dim)


def _item_text(args, pid):
    try:
        r=_run_query(args, """MATCH (p:Product {id:$pid})
            OPTIONAL MATCH (p)-[:HAS_KEYWORD]->(k:Keyword)
            RETURN p.name AS name, collect(k.text) AS kws""", pid=pid)
        if r: return f"{r[0].get('name') or ''} {' '.join(r[0].get('kws') or [])}"
    except Exception: pass
    return str(pid)


def _clip_features(args, items, iidx, dim=128):
    if not args.img_dir or not os.path.isdir(args.img_dir):
        print("[告警] 未提供 --img_dir 或目录不存在，图像特征回退为随机。"
              "★多模态图像分支此时无效——这正是论文里 CLIP/CLIP-ViT 那部分必须用真实图片才成立的地方。")
        return torch.randn(len(items),dim)
    try:
        import open_clip; from PIL import Image
        model,_,pre=open_clip.create_model_and_transforms('ViT-B-32',pretrained='laion2b_s34b_b79k')
        model.eval(); raw=torch.zeros(len(items),512)
        with torch.no_grad():
            for p,i in iidx.items():
                fp=os.path.join(args.img_dir,f"{p}.jpg")
                if os.path.exists(fp):
                    raw[i]=model.encode_image(pre(Image.open(fp).convert('RGB')).unsqueeze(0))[0]
        # 512->dim 线性降维（随机投影占位；训练中 img_proj 会再学一层）
        W=torch.randn(512,dim)/np.sqrt(512)
        return raw@W
    except Exception as e:
        print(f"[告警] CLIP 不可用({e})，图像特征回退为随机。")
        return torch.randn(len(items),dim)
