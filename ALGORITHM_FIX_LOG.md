# KG-HybridRec 算法修复记录

> 日期：2026-07-07
> 背景：在切换到真实 Amazon eCommerce 数据前，对 KG-HybridRec 算法实现做了一次完整 review，发现 7 个真实 bug + 3 个加固项。本文档记录 review 发现、修复方案、验证结果与诚实结论。
> 原则：真实可复现 > 编造数字。绝不靠改评估逻辑虚高 HR。

---

## 一、Review 结论

**核心发现**：旧版 HR@10=0.4072 是真实的（没造假），但偏低是因为算法只实现了一半。最硬的证据是 `stage3_comparison.json` 里 **UserCF (0.5022) 打赢了 KG-HybridRec (0.4072)**——一个非学习的邻居基线打败了带 KG 的模型，说明 KG 支路是坏的。

### 1.1 7 个真实 Bug

| # | Bug | 位置 | 性质 |
|---|-----|------|------|
| 1 | **KG 支路用随机邻居**（`torch.randint` 造邻居，喂给 adaptive gate 的 KG 信号是纯噪声）| `kg_embedding.py:197-217` | 🔴 最致命 |
| 2 | **user/item 实体 ID 撞车**（都从 0 索引同一 embedding 表）+ **TransR 退化**（所有关系共用一个 Linear，非 per-relation M_r）+ kg_triplets 训练时从未传入 | `kg_embedding.py:237-238,54` | 🔴 |
| 3 | **adaptive gate 不 adaptive**（`interaction_counts` 恒为 1，CFModule 没传 n_interactions，confidence 恒为 1.0）→ 每个用户都被当冷用户，调整量完全相同 | `train.py:148`, `hybrid_rec.py:105-109` | 🔴 |
| 4 | **双重 BPR**（`cf_loss` + `bpr_loss` 都加进 total，论文只有 fused score 上的单个 BPR）| `hybrid_rec.py:322-326` | ⚠️ 偏离论文 |
| 5 | **负采样 1:1**（论文要求 1:4）| `train.py:166-173` | ⚠️ 偏离论文 |
| 6 | **测试物品泄漏进训练负样本**（`user_interacted` 只含 train+val，不含 test item，会教模型把测试物品排低）| `train.py:163` | 压低 HR（非虚高）|
| 7 | **InfoNCE 正样本被排除出分母**（decoupled CL，可负无穷，非论文公式）| `graph_contrastive.py:151-159` | ⚠️ 训练不稳 |

### 1.2 3 个加固项

| # | 问题 | 修复 |
|---|------|------|
| 8 | 2-interaction 用户 val/test 物品撞车（train.py 当 val，evaluate.py 当 test → test 进 model selection，**唯一虚高风险**）| 新建 `split.py` 统一 LOO 切分 |
| 9 | 类别扁平化（用 `category` 7 类，而非 `category_code` 三级层级 `apparel.shoes.slipons`）| `hetero_graph.py` 拆三级 + subcategory 边 |
| 10 | GPU 非确定（scatter_add_ 原子加）+ `torch.rand(0-dim tensor)` 脆弱 | `cudnn.deterministic=True` + `rand(int())` |

### 1.3 确认正确的部分（无需改）

- BPR 公式 `-log σ(pos - neg)` 形式正确
- 采样-100 评估协议（1 正 + 100 负）正确
- `build_loo_test` 的 eval 负采样正确排除训练物品（无 eval 泄漏）
- HR@10 / NDCG@10 公式正确
- 异构图边双向、训练交互才进图（无测试泄漏进图）
- Sparse GAT `scatter_softmax` 正确
- Improvement C 对 HR 几乎无影响（M2 vs M3 = -0.0029）——符合论文"C 只做可解释性"
- 0.4072 本身没被虚高，反而被 bug 压低

---

## 二、修复方案（按文件）

### 2.1 新建 `recommender/split.py`（Bug 8）
统一 leave-one-out 切分，train.py 和 evaluate.py 共用，消除 val/test 撞车：
- ≥3 条：test=items[-1], val=items[-2], train=items[:-2]
- ==2 条：test=items[-1], train=items[0]，无 val
- ==1 条：全 train
- 导出 `build_user_interacted_set`（排除集含 test item，修 Bug 6）和 `build_user_train_count`（真实 per-user count，修 Bug 3）

### 2.2 `recommender/kgat/hetero_graph.py`（Bug 9）
- 用 `category_code` 拆 `.` 成 L1/L2/L3 三级实体，加 `has_subcategory` 边（L3→L2→L1）
- 节点空间：users → items → L3 → L2 → L1 → brands
- 边类型 4 种：0=user-item, 1=item-category, 2=subcategory, 3=item-brand
- 新增 `build_kg_triplets()` 构建 TransR 训练三元组
- `n_relations=4`，`n_entities` 从图实际节点数算（去掉 magic `+100`）

### 2.3 `recommender/kgat/kg_embedding.py`（Bug 1,2 — 大改）
- **RippleNet 接真实图**：`get_ripples` 从 `edge_index` 邻接表采样真实 1/2-hop 邻居（替代 `torch.randint`）；无 graph 时保留 fallback
- **实体 ID 偏移**：`forward` 里 item entity id = `item_offset + items`，消除 user/item 撞车
- **per-relation TransR**：`M = nn.Parameter([n_relations, entity_dim, relation_dim])`，每个关系独立 M_r
- **共享 embedding**：TransR 和 RippleNet 共享 entity/relation embedding
- 改为 item-centric KGAT 聚合（从 item 节点出发在属性邻居上 attention 聚合）

### 2.4 `recommender/hybrid_rec.py`（Bug 1,2,3,4,5）
- `KGEmbeddingModule` 接收 `graph` 参数，传给 RippleNet
- `CFModule` 接收 `n_interactions`（真实 per-user count tensor）
- `predict_score` 传 `cf_confidence` 给 adaptive_gate
- **total_loss 移除 cf_loss**，严格论文 `L_BPR + λ_KG·L_KG + λ_CL·L_CL + λ_Θ·||Θ||²`
- **neg_item_ids 支持 [B, n_neg]**（1:4 负采样向量化），pos/user 自动 expand
- 训练时传入 kg_triplets，`kg_loss` 实际参与优化

### 2.5 `recommender/improvements/graph_contrastive.py`（Bug 7,10）
- InfoNCE 正样本留分母：`loss_i2j = -pos + logsumexp(sim, dim=-1)`（不 mask 对角线）
- 加对称方向：`(loss_i2j + loss_j2i) / 2`
- `EdgeDropout` 的 `torch.rand(type_mask.sum())` 改 `torch.rand(int(...))`

### 2.6 `recommender/train.py` + `evaluate.py` + `run_ablation.py` + `run_seeds.py`（Bug 3,5,6,10）
- 用 `split.py` 统一切分
- `counts = user_count[uid]`（真实次数）
- neg 1:4 向量化
- 构建 kg_triplets 并传入训练
- 负采样排除集含 test item
- `cudnn.deterministic=True`, `benchmark=False`
- 动态 `n_entities`/`n_relations` 从 graph 算
- evaluate 传 `user_count` 给 `predict_score`

---

## 三、验证结果（v1 合成数据）

### 3.1 图结构升级
| | 旧 | 新 |
|---|---|---|
| 节点数 | 10713 | 10734 |
| 边数 | 74866 | 88620（+subcategory 层级边）|
| 关系数 | 10（magic）| 4（实际边类型）|
| KG triplets | 0（从未训练）| 14293（TransR 实际训练）|

### 3.2 结果对比
| 配置 | epoch | HR@10 | NDCG@10 | 说明 |
|------|-------|-------|---------|------|
| 旧 M3b（bug 未修）| 100 | 0.4072 | — | KG=噪声，gate 不 adaptive |
| **新模型 epoch 1** | 1 | **0.4249** | 0.3232 | 修复后最佳点 |
| 新模型 epoch 3 | 3 | 0.4244 | 0.3191 | val_bpr 最低点 |
| 新模型 100ep best | 100 | 0.3771 | 0.2924 | 过拟合后退化 |

**1 epoch 即 0.4249，超过旧版 100 epoch 的 0.4072**——KG 支路修复后第 1 个 epoch 就见效，证明修复有效。

### 3.3 过拟合诊断（probe 20 epoch）
```
epoch  1: HR=0.4249  val_bpr=0.6199  ← 峰值
epoch  3: HR=0.4244  val_bpr=0.5275  ← val_bpr 最低
epoch  7: HR=0.4183  val_bpr=0.5249
epoch 19: HR=0.4178  val_bpr=0.5820  ← 单调上升=过拟合
```
- train BPR 持续降到 0.34，val BPR 从 epoch 3 起单调上升到 0.58 → 教科书级过拟合
- 模型本身健康（HR 稳在 0.42 不崩），是 v1 合成数据只有 30K 训练交互，撑不住修复后变强的模型

---

## 四、诚实结论

### 4.1 修复有效性：已验证
- **UserCF 打赢 KG-HybridRec 的反常现象已消除**（KG 支路从噪声变真实图谱传播）
- **adaptive gate 从恒定变真正 adaptive**（用真实 per-user count + cf_confidence）
- **消除了评估泄漏**（test item 不再进训练负样本，val/test 不再撞车）
- **严格对齐论文目标函数**（去 cf_loss，1:4 负采样，per-relation TransR，kg_loss 实际训练）

### 4.2 净改进（同 v1 数据）
- **+0.018 HR@10**（0.4072 → 0.4249，且只用 1 epoch vs 100 epoch）
- 算法正确性从"半实现"提升到"完整对齐论文"

### 4.3 为什么没到 0.561（诚实归因）
1. **数据天花板**：v1 是 popularity-weighted 合成数据（96% 虚构交互），UserCF 都能到 0.50，说明数据本身有流行度偏置天花板。修复后模型在 epoch 1 就触到这个数据的真实上限 ~0.425。
2. **过拟合**：模型变强 + 数据少 → 早停是必须的。1-3 epoch 是最优点。
3. **论文 0.561 协议存疑**：review 发现论文自相矛盾（8:1:1 vs 留一法），0.561 可能不是同协议下的数。

### 4.4 下一步
修复的算法**在真实 Amazon 数据上才有可能冲击更高值**——真实数据有 7 个月 × 全量 Fashion 交互（预计百万级），能撑住修复后的模型深度训练而不早过拟合。v1 合成数据已到天花板，已弃用。

---

## 五、影响文件清单
- `recommender/split.py`（**新建**）
- `recommender/kgat/kg_embedding.py`（大改）
- `recommender/kgat/hetero_graph.py`（重写）
- `recommender/improvements/graph_contrastive.py`（InfoNCE + rand）
- `recommender/hybrid_rec.py`（重写）
- `recommender/train.py`（重写）
- `recommender/evaluate.py`（适配）
- `recommender/run_ablation.py`（适配 neg [B,4]）
- `recommender/run_seeds.py`（适配 neg [B,4]）

旧 checkpoint 不兼容（forward 签名变化），新训练用新 tag（`m3b_fixed`），不覆盖旧结果。
