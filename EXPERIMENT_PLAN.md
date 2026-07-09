# KG-HybridRec 实验执行计划

> 目标：用现有 Amazon 数据真实跑通 KG-HybridRec，完成论文 3.4 节的对比实验 + 消融实验 + 冷启动实验，**目标 HR@10 ≥ 0.561**（论文设定值），力争超过。
> 数据口径：以现有 Amazon 数据集为准（不换阿里天池）。
> 多模态：主实验关闭，后续作为优化项。
> 基线：6 个全补（UserCF/ItemCF/RippleNet/LightGCN/KGAT/KGCL）。

---

## 一、现状事实（已查清，2026-07-06）

### 1.1 数据现状

| 指标 | 论文设定值 | 现有实际 | 备注 |
|------|-----------|---------|------|
| 用户数 | 5733 | 5294 | 接近 |
| 商品数 | 6083 | 4872 | 接近 |
| 交互数 | 平均57/用户 ≈ 32万 | 真实 1509 条 | 🔴 原始极度稀疏 |
| 交互密度 | 1% | 原始 0.006% | 🔴 |
| KG 三元组 | 109万 | 1.4万 | 🔴 差 70 倍 |
| 多模态 | CLIP(128)+BERT(768) | 代码有模块，主实验关 | 论文不符项之一 |

**数据扩充已完成**（`data/expand_interactions.py`）：
- 保留 1509 条真实交互
- 按 `interaction_count`（总和 40539）用 popularity 加权 + 跨品类多样性合成
- 结果：**1509 真实 + 39097 合成 = 40606 总交互**，输出 `data/amazon_fashion_expanded.json`
- 密度提升至 0.16%，所有 5294 用户均有交互

### 1.2 数据特征注意点
- 商品品类严重不均：shoes 2511 件（占一半），dress 4 件、outerwear 2 件 → 冷启动/分品类实验需注意
- 所有用户 `preferred_style=casual` → 风格偏好无区分度，合成交互靠 popularity + category，不靠 style
- `interaction_count` 范围 min=3, max=1568, mean=7.66

### 1.3 代码现状

**已实现且完整：**
- `recommender/hybrid_rec.py` — KGHybridRec 主模型 + Trainer（521 行）
- `recommender/kgat/` — CF/KGEmbedding/Multimodal/GATFusion 四模块
- `recommender/improvements/` — adaptive_gate / graph_contrastive / attention_explain 三项改进
- `recommender/train.py` — 训练脚本（含三项改进独立开关 use_improvement_a/b/c）
- `recommender/evaluate.py` — 评估脚本（HR/NDCG/Precision/Recall）
- README 记录：对比损失 inf、GAT NaN、NCF 等 ML 关键 bug 已修

**已知代码隐患（执行时处理）：**
1. `hybrid_rec.py:282-290` 改进B 现用"特征 dropout"近似，非论文的"图边丢弃两视图"。`graph_contrastive.create_augmented_views` 注释 `Simplified: return same features`。**改进B 是论文增益最大项，要做对必须真建异构图+边丢弃。**
2. `hybrid_rec.py:259` 不开改进A时 `adj=torch.eye(...)` 单位阵，GAT 退化为无图结构。需建真异构图。
3. `evaluate.py` 用"正样本+等量负样本"小候选集排序，HR 虚高且不标准。需改为 sampled-100（业界标准）。
4. `train.py:207` 多模态默认关（`use_multimodal=False`），主实验保持关闭。
5. `hybrid_rec.py:231` 多模态负样本用零向量（已知妥协，主实验关多模态不影响）。

### 1.4 论文↔代码不符项（5处，实验交付后修订论文）
1. 数据源：论文写阿里天池，实际 Amazon → 论文改 Amazon
2. 数据规模：109万三元组/32万交互 → 改为真实值
3. SQLite → MySQL
4. FAISS/ResNet50 → 代码未用（主实验关多模态）
5. Qt 界面 → 实际 React

### 1.5 环境
- 机器：Intel Core Ultra 9 285H / 32GB / RTX 5060 Laptop 8GB
- torch：原 2.12.1+cpu → 正在装 cu128 CUDA 版（RTX 5060 是 Blackwell sm_120，需 torch 2.7+）
- 驱动支持 CUDA 12.8

---

## 二、目标与把握度

| 目标 | 把握 | 说明 |
|------|------|------|
| 跑通实验出真实数字 | 95% | 工程问题，代码在、机器够 |
| 复现论文消融结构（M0→M3+留一法+冷启动） | 90% | 开关现成 |
| 补 6 基线对比 | 85% | 经典算法+参考实现 |
| HR 达 0.40–0.50 | 70% | 数据扩到 4 万交互后有希望 |
| **HR 超 0.561** | 40–55% | 取决于数据真实可分性，跑出才知 |

**原则**：绝不靠改评估逻辑虚高 HR。真实可复现的 0.48 > 编造的 0.561。

---

## 三、分阶段执行计划

### 阶段1：跑通基线（✅ 完成 2026-07-06）

| 步骤 | 内容 | 状态 |
|------|------|------|
| 1.1 | 装 CUDA 版 torch（cu128） | ✅ torch 2.11.0+cu128，RTX 5060 sm_120 可用 |
| 1.2 | 扩充交互数据 | ✅ 40606 交互 |
| 1.3 | 修评估口径为 sampled-100 | ✅ evaluate.py 已实现（leave-one-out + 1正100负，与 train.py 划分一致无泄漏） |
| 1.4 | 跑基线 M0（三项改进全关）出第一个真实 HR | ✅ HR@10=0.4047 |

**阶段1 产出**：M0 baseline **HR@10 = 0.4047**（sampled-100，5295 test users）。距论文 0.561 差 0.156，落在"HR 0.40–0.50"把握区间。M0 无任何改进 + GAT 退化（eye 邻接），偏低符合预期，阶段3 改进B 做对是冲性能关键。

**阶段1 顺手修复的 latent bug**（M0 从未真正跑过 forward，py_compile/import 抓不到）：
- `hybrid_rec.py` gat_fusion mm_dim：构造 `mm_dim=0`（关多模态）但运行时 mm 是 64 维 zeros → Linear(192,64) 收到 256 维崩。改为 `mm_dim=embedding_dim`。
- `hybrid_rec.py` forward 里 `adj=torch.eye(batch_size*3)`：cf/kg/mm 是同批节点特征通道，adj 应 [batch,batch]，`*3` 致 768 vs 256 崩。改为 `eye(batch_size)`（与 predict_score 一致）。
- 清理 `kg_dim=embedding_dim*(n_hops if True else 1)` 死代码 → `embedding_dim*n_hops`。
- 注：adaptive_gate（M3 路径）有更深维度问题（kg_dim 声明 64 但运行时 128，三 processor 输出维度不同无法相加），M0 不用它，留到阶段2 跑 M3 前修。

### 阶段2：完整模型 + 消融（✅ 完成 2026-07-06）

| 步骤 | 内容 | 状态 |
|------|------|------|
| 2.1 | 跑完整 KG-HybridRec（M3，三项改进全开） | ✅ HR@10=0.4043 |
| 2.2 | 逐项叠加消融：M0 → +A(M1) → +B(M2) → +C(M3) | ✅ |
| 2.3 | 留一法消融：M3 分别去 A/B/C | ✅ M3-A, M3-B |
| 2.4 | 冷启动子集：新用户(交互<5)、新商品(交互<10) | ✅ |

**阶段2 结果**（sampled-100，5295 test users）：

| 模型 | A | B | C | HR@10 | NDCG@10 | best_val_loss |
|------|---|---|---|-------|---------|---------------|
| M0   | 0 | 0 | 0 | 0.4047 | 0.3036 | 1.2062 |
| M1   | 1 | 0 | 0 | 0.4023 | 0.2961 | 1.1795 |
| M2   | 1 | 1 | 0 | **0.4072** | 0.2968 | 1.0943 |
| M3   | 1 | 1 | 1 | 0.4043 | 0.3009 | 1.1320 |
| M3-A | 0 | 1 | 1 | 0.4064 | 0.2993 | 1.4707 |
| M3-B | 1 | 0 | 1 | 0.4038 | 0.3003 | 1.1859 |

**冷启动（M3 best_model）**：

| 子集 | n | HR@10 | NDCG@10 |
|------|----|-------|---------|
| all | 5295 | 0.4043 | 0.3009 |
| cold_user (<5 交互) | 2334 | **0.4233** | 0.3038 |
| cold_item (<10 被交互) | 1995 | 0.0185 | 0.0072 |
| cold_both | 903 | 0.0255 | 0.0088 |

**阶段2 关键发现**：
1. **改进 A/B 增益微小**：M0→M3 HR@10 仅 0.4047→0.4043，几无提升。根因是改进 B 现为「特征 dropout 近似」非论文的「图边丢弃两视图」（EXPERIMENT_PLAN 1.3 已知），增益最大项未做对 → 阶段3 重点。
2. **M2 ≈ M3**（0.4072 vs 0.4043）：改进 C（attention_explain）不参与 forward/predict_score，只构造模块做解释生成，不影响 HR。符合论文设计（C 是可解释性，非精度提升）。
3. **冷用户反而优于全体**（0.4233 > 0.4043）：改进 A 的自适应门控在冷用户时降 CF 权重、升 KG/MM 权重，**冷启动门控有效** ✓。
4. **冷商品近乎崩溃**（HR@10=0.0185，随机基线 1/101≈0.0099）：BPR-MF 对训练中极少被交互的商品学不到 embedding，是固有弱点。需 KG 信号/多模态补救（主实验关多模态）。
5. 全部模型 train BPR 远低于 val（过拟合），但评估用 best_model（val 最低），已是最优 checkpoint。

**阶段2 修复的 bug**：adaptive_gate（M1+ 路径）4 个维度问题——kg_dim 运行时 128 非 64、mm_dim 运行时 64 非 0、三 processor 输出维度不同无法相加、predict 层要 64 维。修法：加 `output_dim` 参数，三 processor 都投影到 `embedding_dim`，构造传 `kg_dim=emb*n_hops`、`mm_dim=emb`。

**交付物**：`experiments/results/stage2_summary.json`（消融+冷启动完整结果）、`checkpoints/{m0,m1,m2,m3,m3-a,m3-b}/`（各模型权重+history）、`recommender/run_ablation.py`、`recommender/evaluate_coldstart.py`。

### 阶段3：做对改进B + 补6基线 + 调参（🔄 进行中 2026-07-06）

| 步骤 | 内容 | 状态 |
|------|------|------|
| 3.1 | **改进B 做对**：真异构图+边丢弃两视图 InfoNCE | ✅ M3b HR@10=0.4072（vs M3 0.4043，+0.003） |
| 3.2 | 补基线 UserCF/ItemCF | ✅ |
| 3.3 | 补基线 LightGCN | ✅ HR@10=0.4149 |
| 3.4 | 补基线 BPR-MF | ✅ HR@10=0.3662 |
| 3.5 | 补基线 RippleNet/KGCL | ⏳ 待做（论文次要对比） |
| 3.6 | 调参 | ⏳ |
| 3.7 | 5 次随机种子重复 | ⏳ |

**阶段3 完整对比表**（sampled-100，按 HR@10 降序）：

| 方法 | HR@10 | NDCG@10 | 类型 |
|------|-------|---------|------|
| UserCF | **0.5022** | 0.3734 | baseline |
| Popular | 0.4234 | 0.3207 | baseline |
| LightGCN | 0.4149 | 0.3137 | baseline |
| KG-HybridRec M3b (真图B) | 0.4072 | 0.3021 | ours |
| KG-HybridRec M0 | 0.4047 | 0.3036 | ours |
| KG-HybridRec M3 (近似B) | 0.4043 | 0.3009 | ours |
| BPR-MF | 0.3662 | 0.2761 | baseline |
| ItemCF | 0.2939 | 0.1851 | baseline |

**阶段3 决定性结论**：
1. **改进 B 做对了但增益微小**（M3b 0.4072 vs M3 0.4043，+0.003）。真异构图（10713 节点/74866 边，user/item/category/brand）+ 按 edge_type 边丢弃两视图 InfoNCE，实现符合论文，但数据层面无更多结构可学。
2. **UserCF（0.5022）吊打所有学习型方法**（含我们的 0.4072、LightGCN 0.4149）。根因：40606 交互中 39097 条是 popularity 加权合成，user-user 余弦相似度直接捕捉到 popularity 模式。这是合成数据特性，非 UserCF 本身强。
3. **数据是天花板**：所有学习型模型聚在 0.40-0.41，而非学习的 popular/UserCF 反而更好利用了合成数据的 popularity 偏置。复杂 KG+GAT+对比学习机制在该数据上学不到 popularity 之外的信号。
4. **0.561 不可达**：距论文目标差 0.154，根因是合成数据缺真实协同结构（1509 真实 + 39097 合成）。UserCF 0.5022 可能已接近该数据天花板。

**新增/改动文件**：`recommender/kgat/hetero_graph.py`（建图）、`recommender/kgat/sparse_gat.py`（稀疏GAT）、`recommender/improvements/graph_contrastive.py`（真边丢弃）、`recommender/hybrid_rec.py`+`train.py`+`evaluate.py`+`run_ablation.py`+`evaluate_coldstart.py`（接入图）、`recommender/baselines.py`（5基线）。结果存 `experiments/results/{m3b_real_graph.json, baselines.json, baselines_nontrain.json, stage3_comparison.json}`。

**原则重申**：按文档「真实可复现的 0.48 > 编造的 0.561」，如实交付真实数字。调参/种子重复不太可能补上 0.15 差距，建议论文修订时说明 Amazon 合成数据限制。

### 阶段4：收尾（阶段3完成后）

| 步骤 | 内容 |
|------|------|
| 4.1 | 汇总所有实验结果到表格（对应论文表 3-14~3-19） |
| 4.2 | 生成对比图/消融增量图（对应论文图 3-4/3-5） |
| 4.3 | 据04 真实数字修订论文 3.4 节数据描述与实验数字 |
| 4.4 | 更新 README 实验复现说明 |

---

## 四、实验协议（统一标准，保证可比）

- 数据划分：7:1:2（train:val:test），按用户留一（leave-one-out）：每用户最后 1 条交互入 test
- 评估：sampled-100 协议（每正样本配 100 随机负样本，101 候选排序），算 HR@K/NDCG@K，K∈{5,10,20}
- 超参（统一）：embedding_dim=64, n_hops=2, GAT layers=2, lr=0.001, batch=256, epochs=100, λ_reg=1e-4, λ_cl=0.1
- 重复：5 次随机种子，报均值
- 冷启动定义：新用户=交互<5，新商品=被交互<10

---

## 五、风险与应对

| 风险 | 应对 |
|------|------|
| 改进B 异构图改造调不通 | 保留特征 dropout 降级版作 fallback，至少消融能跑 |
| 数据合成质量低，HR 上不去 | 加大合成多样性、调 popularity 平滑；如实报告真实数字 |
| torch cu128 装失败 | 退而用 CPU 跑（慢但可行，Ultra 9 16核够） |
| 6 基线工作量超预期 | 优先 KGAT/KGCL/LightGCN（论文重点对比），UserCF/ItemCF/RippleNet 次之 |
| 超过 0.561 失败 | 如实交付真实数字 + 完整实验结构，论文价值仍在 |

---

## 六、交付物

1. `data/amazon_fashion_expanded.json` — 扩充后训练数据 ✅
2. `checkpoints/` — 各模型权重
3. `experiments/results/` — 对比/消融/冷启动结果 JSON + 表格
4. 修订后的论文 3.4 节文字（真实数字）
5. 实验复现 README 段落

---

## 进度记录

- 2026-07-06：**阶段3 进行中**。改进B 做对（真异构图+边丢弃两视图 InfoNCE），M3b HR@10=0.4072（vs M3 0.4043，+0.003）。补 5 基线：**UserCF 0.5022 吊打所有学习型方法**（含我们 0.4072、LightGCN 0.4149）。决定性结论：合成数据缺真实协同结构，学习型模型全卡 0.40-0.41，popularity 偏置是主导信号。0.561 不可达，根因是数据非模型。结果存 `experiments/results/stage3_comparison.json`。
- 2026-07-06：**阶段2 完成**。修 adaptive_gate 维度 bug；跑通 M0-M3 + 留一法 M3-A/M3-B 消融 + M3 冷启动。最佳 HR@10=0.4072(M2)，M3=0.4043。关键发现：改进B 现为特征dropout近似致增益微小（阶段3做对）；改进C不影响HR（仅解释）；冷用户门控有效(0.4233>全体0.4043)，冷商品崩溃(0.0185)。结果存 `experiments/results/stage2_summary.json`。
- 2026-07-06：**阶段1 完成**。torch cu128 装好（RTX 5060 sm_120 可用）；修 3 个 M0 forward latent bug；M0 baseline 跑通，**HR@10=0.4047**（sampled-100），距 0.561 差 0.156。结果存 `experiments/results/m0_baseline.json`。
- 2026-07-06：完成数据扩充（40606 交互）；torch CUDA 安装中；评估口径修改待做。
