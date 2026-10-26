表4-5 整体性能对比
Tab.4-5 Overall Performance Comparison

| 算法 | HR@10 | NDCG@10 | P@10 | HR@20 | NDCG@20 | P@20 |
|---|---|---|---|---|---|---|
| KG-HybridRec(本文) | **0.5909** | 0.4041 | **0.0591** | **0.7041** | 0.4327 | **0.0352** |
| KGAT | 0.5841 | 0.3990 | 0.0584 | 0.6929 | 0.4266 | 0.0346 |
| LightGCN | 0.5900 | **0.4609** | 0.0590 | 0.6642 | **0.4797** | 0.0332 |
| BPR-MF | 0.5201 | 0.3909 | 0.0520 | 0.6053 | 0.4124 | 0.0303 |
| RippleNet | 0.5153 | 0.3396 | 0.0515 | 0.6309 | 0.3688 | 0.0315 |
| KGCL | 0.5151 | 0.3896 | 0.0515 | 0.5974 | 0.4103 | 0.0299 |
| MKR | 0.5338 | 0.3600 | 0.0534 | 0.6461 | 0.3884 | 0.0323 |
| KGIN | 0.4960 | 0.3172 | 0.0496 | 0.6242 | 0.3496 | 0.0312 |
| CKAN | 0.5563 | 0.4183 | 0.0556 | 0.6500 | 0.4420 | 0.0325 |

评测协议：sampled-100 leave-one-out，n_neg=100，seed=42，与 baselines_real.py 完全一致。
KGAT/LightGCN/BPR-MF/RippleNet/KGCL 为既有论文数据；MKR/KGIN/CKAN 于 2026-10-21 重跑（修复 baselines_kg.py:_transpose_csr 后），原始日志：experiments/results/kg_baselines_full.log，JSON：experiments/results/kg_baselines.json。
