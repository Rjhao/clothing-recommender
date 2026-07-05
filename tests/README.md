# KG-HybridRec 真实实现（对齐论文 4.3）

把论文 4.3 描述的四层深度模型真正实现出来，替换掉原 `kg_hybrid_complete.py`
里那套 "SVD + 启发式规则 + 固定 0.4/0.6 线性融合" 的伪实现。

## 文件
- `modules.py`  CAGF（模态级/分支级门控）、KG-CSL（对比自监督）、注意力解释路径
- `model.py`    KGHybridRec：BPR协同 + TransR知识图谱 + 多模态CAGF + 多层GAT + KG-CSL + 预测
- `data_neo4j.py` 从你的 Neo4j 真实数据构建张量（复用你现有查询）
- `train.py`    训练/评估/消融入口

## 环境
```bash
pip install torch numpy            # 必需
pip install neo4j transformers     # 真实数据 + BERT 文本特征
pip install open_clip_torch pillow # CLIP 图像特征（多模态图像分支必需）
```

## 1) 先冒烟自测（合成数据，30秒，验证代码能端到端跑）
```bash
cd kg_hybridrec_real
python train.py --source synthetic --epochs 30
```
应打印 loss 下降 + 一组 HR/NDCG/Recall/Coverage 指标。
（注：本仓库已用 NumPy 等价镜像验证了全部张量运算与索引的正确性；
 端到端 torch 运行请在你本机执行上面这条命令确认。）

## 2) 真实数据训练
```bash
python train.py --source neo4j \
  --neo4j_uri bolt://localhost:7687 --neo4j_user neo4j --neo4j_pwd 你的密码 \
  --img_dir /path/to/商品图片目录 \
  --epochs 100 --dim 64
```
图片目录里文件名需为 `{商品id}.jpg`。

## 3) 消融实验（直接产出论文 表4-19 / 表4-20 / 表4-21 的各行）
| 命令 | 对应表格行 |
|---|---|
| `python train.py ...`（默认全开） | A1 完整模型 |
| `--no_cagf` | 表4-20 E1：去CAGF（退化为等权融合） |
| `--no_csl`  | 表4-21 F1：去KG-CSL（λ1=0） |
| `--no_mm`   | 表4-19 A3：移除多模态 |
| `--no_gat`  | 表4-19 A4：移除GAT |
把每次跑出的 HR/NDCG/Recall/Coverage 填进 `补充消融实验_模板.xlsx` 即可。

## 4) 基线 RippleNet / KGAT（强烈建议用 RecBole，别手搓）
```bash
pip install recbole
# 把 Neo4j 数据导成 RecBole 原子文件 (.inter / .kg / .link)，然后：
python run_recbole.py --model=RippleNet --dataset=fashion
python run_recbole.py --model=KGAT     --dataset=fashion
```
RecBole 内置 RippleNet、KGAT 官方实现，跑出的数字才可信、可追溯，
直接替换论文表4-14里那两行（目前那两行在你旧代码里没有来源）。

## ★ 必须正视的三件事（诚实提醒）
1. **多模态依赖真实图片/文本**。没有 `--img_dir` 真图片时，图像特征自动回退为随机向量，
   此时"多模态"是无效的、论文里的 CLIP-ViT 说法不成立。文本 BERT 同理需要真实商品描述。
2. **数字会变**。真实现跑出的结果会覆盖论文现有所有结果表，且不保证优于 KGAT。
   若真跑完打不过基线，要么继续调（加难负采样、调 λ/τ、增多模态质量），要么回到"改论文"路线。
3. **与论文的简化对照**：本实现用多层 GAT 在异构图上做多跳聚合，已涵盖 RippleNet 式
   多跳传播的作用，未单独再写一遍波纹集；TransR 作为 KG 嵌入损失项参与训练。
   若论文要逐字对应"波纹集传播"，需另加一节说明二者等价或补该模块。

## 损失
L = L_BPR + λ_kg · L_TransR + λ_csl · L_contrastive + λ_l2 · ‖Θ‖²
（默认 λ_kg=0.1, λ_csl=0.1, λ_l2=1e-5；建议做 λ_csl、τ 敏感性）
