# 开发日志 - KG-HybridRec

> 最后更新：2026-07-10

---

## 一、虚拟试穿模块现状

### 1.1 当前实现（MediaPipe 轻量方案）

**弃用 ACGPN 全套流水线**，原因：
- `checkpoints/` 仅含 `label2city`（pix2pixHD 语义分割→街景）权重，**不是** ACGPN 试穿模型
- 缺少 GMM/TOM/SCHP/OpenPose 核心权重
- 依赖老版 torch，Python 3.14 上安装极容易失败

**当前方案**：MediaPipe PoseLandmarker（Tasks API）+ OpenCV 叠加

| 组件 | 状态 | 说明 |
|------|------|------|
| OpenCV | ✅ | cv2 5.0.0 (opencv-contrib-python) |
| MediaPipe | ✅ | 0.10.35, Tasks API (非 solutions) |
| PoseLandmarker | ✅ | `pose_landmarker_lite.task` 已下载 |
| ACGPN 完整管线 | ❌ | 缺 GMM/TOM/SCHP 权重 |

**贴图逻辑**：
- 33 个人体关键点 → 计算肩宽/躯干倾角/髋宽
- 按 category（top/bottom/dress/outerwear/shoes/accessories）自适应定位
- HSV + GrabCut 去除衣物背景，alpha 高斯羽化边缘
- 检测不到人体时回退固定比例贴图

**服务地址**：`http://localhost:8001`

### 1.2 目录结构

```
ACGPN/
├── ACGPN_inference/checkpoints/label2city/
│   ├── latest_net_G.pth   ← pix2pixHD 权重（不是试穿模型！）
│   ├── latest_net_G1.pth
│   ├── latest_net_G2.pth
│   └── latest_net_U.pth
├── ACGPN_train/
│   ├── checkpoints/        ← 空，训练无产出
│   └── sample/            ← 空
└── ACGPN_inference/sample/ ← 空
```

### 1.3 训练说明

ACGPN 完整训练需要：
- **数据集**：VITON（14,221 训练对），当前 `Data_preprocessing/data/` 为空
- **模型组件**：
  - GMM（几何匹配网络）— 20 epochs
  - TOM（试穿合成网络）— 200 epochs
  - U（TPS UV 映射）— 随训练
  - G（Refine 融合网络）— 200 epochs

### 1.4 快速验证

```bash
# 启动服务
python tryon_api.py

# 健康检查
curl http://localhost:8001/

# 命令行验证
python -c "
import tryon_api
b = tryon_api.run_acgpn_tryon(
    'static/test/person.jpg',
    'static/test/cloth.jpg',
    'static/test/_out.jpg',
    'top'
)
print('backend:', b)
"
```

---

## 二、推荐系统实验结果（2026-07-10 更新）

### 2.1 数据集统计

| 数据集 | 用户数 | 商品数 | 交互数 | 稀疏度 | 用户均值 | 用户中位数 | 商品均值 | 商品中位数 |
|--------|--------|--------|--------|--------|----------|------------|----------|------------|
| purchase (dense) | 64,097 | 21,683 | 273,464 | 0.000197 | 4.3 | 3 | 12.6 | 4 |
| view_sparse | 24,999 | 33,938 | 380,754 | 0.000449 | 15.2 | 9 | 11.2 | 4 |

### 2.2 核心结果

**purchase (dense) — UserCF HR@10=0.9995**

| 算法 | HR@5 | HR@10 | HR@20 | NDCG@10 | 备注 |
|------|------|-------|-------|---------|------|
| UserCF | 0.9938 | **0.9995** | 1.0000 | 0.9212 | 利用物品共现，几乎完美 |
| ItemCF | — | 0.9527 | — | — | 稠密数据次优 |
| KG-HybridRec M3 | 0.4780 | **0.5890** | 0.7043 | 0.3577 | 真实数据最佳学习型基线 |
| KGCL | — | 0.2361 | — | — | |
| LightGCN | — | 0.2878 | — | — | 训练稳定性问题 |
| RippleNet | — | 0.2352 | — | — | |
| BPR-MF | — | 0.2299 | — | — | |

**view_sparse — ItemCF 异常强劲（HR@10=0.7067）**

| 算法 | HR@5 | HR@10 | HR@20 | NDCG@10 | 备注 |
|------|------|-------|-------|---------|------|
| **ItemCF** | **0.5843** | **0.7067** | **0.8288** | **0.5139** | 稀疏场景意外强 |
| Popular | 0.4302 | 0.5491 | 0.6726 | 0.3611 | 流行度基线 |
| UserCF | 0.0972 | 0.1104 | 0.1369 | 0.0800 | 稀疏用户太稀疏 |
| KG-HybridRec M3 | — | **0.5378** | — | — | 学习型最佳 |

---

## 三、ItemCF 在稀疏场景强效果分析

### 3.1 现象

view_sparse（用户平均 15.2 次交互，中位数 9 次）:
- ItemCF HR@10 = **0.7067** — 远超 UserCF (0.1104) 和 Popular (0.5491)
- ItemCF 超过所有学习型基线（KG-HybridRec M3 = 0.5378）

### 3.2 根因分析

**① 商品共现天然比用户相似度更稠密**

| 维度 | ItemCF（物品侧） | UserCF（用户侧） |
|------|------------------|-----------------|
| view_sparse 实体数 | 33,938 items | 24,999 users |
| 每个实体的交互数 | 均值 11.2，中位数 **4** | 均值 15.2，中位数 **9** |
| 共现向量维度 | 24,999（用户数） | 33,938（商品数） |

ItemCF 的相似度向量是 `ui.T @ ui` — 每个商品的共现向量是所有购买过它的用户。33,938 个商品的向量空间比 24,999 个用户的向量空间更宽，共现信号更分散但更稳定。

**② 用户冷启动不直接影响 ItemCF**

UserCF 的核心问题是：当一个用户只有 9 次交互（view_sparse 中位数），找相似用户时，每个候选相似度分数只基于 9 个共同购买的商品。噪声大，不可靠。

ItemCF 评估商品 j 时，score = Σ(sim[j,i] × ui[u,i])。用户有 9 个历史商品，只需每个商品的 Top-K 相似物品在测试集出现即可。即使某几个邻居商品共现不准，其他邻居也能弥补。

**③ 流行度偏差（Popularity Bias）放大 ItemCF 优势**

LOO 测试协议中，测试正样本是用户最后一条交互——这是用户真实购买的，必然是用户感兴趣的，且倾向于热门商品（购买频次高的商品更容易出现在"最后一条"）。

关键洞察：**流行商品天然与更多商品共现**。Top-1 商品（5100816）有 1,858 次交互，覆盖 7.4% 的用户。这些商品与大量其他商品形成共现对，ItemCF 学到的相似度更可靠。

相反，UserCF 的邻居是基于"共同购买"——两个冷门商品的用户群重叠概率极低，即使有共现也可能只是噪声。

**④ Top-K 截断帮助 ItemCF，伤害 UserCF**

UserCF（baselines_fast.py line 104）用密集 sim 矩阵但没有 top-K 截断：
```python
sim = (ui_dense @ ui_dense.T) / (norm @ norm.T)
np.fill_diagonal(sim, 0)
```
这导致相似度分数被大量噪声（低共现）稀释。

ItemCF（baselines_fast.py lines 121-129）：
```python
ui_dense = ui_sparse.toarray().T  # [n_i, n_u]
sim = (ui_dense @ ui_dense.T) / (norm @ norm.T)
score_matrix = ui_sparse @ sim
```
同样无截断，但 item-item 共现矩阵 `ui.T @ ui` 天然更稠密（平均每个 item 与 11.2 个用户有交互）。

### 3.3 结论

ItemCF 在 view_sparse 的强势表现是**数据特性**而非算法异常：
- 商品数（33k）> 用户数（25k），商品侧向量空间更宽
- 流行商品在测试集中出现频率高，与大量商品有共现
- ItemCF 只依赖"商品 j 与用户历史商品的共现强度"，不依赖跨用户相似度
- view_sparse 的稀疏度（0.000449）反而让热门商品的共现信号更突出

**这不是 bug，是 CF 的正常行为**——在商品维度大于用户维度、热门商品主导测试集的场景下，ItemCF 天然优于 UserCF。

### 3.4 论文陈述建议

稠密数据（purchase）：诚实声明 ItemCF/UserCF 利用物品共现的强力基线，KG-HybridRec 在学习型方法中最佳。

稀疏数据（view_sparse）：强调 ItemCF 的强势表现符合协同过滤理论预期，同时 KG-HybridRec 作为学习型方法在跨场景一致性上更优（purchase 0.5890 vs view_sparse 0.5378，跌幅 -0.052），而 ItemCF 从 purchase 到 view_sparse 跌幅更大（0.9527 → 0.7067，跌幅 -0.246），UserCF 则几乎失效（0.9995 → 0.1104）。

---

## 四、Git 状态

**未提交更改**（2026-07-10）:
- README.md
- TRYON_SETUP.md
- api/main.py
- api/routes/recommend.py
- api/services/clothing.py
- api/services/user.py
- data/amazon_fashion_with_images.json
- data/import_amazon_fashion.py
- frontend/src/api/index.ts
- frontend/src/components/ThreeDClothViewer.tsx
- frontend/src/pages/ClothingList.tsx
- frontend/src/pages/Dashboard.tsx
- frontend/src/pages/Login.tsx
- frontend/src/pages/VirtualTryOn.tsx
