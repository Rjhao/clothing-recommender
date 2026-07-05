# 基于知识图谱的智能个性化服装推荐系统

## 项目概述

基于知识图谱与混合推荐的智能服装推荐系统（KG-HybridRec算法），支持个性化推荐、搭配推荐、虚拟试穿功能。

---

## 开发计划

### 整体规划

| 阶段 | 内容 | 优先级 | 状态 |
|-----|------|--------|------|
| 第一阶段 | 首页重构 | P0 | ✅ 完成 |
| 第二阶段 | 商品列表优化 | P0 | ✅ 完成 |
| 第三阶段 | 商品详情页 | P0 | ✅ 完成 |
| 第四阶段 | 个性化推荐页 | P1 | ✅ 完成 |
| 第五阶段 | 搭配推荐页 | P1 | ✅ 完成 |
| 第六阶段 | 用户中心 | P1 | ✅ 完成 |
| 第七阶段 | AI虚拟试穿 | P0 | ✅ 完成 |

---

### 任务清单

#### ✅ 已完成
1. 基础架构搭建（Neo4j、MySQL、Redis）
2. 数据导入（Amazon 4872件服装）
3. 知识图谱构建
4. 后端API开发
5. 基础前端页面
6. **首页重构** - 任务 #1 ✅
   - [x] 轮播Banner
   - [x] 猜你喜欢推荐
   - [x] 分类快捷入口
   - [x] 热门榜单
   - [x] 新品上市
7. **商品列表优化** - 任务 #2 ✅
   - [x] 分类筛选
   - [x] 属性过滤（风格/价格）
   - [x] 排序功能
   - [x] 无限滚动
   - [x] 商品卡片优化
8. **商品详情页** - 任务 #3 ✅
   - [x] 图片画廊
   - [x] 规格选择
   - [x] 商品信息展示
   - [x] 搭配推荐
   - [x] 相似商品
   - [x] 评价列表
9. **个性化推荐页** - 任务 #4 ✅
   - [x] 推荐理由展示
   - [x] 猜你喜欢
   - [x] 相似用户推荐
   - [x] 足迹历史
10. **搭配推荐页** - 任务 #5 ✅
    - [x] 场景搭配
    - [x] 智能搭配生成
    - [x] 搭配收藏
11. **用户中心** - 任务 #6 ✅
    - [x] 用户信息
    - [x] 我的收藏
    - [x] 浏览历史
    - [x] 风格偏好设置
12. **AI虚拟试穿** - 任务 #7 ✅
    - [x] 3D虚拟人物（Three.js + VRM）
    - [x] 服装选择
    - [x] 虚拟试穿预览
    - [x] 性别切换

---

## 页面功能详细规划

### 一、首页/发现页

| 功能 | 优先级 | 说明 |
|-----|--------|------|
| 轮播Banner | P0 | 活动、专题推广 |
| 猜你喜欢 | P0 | 个性化商品推荐 |
| 分类快捷入口 | P0 | 6大分类快速导航 |
| 热门榜单 | P1 | 销量/收藏TOP榜 |
| 新品上市 | P1 | 最新上架商品 |
| 专题搭配 | P1 | 场景/节日专题 |
| 快捷功能入口 | P1 | 虚拟试穿、智能推荐入口 |
| 浏览历史 | P2 | 最近浏览记录 |

### 二、商品列表页

| 功能 | 优先级 | 说明 |
|-----|--------|------|
| 分类筛选 | P0 | 二级分类筛选 |
| 属性过滤 | P0 | 风格/价格/颜色/尺码 |
| 排序 | P0 | 销量/价格/新品/推荐 |
| 搜索 | P0 | 关键词搜索 |
| 商品卡片 | P0 | 图片+价格+销量 |
| 无限滚动 | P1 | 懒加载更多 |
| 查看模式 | P2 | 列表/网格切换 |

### 三、商品详情页

| 功能 | 优先级 | 说明 |
|-----|--------|------|
| 图片画廊 | P0 | 多图展示/放大 |
| 商品信息 | P0 | 名称/价格/属性 |
| 规格选择 | P0 | 颜色/尺码选择 |
| 搭配推荐 | P1 | 配套商品推荐 |
| 相似商品 | P1 | 同款/相似推荐 |
| 评价展示 | P1 | 用户评价列表 |
| 虚拟试穿入口 | P1 | 进入试穿页面 |
| 收藏/购买 | P0 | 快捷操作 |

### 四、个性化推荐页

| 功能 | 优先级 | 说明 |
|-----|--------|------|
| 推荐理由 | P0 | 显示推荐原因 |
| 猜你喜欢 | P0 | 基于行为的推荐 |
| 相似用户推荐 | P1 | 协同过滤推荐 |
| 搭配套装 | P1 | 一整套搭配推荐 |
| 足迹历史 | P1 | 最近浏览历史 |
| 风格测试入口 | P2 | 偏好测试问卷 |

### 五、搭配推荐页

| 功能 | 优先级 | 说明 |
|-----|--------|------|
| 场景搭配 | P0 | 上班/约会/休闲/运动 |
| 智能搭配 | P1 | AI生成搭配方案 |
| 搭配收藏 | P1 | 收藏的搭配方案 |
| 分享搭配 | P2 | 分享到社交平台 |

### 六、虚拟试穿页

| 功能 | 优先级 | 说明 |
|-----|--------|------|
| 虚拟形象选择 | P0 | 男/女/自定义 |
| 服装选择 | P0 | 分类选择服装 |
| AI试穿 | P0 | 真正穿到身上 |
| 试穿效果预览 | P0 | 360度查看 |
| 保存/分享 | P1 | 保存效果图 |

### 七、用户中心

| 功能 | 优先级 | 说明 |
|-----|--------|------|
| 头像/昵称 | P0 | 用户信息 |
| 我的收藏 | P0 | 收藏的商品 |
| 浏览历史 | P1 | 最近浏览 |
| 风格偏好 | P1 | 偏好测试/设置 |
| 体型设置 | P2 | 身高体重肤色 |

---

## 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                         前端 (React)                        │
│   首页 | 商品列表 | 商品详情 | 推荐 | 搭配 | 试穿 | 用户中心    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    API网关 (FastAPI)                        │
│   /clothing | /recommend | /outfit | /tryon | /user       │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│    Neo4j     │     │    MySQL     │     │    Redis     │
│  知识图谱    │     │  业务数据    │     │    缓存      │
└───────────────┘     └───────────────┘     └───────────────┘
```

---

## 技术栈

| 模块 | 技术 | 用途 |
|-----|------|------|
| 前端 | React 18 + TypeScript | 用户界面 |
| UI库 | Ant Design 5 | 组件库 |
| 3D渲染 | Three.js | 虚拟试穿 |
| 后端 | FastAPI + Python | API服务 |
| 图数据库 | Neo4j 5 | 知识图谱存储 |
| 关系数据库 | MySQL 8 | 业务数据存储 |
| 缓存 | Redis 7 | 推荐结果缓存 |
| 虚拟试穿 | ACGPN (PyTorch) | AI试穿算法 |

---

## 项目结构

```
clothing-recommender/
├── api/                      # FastAPI 应用
│   ├── models/               # Pydantic 数据模型
│   ├── routes/               # API 路由
│   ├── services/             # 业务逻辑服务
│   └── database.py           # 数据库连接配置
├── kg/                       # 知识图谱模块
│   ├── graph/
│   │   ├── builder.py       # 图谱构建
│   │   ├── query.py         # 图查询
│   │   └── init.py          # 初始化
│   └── embedding/           # 图嵌入算法
├── recommender/              # 推荐引擎
│   ├── collaborative/        # 协同过滤
│   ├── knowledge/           # 知识推荐
│   └── hybrid/              # 混合推荐
├── data/                     # 数据处理
│   ├── crawler/             # 数据爬取与清洗
│   ├── import_amazon_fashion.py  # Amazon数据导入
│   └── importer.py          # 数据导入脚本
├── frontend/                  # React 前端
│   ├── src/
│   │   ├── components/      # 通用组件
│   │   │   └── ThreeDClothViewer.tsx  # 3D虚拟试穿组件
│   │   ├── pages/          # 页面组件
│   │   │   ├── Dashboard.tsx    # 首页
│   │   │   ├── ClothingList.tsx # 商品列表
│   │   │   ├── ClothingDetail.tsx # 商品详情
│   │   │   ├── Recommendations.tsx  # 推荐页面
│   │   │   ├── Outfit.tsx    # 搭配推荐
│   │   │   ├── VirtualTryOn.tsx  # 虚拟试穿
│   │   │   ├── UserCenter.tsx  # 用户中心
│   │   │   └── ...
│   │   └── App.tsx
│   └── package.json
├── ACGPN/                    # 虚拟试穿模块
│   ├── ACGPN_inference/    # 推理代码
│   ├── checkpoints/          # 模型权重
│   └── README.md
└── .env                     # 环境变量配置
```

---

## 当前系统状态

### ✅ 已完成功能
- [x] Neo4j 图数据库部署（Docker）
- [x] MySQL 数据库部署（Docker）
- [x] Redis 缓存部署（Docker）
- [x] Amazon 服饰数据集导入（4872件服装）
- [x] 知识图谱构建
- [x] FastAPI 服务 (端口 8000)
- [x] 健康检查接口 `/health`
- [x] 服装列表接口 `/clothing/`
- [x] 服装详情接口 `/clothing/{id}`
- [x] 推荐接口 `/recommend/{user_id}`
- [x] 用户交互接口
- [x] React + TypeScript + Vite 项目
- [x] Ant Design UI 组件库
- [x] **首页重构** - 轮播Banner、猜你喜欢、分类入口、热门榜单、新品上市
- [x] **商品列表优化** - 分类筛选、属性过滤、排序、无限滚动、网格/列表视图
- [x] **商品详情页** - 图片画廊、规格选择、商品信息、搭配推荐、相似商品、评价列表
- [x] **个性化推荐页** - 推荐理由、猜你喜欢、相似用户、足迹历史、搭配套装
- [x] **搭配推荐页** - 场景搭配（上班/约会/休闲/运动）、AI智能生成、搭配收藏
- [x] **用户中心** - 用户信息、风格偏好、颜色偏好、消费偏好、浏览历史、我的收藏
- [x] **虚拟试穿页面** - 3D虚拟人物（Three.js + VRM）、服装选择、男女形象切换
- [x] 商品管理页面
- [x] 用户管理页面
- [x] 知识图谱可视化

### 🔄 开发中
- 无

### ⏳ 待开发
- [ ] AI虚拟试穿（ACGPN）- 后端推理服务
- [ ] 评价功能
- [ ] 社交分享
- [ ] AR预览

---

## 快速开始

### 1. 启动基础服务

```bash
# 检查Docker容器状态
docker ps

# 启动容器（如未启动）
docker-compose up -d
```

### 2. 启动后端

```bash
cd D:/clothing-recommender
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 3. 启动前端

```bash
cd D:/clothing-recommender/frontend
npm run dev -- --host 0.0.0.0 --port 3000
```

### 4. 访问地址

- 前端页面：http://localhost:3000
- API 文档：http://localhost:8000/docs
- Neo4j Browser：http://localhost:7474

---

## 环境变量

```env
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=root123
MYSQL_DATABASE=clothing_recommender

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

SECRET_KEY=your-secret-key-change-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

---

## 开发团队

- 论文题目：基于知识图谱的智能个性化服装推荐系统
- 算法：KG-HybridRec
- 虚拟试穿：ACGPN

---

## 更新日志

### 2026-07-05

**全项目代码审查与缺陷修复（近 50 项，分四轮）**

**一、致命错误（应用无法启动 / 脚本崩溃）— 5 项**
- `api/models/user.py`：`UserPreference` 引用 `StyleEnum/SeasonEnum/OccasionEnum` 但未导入 → 启动 `NameError`
- `data/crawl_wikipedia.py:138`：f-string 替换字段未闭合 → `SyntaxError`
- `data/crawler/tb_scraper.py:109`：正则捕获组未闭合 → `re.error`（淘宝抓取永远返回空）
- `data/update_images_to_db.py:76`：字符串错误解包为 `(pid, url)` → `ValueError`
- `data/crawler/exporter.py:307`：用了 `defaultdict` 但未导入 → `NameError`

**二、后端 HIGH（安全 / 资源）**
- `api/main.py`：CORS `*`+credentials（无效配置）→ `credentials=False`；每请求新建 SQLAlchemy engine → 共享 engine + `Depends(get_db)`；session 异常不关闭 → `try/finally`；同步 I/O 的 `async` 端点 → `def`（线程池）
- `api/routes/interactions.py`：**IDOR 漏洞**——`user_id` 来自请求体可伪造 → 改从 JWT token 取 `current_user.id`；history 加权限校验；500 不再泄露内部错误
- `api/routes/auth.py`：session 泄漏 → `try/finally`；新增 email 唯一性检查；密码校验移到 session 关闭前
- `api/routes/recommend.py`：`exclude_viewed` 参数被接受但从未传递 → 透传到 engine；修复错误泄露
- `api/database.py`：新增共享 MySQL engine + `get_db` 依赖 + `get_db_session`

**三、后端 MEDIUM**
- `api/services/user_profile.py`：`item["s"]["id"]` KeyError + `sum(None)` 崩溃 → `.get()` + `or 0`
- `api/services/cache.py`：缓存 key 缺 `offset`（分页返回首页缓存）→ 补 offset；`KEYS`（阻塞 Redis）→ `SCAN`

**四、数据脚本**
- `data/import_amazon_fashion.py`：`exporter.mysql_conn` 从未赋值 → 商品**静默不导出**；补赋值 + `conn.close` 移入 `finally`
- `data/crawler/base.py:120`：`quote()` 后又被 requests 二次编码 → 关键词乱码
- `data/crawler/washer.py`：HTML 实体替换在 `text_clean_pattern` 之后（`&`/`;` 已被删）→ 死代码致商品名残留 `amp`/`nbsp` → 调整顺序

**五、前端（9 项）**
- `frontend/src/api/index.ts`：加 JWT 请求拦截器（所有鉴权请求自动附 `Authorization`）
- `ClothingList.tsx`：「加载更多」永远抓第 1 页 → 无限重复 → 加 offset 分页
- `Dashboard.tsx`：商品点击跳错路由（`/clothing?detail=`）→ 改 `/clothing/:id`
- `ThreeDViewer.tsx`：`scale` 在 effect 依赖 → 每次滚轮重建场景；每帧 `setRotation` → 60 次/秒重渲染；几何体/材质/纹理不释放 → 全修
- `ThreeDClothViewer.tsx`：GLTF 异步加载竞态；模型资源不释放；`AnimationMixer` 从不 update → 全修
- `ClothingDetail.tsx` / `Recommendations.tsx`：硬编码 `userId=1` → 改用 `useAuth()` 真实用户
- `VirtualTryOn.tsx`：`setTimeout` 无清理 → 卸载后 setState → ref 清理
- `AuthContext.tsx`：token 在 `getMe` 前存 → 失败时遗留 → 移到成功后
- 清理 4 处预存 `React` 未使用导入（阻塞 `npm run build`）

**六、图谱属性不一致（"图片不显示"根因）**

查询层（`query.py`）和主 API 用整数 `id`，但 `exporter.py` 建节点用字符串 `product_id`、`sync_images_to_neo4j.py` 按 `product_id` 匹配、`clothing.py` 读 `Image_url`（大写）——三套 schema 互不相交。
- `api/services/clothing.py`：加 `_image_url()`/`_clothing_id()` 助手，对两套 schema 都兼容
- `data/sync_images_to_neo4j.py`：重写——按 `id` 或 `product_id` 都能匹配、写入小写 `image_url`、凭据读 `settings`
- `data/crawler/exporter.py`：`export_to_neo4j` 弃用（schema 不符，图谱统一走 `import_to_kg`）
- `data/migrate_clothing_schema.py`（**新增**）：一次性幂等迁移，补 `id`、统一 `image_url`、`BELONGS_TO → IN_CATEGORY`

**七、安全 + 配置**
- `tryon_api.py`：上传文件用子串匹配（可越权读任意用户图）→ 改精确 id + glob；加 10MB 上限 + 扩展名白名单；CORS 修正；路径改绝对
- `.env`：`SECRET_KEY` 占位符 → 真随机值（⚠️ 已签发 JWT 失效，需重新登录）

**八、ML 算法核心（5 CRITICAL + 2 HIGH + 6 MEDIUM + 1 连带）**
- CRITICAL：`graph_contrastive` 对角线先 mask 再取 `pos_sim` 致 loss=inf（且 `.clone()` 修视图被改 bug）；`graph_contrastive` `torch.where` 形状不匹配；`multimodal` 用 `F` 但未导入；`hybrid/engine` `int/list` 相除 TypeError；`gat_fusion` matmul 转置错位
- HIGH：`kg_embedding` cat 维度不匹配；`query.COMPUTE_RECOMMEND_SCORE` 笛卡尔积致分数虚高 → 改相关子查询；`builder` 四处 `MERGE {score}` 带属性致重复边
- MEDIUM：`query.FIND_SIMILAR_USERS` 补 `style_overlap`（**让 CF 真正生效**）+ LIMIT；`hybrid/engine:61` KeyError；`gat_fusion` 孤立节点 softmax NaN → 加自环；`hybrid_rec` BPR 正负同嵌入零梯度；`hybrid_rec` `torch.roll` 对比样本错配（训练时活跃，`use_improvement_b=True`）→ 特征 dropout；`collaborative` squeeze 单候选 IndexError
- 连带：`collaborative/engine` 模块级缺 `import torch`（**NCF 类从未能运行**，现修复可用）

**验证**：全部 `py_compile` 通过；`import api.main` 成功；`pytest 8/8 通过`；`tsc --noEmit` 零错误；ML 关键修复通过功能测试（对比损失 inf→2.981、GAT 孤立节点不再 NaN、NCF 单候选正常）。

**待手动操作**：Neo4j 容器启动后跑一次 `python -m data.migrate_clothing_schema`（幂等，统一 schema）。代码层面对两套 schema 已兼容，即使不跑迁移（只要当初是 `import_to_kg` 路径导入）图片和相似推荐也会正常。

### 2026-06-30
- 完成全部7个开发阶段的前端优化
- 首页重构（轮播Banner、猜你喜欢、分类入口）
- 商品列表优化（筛选、排序、无限滚动）
- 商品详情页（图片画廊、规格选择、搭配推荐）
- 个性化推荐页（推荐理由、相似用户、足迹历史）
- 搭配推荐页（场景搭配、AI生成、收藏功能）
- 用户中心（风格偏好、颜色偏好、消费偏好）
- 虚拟试穿页面完善（3D虚拟人物、服装选择）

### 2026-06-29
- 完成数据导入（Amazon 4872件服装）
- 完成Neo4j知识图谱构建
- 完成基础API服务
- 完成前端基础页面
- 完成3D虚拟试穿功能
