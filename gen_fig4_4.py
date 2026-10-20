# -*- coding: utf-8 -*-
"""生成图 4-4:训练数据削减下的冷用户性能鲁棒性(极简折线图)。"""
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

# ---- 数据 ----
x = [0, 1, 2]
xticks = ['1.0', '0.5', '0.2']

series = [
    ('KG-HybridRec', [0.594, 0.591, 0.589], '#ED7D31', 'o'),
    ('LightGCN',     [0.587, 0.402, 0.272], '#5B9BD5', 's'),
    ('BPR-MF',       [0.519, 0.318, 0.245], '#A5A5A5', '^'),
    ('RippleNet',    [0.461, 0.373, 0.330], '#70AD47', 'D'),
    ('KGCL',         [0.525, 0.282, 0.224], '#7030A0', 'v'),
]

# ---- 画布 ----
fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# ---- 折线 ----
for name, vals, color, marker in series:
    ax.plot(x, vals,
            color=color, linewidth=1.8, marker=marker,
            markersize=8, markerfacecolor='white',
            markeredgecolor=color, markeredgewidth=1.8,
            label=name, zorder=3)

# ---- 数据标签:三个 x 都标,左右错开避免重叠 ----
for xi in x:
    pts = []
    for name, vals, color, marker in series:
        pts.append((vals[xi], name, color))
    pts_sorted = sorted(pts, key=lambda p: -p[0])  # 由高到低
    for rank, (v, name, color) in enumerate(pts_sorted):
        dx = -0.10 if rank % 2 == 0 else 0.10
        ha = 'right' if rank % 2 == 0 else 'left'
        ax.text(xi + dx, v, f'{v:.3f}',
                fontsize=9, color=color, ha=ha, va='center', zorder=4,
                bbox=dict(boxstyle='round,pad=0.18', fc='white',
                          ec='none', alpha=0.9))

# ---- 坐标轴范围 ----
ax.set_xlim(-0.3, 2.3)
ax.set_ylim(0.18, 0.66)

# ---- X 轴 ----
ax.set_xticks(x)
ax.set_xticklabels(xticks, fontsize=11, color='#333333')
ax.set_xlabel('训练数据比例', fontsize=11, color='#333333', labelpad=8)

# ---- Y 轴 ----
ax.set_yticks([0.20, 0.30, 0.40, 0.50, 0.60])
ax.set_yticklabels([f'{v:.2f}' for v in [0.20, 0.30, 0.40, 0.50, 0.60]],
                   fontsize=10, color='#333333')
ax.set_ylabel('冷用户 HR@10', fontsize=11, color='#333333', labelpad=8)

# ---- 网格 + 边框 ----
ax.grid(axis='y', linestyle='--', linewidth=0.6, color='#dddddd', zorder=1)
ax.set_axisbelow(True)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
for s in ['left', 'bottom']:
    ax.spines[s].set_color('#888888')
    ax.spines[s].set_linewidth(1.0)

# ---- 图例 ----
leg = ax.legend(loc='best', frameon=True, fontsize=9,
                framealpha=0.95, edgecolor='#cccccc')
leg.get_frame().set_linewidth(0.8)

plt.subplots_adjust(left=0.10, right=0.97, top=0.97, bottom=0.12)

# ---- 保存 ----
out_dir = r'D:/clothing-recommender/figures'
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, '图4-4_数据稀疏度鲁棒性.png')
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
print(f'Saved: {out_path}')
plt.close(fig)