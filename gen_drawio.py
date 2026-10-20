# -*- coding: utf-8 -*-
"""生成第3章 5 张图的 draw.io 文件(.drawio,多页)。
每页一张图:图3-1构建流程 / 图3-2本体结构 / 图3-3 KG-HybridRec框架 / 图3-4 CAGF数据流 / 图3-5三项改进位置。
输出: D:/clothing-recommender/第3章_图.drawio
"""
import html

# ---- draw.io 基础构件 ----
def cell(node_id, value, style, x, y, w, h, parent="1"):
    v = html.escape(value) if value else ""
    return (f'<mxCell id="{node_id}" value="{v}" style="{style}" vertex="1" parent="{parent}">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

def edge(eid, src, tgt, label="", style="endArrow=classic;html=1;rounded=0;", parent="1"):
    v = html.escape(label) if label else ""
    s = (f'<mxCell id="{eid}" value="{v}" style="{style}" edge="1" parent="{parent}" source="{src}" target="{tgt}">'
         f'<mxGeometry relative="1" as="geometry"/></mxCell>')
    return s

# 颜色样式
BOX_BLUE   = "rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;fontSize=11;"
BOX_GREEN  = "rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;fontSize=11;"
BOX_ORANGE = "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffe6cc;strokeColor=#d79b00;fontSize=11;"
BOX_PURPLE = "rounded=1;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;fontSize=11;"
BOX_RED    = "rounded=1;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;fontSize=11;"
BOX_CYAN   = "rounded=1;whiteSpace=wrap;html=1;fillColor=#b1ddf0;strokeColor=#10739e;fontSize=11;"
BOX_YELLOW = "rounded=1;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=11;"
BOX_GREY   = "rounded=1;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#666666;fontSize=10;"
CIRCLE     = "ellipse;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=10;"
TITLE      = "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=13;fontStyle=1;"
ARROW      = "endArrow=classic;html=1;rounded=0;strokeWidth=2;"
ARROW_DASH = "endArrow=classic;html=1;rounded=0;dashed=1;strokeWidth=2;"
ARROW_THICK= "endArrow=classic;html=1;rounded=0;strokeWidth=3;strokeColor=#b85450;"

def page(name, pid, cells_xml):
    return f'<diagram name="{name}" id="{pid}"><mxGraphModel dx="1400" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1600" pageHeight="1200" math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>{cells_xml}</root></mxGraphModel></diagram>'

# ============ 图 3-1 构建流程 (6阶段流水线) ============
def fig_3_1():
    c = ""
    stages = [
        ("① 数据采集", "Amazon eCommerce JSON\n(product_id / name / category\n/ brand / price / styles / season)", BOX_BLUE),
        ("② 规范化清洗", "clean_category()\nclean_style()\nclean_season()\n(非法值→默认枚举)", BOX_GREEN),
        ("③ 本体映射", "字段 → 节点标签与关系\n(Category/Style/Brand\n等本体映射)", BOX_ORANGE),
        ("④ 节点关系导入", "MERGE 幂等写入\nUser/Clothing/Category/Style/Brand\nIN_CATEGORY/HAS_STYLE/FROM_BRAND", BOX_PURPLE),
        ("⑤ 关系生成", "MATCHES_WITH(score=0.6)\n跨品类互补\nSIMILAR_TO(score=0.7)\n同风格相似", BOX_RED),
        ("⑥ 表示学习", "TransE 离线嵌入\n(h+r≈t, dim=100)\n实体/关系向量", BOX_CYAN),
    ]
    x0, y0, w, h, gap = 40, 120, 230, 170, 30
    ids = []
    for i, (t, d, st) in enumerate(stages):
        x = x0 + i*(w+gap)
        cid = f"f1_s{i}"
        ids.append(cid)
        c += cell(cid, f"{t}\n\n{d}", st, x, y0, w, h)
        c += cell(f"f1_t{i}", f"图 3-1 阶段 {i+1}", TITLE if False else "", x, y0-40, w, 30)
    # 箭头
    for i in range(len(ids)-1):
        c += edge(f"f1_e{i}", ids[i], ids[i+1], "", ARROW)
    c += cell("f1_title", "图 3-1  服装领域知识图谱构建总体流程", TITLE, 40, 40, 1500, 40)
    c += cell("f1_note", "数据(蓝)→清洗(绿)→本体(橙)→导入(紫)→关系生成(红)→表示学习(青)", "text;html=1;strokeColor=none;fillColor=none;fontSize=10;", 40, 330, 1500, 20)
    return c

# ============ 图 3-2 本体结构 (辐射图) ============
def fig_3_2():
    c = ""
    c += cell("f2_title", "图 3-2  服装领域知识图谱本体结构图", TITLE, 40, 40, 1500, 40)
    # 中心 Clothing
    cx, cy, cw, ch = 700, 500, 220, 130
    c += cell("f2_cloth", "Clothing\n\nid / name / category / brand\n/ color / style / season / price\n/ image_url / popularity", BOX_BLUE, cx, cy, cw, ch)
    # User 左上
    c += cell("f2_user", "User\n\nid / username / gender\n/ age / height / weight\n/ body_shape / skin_tone", BOX_GREEN, 200, 200, 200, 130)
    # Brand 右上
    c += cell("f2_brand", "Brand\nname", CIRCLE, 1200, 200, 140, 80)
    # Category 左下
    c += cell("f2_cat", "Category\nid / name", CIRCLE, 200, 780, 140, 80)
    c += cell("f2_subcat", "子品类\n(L2/L3)", BOX_GREY, 380, 800, 120, 60)
    # Style 右下
    c += cell("f2_style", "Style\n\nid / name\n/ description", CIRCLE, 1200, 780, 140, 90)
    # 另一件 Clothing (右)
    c += cell("f2_cloth2", "Clothing'\n(另一件)", BOX_BLUE, 1300, 500, 160, 90)
    # BodyShape / Occasion (下方小)
    c += cell("f2_bs", "BodyShape", BOX_GREY, 560, 780, 110, 50)
    c += cell("f2_occ", "Occasion", BOX_GREY, 760, 780, 110, 50)
    # 关系边
    c += edge("f2_e1","f2_user","f2_cloth","BOUGHT\nVIEWED\nRATED", "endArrow=classic;html=1;fontSize=9;strokeWidth=2;strokeColor=#b85450;")
    c += edge("f2_e2","f2_user","f2_style","PREFERS", "endArrow=classic;html=1;fontSize=9;strokeColor=#82b366;")
    c += edge("f2_e3","f2_cloth","f2_brand","FROM_BRAND", "endArrow=classic;html=1;fontSize=9;")
    c += edge("f2_e4","f2_cloth","f2_cat","IN_CATEGORY", "endArrow=classic;html=1;fontSize=9;")
    c += edge("f2_e5","f2_cat","f2_subcat","层级", "endArrow=classic;html=1;fontSize=9;dashed=1;")
    c += edge("f2_e6","f2_cloth","f2_style","HAS_STYLE", "endArrow=classic;html=1;fontSize=9;")
    c += edge("f2_e7","f2_cloth","f2_cloth2","MATCHES_WITH 0.6\nSIMILAR_TO 0.7", "endArrow=classic;startArrow=classic;html=1;fontSize=9;strokeWidth=2;")
    c += edge("f2_e8","f2_cloth","f2_bs","HAS_BODY_SHAPE", "endArrow=classic;html=1;fontSize=9;dashed=1;")
    c += edge("f2_e9","f2_cloth","f2_occ","SUITABLE_FOR", "endArrow=classic;html=1;fontSize=9;dashed=1;")
    # 图例
    c += cell("f2_leg", "图例:矩形=实体节点  圆形=属性/分类节点  实线=强语义关系  虚线=弱/衍生关系  边上数字=score", "text;html=1;strokeColor=none;fillColor=none;fontSize=10;", 200, 920, 1300, 30)
    return c

# ============ 图 3-3 KG-HybridRec 总体框架 ============
def fig_3_3():
    c = ""
    c += cell("f3_title", "图 3-3  KG-HybridRec 算法总体框架", TITLE, 40, 40, 1500, 40)
    # 底层三路
    c += cell("f3_cf", "CF 分支\nUser/Item Embedding\n(BPR-MF)\ncf_confidence=tanh(n_u/10)", BOX_GREEN, 80, 260, 260, 140)
    c += cell("f3_kg", "KG 分支\n异构图(4类边)\nRippleNet 注意力聚合\n(hop1/hop2)\nTransR 约束 h+r≈t", BOX_BLUE, 460, 260, 260, 140)
    c += cell("f3_mm", "MM 分支\nCLIP + BERT\n图像/文本特征\n(主实验关闭)", BOX_ORANGE, 840, 260, 260, 140)
    # 中部融合
    c += cell("f3_gate", "自适应门控 CAGF\n(改进A)\nsoftmax 加权 + 冷启动调整", BOX_PURPLE, 460, 480, 420, 110)
    # 上部打分
    c += cell("f3_fused", "融合嵌入 h_fused", BOX_RED, 560, 640, 220, 50)
    c += cell("f3_mlp", "predict MLP", BOX_GREY, 560, 720, 220, 50)
    c += cell("f3_score", "匹配分数 ŷ", BOX_YELLOW, 560, 800, 220, 50)
    c += cell("f3_bpr", "BPR 损失\nL_BPR", BOX_RED, 880, 780, 200, 90)
    # 辅助: 改进B/C
    c += cell("f3_cl", "改进B\n图对比学习\nL_CL\n(训练独立GAT\n不进 predict_score)", "rounded=1;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;dashed=1;fontSize=10;", 1200, 480, 260, 140)
    c += cell("f3_ex", "改进C\n注意力路径解释\n(不影响打分)", "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffe6cc;strokeColor=#d79b00;dashed=1;fontSize=10;", 1200, 660, 260, 110)
    # 箭头:三路->gate
    c += edge("f3_e1","f3_cf","f3_gate","",ARROW)
    c += edge("f3_e2","f3_kg","f3_gate","",ARROW)
    c += edge("f3_e3","f3_mm","f3_gate","",ARROW_DASH)  # MM虚线(关闭)
    c += edge("f3_e4","f3_gate","f3_fused","",ARROW)
    c += edge("f3_e5","f3_fused","f3_mlp","",ARROW)
    c += edge("f3_e6","f3_mlp","f3_score","",ARROW)
    c += edge("f3_e7","f3_score","f3_bpr","BPR",ARROW)
    c += edge("f3_e8","f3_cl","f3_bpr","λ_CL",ARROW_DASH)
    c += edge("f3_e9","f3_score","f3_ex","解释",ARROW_DASH)
    c += cell("f3_note", "说明:主模型路径融合走自适应门控(改进A);GATFusion仅消融路径。改进B的L_CL进训练损失但训练独立GAT,不进predict_score;改进C纯解释。", "text;html=1;strokeColor=none;fillColor=none;fontSize=9;align=left;", 80, 920, 1400, 30)
    return c

# ============ 图 3-4 CAGF 数据流 ============
def fig_3_4():
    c = ""
    c += cell("f4_title", "图 3-4  改进A 自适应门控融合(CAGF)数据流", TITLE, 40, 40, 1500, 40)
    # 底层输入
    c += cell("f4_cf", "h_cf", BOX_GREEN, 60, 160, 120, 60)
    c += cell("f4_kg", "h_kg", BOX_BLUE, 200, 160, 120, 60)
    c += cell("f4_mm", "h_mm", BOX_ORANGE, 340, 160, 120, 60)
    c += cell("f4_nu", "用户交互次数\nn_u", BOX_YELLOW, 560, 160, 140, 60)
    # 中层门控计算
    c += cell("f4_concat", "三路拼接", BOX_GREY, 80, 290, 200, 50)
    c += cell("f4_mlp", "两层 MLP", BOX_GREY, 80, 370, 200, 50)
    c += cell("f4_g", "基础门控 g\n[g_cf, g_kg, g_mm]", BOX_PURPLE, 80, 450, 200, 60)
    c += cell("f4_cold", "cold_mask =\n1 - clip(n_u/10,0,1)\n(冷→1, 热→0)", BOX_RED, 400, 290, 220, 80)
    c += cell("f4_adj", "g ← g + cold_mask·\n[-0.5, +0.3, +0.3]\n(CF↓ KG/MM↑)", BOX_RED, 400, 410, 220, 80)
    c += cell("f4_conf", "cf_confidence =\ntanh(n_u/10)\n→ g_cf 再缩放\n(仅CF,下限0.1)", BOX_ORANGE, 720, 360, 200, 110)
    c += cell("f4_softmax", "softmax\n→ 权重 w", BOX_PURPLE, 480, 560, 200, 50)
    # 上层融合输出
    c += cell("f4_proc", "MLP_cf / MLP_kg / MLP_mm\n(投影到 d 维)", BOX_GREY, 100, 680, 300, 50)
    c += cell("f4_out", "h_fused =\nw_cf·MLP_cf(h_cf)\n+ w_kg·MLP_kg(h_kg)\n+ w_mm·MLP_mm(h_mm)", BOX_RED, 460, 670, 340, 90)
    # 箭头
    c += edge("f4_e1","f4_cf","f4_concat","",ARROW)
    c += edge("f4_e2","f4_kg","f4_concat","",ARROW)
    c += edge("f4_e3","f4_mm","f4_concat","",ARROW)
    c += edge("f4_e4","f4_concat","f4_mlp","",ARROW)
    c += edge("f4_e5","f4_mlp","f4_g","",ARROW)
    c += edge("f4_e6","f4_nu","f4_cold","",ARROW)
    c += edge("f4_e7","f4_cold","f4_adj","",ARROW)
    c += edge("f4_e8","f4_g","f4_adj","加",ARROW)
    c += edge("f4_e9","f4_nu","f4_conf","",ARROW)
    c += edge("f4_e10","f4_adj","f4_softmax","",ARROW)
    c += edge("f4_e11","f4_conf","f4_softmax","g_cf缩放",ARROW)
    c += edge("f4_e12","f4_softmax","f4_out","w",ARROW)
    c += edge("f4_e13","f4_cf","f4_proc","", "endArrow=classic;html=1;strokeColor=#999999;")
    c += edge("f4_e14","f4_proc","f4_out","",ARROW)
    c += cell("f4_note", "冷用户:CF权重↓ KG/MM权重↑   |   热用户:CF权重↑  (交互数≥10时cold_mask→0)", "text;html=1;strokeColor=none;fillColor=none;fontSize=10;", 80, 800, 1000, 30)
    return c

# ============ 图 3-5 三项改进位置与作用域 ============
def fig_3_5():
    c = ""
    c += cell("f5_title", "图 3-5  三项改进在 KG-HybridRec 中的位置与作用域", TITLE, 40, 40, 1500, 40)
    # 主框架(简化)
    c += cell("f5_cf", "CF 表征", BOX_GREY, 100, 220, 120, 50)
    c += cell("f5_kg", "KG 表征", BOX_GREY, 260, 220, 120, 50)
    c += cell("f5_mm", "MM 表征", BOX_GREY, 420, 220, 120, 50)
    c += cell("f5_fuse", "融合", BOX_GREY, 260, 340, 120, 50)
    c += cell("f5_score", "打分 ŷ", BOX_GREY, 260, 440, 120, 50)
    c += cell("f5_loss", "损失 L", BOX_GREY, 420, 440, 120, 50)
    c += edge("f5_e1","f5_cf","f5_fuse","", "endArrow=classic;html=1;strokeColor=#999999;")
    c += edge("f5_e2","f5_kg","f5_fuse","", "endArrow=classic;html=1;strokeColor=#999999;")
    c += edge("f5_e3","f5_mm","f5_fuse","", "endArrow=classic;html=1;strokeColor=#999999;")
    c += edge("f5_e4","f5_fuse","f5_score","", "endArrow=classic;html=1;strokeColor=#999999;")
    c += edge("f5_e5","f5_score","f5_loss","", "endArrow=classic;html=1;strokeColor=#999999;")
    # 改进A (绿,实线,覆盖融合)
    c += cell("f5_a", "改进 A  自适应门控 CAGF\n作用于融合路径,影响 HR\n• cold_mask 依交互数\n• CF/KG/MM 动态权重", "rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;strokeWidth=3;fontSize=10;", 220, 320, 240, 40)
    c += edge("f5_ea","f5_a","f5_fuse","", "endArrow=classic;html=1;strokeColor=#82b366;strokeWidth=2;dashed=0;")
    # 改进B (蓝,虚线,作用于损失)
    c += cell("f5_b", "改进 B  图对比学习\n作用于训练损失 L_CL\n• 训练独立 GAT\n• 不进 predict_score\n• 消融 M2≈M3 对照点", "rounded=1;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;strokeWidth=2;dashed=1;fontSize=10;", 700, 320, 280, 110)
    c += edge("f5_eb","f5_b","f5_loss","λ_CL·L_CL", "endArrow=classic;html=1;strokeColor=#9673a6;dashed=1;strokeWidth=2;")
    # 改进C (橙,虚线,可解释输出)
    c += cell("f5_c", "改进 C  注意力路径解释\n作用于可解释输出\n• 不进训练损失\n• 不进 predict_score\n• 消融 M2≈M3 对照点", "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffe6cc;strokeColor=#d79b00;strokeWidth=2;dashed=1;fontSize=10;", 1100, 320, 280, 110)
    c += edge("f5_ec","f5_c","f5_score","解释", "endArrow=classic;html=1;strokeColor=#d79b00;dashed=1;strokeWidth=2;")
    # 图例
    c += cell("f5_leg1", "■ 影响精度(实线框)", "text;html=1;strokeColor=none;fillColor=none;fontSize=10;align=left;fontColor=#82b366;", 700, 480, 280, 20)
    c += cell("f5_leg2", "■ 不影响精度(虚线框):B/C 在 predict_score 中不出现", "text;html=1;strokeColor=none;fillColor=none;fontSize=10;align=left;fontColor=#9673a6;", 700, 510, 400, 20)
    c += cell("f5_leg3", "A(绿)=精度核心  B(紫)=训练正则  C(橙)=可解释", "text;html=1;strokeColor=none;fillColor=none;fontSize=10;align=left;", 700, 540, 400, 20)
    return c

# ============ 图 3-6 稠密场景主结果柱状图 ============
def fig_3_6():
    c = ""
    c += cell("f6_title", "图 3-6  稠密场景主结果对比(purchase,HR@10)", TITLE, 40, 40, 1500, 40)
    # 数据: 方法, HR@10, 颜色, 高亮
    data = [
        ("LightGCN", 0.5900, "#4472C4", False),
        ("KG-HybridRec\n(本文)", 0.5890, "#C00000", True),
        ("BPR-MF", 0.5201, "#70AD47", False),
        ("RippleNet", 0.5153, "#ED7D31", False),
        ("KGCL", 0.5151, "#7030A0", False),
    ]
    base_y = 700  # 柱底 y
    max_h = 360   # 0.60 对应高度
    val_min = 0.50
    scale = max_h / (0.60 - val_min)
    bw, gap = 130, 60
    x0 = 120
    for i,(name, val, color, hl) in enumerate(data):
        x = x0 + i*(bw+gap)
        h = int((val - val_min) * scale)
        style = f"rounded=0;whiteSpace=wrap;html=1;fillColor={color};strokeColor={color};strokeWidth={'4' if hl else '1'};fontColor=#FFFFFF;fontSize=12;fontStyle=1;"
        c += cell(f"f6_b{i}", "", style, x, base_y-h, bw, h)
        c += cell(f"f6_v{i}", f"{val:.4f}", "text;html=1;strokeColor=none;fillColor=none;fontSize=11;align=center;fontStyle=1;", x-10, base_y-h-28, bw+20, 22)
        c += cell(f"f6_n{i}", name, "text;html=1;strokeColor=none;fillColor=none;fontSize=11;align=center;", x-20, base_y+8, bw+40, 40)
    # 纵轴
    c += cell("f6_yaxis", "", "endArrow=classic;html=1;rounded=0;", 80, 300, 0, 0)  # placeholder
    c += edge("f6_ye","f6_title","f6_title","", "endArrow=classic;html=1;exitX=0;exitY=0.5;exitDx=0;exitDy=0;") if False else ""
    # 轴线 + 刻度
    c += '<mxCell id="f6_axisY" value="" style="endArrow=classic;html=1;strokeWidth=1.5;" edge="1" parent="1"><mxGeometry relative="1" as="geometry"><mxPoint x="90" y="720" as="sourcePoint"/><mxPoint x="90" y="300" as="targetPoint"/></mxGeometry></mxCell>'
    c += '<mxCell id="f6_axisX" value="" style="endArrow=classic;html=1;strokeWidth=1.5;" edge="1" parent="1"><mxGeometry relative="1" as="geometry"><mxPoint x="90" y="720" as="sourcePoint"/><mxPoint x="1100" y="720" as="targetPoint"/></mxGeometry></mxCell>'
    for v in [0.50,0.52,0.54,0.56,0.58,0.60]:
        yy = base_y - int((v-val_min)*scale)
        c += cell(f"f6_g{int(v*100)}", f"{v:.2f}", "text;html=1;strokeColor=none;fillColor=none;fontSize=9;align=right;", 40, yy-8, 45, 16)
        c += '<mxCell id="f6_gr{0}" value="" style="endArrow=none;html=1;dashed=1;strokeColor=#CCCCCC;" edge="1" parent="1"><mxGeometry relative="1" as="geometry"><mxPoint x="90" y="{1}" as="sourcePoint"/><mxPoint x="1080" y="{1}" as="targetPoint"/></mxGeometry></mxCell>'.format(int(v*100), yy)
    c += cell("f6_yl", "HR@10", "text;html=1;strokeColor=none;fillColor=none;fontSize=11;align=center;fontStyle=1;", 30, 290, 50, 20)
    # 标注
    c += cell("f6_a1", "+0.001 几乎打平", "text;html=1;strokeColor=#C00000;fillColor=none;fontSize=10;align=center;fontStyle=1;", 250, 320, 200, 20)
    c += cell("f6_a2", "+0.07 稳定优势", "text;html=1;strokeColor=#C00000;fillColor=none;fontSize=10;align=center;fontStyle=1;", 600, 430, 250, 20)
    return c

# ============ 图 3-7 冷启动分层分组柱状图 ============
def fig_3_7():
    c = ""
    c += cell("f7_title", "图 3-7  冷启动子集分层(purchase,HR@10)", TITLE, 40, 40, 1500, 40)
    models = [
        ("KG-HybridRec\n(本文)", "#C00000", [0.5944, 0.5422, 0.4452]),
        ("LightGCN", "#4472C4", [0.5873, 0.6159, 0.5753]),
        ("KGCL", "#7030A0", [0.5250, 0.5283, 0.4795]),
        ("BPR-MF", "#70AD47", [0.5195, 0.5265, 0.5068]),
        ("RippleNet", "#ED7D31", [0.4606, 0.4205, 0.3699]),
    ]
    groups = ["cold (<5)\nn=57828", "warm (5-20)\nn=6123", "heavy (>20)\nn=146"]
    base_y = 720
    val_min, val_max = 0.35, 0.65
    max_h = 360
    scale = max_h / (val_max - val_min)
    bw = 42
    grp_gap = 80
    x0 = 130
    for g in range(3):
        gx = x0 + g*(5*bw + 30 + grp_gap)
        c += cell(f"f7_g{g}", groups[g], "text;html=1;strokeColor=none;fillColor=none;fontSize=10;align=center;fontStyle=1;", gx-20, base_y+10, 5*bw+60, 40)
        for m in range(5):
            name, color, vals = models[m]
            x = gx + m*(bw+8)
            h = int((vals[g]-val_min)*scale)
            c += cell(f"f7_b{g}{m}", f"{vals[g]:.4f}", f"rounded=0;whiteSpace=wrap;html=1;fillColor={color};strokeColor={color};fontColor=#FFFFFF;fontSize=8;fontStyle=1;", x, base_y-h, bw, h)
    # 轴
    c += '<mxCell id="f7_axisY" value="" style="endArrow=classic;html=1;strokeWidth=1.5;" edge="1" parent="1"><mxGeometry relative="1" as="geometry"><mxPoint x="90" y="720" as="sourcePoint"/><mxPoint x="90" y="320" as="targetPoint"/></mxGeometry></mxCell>'
    c += '<mxCell id="f7_axisX" value="" style="endArrow=classic;html=1;strokeWidth=1.5;" edge="1" parent="1"><mxGeometry relative="1" as="geometry"><mxPoint x="90" y="720" as="sourcePoint"/><mxPoint x="900" y="720" as="targetPoint"/></mxGeometry></mxCell>'
    for v in [0.4,0.45,0.5,0.55,0.6,0.65]:
        yy = base_y - int((v-val_min)*scale)
        c += cell(f"f7_g{int(v*100)}", f"{v:.2f}", "text;html=1;strokeColor=none;fillColor=none;fontSize=9;align=right;", 40, yy-8, 45, 16)
    c += cell("f7_yl", "HR@10", "text;html=1;strokeColor=none;fillColor=none;fontSize=11;align=center;fontStyle=1;", 30, 310, 50, 20)
    # 图例
    lx = 950
    for i,(name,color,_) in enumerate(models):
        c += cell(f"f7_leg{i}","",f"rounded=0;whiteSpace=wrap;html=1;fillColor={color};strokeColor={color};",lx,200+i*30,24,16)
        c += cell(f"f7_legt{i}",name,"text;html=1;strokeColor=none;fillColor=none;fontSize=10;align=left;",lx+30,196+i*30,160,20)
    c += cell("f7_note1","KG-HybridRec: cold>warm>heavy 单调降", "text;html=1;strokeColor=#C00000;fillColor=none;fontSize=9;fontStyle=1;",950,360,280,20)
    c += cell("f7_note2","LightGCN/BPR-MF: 反序 warm>cold", "text;html=1;strokeColor=#4472C4;fillColor=none;fontSize=9;fontStyle=1;",950,380,280,20)
    c += cell("f7_note3","cold 上 KG-HybridRec 0.5944 学习类最高", "text;html=1;strokeColor=#C00000;fillColor=none;fontSize=9;fontStyle=1;",950,400,280,30)
    return c

# ============ 图 3-8 数据稀疏度鲁棒性折线图 ============
def fig_3_8():
    c = ""
    c += cell("f8_title", "图 3-8  数据稀疏度鲁棒性(cold_user 子集 HR@10)", TITLE, 40, 40, 1500, 40)
    base_y = 720
    val_min, val_max = 0.20, 0.65
    max_h = 360
    scale = max_h / (val_max - val_min)
    # 三个比例的 x 位置
    xs = {1.0: 200, 0.5: 500, 0.2: 800}
    x_labels = {1.0:"ratio=1.0\n(145k 交互)", 0.5:"ratio=0.5\n(72k)", 0.2:"ratio=0.2\n(29k, 削减80%)"}
    series = [
        ("KG-HybridRec", "#C00000", [0.5936,0.5911,0.5891], True),
        ("BPR-MF", "#70AD47", [0.5195,0.2824,0.2238], False),
        ("LightGCN", "#4472C4", [0.5873,0.4034,0.2721], False),
    ]
    ratios = [1.0, 0.5, 0.2]
    # 画点和连线(用矩形点 + edge)
    pid = 100
    for sname, color, vals, hl in series:
        prev_id = None
        for j, r in enumerate(ratios):
            x = xs[r]
            yy = base_y - int((vals[j]-val_min)*scale)
            # 点
            dot_style = f"ellipse;whiteSpace=wrap;html=1;fillColor={color};strokeColor={color};strokeWidth={'3' if hl else '1'};"
            dot_id = f"f8_d{sname[:3]}{j}"
            c += cell(dot_id, "", dot_style, x-7, yy-7, 14, 14)
            # 数值标签
            c += cell(f"f8_v{sname[:3]}{j}", f"{vals[j]:.4f}", f"text;html=1;strokeColor=none;fillColor=none;fontSize=9;fontColor={color};align=center;fontStyle=1;", x-30, yy-30, 60, 18)
            if prev_id:
                eid = f"f8_e{sname[:3]}{j}"
                c += edge(eid, prev_id, dot_id, "", f"endArrow=none;html=1;strokeColor={color};strokeWidth={'3' if hl else '2'};")
            prev_id = dot_id
    # 轴
    c += '<mxCell id="f8_axisY" value="" style="endArrow=classic;html=1;strokeWidth=1.5;" edge="1" parent="1"><mxGeometry relative="1" as="geometry"><mxPoint x="120" y="720" as="sourcePoint"/><mxPoint x="120" y="300" as="targetPoint"/></mxGeometry></mxCell>'
    c += '<mxCell id="f8_axisX" value="" style="endArrow=classic;html=1;strokeWidth=1.5;" edge="1" parent="1"><mxGeometry relative="1" as="geometry"><mxPoint x="120" y="720" as="sourcePoint"/><mxPoint x="950" y="720" as="targetPoint"/></mxGeometry></mxCell>'
    for r in ratios:
        c += cell(f"f8_xl{int(r*10)}", x_labels[r], "text;html=1;strokeColor=none;fillColor=none;fontSize=10;align=center;", xs[r]-60, base_y+10, 120, 40)
    for v in [0.2,0.3,0.4,0.5,0.6]:
        yy = base_y - int((v-val_min)*scale)
        c += cell(f"f8_g{int(v*100)}", f"{v:.2f}", "text;html=1;strokeColor=none;fillColor=none;fontSize=9;align=right;", 60, yy-8, 55, 16)
    c += cell("f8_yl", "HR@10", "text;html=1;strokeColor=none;fillColor=none;fontSize=11;align=center;fontStyle=1;", 50, 290, 60, 20)
    # 标注
    c += cell("f8_a1", "KG-HybridRec: -0.76%\n近乎水平(鲁棒)", "text;html=1;strokeColor=#C00000;fillColor=none;fontSize=10;fontStyle=1;align=center;", 700, 360, 200, 40)
    c += cell("f8_a2", "BPR-MF: -56.9%\nLightGCN: -53.7%\n性能腰斩", "text;html=1;strokeColor=#4472C4;fillColor=none;fontSize=10;fontStyle=1;align=center;", 700, 560, 200, 50)
    # 图例
    for i,(sname,color,_,_) in enumerate(series):
        c += cell(f"f8_leg{i}","",f"rounded=0;whiteSpace=wrap;html=1;fillColor={color};strokeColor={color};",980,250+i*30,24,16)
        c += cell(f"f8_legt{i}",sname,"text;html=1;strokeColor=none;fillColor=none;fontSize=10;align=left;",1010,246+i*30,160,20)
    return c

def build():
    figs = [
        ("图3-1_构建流程", "fig1", fig_3_1()),
        ("图3-2_本体结构", "fig2", fig_3_2()),
        ("图3-3_KG-HybridRec框架", "fig3", fig_3_3()),
        ("图3-4_CAGF数据流", "fig4", fig_3_4()),
        ("图3-5_三项改进位置", "fig5", fig_3_5()),
        ("图3-6_主结果柱状图", "fig6", fig_3_6()),
        ("图3-7_冷启动分层", "fig7", fig_3_7()),
        ("图3-8_密度鲁棒性", "fig8", fig_3_8()),
    ]
    outdir = r"D:/clothing-recommender/figures"
    import os
    os.makedirs(outdir, exist_ok=True)
    # 1) 合并多页文件
    pages = [page(n, pid, xml) for (n, pid, xml) in figs]
    xml_all = '<?xml version="1.0" encoding="UTF-8"?>\n<mxfile host="app.diagrams.net" type="device">\n' + "\n".join(pages) + '\n</mxfile>'
    with open(r"D:/clothing-recommender/第3章_图.drawio", "w", encoding="utf-8") as f:
        f.write(xml_all)
    print("多页文件 OK -> 第3章_图.drawio (", len(pages), "页)")
    # 2) 每张图单独一个 .drawio 文件(便于 CLI 逐个导出)
    for (n, pid, xml) in figs:
        single = '<?xml version="1.0" encoding="UTF-8"?>\n<mxfile host="app.diagrams.net" type="device">\n' + page(n, pid, xml) + '\n</mxfile>'
        p = f"{outdir}/{n}.drawio"
        with open(p, "w", encoding="utf-8") as f:
            f.write(single)
        print("单页 OK ->", p)

if __name__ == "__main__":
    build()
