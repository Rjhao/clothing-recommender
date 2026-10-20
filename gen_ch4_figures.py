# -*- coding: utf-8 -*-
"""生成第4章图的 draw.io 文件
图4-3: 跨场景HR@10跌幅分组柱状图
输出: D:/clothing-recommender/第4章_图.drawio
"""
import xml.etree.ElementTree as ET
import datetime

# ---- draw.io XML 结构 ----
MXGRAPH = "http://graphdraw.mxgraph.com/mxgraph"
MXEDITOR = "http://www.jgraph.com/mxgraph"

def new_page(title, w=900, h=500):
    root = ET.Element('mxfile')
    root.set('host', 'app.diagrams.net')
    root.set('modified', datetime.datetime.now().isoformat())
    root.set('type', 'device')

    diagram = ET.SubElement(root, 'diagram')
    diagram.set('id', '1')
    diagram.set('name', title)

    mxmodel = ET.SubElement(diagram, 'mxGraphModel')
    mxmodel.set('dx', '1200')
    mxmodel.set('dy', '800')
    mxmodel.set('grid', '1')
    mxmodel.set('guides', '1')
    mxmodel.set('toolver', '1')
    mxmodel.set('page', '1')
    mxmodel.set('pageScale', '1')
    mxmodel.set('pageWidth', str(w))
    mxmodel.set('pageHeight', str(h))

    root_page = ET.SubElement(mxmodel, 'root')
    ET.SubElement(root_page, 'mxCell')  # layer
    return root_page, diagram

def cell(node_id, value, style, x, y, w, h, parent="1"):
    v = value.replace('"', '&quot;') if value else ""
    c = ET.SubElement(parent, 'mxCell')
    c.set('id', str(node_id))
    c.set('value', v)
    c.set('style', style)
    c.set('vertex', '1')
    c.set('parent', str(parent))
    geo = ET.SubElement(c, 'mxGeometry')
    geo.set('x', str(x))
    geo.set('y', str(y))
    geo.set('width', str(w))
    geo.set('height', str(h))
    geo.set('as', 'geometry')
    return c

def text_node(node_id, value, style, x, y, w, h, parent="1"):
    """纯文本节点"""
    return cell(node_id, value, style, x, y, w, h, parent)

# ---- 图4-3: 跨场景 HR@10 跌幅柱状图 ----
def make_fig3(root):
    """跨场景 HR@10 跌幅对比柱状图"""
    page_w, page_h = 900, 500

    # 标题
    text_node(1, '图 4-3 跨场景 HR@10 跌幅对比', 'text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=14;fontStyle=1;', 250, 10, 400, 30, root)

    # 柱子数据 (dense_hr10, sparse_hr10, delta)
    data = [
        ('LightGCN',   0.5900, 0.7115, +0.1215, '#5B9BD5'),  # 反序上涨 蓝
        ('BPR-MF',     0.5201, 0.6839, +0.1638, '#5B9BD5'),  # 反序上涨 蓝
        ('RippleNet',   0.5153, 0.5276, +0.0123, '#70AD47'),   # 略涨 绿
        ('KG-HybridRec',0.5884, 0.5390, -0.0494, '#ED7D31'),   # 下跌 橙
    ]

    bar_w = 40
    base_y = 420
    max_h = 250
    max_val = 0.75

    # Y轴
    cell(100, '', 'endArrow=none;strokeWidth=2;strokeColor=#666666;', 80, 60, 0, 370, root)
    # X轴
    cell(101, '', 'endArrow=none;strokeWidth=2;strokeColor=#666666;', 80, 420, 780, 0, root)

    # Y轴刻度标签
    for val, y_pos in [(0.75, 80), (0.50, 180), (0.25, 280), (0.00, 380)]:
        text_node(200+val, f'{val:.2f}', 'text;html=1;align=right;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=10;', 35, y_pos-7, 35, 14, root)
        cell(300+val, '', 'dashed=1;strokeWidth=1;strokeColor=#cccccc;strokeDashPattern=1 3;', 80, y_pos, 750, 0, root)

    x_start = 130
    gap = 160

    for i, (name, dense_hr, sparse_hr, delta, color) in enumerate(data):
        x = x_start + i * gap

        # 稠密柱
        dense_h = dense_hr / max_val * max_h
        cell(400+i*2, f'{dense_hr:.4f}', f'rounded=0;whiteSpace=wrap;html=1;fillColor={color};strokeColor=#ffffff;strokeWidth=1;align=center;verticalAlign=middle;fontSize=9;', x, base_y - dense_h, bar_w, dense_h, root)

        # 稀疏柱
        sparse_h = sparse_hr / max_val * max_h
        cell(401+i*2, f'{sparse_hr:.4f}', f'rounded=0;whiteSpace=wrap;html=1;fillColor={color};strokeColor=#ffffff;strokeWidth=1;align=center;verticalAlign=middle;fontSize=9;opacity=60;', x + bar_w + 5, base_y - sparse_h, bar_w, sparse_h, root)

        # 跌幅标注
        if delta > 0:
            delta_text = f'+{delta:.4f}'
            delta_color = '#5B9BD5'
        else:
            delta_text = f'{delta:.4f}'
            delta_color = '#ED7D31'
        text_node(500+i, delta_text, f'text;html=1;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=11;fontStyle=1;color={delta_color};', x, base_y - max_h - 25, bar_w*2+5, 20, root)

        # 底部标签
        text_node(600+i, name, 'text;html=1;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=11;', x - 10, base_y + 5, bar_w*2+15, 20, root)

    # 图例
    cell(700, '稠密场景(HR@10)', 'rounded=0;whiteSpace=wrap;html=1;fillColor=#5B9BD5;strokeColor=#ffffff;strokeWidth=1;align=center;verticalAlign=middle;fontSize=10;', 650, 50, 100, 20, root)
    cell(701, '稀疏场景(HR@10)', 'rounded=0;whiteSpace=wrap;html=1;fillColor=#5B9BD5;strokeColor=#ffffff;strokeWidth=1;align=center;verticalAlign=middle;fontSize=10;opacity=60;', 650, 75, 100, 20, root)
    cell(702, '涨幅', 'text;html=1;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=10;color=#5B9BD5;fontStyle=1;', 650, 100, 40, 15, root)
    cell(703, '跌幅', 'text;html=1;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=10;color=#ED7D31;fontStyle=1;', 700, 100, 40, 15, root)

    # 标注"反序上涨"
    text_node(800, '← 反序上涨', 'text;html=1;align=left;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=10;color=#5B9BD5;', 680, 130, 80, 15, root)
    text_node(801, '← 本文稳定', 'text;html=1;align=left;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=10;color=#ED7D31;', 680, 150, 80, 15, root)

    # 坐标轴标签
    text_node(900, 'HR@10', 'text;html=1;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=11;fontStyle=1;', 25, 250, 40, 20, root)
    text_node(901, '方法', 'text;html=1;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=11;fontStyle=1;', 430, 450, 40, 20, root)
    text_node(902, 'ΔHR@10', 'text;html=1;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=11;fontStyle=1;', 730, 450, 40, 20, root)

def save(root, path):
    ET.indent(root, space='')
    tree = ET.ElementTree(root)
    tree.write(path, encoding='utf-8', xml_declaration=True)
    print(f"Saved: {path}")

if __name__ == '__main__':
    root, diagram = new_page('图 4-3 跨场景 HR@10 跌幅对比')
    make_fig3(root)
    save(root, 'D:/clothing-recommender/第4章_图.drawio')
