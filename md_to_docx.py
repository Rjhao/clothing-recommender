# -*- coding: utf-8 -*-
"""Markdown -> Word 转换器(论文第3章加强版)。
- 块公式 $$...$$ 与行内公式 $...$ -> Word 原生 OMML 公式(LaTeX->MathML->MML2OMML.XSL->OMML)
- \tag{3-x} 分离为公式编号(右对齐括号)
- 图引用块(> **图 3-x ...**)-> 在其前插入对应 SVG 图片 + 图题
运行: PYTHONUTF8=1 python md_to_docx.py <input.md> <output.docx> [figures_dir]
"""
import re
import sys
import os
import html
from docx import Document
from docx.shared import Pt, RGBColor, Inches, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn, nsmap
from docx.oxml import OxmlElement, parse_xml
import lxml.etree as ET
from latex2mathml.converter import convert as latex2mathml

# ---- 字体 ----
CN_BODY="宋体"; CN_HEAD="黑体"; EN_FONT="Times New Roman"
MATH_FONT="Cambria Math"; CODE_FONT="Consolas"
FIG_DIR = "D:/clothing-recommender/figures"

# ---- OMML 转换 ----
_XSLT = None
def get_xslt():
    global _XSLT
    if _XSLT is None:
        xsl_path = r"C:/Program Files/Microsoft Office/root/Office16/MML2OMML.XSL"
        _XSLT = ET.XSLT(ET.parse(xsl_path))
    return _XSLT

MML_NS = "http://www.w3.org/1998/Math/MathML"
def latex_to_omml(latex):
    """LaTeX -> OMML XML 元素(lxml element),失败返回 None。"""
    latex = latex.strip()
    # 去掉 \tag{...} 与 \label{...},单独提取编号
    tag_m = re.search(r'\\tag\{([^}]*)\}', latex)
    tag = tag_m.group(1) if tag_m else None
    latex_clean = re.sub(r'\\tag\{[^}]*\}', '', latex).strip()
    latex_clean = re.sub(r'\\label\{[^}]*\}', '', latex_clean).strip()
    try:
        mathml = latex2mathml(latex_clean)
        # latex2mathml 输出 <math xmlns=...>...</math>,需包成文档供 XSLT
        doc = ET.fromstring(mathml)
        omml = get_xslt()(doc)
        omml_str = ET.tostring(omml, encoding='unicode')
        # 取出 <m:oMath>...</m:oMath> 块(去掉外层 oMathPara)
        m = re.search(r'<m:oMath\b.*?</m:oMath>', omml_str, re.S)
        if not m:
            m = re.search(r'<m:oMath\b[^>]*/>', omml_str)
        if m:
            return m.group(0), tag
        return omml_str, tag
    except Exception as e:
        return None, tag

def add_omml_run(paragraph, omml_xml):
    """把 OMML XML 插入 paragraph(作为 math 元素)。"""
    # parse_xml 需要 w: 命名空间上下文,但 m: 是独立命名空间,直接用 lxml 包进 run
    # python-docx 的 run._element 可 append 子元素
    try:
        omath = ET.fromstring(omml_xml)
    except Exception:
        # 回退:纯文本
        r = paragraph.add_run(omml_xml)
        set_run_font(r, en=MATH_FONT, size=11, italic=True)
        return
    # 把 m: 命名空间元素直接 append 到 paragraph 的 XML
    paragraph._element.append(omath)

def add_block_math(doc, latex):
    omml_xml, tag = latex_to_omml(latex)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.space_before = Pt(8); pf.space_after = Pt(8)
    if omml_xml:
        add_omml_run(p, omml_xml)
    else:
        # 回退:Cambria Math 文本
        r = p.add_run(render_math_text_fallback(latex))
        set_run_font(r, en=MATH_FONT, size=11, italic=True)
    if tag:
        # 公式编号右对齐:在公式后加 tab + (tag)
        r = p.add_run(f"          ({tag})")
        set_run_font(r, cn=CN_BODY, en=EN_FONT, size=11)

def render_math_text_fallback(latex):
    t = latex
    t = re.sub(r'\\tag\{([^}]*)\}', r'  (\1)', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

# ---- 字体设置 ----
def set_run_font(run, cn=CN_BODY, en=EN_FONT, size=None, bold=None, italic=None, color=None):
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn('w:rFonts'))
    if rfonts is None:
        rfonts = OxmlElement('w:rFonts'); rpr.append(rfonts)
    rfonts.set(qn('w:ascii'), en); rfonts.set(qn('w:hAnsi'), en)
    rfonts.set(qn('w:eastAsia'), cn); rfonts.set(qn('w:cs'), en)
    run.font.name = en
    if size is not None: run.font.size = Pt(size)
    if bold is not None: run.font.bold = bold
    if italic is not None: run.font.italic = italic
    if color is not None: run.font.color.rgb = RGBColor(*color)

def shade_cell(cell, color_hex):
    tcpr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'),'clear'); shd.set(qn('w:color'),'auto'); shd.set(qn('w:fill'),color_hex)
    tcpr.append(shd)

# ---- 行内解析(行内 $...$ 也转 OMML) ----
INLINE_RE = re.compile(r'(\*\*[^*]+\*\*)|(\$[^$]+\$)|(`[^`]+`)|(\[[^\]]+\]\([^)]+\))')

def add_inline(paragraph, text, cn=CN_BODY, en=EN_FONT, size=None, bold=None, color=None):
    pos = 0
    for m in INLINE_RE.finditer(text):
        if m.start() > pos:
            r = paragraph.add_run(text[pos:m.start()])
            set_run_font(r, cn=cn, en=en, size=size, bold=bold, color=color)
        tok = m.group(0)
        if tok.startswith('**') and tok.endswith('**'):
            r = paragraph.add_run(tok[2:-2]); set_run_font(r, cn=cn, en=en, size=size, bold=True, color=color)
        elif tok.startswith('$') and tok.endswith('$'):
            omml_xml, _ = latex_to_omml(tok[1:-1])
            if omml_xml:
                add_omml_run(paragraph, omml_xml)
            else:
                r = paragraph.add_run(tok[1:-1]); set_run_font(r, en=MATH_FONT, size=size, italic=True, color=color)
        elif tok.startswith('`') and tok.endswith('`'):
            r = paragraph.add_run(tok[1:-1]); set_run_font(r, cn=cn, en=CODE_FONT, size=(size-1 if size else None), color=color)
        elif tok.startswith('['):
            mm = re.match(r'\[([^\]]+)\]\(([^)]+)\)', tok)
            label = mm.group(1) if mm else tok
            r = paragraph.add_run(label); set_run_font(r, cn=cn, en=en, size=size, bold=bold, color=(0x05,0x63,0xC1))
        pos = m.end()
    if pos < len(text):
        r = paragraph.add_run(text[pos:]); set_run_font(r, cn=cn, en=en, size=size, bold=bold, color=color)

# ---- 图识别:引用块首行 "> **图 N-x ...**"(N 为任意章号) ----
FIG_TITLE_RE = re.compile(r'^\s*\*?\*?图\s*(\d+)-(\d+)\b')

def is_special(line):
    s = line.strip()
    return (s.startswith('```') or s.startswith('|') or s.startswith('$$')
            or re.match(r'^#{1,6}\s', s) or s.startswith('>') or re.match(r'^\s*[-*]\s+', s))

def parse_blocks(md):
    lines = md.split('\n')
    blocks = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]; s = line.strip()
        if s.startswith('```'):
            code = []; i += 1
            while i < n and not lines[i].strip().startswith('```'):
                code.append(lines[i]); i += 1
            i += 1; blocks.append(('code', '\n'.join(code))); continue
        if s.startswith('$$'):
            rest = s[2:]
            if rest.endswith('$$') and len(rest) >= 2:
                blocks.append(('math', rest[:-2].strip())); i += 1; continue
            buf = [rest] if rest else []; i += 1
            while i < n and not lines[i].strip().endswith('$$'):
                buf.append(lines[i].strip()); i += 1
            if i < n:
                buf.append(lines[i].strip()[:-2].strip()); i += 1
            blocks.append(('math', ' '.join(b for b in buf if b).strip())); continue
        if s.startswith('|'):
            tbl = []
            while i < n and lines[i].strip().startswith('|'):
                tbl.append(lines[i].strip()); i += 1
            blocks.append(('table', tbl)); continue
        m = re.match(r'^(#{1,6})\s+(.*)$', s)
        if m:
            blocks.append(('h', (len(m.group(1)), m.group(2).strip()))); i += 1; continue
        if s.startswith('>'):
            qs = []
            while i < n and lines[i].strip().startswith('>'):
                qs.append(lines[i].strip()[1:].strip()); i += 1
            blocks.append(('quote', qs)); continue
        if re.match(r'^\s*[-*]\s+', s):
            items = []
            while i < n and re.match(r'^\s*[-*]\s+', lines[i]):
                items.append(re.sub(r'^\s*[-*]\s+','',lines[i].strip())); i += 1
            blocks.append(('list', items)); continue
        if s == '':
            i += 1; continue
        para = []
        while i < n and lines[i].strip() != '' and not is_special(lines[i]):
            para.append(lines[i].strip()); i += 1
        blocks.append(('p', ' '.join(para)))
    return blocks

# ---- 渲染 ----
def add_heading(doc, level, text):
    sizes = {1:18,2:16,3:14,4:13,5:12,6:11}
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT if level>1 else WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.space_before = Pt(12 if level<=2 else 8); pf.space_after = Pt(6)
    if level == 1: pf.space_before = Pt(0)
    r = p.add_run(text); set_run_font(r, cn=CN_HEAD, en=EN_FONT, size=sizes.get(level,12), bold=True)

def add_paragraph(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(24)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.5
    add_inline(p, text, size=12)

def add_quote(doc, lines):
    # 检测是否是图题块(首行含"图 N-x")
    first = lines[0] if lines else ""
    fig_m = FIG_TITLE_RE.match(first)
    if fig_m:
        chap, fig_num = fig_m.group(1), fig_m.group(2)
        svg = os.path.join(FIG_DIR, f"图{chap}-{fig_num}.svg")
        if os.path.exists(svg):
            add_image(doc, svg, first)
        else:
            # 没图,只放图题
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(f"【图 {chap}-{fig_num} 缺失】{first}"); set_run_font(r, cn=CN_HEAD, size=10.5, bold=True)
    # 引用内容(构图说明)作为图注下方小字
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.left_indent = Inches(0.3); pf.space_after = Pt(2); pf.line_spacing = 1.3
    for idx, ln in enumerate(lines):
        if idx > 0:
            r = p.add_run('\n')
        add_inline(p, ln, size=9.5, color=(0x66,0x66,0x66))

def add_image(doc, svg_path, caption):
    """插入图片 PNG + 居中图题。优先用同名 PNG(drawio 导出)。"""
    png_path = svg_path.replace('.svg', '.png')
    if not os.path.exists(png_path):
        # 尝试用 drawio 源文件导出
        drawio_src = svg_path.replace('.svg', '') + '.drawio'
        # 找带中文名的 drawio 源(可能没有,因为导出用英文名)
        if not os.path.exists(drawio_src):
            # 在 figures 目录里找任意 图3-x_*.drawio
            base = os.path.basename(svg_path).replace('.svg','')  # 图3-1
            for f in os.listdir(FIG_DIR):
                if f.startswith(base + '_') and f.endswith('.drawio'):
                    drawio_src = os.path.join(FIG_DIR, f); break
        if os.path.exists(drawio_src):
            import subprocess
            tmp_drawio = drawio_src
            subprocess.run(['D:/draw.io/draw.io.exe','--export','--format','png','--scale','2','--border','20','--output',png_path,tmp_drawio],
                           capture_output=True, timeout=120)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format; pf.space_before = Pt(6); pf.space_after = Pt(2)
    if os.path.exists(png_path):
        run = p.add_run()
        # 根据图类型设宽度:柱状图/折线图宽一点
        width = Inches(6.2)
        run.add_picture(png_path, width=width)
    else:
        r = p.add_run(f"[图片缺失 {svg_path}]"); set_run_font(r, color=(0xCC,0x00,0x00))
    pc = doc.add_paragraph(); pc.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = pc.add_run(caption); set_run_font(r, cn=CN_HEAD, size=10.5, bold=True)

def add_list(doc, items):
    for it in items:
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.line_spacing = 1.5
        add_inline(p, it, size=12)

def add_code(doc, code):
    for cl in code.split('\n'):
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.left_indent = Inches(0.3); pf.space_after = Pt(0); pf.line_spacing = 1.0
        r = p.add_run(cl if cl else ' '); set_run_font(r, cn=CN_BODY, en=CODE_FONT, size=9.5, color=(0x33,0x33,0x33))

def add_table(doc, table_lines):
    rows = []
    for ln in table_lines:
        cells = [c.strip() for c in ln.strip().strip('|').split('|')]
        rows.append(cells)
    if len(rows) < 2: return
    if re.match(r'^[\s:-]+$', rows[1][0]) and all(re.match(r'^[\s:-]*$', c) for c in rows[1]):
        header = rows[0]; body = rows[2:]
    else:
        header = rows[0]; body = rows[1:]
    ncol = len(header)
    table = doc.add_table(rows=1, cols=ncol)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = 'Light Grid Accent 1'
    for j, h in enumerate(header):
        cell = table.rows[0].cells[j]; cell.text = ''
        p = cell.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_inline(p, h, size=10.5, bold=True); shade_cell(cell, 'D9E2F3')
    for row in body:
        cells = table.add_row().cells
        for j in range(ncol):
            txt = row[j] if j < len(row) else ''
            cells[j].text = ''; p = cells[j].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT; add_inline(p, txt, size=10.5)

def md_to_docx(md_path, docx_path, fig_dir=None):
    global FIG_DIR
    if fig_dir: FIG_DIR = fig_dir
    with open(md_path, 'r', encoding='utf-8') as f:
        md = f.read()
    doc = Document()
    style = doc.styles['Normal']
    style.font.name = EN_FONT; style.font.size = Pt(12)
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn('w:rFonts'))
    if rfonts is None:
        rfonts = OxmlElement('w:rFonts'); rpr.append(rfonts)
    rfonts.set(qn('w:eastAsia'), CN_BODY)
    for sec in doc.sections:
        sec.top_margin = Inches(1); sec.bottom_margin = Inches(1)
        sec.left_margin = Inches(1.1); sec.right_margin = Inches(1.1)
    blocks = parse_blocks(md)
    for b in blocks:
        kind = b[0]
        if kind == 'h': add_heading(doc, b[1][0], b[1][1])
        elif kind == 'p': add_paragraph(doc, b[1])
        elif kind == 'math': add_block_math(doc, b[1])
        elif kind == 'table': add_table(doc, b[1])
        elif kind == 'quote': add_quote(doc, b[1])
        elif kind == 'list': add_list(doc, b[1])
        elif kind == 'code': add_code(doc, b[1])
    doc.save(docx_path)
    print(f'OK -> {docx_path}')

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('用法: python md_to_docx.py <input.md> <output.docx> [figures_dir]'); sys.exit(1)
    md_to_docx(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv)>3 else None)
