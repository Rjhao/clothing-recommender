"""将 论文修订_3.4节_2026-07-07.md 转为 .docx 输出到桌面。

解析:
- 标题 (# ## ###) → Heading 1/2/3
- 表格 (|...|) → docx Table
- 普通段落 → 普通段落
- 加粗 (**...**) → bold run
"""
import re
import sys
from pathlib import Path

from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH


def parse_md(text):
    """返回 [(type, content), ...] 的 token 列表。
    type ∈ {'h1','h2','h3','h4','p','table','blank','code','hr'}。
    """
    lines = text.split('\n')
    tokens = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # 标题
        m = re.match(r'^(#{1,6})\s+(.+)$', stripped)
        if m:
            level = len(m.group(1))
            content = m.group(2).strip()
            tokens.append((f'h{level}', content))
            i += 1
            continue

        # 表格 (|...| 开头)
        if stripped.startswith('|') and stripped.endswith('|') and i + 1 < len(lines) and re.match(r'^\|[\s\-:|]+\|$', lines[i+1].strip()):
            table_lines = [stripped, lines[i+1].strip()]
            j = i + 2
            while j < len(lines) and lines[j].strip().startswith('|') and lines[j].strip().endswith('|'):
                table_lines.append(lines[j].strip())
                j += 1
            # 解析:row 0 表头,row 1 分隔,row 2+ 数据
            rows = []
            for tl in table_lines:
                cells = [c.strip() for c in tl.strip('|').split('|')]
                rows.append(cells)
            tokens.append(('table', rows))
            i = j
            continue

        # 分隔
        if stripped == '---':
            tokens.append(('hr', ''))
            i += 1
            continue

        # 空行
        if not stripped:
            tokens.append(('blank', ''))
            i += 1
            continue

        # 默认段落(可能跨多行,直到空行/标题/表格/分隔)
        para = [stripped]
        j = i + 1
        while j < len(lines):
            nxt = lines[j].strip()
            if not nxt or nxt.startswith('#') or nxt.startswith('|') or nxt == '---':
                break
            para.append(nxt)
            j += 1
        tokens.append(('p', ' '.join(para)))
        i = j
    return tokens


def add_paragraph_with_bold(doc, text, style=None):
    """在段落中识别 **text** 加粗。"""
    p = doc.add_paragraph(style=style) if style else doc.add_paragraph()
    parts = re.split(r'(\*\*[^*]+\*\*)', text)
    for part in parts:
        if not part: continue
        if part.startswith('**') and part.endswith('**'):
            run = p.add_run(part[2:-2])
            run.bold = True
        else:
            p.add_run(part)
    return p


def add_table(doc, rows):
    """rows[0] = header, rows[1] = separator, rows[2:] = data."""
    if len(rows) < 3:
        return
    headers = rows[0]
    n_cols = len(headers)
    table = doc.add_table(rows=1, cols=n_cols)
    table.style = 'Light Grid Accent 1'
    # header
    hdr_cells = table.rows[0].cells
    for ci, h in enumerate(headers):
        hdr_cells[ci].text = ''
        p = hdr_cells[ci].paragraphs[0]
        run = p.add_run(h)
        run.bold = True
        run.font.size = Pt(10)
    # data
    for row in rows[2:]:
        cells = table.add_row().cells
        for ci, val in enumerate(row):
            if ci >= n_cols: break
            cells[ci].text = ''
            p = cells[ci].paragraphs[0]
            # 加粗
            parts = re.split(r'(\*\*[^*]+\*\*)', val)
            for part in parts:
                if not part: continue
                if part.startswith('**') and part.endswith('**'):
                    r = p.add_run(part[2:-2])
                    r.bold = True
                else:
                    r = p.add_run(part)
                r.font.size = Pt(10)


def main():
    md_path = Path('论文修订_3.4节_2026-07-07.md')
    out_path = Path.home() / 'Desktop' / '论文修订_3.4节_2026-07-07.docx'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    text = md_path.read_text(encoding='utf-8')
    tokens = parse_md(text)

    doc = Document()
    # 字体设置
    style = doc.styles['Normal']
    style.font.name = '宋体'
    style.font.size = Pt(10.5)
    # 标题样式
    for lvl, sz in [(1, 16), (2, 14), (3, 12), (4, 11)]:
        try:
            h_style = doc.styles[f'Heading {lvl}']
            h_style.font.name = '黑体'
            h_style.font.size = Pt(sz)
            h_style.font.bold = True
        except KeyError:
            pass

    # 首页标题
    title = doc.add_heading('论文 3.4 节修订稿', level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    for typ, content in tokens:
        if typ == 'blank':
            continue
        elif typ == 'h1':
            doc.add_heading(content, level=1)
        elif typ == 'h2':
            doc.add_heading(content, level=2)
        elif typ == 'h3':
            doc.add_heading(content, level=3)
        elif typ == 'h4':
            doc.add_heading(content, level=4)
        elif typ == 'hr':
            p = doc.add_paragraph()
            p.add_run('─' * 40)
        elif typ == 'p':
            add_paragraph_with_bold(doc, content)
        elif typ == 'table':
            add_table(doc, content)
            # 表格后空一行
            doc.add_paragraph()

    doc.save(str(out_path))
    print(f"Saved: {out_path}")
    print(f"Size: {out_path.stat().st_size:,} bytes")


if __name__ == '__main__':
    main()
