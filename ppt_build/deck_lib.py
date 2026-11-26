"""발표 pptx를 만드는 작은 도구 (python-pptx) + 같은 좌표로 그리는 미리보기 PNG (PIL).

서버에 LibreOffice가 없어서 검수는 미리보기로 한다. 미리보기는 NanumGothic(한글) 글자 폭에 1.06배 여유를 두고
줄바꿈을 흉내 내서, 글자가 상자를 넘치면 경고를 찍는다. 실제 PowerPoint 렌더와 똑같지는 않다.
글꼴은 Arial(라틴·동아시아 모두 Arial로 지정, 한글은 PowerPoint가 시스템 한글 글꼴로 대체).
"""
import copy
import re
import uuid

from lxml import etree
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION, XL_MARKER_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

W, H = 13.333, 7.5
FONT = 'Arial'
PX = 100                      # 미리보기: 1인치 = 100px
FONT_DIR = '/home/junho/miniforge3/envs/pptx/lib/python3.11/site-packages/koreanize_matplotlib/fonts/'
WIDTH_SLACK = 1.06

# 색 (PCB 솔더마스크 초록 + 구리 강조)
DARK = '10261F'
GREEN = '1E6B4E'
GREEN_M = '3C8C68'
TINT = 'E8F2EC'
COPPER = 'B8661F'
COPPER_T = 'F7E9DA'
GRAY_T = 'EEF1EF'
GRAY_L = '9AA59F'
INK = '1D2420'
MUTED = '5E6A64'
LINE = 'C9D4CE'
WHITE = 'FFFFFF'


def rgb(h):
    return RGBColor.from_string(h)


# 미리보기 글꼴 파일 (보통, 굵게). 빌드 스크립트에서 FONT와 함께 바꿀 수 있다.
PREVIEW_FONTS = None


def _font(size_pt, bold=False):
    if PREVIEW_FONTS:
        path = PREVIEW_FONTS[1] if bold else PREVIEW_FONTS[0]
    else:
        path = FONT_DIR + ('NanumGothicBold.ttf' if bold else 'NanumGothic.ttf')
    return ImageFont.truetype(path, max(1, int(round(size_pt / 72 * PX))))


def _hex(h):
    return '#' + h


def parse_runs(text):
    """'**굵게**' 표기를 (글자, 굵게?) 조각으로."""
    out = []
    for i, part in enumerate(re.split(r'\*\*', text)):
        if part:
            out.append((part, i % 2 == 1))
    return out or [('', False)]


class Para:
    def __init__(self, text, size=14, color=INK, bold=False, bullet=False, align='l', after=4, italic=False):
        self.text, self.size, self.color, self.bold = text, size, color, bold
        self.bullet, self.align, self.after, self.italic = bullet, align, after, italic


def as_paras(x, size, color, bold, align, after):
    if isinstance(x, Para):
        return [x]
    if isinstance(x, str):
        return [Para(line, size, color, bold, False, align, after) for line in x.split('\n')]
    out = []
    for p in x:
        out += as_paras(p, size, color, bold, align, after)
    return out


# ------------------------------------------------------------------ 미리보기 줄바꿈
def wrap_lines(para, width_in):
    """한 문단을 줄로 나눈다. 돌려주는 값: [(줄의 조각들, 줄 높이 px)]"""
    pieces = parse_runs(para.text)
    indent = 0.22 if para.bullet else 0
    avail = (width_in - indent) * PX
    words = []   # (단어, 굵게?) 공백 포함 단위
    for t, b in pieces:
        for w in re.split(r'(\s+)', t):
            if w:
                words.append((w, b or para.bold))
    lines, cur, cur_w = [], [], 0.0

    def wlen(w, b):
        return _font(para.size, b).getlength(w) * WIDTH_SLACK

    for w, b in words:
        lw = wlen(w, b)
        if w.isspace():
            if cur:
                cur.append((w, b))
                cur_w += lw
            continue
        if cur_w + lw <= avail or not cur:
            if lw > avail:            # 한 단어가 너무 길면 글자 단위로 자른다
                for ch in w:
                    cw = wlen(ch, b)
                    if cur_w + cw > avail and cur:
                        lines.append(cur)
                        cur, cur_w = [], 0.0
                    cur.append((ch, b))
                    cur_w += cw
            else:
                cur.append((w, b))
                cur_w += lw
        else:
            while cur and cur[-1][0].isspace():
                cur.pop()
            lines.append(cur)
            cur, cur_w = [(w, b)], lw
    if cur or not lines:
        lines.append(cur)
    lh = para.size / 72 * PX * 1.2
    return [(ln, lh) for ln in lines]


# ------------------------------------------------------------------ 덱
class Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(W), Inches(H)
        self._set_theme_fonts()
        self.slides = []
        self.sections = []          # (이름, [Slide])
        self.warnings = []

    def _set_theme_fonts(self):
        master = self.prs.slide_masters[0]
        theme_part = [r.target_part for r in master.part.rels.values() if r.reltype.endswith('/theme')][0]
        root = etree.fromstring(theme_part.blob)
        ns = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}
        for tag in ('majorFont', 'minorFont'):
            f = root.find(f'.//a:fontScheme/a:{tag}', ns)
            f.find('a:latin', ns).set('typeface', FONT)
            f.find('a:ea', ns).set('typeface', FONT)
        root.find('.//a:fontScheme', ns).set('name', FONT)
        theme_part._blob = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)

    def section(self, name):
        self.sections.append((name, []))

    def new(self, title, label=None, num=None, dark=False, notes=None, title_size=28):
        layout = self.prs.slide_layouts[5]          # Title Only
        s = self.prs.slides.add_slide(layout)
        sl = Slide(self, s, dark, len(self.slides) + 1)
        self.slides.append(sl)
        self.sections[-1][1].append(sl)
        if dark:
            bg = s.background.fill
            bg.solid()
            bg.fore_color.rgb = rgb(DARK)
        sl.set_title(title, title_size)
        if label:
            sl.badge(num, label)
        if notes:
            s.notes_slide.notes_text_frame.text = notes
        return sl

    def save(self, path):
        for sl in self.slides:
            if sl.number > 1:
                sl.text(W - 1.1, H - 0.45, 0.6, 0.3, str(sl.number), size=11, color=('B8C7BF' if sl.dark else MUTED), align='r')
        self._write_sections()
        self.prs.save(path)

    def _write_sections(self):
        pres = self.prs.part._element
        ids = {sl.pslide.slide_id: sl for sl in self.slides}
        P14 = 'http://schemas.microsoft.com/office/powerpoint/2010/main'
        extlst = pres.find(qn('p:extLst'))
        if extlst is None:
            extlst = etree.SubElement(pres, qn('p:extLst'))
        ext = etree.SubElement(extlst, qn('p:ext'), uri='{521415D9-36F7-43E2-AB2F-B90AF26B5E84}')
        lst = etree.SubElement(ext, '{%s}sectionLst' % P14, nsmap={'p14': P14})
        for name, sls in self.sections:
            sec = etree.SubElement(lst, '{%s}section' % P14, name=name, id='{%s}' % str(uuid.uuid4()).upper())
            sl_lst = etree.SubElement(sec, '{%s}sldIdLst' % P14)
            for sl in sls:
                etree.SubElement(sl_lst, '{%s}sldId' % P14, id=str(sl.pslide.slide_id))

    # ------------------------------------------------------------ 미리보기
    def preview(self, out_dir, only=None):
        import pathlib
        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = []
        for sl in self.slides:
            if only and sl.number not in only:
                continue
            p = out / f'slide-{sl.number:02d}.png'
            sl.render(p)
            paths.append(p)
        return paths


class Slide:
    def __init__(self, deck, pslide, dark, number):
        self.deck, self.pslide, self.dark, self.number = deck, pslide, dark, number
        self.shapes = pslide.shapes
        self.ops = []     # 미리보기용 기록

    # -------------------------------------------------------------- 공통 글자 채우기
    def _fill_tf(self, tf, paras, margin=(0.08, 0.04), anchor='t', wrap=True):
        tf.word_wrap = wrap
        tf.auto_size = None
        tf.margin_left = tf.margin_right = Inches(margin[0])
        tf.margin_top = tf.margin_bottom = Inches(margin[1])
        tf.vertical_anchor = {'t': MSO_ANCHOR.TOP, 'm': MSO_ANCHOR.MIDDLE, 'b': MSO_ANCHOR.BOTTOM}[anchor]
        bodyPr = tf._txBody.find(qn('a:bodyPr'))
        for child in list(bodyPr):
            if child.tag in (qn('a:spAutoFit'), qn('a:normAutofit'), qn('a:noAutofit')):
                bodyPr.remove(child)
        etree.SubElement(bodyPr, qn('a:noAutofit'))
        for i, para in enumerate(paras):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = {'l': PP_ALIGN.LEFT, 'c': PP_ALIGN.CENTER, 'r': PP_ALIGN.RIGHT}[para.align]
            p.space_after = Pt(para.after)
            p.line_spacing = 1.0
            if para.bullet:
                pPr = p._p.get_or_add_pPr()
                pPr.set('marL', str(int(Inches(0.22))))
                pPr.set('indent', str(int(-Inches(0.22))))
                bf = etree.SubElement(pPr, qn('a:buFont'))
                bf.set('typeface', FONT)
                bc = etree.SubElement(pPr, qn('a:buChar'))
                bc.set('char', '•')
            for t, b in parse_runs(para.text):
                r = p.add_run()
                r.text = t
                f = r.font
                f.size = Pt(para.size)
                f.bold = b or para.bold
                f.italic = para.italic
                f.color.rgb = rgb(para.color)
                f.name = FONT
                rPr = r._r.get_or_add_rPr()
                rPr.set('lang', 'ko-KR')
                ea = etree.SubElement(rPr, qn('a:ea'))
                ea.set('typeface', FONT)

    def _record_text(self, x, y, w, h, paras, margin, anchor, what):
        self.ops.append(('text', x, y, w, h, paras, margin, anchor, what))

    # -------------------------------------------------------------- 제목·배지
    def set_title(self, title, size):
        t = self.pslide.shapes.title
        x, y, w, h = 0.6, 0.62, W - 1.2, 0.8
        t.left, t.top, t.width, t.height = Inches(x), Inches(y), Inches(w), Inches(h)
        color = WHITE if self.dark else INK
        paras = as_paras(title, size, color, True, 'l', 0)
        self._fill_tf(t.text_frame, paras, margin=(0.0, 0.02), anchor='m')
        self._record_text(x, y, w, h, paras, (0.0, 0.02), 'm', 'title')

    def badge(self, num, label):
        c = self.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.6), Inches(0.27), Inches(0.36), Inches(0.36))
        self._style(c, COPPER, None)
        paras = [Para(str(num), 12, WHITE, True, False, 'c', 0)]
        self._fill_tf(c.text_frame, paras, margin=(0, 0), anchor='m')
        c.name = 'section badge'
        self.ops.append(('oval', 0.6, 0.27, 0.36, 0.36, COPPER, None))
        self._record_text(0.6, 0.27, 0.36, 0.36, paras, (0, 0), 'm', 'badge')
        self.text(1.05, 0.25, 6, 0.4, label, size=13, color=('B8C7BF' if self.dark else MUTED), bold=True, anchor='m')

    # -------------------------------------------------------------- 도형
    def _style(self, shape, fill, line, line_w=1.0, dash=False):
        st = shape._element.find(qn('p:style'))
        if st is not None:
            shape._element.remove(st)
        if fill:
            shape.fill.solid()
            shape.fill.fore_color.rgb = rgb(fill)
        else:
            shape.fill.background()
        if line:
            shape.line.color.rgb = rgb(line)
            shape.line.width = Pt(line_w)
            if dash:
                from pptx.enum.dml import MSO_LINE
                shape.line.dash_style = MSO_LINE.DASH
        else:
            shape.line.fill.background()

    def box(self, x, y, w, h, text='', fill=TINT, line=None, size=14, color=INK, bold=False, align='c',
            anchor='m', radius=0.08, margin=(0.1, 0.06), line_w=1.0, dash=False, after=2, name=None):
        kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
        s = self.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
        if radius:
            s.adjustments[0] = min(0.5, radius / min(w, h))
        self._style(s, fill, line, line_w, dash)
        paras = as_paras(text, size, color, bold, align, after) if text != '' else []
        if paras:
            self._fill_tf(s.text_frame, paras, margin, anchor)
        if name:
            s.name = name
        self.ops.append(('box', x, y, w, h, fill, line, radius, line_w, dash))
        if paras:
            self._record_text(x, y, w, h, paras, margin, anchor, f'box@{x:.2f},{y:.2f}')
        return s

    def oval(self, x, y, d, fill, text='', size=14, color=WHITE, bold=True):
        s = self.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
        self._style(s, fill, None)
        self.ops.append(('oval', x, y, d, d, fill, None))
        if text:
            paras = [Para(text, size, color, bold, False, 'c', 0)]
            self._fill_tf(s.text_frame, paras, (0, 0), 'm')
            self._record_text(x, y, d, d, paras, (0, 0), 'm', 'oval')
        return s

    def text(self, x, y, w, h, text, size=14, color=INK, bold=False, align='l', anchor='t', margin=(0.0, 0.02),
             after=4, name=None):
        tb = self.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        paras = as_paras(text, size, color, bold, align, after)
        self._fill_tf(tb.text_frame, paras, margin, anchor)
        if name:
            tb.name = name
        self._record_text(x, y, w, h, paras, margin, anchor, f'text@{x:.2f},{y:.2f}')
        return tb

    def line(self, x1, y1, x2, y2, color=GRAY_L, width=1.5, arrow=True, dash=False):
        c = self.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
        c.line.color.rgb = rgb(color)
        c.line.width = Pt(width)
        ln = c.line._get_or_add_ln()
        if dash:
            pd = etree.SubElement(ln, qn('a:prstDash'))
            pd.set('val', 'dash')
        if arrow:
            te = etree.SubElement(ln, qn('a:tailEnd'))
            te.set('type', 'triangle')
            te.set('w', 'med')
            te.set('len', 'med')
        self.ops.append(('line', x1, y1, x2, y2, color, width, arrow, dash))
        return c

    def image(self, path, x, y, w=None, h=None, crop=None, name=None):
        """w나 h 중 하나만 주면 비율 유지. crop = 원본 픽셀 (l, t, r, b)."""
        im = Image.open(path)
        iw, ih = im.size
        if crop:
            l, t, r, b = crop
            cw, ch = r - l, b - t
        else:
            cw, ch = iw, ih
        if w is None:
            w = h * cw / ch
        if h is None:
            h = w * ch / cw
        pic = self.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))
        if crop:
            pic.crop_left, pic.crop_top = l / iw, t / ih
            pic.crop_right, pic.crop_bottom = 1 - r / iw, 1 - b / ih
        if name:
            pic.name = name
        self.ops.append(('image', x, y, w, h, str(path), crop))
        return w, h

    # -------------------------------------------------------------- 표
    def table(self, x, y, col_w, rows, size=13, header_fill=GREEN, header_color=WHITE, fills=None, row_h=0.36,
              bold_first_col=True, align=None, colors=None, header_h=None):
        """header_h: 머리 행 높이 (없으면 row_h)."""
        nr, nc = len(rows), len(col_w)
        rh = [header_h if (i == 0 and header_h) else row_h for i in range(nr)]
        shp = self.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(sum(col_w)), Inches(sum(rh)))
        tbl = shp.table
        tblPr = tbl._tbl.tblPr
        tblPr.set('firstRow', '1')
        tblPr.set('bandRow', '0')
        sid = tblPr.find(qn('a:tableStyleId'))
        if sid is None:
            sid = etree.SubElement(tblPr, qn('a:tableStyleId'))
        sid.text = '{2D5ABB26-0587-4C30-8999-92F81FD0307C}'      # No Style, No Grid
        for j, cw in enumerate(col_w):
            tbl.columns[j].width = Inches(cw)
        heights = []
        for i, row in enumerate(rows):
            tbl.rows[i].height = Inches(rh[i])
            line_h = 0
            for j, val in enumerate(row):
                cell = tbl.cell(i, j)
                fill = header_fill if i == 0 else (fills[i][j] if fills and fills[i][j] else WHITE)
                cell.fill.solid()
                cell.fill.fore_color.rgb = rgb(fill)
                col = header_color if i == 0 else (colors[i][j] if colors and colors[i][j] else INK)
                al = 'l' if j == 0 else 'c'
                if align:
                    al = align[j]
                b = i == 0 or (j == 0 and bold_first_col)
                paras = as_paras(str(val), size, col, b, al, 0)
                cell.margin_left = cell.margin_right = Inches(0.06)
                cell.margin_top = cell.margin_bottom = Inches(0.03)
                cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                tf = cell.text_frame
                tf.word_wrap = True
                for k, para in enumerate(paras):
                    p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
                    p.alignment = {'l': PP_ALIGN.LEFT, 'c': PP_ALIGN.CENTER, 'r': PP_ALIGN.RIGHT}[para.align]
                    for t, bb in parse_runs(para.text):
                        r = p.add_run()
                        r.text = t
                        r.font.size = Pt(size)
                        r.font.bold = bb or para.bold
                        r.font.color.rgb = rgb(para.color)
                        r.font.name = FONT
                        rPr = r._r.get_or_add_rPr()
                        rPr.set('lang', 'ko-KR')
                        ea = etree.SubElement(rPr, qn('a:ea'))
                        ea.set('typeface', FONT)
                # 아래 테두리 (얇은 선)
                tcPr = cell._tc.get_or_add_tcPr()
                for side in ('a:lnB',):
                    ln = etree.SubElement(tcPr, qn(side), w=str(int(Pt(0.75))))
                    sf = etree.SubElement(ln, qn('a:solidFill'))
                    etree.SubElement(sf, qn('a:srgbClr'), val=LINE)
                # 미리보기 높이
                hh = sum(lh for para in paras for _, lh in wrap_lines(para, col_w[j] - 0.12)) / PX + 0.06
                line_h = max(line_h, hh)
            heights.append(max(rh[i], line_h))
        # 탭(tcPr) 안 요소 순서: lnB 뒤에 solidFill이 와야 한다 → fill을 다시 붙여 순서를 맞춘다
        for i in range(nr):
            for j in range(nc):
                tcPr = tbl.cell(i, j)._tc.tcPr
                fills_el = [c for c in tcPr if c.tag == qn('a:solidFill')]
                for f in fills_el:
                    tcPr.remove(f)
                    tcPr.append(f)
        self.ops.append(('table', x, y, col_w, rows, size, header_fill, header_color, fills, heights, align, colors,
                         bold_first_col))
        return sum(heights)

    # -------------------------------------------------------------- 차트
    def _plot_layout(self, chart, lx, ly, lw, lh):
        pa = chart._chartSpace.find('.//' + qn('c:plotArea'))
        old = pa.find(qn('c:layout'))
        if old is not None:
            pa.remove(old)
        lay = etree.Element(qn('c:layout'))
        ml = etree.SubElement(lay, qn('c:manualLayout'))
        for tag, val in (('c:layoutTarget', 'inner'), ('c:xMode', 'edge'), ('c:yMode', 'edge'),
                         ('c:x', lx), ('c:y', ly), ('c:w', lw), ('c:h', lh)):
            etree.SubElement(ml, qn(tag), val=str(val))
        pa.insert(0, lay)

    def _axis_style(self, chart, vmin, vmax, major, fmt='0.00'):
        va = chart.value_axis
        va.minimum_scale, va.maximum_scale, va.major_unit = vmin, vmax, major
        va.has_major_gridlines = True
        va.major_gridlines.format.line.color.rgb = rgb('E1E7E3')
        va.major_gridlines.format.line.width = Pt(0.75)
        va.format.line.fill.background()
        va.tick_labels.font.size = Pt(12)
        va.tick_labels.font.color.rgb = rgb(MUTED)
        va.tick_labels.number_format = fmt
        va.tick_labels.number_format_is_linked = False
        ca = chart.category_axis
        ca.format.line.color.rgb = rgb(LINE)
        ca.tick_labels.font.size = Pt(12)
        ca.tick_labels.font.color.rgb = rgb(INK)
        ca.has_major_gridlines = False

    def bar_chart(self, x, y, w, h, cats, series, colors, vmax, major, lay=(0.09, 0.05, 0.88, 0.72),
                  point_colors=None, legend=True, fmt='0.000', ref=None, gap=80):
        cd = CategoryChartData()
        cd.categories = cats
        for name, vals in series:
            cd.add_series(name, vals)
        gf = self.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), cd)
        ch = gf.chart
        ch.font.name = FONT
        ch.font.size = Pt(12)
        self._axis_style(ch, 0, vmax, major, '0.00')
        plot = ch.plots[0]
        plot.gap_width = gap
        plot.overlap = -10
        plot.has_data_labels = True
        dl = plot.data_labels
        dl.number_format = fmt
        dl.number_format_is_linked = False
        dl.position = XL_LABEL_POSITION.OUTSIDE_END
        dl.font.size = Pt(12)
        dl.font.color.rgb = rgb(INK)
        for i, s in enumerate(plot.series):
            s.format.fill.solid()
            s.format.fill.fore_color.rgb = rgb(colors[i])
            if point_colors:
                for k, c in enumerate(point_colors):
                    pt = s.points[k]
                    pt.format.fill.solid()
                    pt.format.fill.fore_color.rgb = rgb(c)
        ch.has_legend = legend
        if legend:
            ch.legend.position = XL_LEGEND_POSITION.BOTTOM
            ch.legend.include_in_layout = False
            ch.legend.font.size = Pt(12)
        self._plot_layout(ch, *lay)
        self.ops.append(('bar', x, y, w, h, cats, series, colors, vmax, major, lay, point_colors, legend, fmt))
        if ref is not None:
            self._ref_line(x, y, w, h, lay, vmax, *ref)
        return ch

    def line_chart(self, x, y, w, h, cats, series, colors, vmin, vmax, major, lay=(0.11, 0.05, 0.86, 0.68),
                   ref=None, legend=True, markers=None):
        cd = CategoryChartData()
        cd.categories = cats
        for name, vals in series:
            cd.add_series(name, vals)
        gf = self.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(x), Inches(y), Inches(w), Inches(h), cd)
        ch = gf.chart
        ch.font.name = FONT
        ch.font.size = Pt(12)
        self._axis_style(ch, vmin, vmax, major, '0.00')
        marks = markers or [XL_MARKER_STYLE.CIRCLE, XL_MARKER_STYLE.SQUARE, XL_MARKER_STYLE.TRIANGLE, XL_MARKER_STYLE.DIAMOND]
        for i, s in enumerate(ch.plots[0].series):
            s.smooth = False
            s.format.line.color.rgb = rgb(colors[i])
            s.format.line.width = Pt(2.25)
            s.marker.style = marks[i % len(marks)]
            s.marker.size = 8
            s.marker.format.fill.solid()
            s.marker.format.fill.fore_color.rgb = rgb(colors[i])
            s.marker.format.line.color.rgb = rgb(WHITE)
        ch.has_legend = legend
        if legend:
            ch.legend.position = XL_LEGEND_POSITION.BOTTOM
            ch.legend.include_in_layout = False
            ch.legend.font.size = Pt(12)
        self._plot_layout(ch, *lay)
        self.ops.append(('linechart', x, y, w, h, cats, series, colors, vmin, vmax, major, lay, legend))
        if ref is not None:
            self._ref_line(x, y, w, h, lay, vmax, *ref, vmin=vmin)
        return ch

    def _ref_line(self, x, y, w, h, lay, vmax, value, label, vmin=0):
        lx, ly, lw, lh = lay
        yy = y + h * (ly + lh * (1 - (value - vmin) / (vmax - vmin)))
        x1, x2 = x + w * lx, x + w * (lx + lw)
        self.line(x1, yy, x2, yy, color=COPPER, width=1.5, arrow=False, dash=True)
        if label:
            self.text(x2 - 3.2, yy - 0.3, 3.2, 0.28, label, size=11, color=COPPER, bold=True, align='r')

    # -------------------------------------------------------------- 미리보기 그리기
    def render(self, path):
        img = Image.new('RGB', (int(W * PX), int(H * PX)), _hex(DARK if self.dark else WHITE))
        d = ImageDraw.Draw(img)
        P = lambda v: int(round(v * PX))  # noqa: E731
        for op in self.ops:
            k = op[0]
            if k == 'box':
                _, x, y, w, h, fill, line, radius, lw, dash = op
                kw = dict(fill=_hex(fill) if fill else None, outline=_hex(line) if line else None,
                          width=max(1, int(lw)) if line else 0)
                if radius:
                    d.rounded_rectangle([P(x), P(y), P(x + w), P(y + h)], radius=P(radius), **kw)
                else:
                    d.rectangle([P(x), P(y), P(x + w), P(y + h)], **kw)
            elif k == 'oval':
                _, x, y, w, h, fill, line = op
                d.ellipse([P(x), P(y), P(x + w), P(y + h)], fill=_hex(fill))
            elif k == 'line':
                _, x1, y1, x2, y2, color, width, arrow, dash = op
                wpx = max(1, int(width * PX / 72 + 0.5))
                if dash:
                    import math
                    L = math.hypot(x2 - x1, y2 - y1)
                    n = max(1, int(L / 0.08))
                    for i in range(0, n, 2):
                        a, b = i / n, min(1, (i + 1) / n)
                        d.line([P(x1 + (x2 - x1) * a), P(y1 + (y2 - y1) * a), P(x1 + (x2 - x1) * b), P(y1 + (y2 - y1) * b)],
                               fill=_hex(color), width=wpx)
                else:
                    d.line([P(x1), P(y1), P(x2), P(y2)], fill=_hex(color), width=wpx)
                if arrow:
                    import math
                    ang = math.atan2(y2 - y1, x2 - x1)
                    s = 0.11
                    pts = [(x2, y2), (x2 - s * math.cos(ang - 0.45), y2 - s * math.sin(ang - 0.45)),
                           (x2 - s * math.cos(ang + 0.45), y2 - s * math.sin(ang + 0.45))]
                    d.polygon([(P(a), P(b)) for a, b in pts], fill=_hex(color))
            elif k == 'image':
                _, x, y, w, h, path_, crop = op
                im = Image.open(path_).convert('RGB')
                if crop:
                    im = im.crop(crop)
                im = im.resize((max(1, P(w)), max(1, P(h))))
                img.paste(im, (P(x), P(y)))
            elif k == 'table':
                self._render_table(d, op)
            elif k in ('bar', 'linechart'):
                self._render_chart(d, op)
        for op in self.ops:
            if op[0] == 'text':
                self._render_text(d, op)
        img.save(path)

    def _render_text(self, d, op):
        _, x, y, w, h, paras, margin, anchor, what = op
        P = lambda v: int(round(v * PX))  # noqa: E731
        width = w - 2 * margin[0]
        blocks = []
        total = 0.0
        for para in paras:
            lines = wrap_lines(para, width)
            blocks.append((para, lines))
            total += sum(lh for _, lh in lines) + para.after / 72 * PX
        if paras:
            total -= paras[-1].after / 72 * PX
        avail = (h - 2 * margin[1]) * PX
        if total > avail + 1.5:
            self.deck.warnings.append(f'slide {self.number}: 글자 넘침 {what} ({total / PX:.2f}in > {avail / PX:.2f}in): '
                                      f'{paras[0].text[:30]}')
        ty = (y + margin[1]) * PX
        if anchor == 'm':
            ty += (avail - total) / 2
        elif anchor == 'b':
            ty += avail - total
        for para, lines in blocks:
            for ln, lh in lines:
                lw = sum(_font(para.size, b).getlength(t) for t, b in ln)
                if para.align == 'c':
                    tx = (x + margin[0]) * PX + (width * PX - lw) / 2
                elif para.align == 'r':
                    tx = (x + w - margin[0]) * PX - lw
                else:
                    tx = (x + margin[0] + (0.22 if para.bullet else 0)) * PX
                if para.bullet and ln is lines[0][0]:
                    d.text(((x + margin[0]) * PX, ty + lh * 0.08), '•', font=_font(para.size), fill=_hex(para.color))
                for t, b in ln:
                    f = _font(para.size, b)
                    d.text((tx, ty + lh * 0.08), t, font=f, fill=_hex(para.color))
                    tx += f.getlength(t)
                ty += lh
            ty += para.after / 72 * PX
        if x < 0 or y < 0 or x + w > W + 0.01 or y + h > H + 0.01:
            self.deck.warnings.append(f'slide {self.number}: 슬라이드 밖 {what}')

    def _render_table(self, d, op):
        _, x, y, col_w, rows, size, hf, hc, fills, heights, align, colors, bfc = op
        P = lambda v: int(round(v * PX))  # noqa: E731
        yy = y
        for i, row in enumerate(rows):
            xx = x
            for j, val in enumerate(row):
                fill = hf if i == 0 else (fills[i][j] if fills and fills[i][j] else WHITE)
                d.rectangle([P(xx), P(yy), P(xx + col_w[j]), P(yy + heights[i])], fill=_hex(fill))
                d.line([P(xx), P(yy + heights[i]), P(xx + col_w[j]), P(yy + heights[i])], fill=_hex(LINE), width=1)
                col = hc if i == 0 else (colors[i][j] if colors and colors[i][j] else INK)
                al = align[j] if align else ('l' if j == 0 else 'c')
                b = i == 0 or (j == 0 and bfc)
                paras = as_paras(str(val), size, col, b, al, 0)
                op2 = ('text', xx, yy, col_w[j], heights[i], paras, (0.06, 0.03), 'm', 'cell')
                self._render_text(d, op2)
                xx += col_w[j]
            yy += heights[i]
        if yy > H - 0.3:
            self.deck.warnings.append(f'slide {self.number}: 표가 아래로 넘침 ({yy:.2f}in)')

    def _render_chart(self, d, op):
        P = lambda v: int(round(v * PX))  # noqa: E731
        if op[0] == 'bar':
            _, x, y, w, h, cats, series, colors, vmax, major, lay, pc, legend, fmt = op
            vmin = 0
        else:
            _, x, y, w, h, cats, series, colors, vmin, vmax, major, lay, legend = op
        lx, ly, lw, lh = lay
        px0, py0, pw, ph = x + w * lx, y + h * ly, w * lw, h * lh
        f = _font(12)
        v = vmin
        while v <= vmax + 1e-9:
            yy = py0 + ph * (1 - (v - vmin) / (vmax - vmin))
            d.line([P(px0), P(yy), P(px0 + pw), P(yy)], fill='#E1E7E3', width=1)
            d.text((P(px0) - 42, P(yy) - 8), f'{v:.2f}', font=f, fill=_hex(MUTED))
            v += major
        n = len(cats)
        slot = pw / n
        for ci, c in enumerate(cats):
            tw = f.getlength(c)
            d.text((P(px0 + slot * (ci + 0.5)) - tw / 2, P(py0 + ph) + 6), c, font=f, fill=_hex(INK))
        if op[0] == 'bar':
            ns = len(series)
            bw = slot / (ns + 0.8)
            for si, (name, vals) in enumerate(series):
                for ci, val in enumerate(vals):
                    if val is None:
                        continue
                    bx = px0 + slot * ci + slot * 0.4 / (ns + 0.8) * 1 + bw * si + bw * 0.4
                    by = py0 + ph * (1 - val / vmax)
                    col = pc[ci] if pc else colors[si]
                    d.rectangle([P(bx), P(by), P(bx + bw * 0.92), P(py0 + ph)], fill=_hex(col))
                    lbl = format(val, '.3f')
                    d.text((P(bx + bw * 0.46) - f.getlength(lbl) / 2, P(by) - 17), lbl, font=f, fill=_hex(INK))
        else:
            for si, (name, vals) in enumerate(series):
                pts = [(P(px0 + slot * (ci + 0.5)), P(py0 + ph * (1 - (val - vmin) / (vmax - vmin)))) for ci, val in enumerate(vals)]
                d.line(pts, fill=_hex(colors[si]), width=3)
                for p in pts:
                    d.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], fill=_hex(colors[si]))
        if legend:
            lx_ = x + 0.3
            for si, (name, _) in enumerate(series):
                d.rectangle([P(lx_), P(y + h - 0.22), P(lx_ + 0.12), P(y + h - 0.1)], fill=_hex(colors[si]))
                d.text((P(lx_ + 0.17), P(y + h - 0.27)), name, font=f, fill=_hex(INK))
                lx_ += 0.35 + f.getlength(name) / PX
