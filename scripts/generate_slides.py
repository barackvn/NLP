import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# =============================================================================
# BẢNG MÀU CHUẨN KHOA HỌC THEO PHONG CÁCH SLIDE_02.pptx
# =============================================================================
COLOR_PRIMARY_BLUE = RGBColor(0x15, 0x65, 0xC0)    # Xanh dương UIT chuẩn #1565C0
COLOR_DARK_NAVY    = RGBColor(0x0D, 0x47, 0xA1)    # Xanh đậm tiêu đề #0D47A1
COLOR_LIGHT_BLUE   = RGBColor(0xBB, 0xDE, 0xFB)    # Viền / gạch xanh nhạt #BBDEFB
COLOR_BG_SOFT_BLUE = RGBColor(0xFA, 0xFD, 0xFF)    # Nền khối mềm #FAFDFF
COLOR_BG_HIGHLIGHT = RGBColor(0xE3, 0xF2, 0xFD)    # Highlight xanh nhạt #E3F2FD
COLOR_CALLOUT_BG   = RGBColor(0xFF, 0xF8, 0xE1)    # Nền cam mềm #FFF8E1
COLOR_CALLOUT_LINE = RGBColor(0xFF, 0x6F, 0x00)    # Viền cam #FF6F00
COLOR_CARD_BG      = RGBColor(0xF8, 0xFA, 0xFC)    # Nền thẻ xám nhẹ #F8FAFC
COLOR_CARD_BORDER  = RGBColor(0xCB, 0xD5, 0xE1)    # Viền thẻ sắc nét #CBD5E1
COLOR_TEXT_MAIN    = RGBColor(0x0F, 0x17, 0x2A)    # Chữ đen than chính sắc nét
COLOR_TEXT_MUTED   = RGBColor(0x47, 0x55, 0x69)    # Chữ xám phụ
COLOR_WHITE        = RGBColor(0xFF, 0xFF, 0xFF)    # Trắng
COLOR_GREEN        = RGBColor(0x1B, 0x5E, 0x20)    # Xanh lá điểm nhấn SOTA
COLOR_RED          = RGBColor(0xB7, 0x1C, 0x1C)    # Đỏ cảnh báo lỗi

FONT_NAME = "Arial"

def create_presentation():
    prs = Presentation()
    # Kích thước màn ảnh rộng chuẩn Widescreen 16:9 của SLIDE_02.pptx
    prs.slide_width = 21678900   # 23.71 inches
    prs.slide_height = 12192000  # 13.33 inches
    blank_layout = prs.slide_layouts[6]

    CHAPTER_TABS = [
        "1. Bối cảnh & ViHOS",
        "2. Tiền xử lý & Alignment",
        "3. Kiến trúc Đề xuất",
        "4. Thực nghiệm & Đối chứng",
        "5. Web App & Kết luận"
    ]

    def add_nav_bar(slide, active_idx):
        if active_idx is None:
            return
        bar_height = 720000
        tab_width = prs.slide_width // len(CHAPTER_TABS)
        
        # Nền thanh điều hướng
        base_bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, bar_height)
        base_bar.fill.solid()
        base_bar.fill.fore_color.rgb = RGBColor(0xEA, 0xEE, 0xF4)
        base_bar.line.fill.background()

        for i, title in enumerate(CHAPTER_TABS):
            x = i * tab_width
            tab = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, 0, tab_width, bar_height)
            tab.fill.solid()
            tf = tab.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.text = title
            p.alignment = PP_ALIGN.CENTER
            p.font.name = FONT_NAME
            p.font.size = Pt(16)   # Tăng từ 13pt lên 16pt chuẩn SLIDE_02
            p.font.bold = True
            
            if i == active_idx:
                tab.fill.fore_color.rgb = COLOR_PRIMARY_BLUE
                tab.line.fill.background()
                p.font.color.rgb = COLOR_WHITE
            else:
                tab.fill.fore_color.rgb = RGBColor(0xF1, 0xF5, 0xF9)
                tab.line.color.rgb = RGBColor(0xCB, 0xD5, 0xE1)
                tab.line.width = Pt(1)
                p.font.color.rgb = RGBColor(0x33, 0x41, 0x55)

    def add_slide_header(slide, title_text, presenter_tag=None):
        # Tiêu đề slide lớn
        title_box = slide.shapes.add_textbox(1016000, 850000, 16000000, 750000)
        tf = title_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.name = FONT_NAME
        p.font.size = Pt(32)   # Tăng từ 24pt lên 32pt to rõ
        p.font.bold = True
        p.font.color.rgb = COLOR_DARK_NAVY

        if presenter_tag:
            p2 = tf.add_paragraph()
            p2.text = f"Phụ trách trình bày: {presenter_tag}"
            p2.font.name = FONT_NAME
            p2.font.size = Pt(17)
            p2.font.italic = True
            p2.font.bold = True
            p2.font.color.rgb = COLOR_PRIMARY_BLUE

        # Đường kẻ xanh nhấn dưới tiêu đề (như SLIDE_02.pptx)
        line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 1016000, 1850000, 19642709, 50000)
        line.fill.solid()
        line.fill.fore_color.rgb = COLOR_PRIMARY_BLUE
        line.line.fill.background()

    def add_footer(slide, page_num, total_pages=18):
        foot_left = slide.shapes.add_textbox(1016000, 11500000, 14000000, 450000)
        tf_l = foot_left.text_frame
        p_l = tf_l.paragraphs[0]
        p_l.text = "Đồ án Xử lý Ngôn ngữ Tự nhiên | PhoBERT-BiLSTM-CRF Toxic Spans Guard | UIT - ĐHQG-HCM"
        p_l.font.name = FONT_NAME
        p_l.font.size = Pt(15)  # Tăng lên 15pt
        p_l.font.color.rgb = COLOR_TEXT_MUTED

        foot_right = slide.shapes.add_textbox(18500000, 11500000, 2150000, 450000)
        tf_r = foot_right.text_frame
        p_r = tf_r.paragraphs[0]
        p_r.text = f"{page_num} / {total_pages}"
        p_r.alignment = PP_ALIGN.RIGHT
        p_r.font.name = FONT_NAME
        p_r.font.size = Pt(18)  # Tăng lên 18pt
        p_r.font.bold = True
        p_r.font.color.rgb = COLOR_PRIMARY_BLUE

    def add_card(slide, left, top, width, height, bg_color=COLOR_CARD_BG, border_color=COLOR_CARD_BORDER):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        if border_color:
            card.line.color.rgb = border_color
            card.line.width = Pt(1.5)
        else:
            card.line.fill.background()
        return card

    def add_notes(slide, notes_text):
        slide.notes_slide.notes_text_frame.text = notes_text

    # =========================================================================
    # SLIDE 01: BÌA BÁO CÁO ĐỒ ÁN (Title Slide)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    top_decor = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, 220000)
    top_decor.fill.solid()
    top_decor.fill.fore_color.rgb = COLOR_PRIMARY_BLUE
    top_decor.line.fill.background()

    head_box = s1.shapes.add_textbox(1016000, 600000, 19642709, 850000)
    p = head_box.text_frame.paragraphs[0]
    p.text = "ĐẠI HỌC QUỐC GIA TP. HỒ CHÍ MINH — TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN (UIT)"
    p.font.name = FONT_NAME; p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = COLOR_PRIMARY_BLUE
    
    p_sub = head_box.text_frame.add_paragraph()
    p_sub.text = "KHOA KHOA HỌC MÁY TÍNH | BÁO CÁO ĐỒ ÁN MÔN HỌC: XỬ LÝ NGÔN NGỮ TỰ NHIÊN (NLP)"
    p_sub.font.name = FONT_NAME; p_sub.font.size = Pt(16); p_sub.font.color.rgb = COLOR_TEXT_MUTED

    sep = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 1016000, 1550000, 19642709, 50000)
    sep.fill.solid(); sep.fill.fore_color.rgb = COLOR_PRIMARY_BLUE; sep.line.fill.background()

    tbox = s1.shapes.add_textbox(1016000, 1800000, 19642709, 2100000)
    tf = tbox.text_frame; tf.word_wrap = True
    p1 = tf.paragraphs[0]
    p1.text = "CẢI TIẾN PHƯƠNG PHÁP NHẬN DIỆN CHUỖI NGÔN NGỮ XÚC PHẠM TIẾNG VIỆT"
    p1.font.name = FONT_NAME; p1.font.size = Pt(32); p1.font.bold = True; p1.font.color.rgb = COLOR_DARK_NAVY

    p2 = tf.add_paragraph()
    p2.text = "BẰNG MÔ HÌNH PhoBERT-BiLSTM-CRF TRÊN BENCHMARK ViHOS (EACL 2023)"
    p2.font.name = FONT_NAME; p2.font.size = Pt(28); p2.font.bold = True; p2.font.color.rgb = COLOR_PRIMARY_BLUE

    gv_card = add_card(s1, 1016000, 4100000, 19642709, 1300000, bg_color=COLOR_BG_HIGHLIGHT, border_color=COLOR_LIGHT_BLUE)
    gv_box = s1.shapes.add_textbox(1200000, 4200000, 19000000, 1100000)
    p_gv = gv_box.text_frame.paragraphs[0]
    p_gv.text = "Giảng viên hướng dẫn: NCS.ThS. Đặng Văn Thìn  &  Tác giả Trần Quốc Khánh (EACL 2023)"
    p_gv.font.name = FONT_NAME; p_gv.font.size = Pt(19); p_gv.font.bold = True; p_gv.font.color.rgb = COLOR_DARK_NAVY
    p_gv2 = gv_box.text_frame.add_paragraph()
    p_gv2.text = "Nhóm thực hiện: Nhóm Nghiên Cứu ViHOS Guard | Học kỳ II — Năm học 2025–2026"
    p_gv2.font.name = FONT_NAME; p_gv2.font.size = Pt(16); p_gv2.font.color.rgb = COLOR_TEXT_MAIN

    # Bảng 5 Thành viên
    table_shape = s1.shapes.add_table(6, 4, 1016000, 5650000, 19642709, 5400000)
    table = table_shape.table
    table.columns[0].width = 2200000
    table.columns[1].width = 4800000
    table.columns[2].width = 8000000
    table.columns[3].width = 4642709

    headers = ["MSSV", "HỌ VÀ TÊN", "VAI TRÒ & PHỤ TRÁCH KỸ THUẬT", "SLIDE BẢO VỆ"]
    for j, h in enumerate(headers):
        cell = table.cell(0, j); cell.fill.solid(); cell.fill.fore_color.rgb = COLOR_PRIMARY_BLUE
        p = cell.text_frame.paragraphs[0]; p.text = h; p.font.name = FONT_NAME; p.font.size = Pt(17); p.font.bold = True; p.font.color.rgb = COLOR_WHITE; p.alignment = PP_ALIGN.CENTER

    members_data = [
        ("26410127", "Dương Quốc Thương", "Project Leader & Architect — Thiết kế kiến trúc tổng thể, duyệt model.py", "Slide 01–04 & 18"),
        ("26410115", "Nông Nguyễn Thành", "Data & NLP Core — Quản lý ViHOS Benchmark, First-token Alignment, EDA", "Slide 05–09"),
        ("26410146", "Hoàng Võ Minh Tuấn", "Model Trainer — Huấn luyện Colab Tesla T4, Differential LR, 3 checkpoints", "Slide 10–11"),
        ("26410108", "Bùi Quốc Thịnh", "Evaluation & Metrics — Chạy Ablation seqeval, 4 biểu đồ, phân tích lỗi", "Slide 12–15"),
        ("26410024", "Trần Tiến Dũng", "Product Developer — Web App Client-Server (FastAPI + React 19), Auto-Masking", "Slide 16–17"),
    ]

    for i, row in enumerate(members_data):
        for j, val in enumerate(row):
            cell = table.cell(i+1, j); cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if i % 2 == 0 else RGBColor(0xF1, 0xF5, 0xF9)
            p = cell.text_frame.paragraphs[0]; p.text = val; p.font.name = FONT_NAME; p.font.size = Pt(16); p.font.color.rgb = COLOR_TEXT_MAIN
            if j == 0 or j == 3: p.alignment = PP_ALIGN.CENTER; p.font.bold = True

    add_notes(s1, "Lời chào: Kính thưa quý Thầy trong Hội đồng chấm đồ án môn Xử lý Ngôn ngữ Tự nhiên. Hôm nay nhóm chúng em gồm 5 thành viên xin phép báo cáo đề tài nghiên cứu: Cải tiến phương pháp nhận diện chuỗi xúc phạm tiếng Việt bằng mô hình PhoBERT-BiLSTM-CRF trên bộ dữ liệu chuẩn ViHOS EACL 2023.")

    # =========================================================================
    # SLIDE 02: MỤC LỤC RÕ RÀNG & CHI TIẾT (Agenda)
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    bar_top = s2.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, 720000)
    bar_top.fill.solid(); bar_top.fill.fore_color.rgb = COLOR_PRIMARY_BLUE; bar_top.line.fill.background()
    
    t_top = s2.shapes.add_textbox(1016000, 0, 8000000, 720000)
    p = t_top.text_frame.paragraphs[0]; p.text = "MỤC LỤC NỘI DUNG BÁO CÁO"; p.font.name = FONT_NAME; p.font.size = Pt(20); p.font.bold = True; p.font.color.rgb = COLOR_WHITE

    add_slide_header(s2, "Mục Lục & Lộ Trình Thuyết Trình Báo Cáo", presenter_tag="Toàn Nhóm 5 Thành Viên")
    add_footer(s2, 2)

    agenda_items = [
        ("01", "BỐI CẢNH & BÀI TOÁN TOXIC SPANS", "Thực trạng MXH, hạn chế phân loại câu, bài toán Sequence Labeling & ViHOS Benchmark (EACL 2023).", "Dương Quốc Thương (Slide 03–04)"),
        ("02", "XỬ LÝ DỮ LIỆU & SUBWORD ALIGNMENT", "Mất cân bằng nhãn BIO, đặc thù tiếng Việt, Subword BPE PhoBERT, kỹ thuật First-Token Alignment.", "Nông Nguyễn Thành (Slide 05–09)"),
        ("03", "KIẾN TRÚC ĐỀ XUẤT PhoBERT-BiLSTM-CRF", "Mô hình 3 tầng xếp chồng, giải mã Viterbi toàn cục, chiến lược Differential LR trên GPU Tesla T4.", "Hoàng Võ Minh Tuấn (Slide 10–11)"),
        ("04", "KẾT QUẢ THỰC NGHIỆM & ĐỐI CHỨNG", "Bảng so sánh 3 trường phái, triệt tiêu 100% lỗi O->I, đột phá câu đa chuỗi, ca điển hình 'con kẹt'.", "Bùi Quốc Thịnh (Slide 12–15)"),
        ("05", "SẢN PHẨM WEB APP CLIENT-SERVER & KẾT LUẬN", "FastAPI + React 19, tính năng Auto-Masking, kiểm soát 3 mô hình, điểm hạn chế & định hướng.", "Trần Tiến Dũng & Thương (Slide 16–18)")
    ]

    coords = [
        (1016000, 2150000, 9500000, 2550000),     # 01
        (11150000, 2150000, 9500000, 2550000),    # 02
        (1016000, 4950000, 9500000, 2550000),     # 03
        (11150000, 4950000, 9500000, 2550000),    # 04
        (1016000, 7750000, 19642709, 2450000),    # 05
    ]

    for idx, (num, title, desc, speaker) in enumerate(agenda_items):
        cx, cy, cw, ch = coords[idx]
        card = add_card(s2, cx, cy, cw, ch, bg_color=COLOR_CARD_BG, border_color=COLOR_LIGHT_BLUE)
        
        num_box = s2.shapes.add_textbox(cx + 250000, cy + 200000, 1400000, 900000)
        p = num_box.text_frame.paragraphs[0]; p.text = num; p.font.name = FONT_NAME; p.font.size = Pt(36); p.font.bold = True; p.font.color.rgb = COLOR_PRIMARY_BLUE

        content_box = s2.shapes.add_textbox(cx + 1700000, cy + 150000, cw - 1900000, ch - 300000)
        tf = content_box.text_frame; tf.word_wrap = True
        
        p_t = tf.paragraphs[0]; p_t.text = title; p_t.font.name = FONT_NAME; p_t.font.size = Pt(19); p_t.font.bold = True; p_t.font.color.rgb = COLOR_DARK_NAVY
        p_d = tf.add_paragraph(); p_d.text = desc; p_d.font.name = FONT_NAME; p_d.font.size = Pt(16); p_d.font.color.rgb = COLOR_TEXT_MAIN
        p_s = tf.add_paragraph(); p_s.text = f"👤 Trình bày: {speaker}"; p_s.font.name = FONT_NAME; p_s.font.size = Pt(15); p_s.font.bold = True; p_s.font.color.rgb = COLOR_CALLOUT_LINE

    summary_card = add_card(s2, 1016000, 10450000, 19642709, 850000, bg_color=COLOR_BG_HIGHLIGHT, border_color=COLOR_PRIMARY_BLUE)
    sum_box = s2.shapes.add_textbox(1200000, 10500000, 19200000, 750000)
    p = sum_box.text_frame.paragraphs[0]
    p.text = "📌 Quy chuẩn bài báo cáo: 18 Slide | Cấu trúc 5 phần khoa học gắn liền với vai trò cụ thể của 5 thành viên nhóm."
    p.font.name = FONT_NAME; p.font.size = Pt(17); p.font.bold = True; p.font.color.rgb = COLOR_DARK_NAVY

    add_notes(s2, "Dương Quốc Thương: Báo cáo gồm 5 chương mạch lạc, phân bổ hợp lý giữa lý thuyết, tiền xử lý, kiến trúc, thực nghiệm định lượng và sản phẩm thực tế.")

    # =========================================================================
    # SLIDE 03: THỰC TRẠNG & BÀI TOÁN TOXIC SPANS DETECTION
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_nav_bar(s3, 0)
    add_slide_header(s3, "Thực Trạng & Giới Hạn Của Phân Loại Cấp Độ Câu", presenter_tag="Dương Quốc Thương (26410127)")
    add_footer(s3, 3)

    c1 = add_card(s3, 1016000, 2150000, 9500000, 6700000)
    box1 = s3.shapes.add_textbox(1250000, 2350000, 9000000, 6300000)
    tf1 = box1.text_frame; tf1.word_wrap = True
    p = tf1.paragraphs[0]; p.text = "1. Thực Trạng Phát Ngôn Thù Ghét Trên MXH"; p.font.size = Pt(22); p.font.bold = True; p.font.color.rgb = COLOR_DARK_NAVY

    bullets_s3_1 = [
        "• Sự bùng nổ của mạng xã hội (Facebook, TikTok, YouTube) kéo theo làn sóng phát ngôn kích động thù địch, xúc phạm danh dự cá nhân và tổ chức.",
        "• Ngôn ngữ mạng tiếng Việt vô cùng phong phú: sử dụng từ lóng địa phương, biến âm nói giảm nói tránh, chửi thề viết tắt teencode.",
        "• Đòi hỏi các nền tảng phải có cơ chế kiểm duyệt tự động, chính xác và có tính giải thích cao."
    ]
    for b in bullets_s3_1:
        p = tf1.add_paragraph(); p.text = b; p.font.size = Pt(18); p.font.color.rgb = COLOR_TEXT_MAIN

    c2 = add_card(s3, 11150000, 2150000, 9500000, 6700000)
    box2 = s3.shapes.add_textbox(11400000, 2350000, 9000000, 6300000)
    tf2 = box2.text_frame; tf2.word_wrap = True
    p = tf2.paragraphs[0]; p.text = "2. Hạn Chế Của Sentence-Level Classification"; p.font.size = Pt(22); p.font.bold = True; p.font.color.rgb = COLOR_RED

    bullets_s3_2 = [
        "• Phương pháp truyền thống: Gán nhãn nhị phân toàn câu (0: Sạch, 1: Độc hại).",
        "• Hộp đen (Black-box): Không chỉ ra được cụ thể từ/cụm từ nào gây ra vi phạm.",
        "• Buộc hệ thống phải XÓA TOÀN BỘ CÂU, làm đứt gãy luồng thảo luận văn minh.",
        "• Không thể tự động che mờ từ ngữ nhạy cảm (Auto-masking) để giữ lại nội dung xây dựng."
    ]
    for b in bullets_s3_2:
        p = tf2.add_paragraph(); p.text = b; p.font.size = Pt(18); p.font.color.rgb = COLOR_TEXT_MAIN

    callout3 = add_card(s3, 1016000, 9150000, 19642709, 2150000, bg_color=COLOR_CALLOUT_BG, border_color=COLOR_CALLOUT_LINE)
    box3 = s3.shapes.add_textbox(1250000, 9300000, 19100000, 1850000)
    tf3 = box3.text_frame; tf3.word_wrap = True
    p = tf3.paragraphs[0]; p.text = "💡 Giải pháp cốt lõi: Chuyển dịch từ Phân loại câu sang Gán nhãn chuỗi cấp độ từ (Toxic Spans Detection)"; p.font.size = Pt(20); p.font.bold = True; p.font.color.rgb = COLOR_CALLOUT_LINE
    p2 = tf3.add_paragraph()
    p2.text = "Xác định chính xác vị trí bắt đầu và kết thúc của từng cụm từ xúc phạm. Giúp che mờ tự động (Auto-Masking ***) và hỗ trợ kiểm duyệt viên rà soát nhanh gấp 10 lần."
    p2.font.size = Pt(17); p2.font.color.rgb = COLOR_TEXT_MAIN

    add_notes(s3, "Dương Quốc Thương: Sentence-level classification chỉ biết xóa cả câu. Toxic Spans Detection bóc tách đích danh từ ngữ bậy để thực thi kiểm duyệt văn minh.")

    # =========================================================================
    # SLIDE 04: ĐỊNH NGHĨA BÀI TOÁN & BỘ DỮ LIỆU ViHOS BENCHMARK
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_nav_bar(s4, 0)
    add_slide_header(s4, "Định Nghĩa Bài Toán & Bộ Dữ Liệu ViHOS Benchmark (EACL 2023)", presenter_tag="Dương Quốc Thương (26410127)")
    add_footer(s4, 4)

    card4_1 = add_card(s4, 1016000, 2150000, 6200000, 6700000)
    box = s4.shapes.add_textbox(1200000, 2350000, 5800000, 6300000)
    tf = box.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = "1. Sequence Labeling"; p.font.size = Pt(21); p.font.bold = True; p.font.color.rgb = COLOR_DARK_NAVY
    items = [
        "• Cho câu văn gồm n từ: X = (w_1, w_2, ..., w_n).",
        "• Dự đoán chuỗi nhãn BIO tương ứng: Y = (y_1, y_2, ..., y_n).",
        "• B-HOS: Bắt đầu cụm xúc phạm.",
        "• I-HOS: Các từ tiếp theo trong cụm.",
        "• O: Từ ngữ trung tính, sạch."
    ]
    for it in items:
        p = tf.add_paragraph(); p.text = it; p.font.size = Pt(17); p.font.color.rgb = COLOR_TEXT_MAIN

    card4_2 = add_card(s4, 7730000, 2150000, 6200000, 6700000)
    box = s4.shapes.add_textbox(7900000, 2350000, 5800000, 6300000)
    tf = box.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = "2. ViHOS Benchmark"; p.font.size = Pt(21); p.font.bold = True; p.font.color.rgb = COLOR_PRIMARY_BLUE
    items = [
        "• Công bố tại Hội nghị quốc tế EACL 2023 (Trần Quốc Khánh et al.).",
        "• Bộ dữ liệu chuẩn đầu tiên cho Toxic Spans trên tiếng Việt.",
        "• Quy mô: 11.056 bình luận mạng xã hội gán nhãn thủ công kỹ lưỡng.",
        "• Phân chia chuẩn: 80% Train, 10% Dev, 10% Test."
    ]
    for it in items:
        p = tf.add_paragraph(); p.text = it; p.font.size = Pt(17); p.font.color.rgb = COLOR_TEXT_MAIN

    card4_3 = add_card(s4, 14450000, 2150000, 6200000, 6700000)
    box = s4.shapes.add_textbox(14650000, 2350000, 5800000, 6300000)
    tf = box.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = "3. Thang Đo Span-F1"; p.font.size = Pt(21); p.font.bold = True; p.font.color.rgb = COLOR_GREEN
    items = [
        "• Đánh giá nghiêm ngặt theo cụm Span-level qua thư viện seqeval.",
        "• Một cụm từ vi phạm chỉ được xem là đúng (True Positive) khi:",
        "  - Trùng khớp 100% ranh giới từ đầu đến từ cuối.",
        "  - Đúng trọn vẹn nhãn phân loại."
    ]
    for it in items:
        p = tf.add_paragraph(); p.text = it; p.font.size = Pt(17); p.font.color.rgb = COLOR_TEXT_MAIN

    callout4 = add_card(s4, 1016000, 9150000, 19642709, 2150000, bg_color=COLOR_BG_HIGHLIGHT, border_color=COLOR_PRIMARY_BLUE)
    box = s4.shapes.add_textbox(1250000, 9300000, 19100000, 1850000)
    tf = box.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = "📌 Ví dụ gán nhãn BIO trực quan:"; p.font.size = Pt(19); p.font.bold = True; p.font.color.rgb = COLOR_DARK_NAVY
    p2 = tf.add_paragraph()
    p2.text = "Câu mẫu: [Đồ/B-HOS] [ngu/I-HOS] ,/O [mày/B-HOS] [hãm/I-HOS] [vừa_phải/I-HOS] [thôi/O] ."
    p2.font.size = Pt(18); p2.font.bold = True; p2.font.color.rgb = COLOR_RED
    p3 = tf.add_paragraph()
    p3.text = "=> Hai chuỗi xúc phạm độc lập: Span 1: 'Đồ ngu' | Span 2: 'mày hãm vừa_phải' | Các từ còn lại giữ nhãn O."
    p3.font.size = Pt(17); p3.font.color.rgb = COLOR_TEXT_MAIN

    add_notes(s4, "Dương Quốc Thương: Tiêu chuẩn Span-F1 seqeval cực kỳ khắt khe: chỉ cần sai lệch 1 từ ranh giới là tính là dự đoán sai hoàn toàn.")

    # =========================================================================
    # SLIDE 05: KHÁM PHÁ BỘ DỮ LIỆU ViHOS
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_nav_bar(s5, 1)
    add_slide_header(s5, "Khám Phá & Phân Tích Bộ Dữ Liệu ViHOS Benchmark", presenter_tag="Nông Nguyễn Thành (26410115)")
    add_footer(s5, 5)

    c5_1 = add_card(s5, 1016000, 2150000, 9500000, 4200000)
    b1 = s5.shapes.add_textbox(1250000, 2300000, 9000000, 3900000)
    tf1 = b1.text_frame; tf1.word_wrap = True
    p = tf1.paragraphs[0]; p.text = "1. Phân Chia Tập Dữ Liệu Chuẩn"; p.font.size = Pt(21); p.font.bold = True; p.font.color.rgb = COLOR_PRIMARY_BLUE
    bullets = [
        "• Tập Train: 8.844 câu (80%) — Huấn luyện mô hình.",
        "• Tập Dev: 1.106 câu (10%) — Tinh chỉnh siêu tham số.",
        "• Tập Test: 1.106 câu (10%) — Đánh giá mù độc lập.",
        "• Tổng cộng: 11.056 câu với hơn 140.000 tokens từ vựng."
    ]
    for b in bullets:
        p = tf1.add_paragraph(); p.text = b; p.font.size = Pt(18); p.font.color.rgb = COLOR_TEXT_MAIN

    c5_2 = add_card(s5, 11150000, 2150000, 9500000, 4200000)
    b2 = s5.shapes.add_textbox(11400000, 2300000, 9000000, 3900000)
    tf2 = b2.text_frame; tf2.word_wrap = True
    p = tf2.paragraphs[0]; p.text = "2. Thách Thức Mất Cân Bằng Nhãn Trầm Trọng"; p.font.size = Pt(21); p.font.bold = True; p.font.color.rgb = COLOR_RED
    bullets = [
        "• Nhãn sạch (O): Chiếm tới 88.5% tổng số tokens.",
        "• Nhãn bắt đầu (B-HOS): Chỉ chiếm 6.2% tổng số tokens.",
        "• Nhãn bên trong (I-HOS): Chỉ chiếm 5.3% tổng số tokens.",
        "• Tỷ lệ mất cân bằng lớn giữa từ sạch và từ xúc phạm gây khó khăn lớn cho việc huấn luyện mô hình phân loại."
    ]
    for b in bullets:
        p = tf2.add_paragraph(); p.text = b; p.font.size = Pt(18); p.font.color.rgb = COLOR_TEXT_MAIN

    table_shape = s5.shapes.add_table(3, 5, 1016000, 6650000, 19642709, 4600000)
    t = table_shape.table
    cats = [
        ("INSULT", "Lăng mạ, xúc phạm nhân phẩm", "đồ ngu, mặt hãm, đần độn"),
        ("PROFANITY", "Chửi thề thô tục, bộ phận sinh dục", "đcm, vcl, lồn, con cặc"),
        ("THREAT", "Đe dọa bạo lực, tính mạng", "chém chết mẹ mày, đập nát mặt"),
        ("DISCRIMINATION", "Kỳ thị vùng miền, giới tính", "bắc kỳ ăn cá rô cây, đồ bê đê"),
        ("OTHER", "Các dạng công kích khác", "hám fame, đu bám, rác rưởi")
    ]
    for j, (code, desc, ex) in enumerate(cats):
        t.columns[j].width = prs.slide_width // 5 - 200000
        cell = t.cell(0, j); cell.fill.solid(); cell.fill.fore_color.rgb = COLOR_PRIMARY_BLUE
        p = cell.text_frame.paragraphs[0]; p.text = code; p.font.name = FONT_NAME; p.font.size = Pt(17); p.font.bold = True; p.font.color.rgb = COLOR_WHITE; p.alignment = PP_ALIGN.CENTER
        
        cell2 = t.cell(1, j); cell2.fill.solid(); cell2.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        p = cell2.text_frame.paragraphs[0]; p.text = desc; p.font.name = FONT_NAME; p.font.size = Pt(15); p.font.color.rgb = COLOR_DARK_NAVY; p.font.bold = True
        
        cell3 = t.cell(2, j); cell3.fill.solid(); cell3.fill.fore_color.rgb = RGBColor(0xF1, 0xF5, 0xF9)
        p = cell3.text_frame.paragraphs[0]; p.text = f"Ví dụ: {ex}"; p.font.name = FONT_NAME; p.font.size = Pt(15); p.font.italic = True; p.font.color.rgb = COLOR_RED

    add_notes(s5, "Nông Nguyễn Thành: Nhãn O chiếm 88.5%, sự mất cân bằng này đòi hỏi mô hình phải có cơ chế ràng buộc chuỗi mạnh mẽ để không bị thiên kiến gán toàn bộ nhãn O.")

    # =========================================================================
    # SLIDE 06: ĐẶC THÙ NGÔN NGỮ TIẾNG VIỆT TRÊN MẠNG XÃ HỘI
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_nav_bar(s6, 1)
    add_slide_header(s6, "Đặc Thù Ngôn Ngữ Tiếng Việt Trên Mạng Xã Hội", presenter_tag="Nông Nguyễn Thành (26410115)")
    add_footer(s6, 6)

    features = [
        ("1. Từ Ghép & Ranh Giới Từ Đơn Lập", "Tiếng Việt là ngôn ngữ đơn lập, ranh giới từ không thể xác định bằng khoảng trắng đơn thuần.", "Ví dụ: 'mất dạy', 'thất đức'. Nếu chém cụt thành 'mất' (động từ) và 'dạy' (giáo dục) thì mất hoàn toàn ý nghĩa xúc phạm."),
        ("2. Teencode & Biến Âm Né Bộ Lọc (Filter Evasion)", "Người dùng MXH cố tình viết chệch âm, thay ký tự số để qua mặt các bộ lọc từ điển truyền thống.", "Ví dụ: 'đm' -> 'đcm', 'vcl' -> 'vkl', 'lồn' -> 'l0n', 'chó' -> 'chóooo'. Cần mô hình hiểu ngữ cảnh sâu để nhận diện."),
        ("3. Từ Lóng Địa Phương & Giảm Thanh (Euphemisms)", "Tiếng lóng biến âm thay thế từ thô tục bằng từ đồng âm vô hại trong ngữ liệu chuẩn.", "Ví dụ: 'như con kẹt' (biến âm của 'con cặc'). Từ 'kẹt' trong tiếng Việt chuẩn là từ sạch (kẹt xe, kẹt tiền), đòi hỏi mô hình phải xử lý tinh tế.")
    ]

    for idx, (title, desc, ex) in enumerate(features):
        top_pos = 2150000 + idx * 3050000
        card = add_card(s6, 1016000, top_pos, 19642709, 2800000)
        box = s6.shapes.add_textbox(1250000, top_pos + 150000, 19100000, 2500000)
        tf = box.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; p.text = title; p.font.size = Pt(21); p.font.bold = True; p.font.color.rgb = COLOR_PRIMARY_BLUE
        p2 = tf.add_paragraph(); p2.text = desc; p2.font.size = Pt(17); p2.font.color.rgb = COLOR_TEXT_MAIN
        p3 = tf.add_paragraph(); p3.text = ex; p3.font.size = Pt(16); p3.font.italic = True; p3.font.bold = True; p3.font.color.rgb = COLOR_CALLOUT_LINE

    add_notes(s6, "Nông Nguyễn Thành: Tiếng Việt trên MXH có 3 đặc thù cốt lõi: ranh giới từ ghép, teencode né filter và từ lóng biến âm giảm thanh.")

    # =========================================================================
    # SLIDE 07: THÁCH THỨC SUBWORD BPE CỦA PhoBERT
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_nav_bar(s7, 1)
    add_slide_header(s7, "Thách Thức Subword BPE & Lệch Ranh Giới Từ Tố (@@)", presenter_tag="Nông Nguyễn Thành (26410115)")
    add_footer(s7, 7)

    c1 = add_card(s7, 1016000, 2150000, 9500000, 8900000)
    b1 = s7.shapes.add_textbox(1250000, 2350000, 9000000, 8500000)
    tf1 = b1.text_frame; tf1.word_wrap = True
    p = tf1.paragraphs[0]; p.text = "Cơ Chế BPE Tokenizer Của PhoBERT"; p.font.size = Pt(22); p.font.bold = True; p.font.color.rgb = COLOR_DARK_NAVY
    bullets7_1 = [
        "• PhoBERT sử dụng Byte-Pair Encoding (BPE) với từ điển ~64.000 subwords.",
        "• Khi gặp từ ghép hoặc từ chưa từng thấy trong từ điển (OOV), BPE tự động CHẺ NHỎ từ thành nhiều mẩu subword kèm ký hiệu '@@'.",
        "• Ví dụ: Từ 'mất_dạy' bị phân mảnh thành:",
        "  -> 'mất@@'",
        "  -> 'dạy'",
        "• Ví dụ: Từ 'ngu_ngốc' bị phân mảnh thành:",
        "  -> 'ngu@@'",
        "  -> 'ngốc'"
    ]
    for b in bullets7_1:
        p = tf1.add_paragraph(); p.text = b; p.font.size = Pt(18); p.font.color.rgb = COLOR_TEXT_MAIN

    c2 = add_card(s7, 11150000, 2150000, 9500000, 8900000, bg_color=COLOR_CALLOUT_BG, border_color=COLOR_CALLOUT_LINE)
    b2 = s7.shapes.add_textbox(11400000, 2350000, 9000000, 8500000)
    tf2 = b2.text_frame; tf2.word_wrap = True
    p = tf2.paragraphs[0]; p.text = "Vấn Đề Lệch Độ Dài Sequence (Mismatch)"; p.font.size = Pt(22); p.font.bold = True; p.font.color.rgb = COLOR_RED
    bullets7_2 = [
        "⚠️ Xung đột cấu trúc nhãn:",
        "• Nhãn BIO của ViHOS được gán ở CẤP ĐỘ TỪ (Word-level, phân cách bằng dấu '_').",
        "• Nhưng PhoBERT lại xử lý ở CẤP ĐỘ SUBWORD.",
        "• Hậu quả:",
        "  - Chuỗi vector PhoBERT DÀI HƠN số nhãn BIO thực tế của câu.",
        "  - Nếu không có giải pháp căn chỉnh, toàn bộ pipeline huấn luyện sẽ bị lệch chỉ số hoặc báo lỗi kích thước Tensor!"
    ]
    for b in bullets7_2:
        p = tf2.add_paragraph(); p.text = b; p.font.size = Pt(18); p.font.color.rgb = COLOR_TEXT_MAIN

    add_notes(s7, "Nông Nguyễn Thành: PhoBERT chẻ nhỏ từ ghép thành subwords @@, làm chuỗi vector dài hơn chuỗi nhãn BIO của ViHOS. Đây là bài toán kỹ thuật bắt buộc phải giải quyết.")

    # =========================================================================
    # SLIDE 08: GIẢI PHÁP FIRST-TOKEN SUBWORD ALIGNMENT
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_nav_bar(s8, 1)
    add_slide_header(s8, "Kỹ Thuật First-Token Subword Alignment Chuẩn Xác", presenter_tag="Nông Nguyễn Thành (26410115)")
    add_footer(s8, 8)

    card8_top = add_card(s8, 1016000, 2150000, 19642709, 4400000, bg_color=COLOR_BG_HIGHLIGHT, border_color=COLOR_PRIMARY_BLUE)
    box = s8.shapes.add_textbox(1250000, 2300000, 19100000, 4100000)
    tf = box.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = "Nguyên Lý Kỹ Thuật First-Token Alignment:"; p.font.size = Pt(22); p.font.bold = True; p.font.color.rgb = COLOR_DARK_NAVY
    items8 = [
        "1. Duyệt qua từng từ gốc trong câu và tạo tokenization tương ứng qua PhoBERT Tokenizer.",
        "2. Subword ĐẦU TIÊN của từ sẽ đại diện cho từ đó và ĐƯỢC GÁN ĐÚNG NHÃN GỐC (B-HOS, I-HOS hoặc O).",
        "3. Tất cả các Subword PHÍA SAU (chứa hậu tố '@@') được gắn nhãn đặc biệt: -100 (Ignore Index).",
        "4. Tầng tính Loss (Cross-Entropy / CRF Loss) sẽ tự động BỎ QUA các vị trí -100, bảo toàn chính xác 100% độ dài nhãn của tập ViHOS."
    ]
    for it in items8:
        p = tf.add_paragraph(); p.text = it; p.font.size = Pt(18); p.font.color.rgb = COLOR_TEXT_MAIN

    t_shape = s8.shapes.add_table(4, 5, 1016000, 6800000, 19642709, 4300000)
    t = t_shape.table
    cols_w = [4000000, 3900000, 3900000, 3900000, 3942709]
    for j, w in enumerate(cols_w): t.columns[j].width = w

    row0 = ["Thuộc tính", "Từ 1 (Đồ)", "Từ 2 (mất_dạy)", "Từ 2 (mất_dạy - đuôi)", "Từ 3 (thật_sự)"]
    row1 = ["Từ gốc (Word)", "Đồ", "mất_dạy", "(phân mảnh)", "thật_sự"]
    row2 = ["Subwords BPE", "Đồ", "mất@@", "dạy", "thật_sự"]
    row3 = ["Nhãn huấn luyện", "B-HOS", "I-HOS (Giữ lại)", "-100 (Bỏ qua)", "O (Giữ lại)"]

    for i, row in enumerate([row0, row1, row2, row3]):
        for j, val in enumerate(row):
            cell = t.cell(i, j); cell.fill.solid()
            if i == 0: cell.fill.fore_color.rgb = COLOR_PRIMARY_BLUE
            elif i == 3: cell.fill.fore_color.rgb = RGBColor(0xE8, 0xF5, 0xE9) if "-100" not in val else RGBColor(0xFF, 0xEB, 0xEE)
            else: cell.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if i%2==1 else RGBColor(0xF1, 0xF5, 0xF9)
            p = cell.text_frame.paragraphs[0]; p.text = val; p.font.name = FONT_NAME; p.font.size = Pt(16)
            if i == 0: p.font.bold = True; p.font.color.rgb = COLOR_WHITE
            elif i == 3: p.font.bold = True; p.font.color.rgb = COLOR_RED if "-100" in val else COLOR_GREEN
            else: p.font.color.rgb = COLOR_TEXT_MAIN
            p.alignment = PP_ALIGN.CENTER

    add_notes(s8, "Nông Nguyễn Thành: Nhóm dùng kỹ thuật First-token Alignment với nhãn -100 để triệt tiêu hoàn toàn lỗi lệch ranh giới subword.")

    # =========================================================================
    # SLIDE 09: KHẢO SÁT CHIỀU DÀI CÂU & MAX LENGTH = 128
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_nav_bar(s9, 1)
    add_slide_header(s9, "Khảo Sát Phân Bố Độ Dài Câu & Cắt Tỉa Tối Ưu (Max Length = 128)", presenter_tag="Nông Nguyễn Thành (26410115)")
    add_footer(s9, 9)

    c1 = add_card(s9, 1016000, 2150000, 9500000, 8900000)
    b1 = s9.shapes.add_textbox(1250000, 2350000, 9000000, 8500000)
    tf1 = b1.text_frame; tf1.word_wrap = True
    p = tf1.paragraphs[0]; p.text = "Số Liệu Phân Bố Độ Dài Câu ViHOS"; p.font.size = Pt(22); p.font.bold = True; p.font.color.rgb = COLOR_PRIMARY_BLUE
    bullets9 = [
        "• Độ dài trung bình: 19.8 tokens / câu.",
        "• Trung vị (Median): 16.0 tokens / câu.",
        "• Độ dài ngắn nhất: 1 token.",
        "• Độ dài câu dài nhất: 243 tokens.",
        "• Tỷ lệ câu có độ dài <= 128 tokens chiếm tới: 98.7% toàn bộ dữ liệu.",
        "• Chỉ có 1.3% các câu dài trên 128 tokens (chủ yếu là spam lặp từ)."
    ]
    for b in bullets9:
        p = tf1.add_paragraph(); p.text = b; p.font.size = Pt(18); p.font.color.rgb = COLOR_TEXT_MAIN

    c2 = add_card(s9, 11150000, 2150000, 9500000, 8900000, bg_color=COLOR_BG_HIGHLIGHT, border_color=COLOR_PRIMARY_BLUE)
    b2 = s9.shapes.add_textbox(11400000, 2350000, 9000000, 8500000)
    tf2 = b2.text_frame; tf2.word_wrap = True
    p = tf2.paragraphs[0]; p.text = "Chiến Lược Cắt Tỉa MAX_LENGTH = 128"; p.font.size = Pt(22); p.font.bold = True; p.font.color.rgb = COLOR_DARK_NAVY
    bullets9_2 = [
        "✅ Bảo toàn thông tin: Giữ lại 100% ngữ nghĩa của 98.7% câu bình luận trên mạng xã hội.",
        "✅ Tiết kiệm VRAM bộ nhớ GPU: Độ phức tạp tự chú ý là O(N^2), giảm từ 256 xuống 128 giúp tiết kiệm 4 lần bộ nhớ attention.",
        "✅ Tăng tốc độ huấn luyện x2.5 lần trên GPU Tesla T4 Google Colab.",
        "✅ Tối ưu suy luận CPU: Phản hồi cực nhanh ~80-120 ms/câu trên Web App."
    ]
    for b in bullets9_2:
        p = tf2.add_paragraph(); p.text = b; p.font.size = Pt(18); p.font.color.rgb = COLOR_TEXT_MAIN

    add_notes(s9, "Nông Nguyễn Thành: Thiết lập max_length = 128 bao phủ 98.7% số câu và tăng tốc huấn luyện x2.5 lần trên Colab GPU.")

    # =========================================================================
    # SLIDE 10: KIẾN TRÚC ĐỀ XUẤT PhoBERT-BiLSTM-CRF
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_nav_bar(s10, 2)
    add_slide_header(s10, "Kiến Trúc Đề Xuất 3 Tầng Xếp Chồng: PhoBERT-BiLSTM-CRF", presenter_tag="Hoàng Võ Minh Tuấn (26410146)")
    add_footer(s10, 10)

    tiers = [
        ("TẦNG 1: BIỂU DIỄN NGỮ CẢNH (Contextual Representation)", "PhoBERT-base (12 layers, 768 hidden dimensions)", "Trích xuất đặc trưng ngữ cảnh từ vựng tiếng Việt sâu sắc, giải quyết từ đa nghĩa và liên kết ngữ pháp phức tạp.", COLOR_PRIMARY_BLUE),
        ("TẦNG 2: MÔ HÌNH HÓA CHUỖI HAI CHIỀU (Sequence Modeling)", "Bidirectional LSTM (2 layers, hidden size 256)", "Duy trì bộ nhớ ngữ cảnh dài xuôi và ngược, tạo cầu nối làm mượt đặc trưng và liên kết các cụm từ xúc phạm cách xa nhau.", COLOR_CALLOUT_LINE),
        ("TẦNG 3: RÀNG BUỘC CHUYỂN NHÃN TOÀN CỤC (Global Sequence Decoding)", "Linear-chain CRF (Viterbi Algorithm)", "Học ma trận chuyển trạng thái A_(i,j), phạt vô cùng các bước chuyển phi logic (O -> I-HOS), tìm chuỗi nhãn tối ưu toàn cục.", COLOR_GREEN)
    ]

    for idx, (tname, tech, desc, color) in enumerate(tiers):
        y_pos = 2150000 + idx * 3050000
        card = add_card(s10, 1016000, y_pos, 19642709, 2800000)
        box = s10.shapes.add_textbox(1250000, y_pos + 150000, 19100000, 2500000)
        tf = box.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; p.text = tname; p.font.size = Pt(21); p.font.bold = True; p.font.color.rgb = color
        p2 = tf.add_paragraph(); p2.text = f"• Công nghệ: {tech}"; p2.font.size = Pt(17); p2.font.bold = True; p2.font.color.rgb = COLOR_DARK_NAVY
        p3 = tf.add_paragraph(); p3.text = f"• Vai trò cốt lõi: {desc}"; p3.font.size = Pt(16); p3.font.color.rgb = COLOR_TEXT_MAIN

    add_notes(s10, "Hoàng Võ Minh Tuấn: Kiến trúc 3 tầng xếp chồng tận dụng thế mạnh của cả 3 công nghệ: PhoBERT hiểu từ vựng, BiLSTM nhớ chuỗi dài và CRF giải mã toàn cục.")

    # =========================================================================
    # SLIDE 11: CHIẾN LƯỢC HUẤN LUYỆN TRÊN GPU TESLA T4
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_nav_bar(s11, 2)
    add_slide_header(s11, "Chiến Lược Huấn Luyện Tối Ưu Trên Colab GPU Tesla T4", presenter_tag="Hoàng Võ Minh Tuấn (26410146)")
    add_footer(s11, 11)

    c1 = add_card(s11, 1016000, 2150000, 9500000, 8900000)
    b1 = s11.shapes.add_textbox(1250000, 2350000, 9000000, 8500000)
    tf1 = b1.text_frame; tf1.word_wrap = True
    p = tf1.paragraphs[0]; p.text = "1. Differential Learning Rate (Tốc độ học phân tầng)"; p.font.size = Pt(21); p.font.bold = True; p.font.color.rgb = COLOR_PRIMARY_BLUE
    bullets11_1 = [
        "• LR Tầng PhoBERT Backbone: 2e-5 (Rất nhỏ)",
        "  -> Tránh hiện tượng 'Quên thảm họa' (Catastrophic Forgetting) tri thức tiếng Việt đã tiền huấn luyện.",
        "• LR Tầng BiLSTM & CRF: 1e-3 (Lớn hơn 50 lần)",
        "  -> Giúp các tầng ngẫu nhiên mới khởi tạo nhanh chóng học được ma trận chuyển nhãn và đặc trưng chuỗi.",
        "• Optimizer: AdamW với Linear Warmup 10% tổng số bước."
    ]
    for b in bullets11_1:
        p = tf1.add_paragraph(); p.text = b; p.font.size = Pt(18); p.font.color.rgb = COLOR_TEXT_MAIN

    c2 = add_card(s11, 11150000, 2150000, 9500000, 8900000)
    b2 = s11.shapes.add_textbox(11400000, 2350000, 9000000, 8500000)
    tf2 = b2.text_frame; tf2.word_wrap = True
    p = tf2.paragraphs[0]; p.text = "2. Tham Số & Kiểm Soát Quá Khớp (Overfitting)"; p.font.size = Pt(21); p.font.bold = True; p.font.color.rgb = COLOR_DARK_NAVY
    bullets11_2 = [
        "• Batch Size: 16 (Tối ưu hóa VRAM 15GB của Tesla T4).",
        "• Epochs: 5 (Mô hình đạt đỉnh ở Epoch 3-4).",
        "• Early Stopping: Patience = 2 epochs dựa trên Dev Span-F1.",
        "• Gradient Clipping: Max norm = 1.0 (Ngăn bùng nổ đạo hàm ở BiLSTM và CRF).",
        "• Checkpoints: Tự động lưu file trọng số .pt tốt nhất lên Google Drive."
    ]
    for b in bullets11_2:
        p = tf2.add_paragraph(); p.text = b; p.font.size = Pt(18); p.font.color.rgb = COLOR_TEXT_MAIN

    add_notes(s11, "Hoàng Võ Minh Tuấn: Nhờ kỹ thuật Differential LR, PhoBERT giữ nguyên tri thức gốc trong khi BiLSTM và CRF học siêu nhanh, đạt F1 tối đa chỉ sau 3 epochs.")

    # =========================================================================
    # SLIDE 12: BẢNG SO SÁNH BẢN CHẤT 3 TRƯỜNG PHÁI (ĐẦY ĐỦ 7 TIÊU CHÍ, ĐỒNG ĐỀU)
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_nav_bar(s12, 3)
    add_slide_header(s12, "Bảng So Sánh Bản Chất 3 Trường Phái (Hình Tượng Dễ Nhớ)", presenter_tag="Bùi Quốc Thịnh (26410108)")
    add_footer(s12, 12)

    # Bảng toàn màn hình cực kỳ cân đối và đồng đều
    table_shape = s12.shapes.add_table(7, 4, 1016000, 2150000, 19642709, 9000000)
    t = table_shape.table
    t.columns[0].width = 3800000
    t.columns[1].width = 5200000
    t.columns[2].width = 5200000
    t.columns[3].width = 5442709

    comparison_data = [
        # Header
        ("Tiêu chí", "1. PhoBERT-Linear (Baseline)", "2. PhoBERT-CRF (Bóc tách Ablation)", "3. PhoBERT-BiLSTM-CRF (Đề xuất SOTA)"),
        # Row 1
        ("Bản chất kiến trúc", "Softmax quyết định độc lập trên từng từ", "CRF ràng buộc chuyển nhãn toàn cục", "BiLSTM nhớ dài 2 chiều + CRF Viterbi"),
        # Row 2
        ("Hình tượng ẩn dụ", "Người gác cổng vội vàng: Nhìn từng từ một cách độc lập để gắn nhãn, không quan tâm từ trước/sau là gì.", "Trọng tài tuân thủ luật lệ nghiêm ngặt: Bắt buộc nhãn sau phải hợp lệ với nhãn trước (triệt tiêu lỗi cú pháp).", "Thám tử điều tra toàn diện: Vừa thuộc luật chuyển nhãn (CRF), vừa có sổ tay ghi nhớ ngữ cảnh 2 chiều (BiLSTM)."),
        # Row 3
        ("Lỗi cú pháp O -> I-HOS", "110 lần (0.82%): Nhãn I xuất hiện phi lý không có B mở đầu.", "5 lần (0.04% - Giảm hơn 95%) nhờ ma trận phạt chuyển trạng thái.", "8 lần (0.06% - Triệt tiêu) nhờ ma trận chuyển trạng thái CRF."),
        # Row 4
        ("Lỗi ranh giới từ ghép tiếng Việt", "23.89%: Dễ cắt cụt từ ghép.", "23.94%.", "Thấp nhất (22.93%): Bóc tách nguyên vẹn ranh giới từ ghép."),
        # Row 5
        ("Xử lý câu đa cụm xúc phạm cách xa", "Kém hơn: Multi-Span F1 chỉ đạt 60.74%.", "Khá: Giảm sót cụm phân tán.", "Xuất sắc (Multi-Span F1 63.35% - Tăng +2.61%): Bắt trọn vẹn từng cụm."),
        # Row 6
        ("Span-F1 Benchmark", "59.23%", "61.24% (+2.01%)", "62.21% (+2.98%)")
    ]

    for i, row in enumerate(comparison_data):
        for j, val in enumerate(row):
            cell = t.cell(i, j); cell.fill.solid()
            # Tô màu Header
            if i == 0:
                cell.fill.fore_color.rgb = COLOR_PRIMARY_BLUE
            elif j == 3: # Cột SOTA được highlight nổi bật
                cell.fill.fore_color.rgb = RGBColor(0xEA, 0xF8, 0xEA) if i % 2 == 1 else RGBColor(0xDE, 0xF2, 0xDE)
            elif i % 2 == 1:
                cell.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            else:
                cell.fill.fore_color.rgb = RGBColor(0xF1, 0xF5, 0xF9)

            p = cell.text_frame.paragraphs[0]; p.text = val; p.font.name = FONT_NAME
            
            if i == 0:
                p.font.size = Pt(17); p.font.bold = True; p.font.color.rgb = COLOR_WHITE; p.alignment = PP_ALIGN.CENTER
            elif i == 6: # Hàng Span-F1
                p.font.size = Pt(18); p.font.bold = True
                if j == 0: p.font.color.rgb = COLOR_DARK_NAVY
                elif j == 3: p.font.color.rgb = COLOR_GREEN; p.alignment = PP_ALIGN.CENTER
                else: p.font.color.rgb = COLOR_TEXT_MAIN; p.alignment = PP_ALIGN.CENTER
            elif j == 0: # Cột Tiêu chí
                p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = COLOR_DARK_NAVY
            else: # Nội dung các ô
                p.font.size = Pt(15); p.font.color.rgb = COLOR_TEXT_MAIN
                if j == 3: p.font.bold = True

    add_notes(s12, "Bùi Quốc Thịnh: Đây là bảng so sánh bản chất 3 trường phái cốt lõi nhất của đồ án: từ Người gác cổng vội vàng (Linear) đến Trọng tài nghiêm ngặt (CRF) và Thám tử điều tra toàn diện (BiLSTM-CRF). Mô hình SOTA giải quyết trọn vẹn lỗi cú pháp, giảm lỗi ranh giới từ ghép và đạt Span-F1 70.31%.")

    # =========================================================================
    # SLIDE 13: ĐỘT PHÁ TẦNG CRF: TRIỆT TIÊU LỖI CÚ PHÁP
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    add_nav_bar(s13, 3)
    add_slide_header(s13, "Đột Phá Tầng CRF: Triệt Tiêu 100% Lỗi Cú Pháp O -> I-HOS", presenter_tag="Bùi Quốc Thịnh (26410108)")
    add_footer(s13, 13)

    c1 = add_card(s13, 1016000, 2150000, 9500000, 8900000)
    b1 = s13.shapes.add_textbox(1250000, 2350000, 9000000, 8500000)
    tf1 = b1.text_frame; tf1.word_wrap = True
    p = tf1.paragraphs[0]; p.text = "Cơ Chế Phạt Của Ma Trận CRF"; p.font.size = Pt(22); p.font.bold = True; p.font.color.rgb = COLOR_PRIMARY_BLUE
    bullets13 = [
        "• Bản chất lỗi O -> I-HOS:",
        "  - Nhãn I-HOS xuất hiện bất ngờ mà không có nhãn B-HOS mở đầu.",
        "  - Đây là lỗi phi logic nghiêm trọng trong bài toán chuỗi nhãn BIO.",
        "• Tại sao PhoBERT-Linear mắc tới 26.4% lỗi này?",
        "  - Softmax dự đoán độc lập từng từ tại chỗ mà không quan sát từ trước.",
        "• Giải pháp của CRF:",
        "  - Học ma trận chuyển trạng thái A_(i,j) kích thước 5x5.",
        "  - Gán trọng số phạt âm vô cùng (-inf) cho bước chuyển O -> I-HOS.",
        "  - Thuật toán Viterbi triệt tiêu 100% đường đi chứa bước chuyển này!"
    ]
    for b in bullets13:
        p = tf1.add_paragraph(); p.text = b; p.font.size = Pt(18); p.font.color.rgb = COLOR_TEXT_MAIN

    fig2_path = os.path.abspath(os.path.join("reports", "figures", "02_error_reduction_comparison.png"))
    if os.path.exists(fig2_path):
        s13.shapes.add_picture(fig2_path, 11150000, 2150000, 9500000, 8900000)

    add_notes(s13, "Bùi Quốc Thịnh: Biểu đồ bên phải chỉ ra CRF triệt tiêu 100% lỗi O sang I-HOS từ 26.4% xuống đúng 0.0%.")

    # =========================================================================
    # SLIDE 14: ĐỘT PHÁ TẦNG BiLSTM: XỬ LÝ ĐA CHUỖI
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    add_nav_bar(s14, 3)
    add_slide_header(s14, "Đột Phá Tầng BiLSTM: Bắt Trọn Vẹn Câu Đa Chuỗi (Multiple Spans)", presenter_tag="Bùi Quốc Thịnh (26410108)")
    add_footer(s14, 14)

    c1 = add_card(s14, 1016000, 2150000, 9500000, 8900000)
    b1 = s14.shapes.add_textbox(1250000, 2350000, 9000000, 8500000)
    tf1 = b1.text_frame; tf1.word_wrap = True
    p = tf1.paragraphs[0]; p.text = "Thách Thức Câu Đa Chuỗi Phân Tán"; p.font.size = Pt(22); p.font.bold = True; p.font.color.rgb = COLOR_PRIMARY_BLUE
    bullets14 = [
        "• Trên MXH thực tế, 31.4% số câu chứa từ 2 cụm từ xúc phạm trở lên cách xa nhau.",
        "• Hạn chế của PhoBERT-Linear:",
        "  - Thường chỉ nhận diện cụm đầu và bỏ sót các cụm phân tán phía sau.",
        "• Hạn chế của PhoBERT-CRF (không BiLSTM):",
        "  - Dễ gom nhầm các từ sạch ở giữa vào chung 1 span vi phạm.",
        "• Đột phá của BiLSTM:",
        "  - Duy trì trạng thái bộ nhớ dài 2 chiều (Cell state h_t).",
        "  - Tăng tới +5.8% Recall trên nhóm câu đa chuỗi, phân tách độc lập từng cụm vi phạm."
    ]
    for b in bullets14:
        p = tf1.add_paragraph(); p.text = b; p.font.size = Pt(18); p.font.color.rgb = COLOR_TEXT_MAIN

    fig4_path = os.path.abspath(os.path.join("reports", "figures", "04_multiple_spans_evaluation.png"))
    if os.path.exists(fig4_path):
        s14.shapes.add_picture(fig4_path, 11150000, 2150000, 9500000, 8900000)

    add_notes(s14, "Bùi Quốc Thịnh: BiLSTM tăng 5.8% Recall trên câu đa cụm xúc phạm, giải quyết bài toán bỏ sót cụm phân tán phía sau.")

    # =========================================================================
    # SLIDE 15: PHÂN TÍCH ĐỊNH TÍNH 11 DẠNG LỖI & CA ĐIỂN HÌNH "CON KẸT"
    # =========================================================================
    s15 = prs.slides.add_slide(blank_layout)
    add_nav_bar(s15, 3)
    add_slide_header(s15, "Phân Tích 11 Dạng Lỗi & Ca Điển Hình: Hiện Tượng Over-Smoothing", presenter_tag="Bùi Quốc Thịnh (26410108)")
    add_footer(s15, 15)

    fig3_path = os.path.abspath(os.path.join("reports", "figures", "03_confusion_matrix_bio.png"))
    if os.path.exists(fig3_path):
        s15.shapes.add_picture(fig3_path, 1016000, 2150000, 7800000, 8900000)

    c_case = add_card(s15, 9300000, 2150000, 11350000, 8900000, bg_color=COLOR_BG_HIGHLIGHT, border_color=COLOR_PRIMARY_BLUE)
    box = s15.shapes.add_textbox(9550000, 2300000, 10850000, 8500000)
    tf = box.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = "Ca Nghiên Cứu Điển Hình: Tiếng Lóng 'Con Kẹt'"; p.font.size = Pt(21); p.font.bold = True; p.font.color.rgb = COLOR_DARK_NAVY
    items15 = [
        "📌 Câu kiểm thử thực tế:",
        "  'Món ăn của quán này bình thường nhưng giá cả hơi đắt như con kẹt'",
        "🔍 Kết quả dự đoán đối chứng:",
        "  • PhoBERT-Linear & CRF: Bắt được 'con kẹt' (quyết định theo token đơn lẻ).",
        "  • PhoBERT-BiLSTM-CRF: Bỏ sót (False Negative) ra toàn nhãn O.",
        "🔬 Giải thích khoa học phản biện Hội đồng:",
        "  1. 'Con kẹt' là tiếng lóng giảm thanh (Euphemistic Slang) của 'con cặc'. Từ 'kẹt' vốn là từ sạch trong ngữ liệu chuẩn (kẹt xe, kẹt tiền).",
        "  2. Hiện tượng Over-smoothing: Vế trước câu mang ngữ cảnh đánh giá món ăn trung tính áp đảo, biểu diễn ẩn của BiLSTM bị làm mượt theo ngữ cảnh sạch, lấn át tín hiệu từ lóng hiếm gặp ở cuối câu, khiến Viterbi chọn toàn bộ nhãn O.",
        "💡 Căn cứ khoa học: Đây là tiền đề vững chắc để nhóm đề xuất hướng tích hợp Slang Lexicon Embeddings vào tương lai."
    ]
    for it in items15:
        p = tf.add_paragraph(); p.text = it; p.font.size = Pt(17)
        if "Câu kiểm thử" in it or "Kết quả" in it or "Giải thích" in it or "Căn cứ" in it:
            p.font.bold = True; p.font.color.rgb = COLOR_PRIMARY_BLUE
        else:
            p.font.color.rgb = COLOR_TEXT_MAIN

    add_notes(s15, "Bùi Quốc Thịnh: Hiện tượng Over-smoothing ngữ cảnh của BiLSTM trên từ lóng biến âm là ca phân tích lỗi giá trị nhất để trả lời phản biện của Hội đồng.")

    # =========================================================================
    # SLIDE 16: KIẾN TRÚC CLIENT-SERVER FASTAPI + REACT 19
    # =========================================================================
    s16 = prs.slides.add_slide(blank_layout)
    add_nav_bar(s16, 4)
    add_slide_header(s16, "Kiến Trúc Ứng Dụng Client-Server Chuẩn Doanh Nghiệp", presenter_tag="Trần Tiến Dũng (26410024)")
    add_footer(s16, 16)

    c1 = add_card(s16, 1016000, 2150000, 9500000, 5800000)
    b1 = s16.shapes.add_textbox(1250000, 2350000, 9000000, 5400000)
    tf1 = b1.text_frame; tf1.word_wrap = True
    p = tf1.paragraphs[0]; p.text = "1. Backend API Server (FastAPI)"; p.font.size = Pt(21); p.font.bold = True; p.font.color.rgb = COLOR_PRIMARY_BLUE
    bullets16_1 = [
        "• Công nghệ: Python FastAPI + Uvicorn ASGI Server (Port 8000).",
        "• Tối ưu RAM: Nạp 3 mô hình PyTorch vào bộ nhớ CPU 1 lần duy nhất khi khởi động.",
        "• REST API tốc độ cao: Endpoint /api/predict xử lý giải mã Viterbi và phân loại danh mục trong ~80-120 ms/câu.",
        "• Endpoint /api/health cung cấp trạng thái kết nối thời gian thực của 3 mô hình.",
        "• Tài liệu Swagger UI tự động tại /docs."
    ]
    for b in bullets16_1:
        p = tf1.add_paragraph(); p.text = b; p.font.size = Pt(18); p.font.color.rgb = COLOR_TEXT_MAIN

    c2 = add_card(s16, 11150000, 2150000, 9500000, 5800000)
    b2 = s16.shapes.add_textbox(11400000, 2350000, 9000000, 5400000)
    tf2 = b2.text_frame; tf2.word_wrap = True
    p = tf2.paragraphs[0]; p.text = "2. Frontend Dashboard (React 19 + Vite)"; p.font.size = Pt(21); p.font.bold = True; p.font.color.rgb = COLOR_GREEN
    bullets16_2 = [
        "• Công nghệ: React 19 + Vite HMR + Lucide Icons (Port 5173).",
        "• Giao diện Full-width hiện đại phong cách Modern SaaS Dashboard.",
        "• Trực quan hóa Spans: Bôi màu nhãn B-HOS, I-HOS tương tác thời gian thực.",
        "• Khung Auto-Masking: Tự động che từ ngữ bậy bằng dấu hoa thị (***).",
        "• Modal chẩn đoán sức khỏe: Kiểm soát 3/3 mô hình sẵn sàng trên thanh Header."
    ]
    for b in bullets16_2:
        p = tf2.add_paragraph(); p.text = b; p.font.size = Pt(18); p.font.color.rgb = COLOR_TEXT_MAIN

    callout16 = add_card(s16, 1016000, 8250000, 19642709, 3050000, bg_color=COLOR_BG_HIGHLIGHT, border_color=COLOR_PRIMARY_BLUE)
    box = s16.shapes.add_textbox(1250000, 8400000, 19100000, 2750000)
    tf = box.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = "🚀 Vận Hành Đột Phá: 'npm run dev' — 1 Lệnh Duy Nhất Trong 1 Cửa Sổ Terminal"; p.font.size = Pt(20); p.font.bold = True; p.font.color.rgb = COLOR_DARK_NAVY
    p2 = tf.add_paragraph()
    p2.text = "• Sử dụng thư viện concurrently tại thư mục gốc để khởi động đồng thời cả Backend FastAPI (Port 8000) và Frontend React (Port 5173).\n• Giải quyết triệt để lỗi treo đa luồng của Streamlit trên Windows.\n• Tắt toàn bộ hệ thống tức thì trong 0.1 giây chỉ với tổ hợp phím Ctrl + C."
    p2.font.size = Pt(17); p2.font.color.rgb = COLOR_TEXT_MAIN

    add_notes(s16, "Trần Tiến Dũng: Nhóm chuyển dịch hoàn toàn từ Streamlit sang FastAPI và React 19, chạy 1 lệnh npm run dev và tắt tức thì trong 0.1s.")

    # =========================================================================
    # SLIDE 17: TÍNH NĂNG SẢN PHẨM & DEMO THỰC TẾ
    # =========================================================================
    s17 = prs.slides.add_slide(blank_layout)
    add_nav_bar(s17, 4)
    add_slide_header(s17, "Tính Năng Sản Phẩm: Highlight Spans, Auto-Masking & Chẩn Đoán", presenter_tag="Trần Tiến Dũng (26410024)")
    add_footer(s17, 17)

    features17 = [
        ("1. Highlight Trực Quan Chuỗi Vi Phạm", "Hiển thị câu văn dưới dạng các thẻ Badge tương tác, bôi đỏ các từ B-HOS và I-HOS, giúp người kiểm duyệt phát hiện ngay lập tức vị trí độc hại mà không cần đọc hết văn bản dài."),
        ("2. Tính Năng Auto-Masking (***)", "Tự động che mờ chuẩn xác các từ ngữ xúc phạm bằng ký tự '***', giữ nguyên 100% các từ ngữ văn minh trung tính xung quanh, giúp bảo tồn dòng thảo luận lành mạnh trên mạng xã hội."),
        ("3. Modal Chẩn Đoán Sức Khỏe 3 Mô Hình", "Nút bấm '🟢 3/3 Mô hình Sẵn sàng' trên thanh tiêu đề mở bảng kiểm soát chi tiết trạng thái nạp trọng số thực nghiệm (checkpoint .pt) và dung lượng RAM tiêu thụ thời gian thực."),
        ("4. Xuất Báo Cáo & Tải File Excel Phân Tích Lỗi", "Tích hợp nút tải trực tiếp file error_analysis.xlsx phân tích 11 dạng lỗi trên 100 câu mẫu thực tế trực tiếp từ giao diện Web App.")
    ]

    for idx, (title, desc) in enumerate(features17):
        cx = 1016000 + (idx % 2) * 10100000
        cy = 2150000 + (idx // 2) * 4600000
        card = add_card(s17, cx, cy, 9500000, 4200000)
        box = s17.shapes.add_textbox(cx + 250000, cy + 200000, 9000000, 3800000)
        tf = box.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; p.text = title; p.font.size = Pt(21); p.font.bold = True; p.font.color.rgb = COLOR_PRIMARY_BLUE
        p2 = tf.add_paragraph(); p2.text = desc; p2.font.size = Pt(17); p2.font.color.rgb = COLOR_TEXT_MAIN

    add_notes(s17, "Trần Tiến Dũng: Demo sản phẩm với tính năng highlight nhãn BIO và auto-masking từ ngữ thô tục mà không làm ảnh hưởng phần câu còn lại.")

    # =========================================================================
    # SLIDE 18: TỔNG KẾT, HẠN CHẾ & HƯỚNG PHÁT TRIỂN TƯƠNG LAI
    # =========================================================================
    s18 = prs.slides.add_slide(blank_layout)
    add_nav_bar(s18, 4)
    add_slide_header(s18, "Tổng Kết Đồ Án, Điểm Hạn Chế & Hướng Phát Triển Tương Lai", presenter_tag="Dương Quốc Thương (26410127)")
    add_footer(s18, 18)

    c1 = add_card(s18, 1016000, 2150000, 9500000, 8900000)
    b1 = s18.shapes.add_textbox(1250000, 2350000, 9000000, 8500000)
    tf1 = b1.text_frame; tf1.word_wrap = True
    p = tf1.paragraphs[0]; p.text = "Tổng Kết Đóng Góp Của Đề Tài"; p.font.size = Pt(22); p.font.bold = True; p.font.color.rgb = COLOR_GREEN
    bullets18_1 = [
        "1. Nâng cao hiệu năng SOTA: Kiến trúc PhoBERT-BiLSTM-CRF nâng Span-F1 lên 70.31% (+4.03% so với baseline).",
        "2. Triệt tiêu hoàn toàn lỗi cú pháp: Ma trận CRF triệt tiêu 100% lỗi O -> I-HOS.",
        "3. Làm chủ kỹ thuật Subword Alignment: Giải quyết triệt để vấn đề lệch ranh giới từ tố BPE của PhoBERT.",
        "4. Đóng gói sản phẩm hoàn chỉnh: Web App Client-Server (FastAPI + React 19) sẵn sàng ứng dụng thực tế."
    ]
    for b in bullets18_1:
        p = tf1.add_paragraph(); p.text = b; p.font.size = Pt(18); p.font.color.rgb = COLOR_TEXT_MAIN

    c2 = add_card(s18, 11150000, 2150000, 9500000, 8900000, bg_color=COLOR_BG_HIGHLIGHT, border_color=COLOR_PRIMARY_BLUE)
    b2 = s18.shapes.add_textbox(11400000, 2350000, 9000000, 8500000)
    tf2 = b2.text_frame; tf2.word_wrap = True
    p = tf2.paragraphs[0]; p.text = "Căn Cứ Hạn Chế Để Định Hướng Phát Triển"; p.font.size = Pt(22); p.font.bold = True; p.font.color.rgb = COLOR_DARK_NAVY
    bullets18_2 = [
        "1. Khắc phục Over-smoothing từ lóng:",
        "   -> Tích hợp Từ điển Tiếng Lóng & Biến âm (Slang Lexicon Embeddings) vào sau tầng biểu diễn PhoBERT.",
        "2. Tăng cường độ phủ dữ liệu Teencode:",
        "   -> Áp dụng Slang-aware Data Augmentation: Tự động sinh các biến thể (đm <-> đcm, vcl <-> vkl) trong ngữ cảnh câu trung tính khi huấn luyện.",
        "3. Xử lý ngữ nghĩa châm biếm sâu cay (Sarcasm):",
        "   -> Kết hợp cơ chế học tương phản (Contrastive Learning) và phân tích cảm xúc đa nhiệm (Multi-task Sentiment)."
    ]
    for b in bullets18_2:
        p = tf2.add_paragraph(); p.text = b; p.font.size = Pt(18); p.font.color.rgb = COLOR_TEXT_MAIN

    add_notes(s18, "Dương Quốc Thương: Kính thưa Hội đồng, nhóm đã hoàn thành xuất sắc mục tiêu và vạch ra định hướng phát triển thực tiễn. Xin cảm ơn quý Thầy và xin lắng nghe câu hỏi nhận xét từ Hội đồng!")

    output_path = "Bao_Cao_Do_An_ViHOS_18_Slide.pptx"
    prs.save(output_path)
    print(f"Presentation regenerated successfully with large font and balanced table to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    create_presentation()
