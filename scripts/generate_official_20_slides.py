import sys
import io
import copy
import os
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def build_presentation():
    input_path = 'outputs/Bao_cao_ViHOS_20_slide_Muc_luc_Logic.pptx'
    output_path = 'outputs/Bao_cao_ViHOS_20_slide_Chinh_thuc_Log_Test.pptx'

    print(f"Loading base presentation from: {input_path}")
    prs = pptx.Presentation(input_path)

    # -------------------------------------------------------------
    # SLIDE 5: Các cách làm và cấu hình nhóm đề xuất
    # -------------------------------------------------------------
    slide5 = prs.slides[4]
    for s in slide5.shapes:
        if s.name == 'Text 42' and s.has_text_frame:
            s.text_frame.text = (
                "Nghiên cứu bóc tách (Ablation Study) 4 cấu hình: bóc tách vai trò của "
                "PhoBERT-Linear, tầng giải mã chuỗi CRF, tầng BiLSTM và đầu kiểm tra Intent Head."
            )
            for p in s.text_frame.paragraphs:
                p.font.name = 'Times New Roman'
                p.font.size = Pt(14)
                p.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

    # -------------------------------------------------------------
    # SLIDE 9: Bộ dữ liệu ViHOS
    # -------------------------------------------------------------
    slide9 = prs.slides[8]
    for s in slide9.shapes:
        if s.has_table:
            t = s.table
            # Row 3 is Test: update token count to 13.444
            t.cell(3, 2).text = "13.444"
            for p in t.cell(3, 2).text_frame.paragraphs:
                p.font.name = 'Times New Roman'
                p.font.size = Pt(13)
                p.alignment = PP_ALIGN.CENTER

    # -------------------------------------------------------------
    # SLIDE 10: So sánh kết quả mô hình cơ sở và đề xuất (TABLE)
    # -------------------------------------------------------------
    slide10 = prs.slides[9]
    # Update subtitle
    for s in slide10.shapes:
        if s.name == 'Text 19' and s.has_text_frame:
            s.text_frame.text = (
                "Kết quả chính thức trên toàn bộ 1.106 câu test (13.444 từ). "
                "PhoBERT–BiLSTM–CRF đạt F1 cao nhất 62,21% (+2,98% so với Baseline)."
            )
            for p in s.text_frame.paragraphs:
                p.font.name = 'Times New Roman'
                p.font.size = Pt(13)
                p.font.bold = True
                p.font.color.rgb = RGBColor(0x02, 0x84, 0xC7)

    # Update Table
    tbl_shape = [s for s in slide10.shapes if s.has_table][0]
    t = tbl_shape.table

    # If only 4 rows, clone 4th row to create 5 rows
    if len(t.rows) == 4:
        last_row_tr = t.rows[len(t.rows) - 1]._tr
        new_tr = copy.deepcopy(last_row_tr)
        tbl_shape._element.xpath('.//a:tbl')[0].append(new_tr)

    # Re-fetch table rows
    table_data = [
        ["Mô hình", "Precision (%)", "Recall (%)", "Span-F1 (%)", "Lỗi O → I-HOS", "Lỗi ranh giới (%)"],
        ["1. PhoBERT–Linear (Baseline)", "60,34", "58,15", "59,23", "110 lần (0,82%)", "23,89"],
        ["2. PhoBERT–CRF (Ablation)", "63,37", "59,26", "61,24 (+2,01)", "5 lần (0,04%)", "23,94"],
        ["3. PhoBERT–BiLSTM–CRF (SOTA)", "65,00", "59,65", "62,21 (+2,98)", "8 lần (0,06%)", "22,93"],
        ["4. PhoBERT–DualHead (Đa nhiệm)", "62,99", "58,31", "60,56 (+1,33)", "18 lần (0,13%)", "24,10"],
    ]

    for row_idx, row_vals in enumerate(table_data):
        for col_idx, val in enumerate(row_vals):
            cell = t.cell(row_idx, col_idx)
            cell.text = val
            p = cell.text_frame.paragraphs[0]
            p.font.name = 'Times New Roman'
            if row_idx == 0:
                p.font.size = Pt(13)
                p.font.bold = True
                p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                p.alignment = PP_ALIGN.CENTER
            elif row_idx == 3: # SOTA row highlight
                p.font.size = Pt(13)
                p.font.bold = True
                p.font.color.rgb = RGBColor(0x02, 0x84, 0xC7)
                p.alignment = PP_ALIGN.LEFT if col_idx == 0 else PP_ALIGN.CENTER
            else:
                p.font.size = Pt(12)
                p.font.bold = (col_idx == 3)
                p.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
                p.alignment = PP_ALIGN.LEFT if col_idx == 0 else PP_ALIGN.CENTER

    # Update Slide 10 Speaker Notes
    slide10.notes_slide.notes_text_frame.text = (
        "Thưa Thầy và các bạn, đây là bảng kết quả đánh giá chính thức trên toàn bộ 1.106 câu của tập Test (tổng cộng 13.444 từ).\n"
        "1. PhoBERT-Linear đạt Span-F1 59,23%. Điểm yếu lớn nhất là gặp 110 lần vi phạm ngữ pháp chuyển nhãn O -> I-HOS.\n"
        "2. PhoBERT-CRF bổ sung tầng giải mã chuỗi, giúp tăng F1 lên 61,24% (+2,01%), đồng thời giảm lỗi O -> I-HOS xuống chỉ còn 5 lần.\n"
        "3. PhoBERT-BiLSTM-CRF đạt kết quả cao nhất toàn diện: Precision 65,00%, Recall 59,65%, Span-F1 62,21% (+2,98% so với Baseline), lỗi ranh giới thấp nhất 22,93%.\n"
        "4. Cấu hình PhoBERT-DualHead đạt F1 60,56% (+1,33%). Toàn bộ số liệu đều được trích xuất từ log đánh giá test chính thức."
    )

    # -------------------------------------------------------------
    # SLIDE 11: Span-F1 của bốn mô hình
    # -------------------------------------------------------------
    slide11 = prs.slides[10]
    for s in slide11.shapes:
        if s.name == 'Text 15' and s.has_text_frame:
            s.text_frame.text = "Span-F1 của bốn mô hình trên tập Test"
            for p in s.text_frame.paragraphs:
                p.font.name = 'Times New Roman'
                p.font.bold = True
        elif s.name == 'Rectangle 28' and s.has_text_frame:
            s.text_frame.text = (
                "PhoBERT–Linear: 59,23%\n"
                "PhoBERT–CRF: 61,24% (+2,01%)\n"
                "PhoBERT–BiLSTM–CRF: 62,21% (+2,98%)\n"
                "PhoBERT–DualHead: 60,56% (+1,33%)\n\n"
                "Điểm nhấn then chốt:\n"
                "• BiLSTM–CRF đạt Span-F1 cao nhất (62,21%).\n"
                "• Cả 3 cấu hình cải tiến đều vượt trội hơn Baseline Linear.\n"
                "• Đánh giá trên toàn bộ 1.106 câu test."
            )
            for p in s.text_frame.paragraphs:
                p.font.name = 'Times New Roman'
                p.font.size = Pt(13)
        elif s.shape_type == pptx.enum.shapes.MSO_SHAPE_TYPE.PICTURE:
            with open('reports/figures/01_ablation_span_f1.png', 'rb') as img_f:
                s.part.related_part(s._element.blip_rId)._blob = img_f.read()

    slide11.notes_slide.notes_text_frame.text = (
        "Biểu đồ thể hiện trực quan Span-F1 của 4 mô hình trên tập Test.\n"
        "Cột cao nhất là PhoBERT-BiLSTM-CRF đạt 62,21%, cải thiện +2,98% so với Baseline Linear (59,23%).\n"
        "Việc thêm CRF giúp tăng 2,01%, và việc bổ sung tiếp BiLSTM giúp tăng thêm 0,97% nữa nhờ khả năng làm mượt biểu diễn chuỗi hai chiều."
    )

    # -------------------------------------------------------------
    # SLIDE 12: Giảm lỗi chuyển nhãn O -> I-HOS
    # -------------------------------------------------------------
    slide12 = prs.slides[11]
    for s in slide12.shapes:
        if s.name == 'Text 15' and s.has_text_frame:
            s.text_frame.text = "Hiệu quả triệt tiêu lỗi cú pháp chuyển nhãn"
            for p in s.text_frame.paragraphs:
                p.font.name = 'Times New Roman'
                p.font.bold = True
        elif s.name == 'Rectangle 28' and s.has_text_frame:
            s.text_frame.text = (
                "Lỗi vi phạm BIO (O → I-HOS):\n"
                "• PhoBERT–Linear: 110 lần (0,82%)\n"
                "• PhoBERT–CRF: 5 lần (0,04%)\n"
                "• PhoBERT–BiLSTM–CRF: 8 lần (0,06%)\n"
                "• PhoBERT–DualHead: 18 lần (0,13%)\n\n"
                "Hiệu quả của tầng CRF:\n"
                "• Triệt tiêu 92,7% – 95,5% lỗi cú pháp phi lý.\n"
                "• Lỗi ranh giới thấp nhất: BiLSTM–CRF (22,93%)."
            )
            for p in s.text_frame.paragraphs:
                p.font.name = 'Times New Roman'
                p.font.size = Pt(13)
        elif s.name == 'Rectangle 29' and s.has_text_frame:
            s.text_frame.text = "Tầng CRF học ma trận chuyển đổi nhãn hợp lệ, loại trừ việc dự đoán I-HOS ngay sau nhãn O."
            for p in s.text_frame.paragraphs:
                p.font.name = 'Times New Roman'
                p.font.size = Pt(12)
        elif s.shape_type == pptx.enum.shapes.MSO_SHAPE_TYPE.PICTURE:
            with open('reports/figures/02_error_reduction_comparison.png', 'rb') as img_f:
                s.part.related_part(s._element.blip_rId)._blob = img_f.read()

    slide12.notes_slide.notes_text_frame.text = (
        "Biểu đồ này chỉ rõ lý do vì sao tầng giải mã CRF lại đóng vai trò tối quan trọng trong bài toán trích xuất thực thể chuỗi.\n"
        "Ở mô hình Linear, do từng vị trí dự đoán độc lập, mô hình vi phạm nguyên tắc BIO tới 110 lần (gán I-HOS ngay sau O mà không có B-HOS mở đầu).\n"
        "Khi thêm CRF, lỗi này giảm tới hơn 92%, xuống chỉ còn 5 đến 8 lần trên toàn bộ 13.444 từ."
    )

    # -------------------------------------------------------------
    # SLIDE 13: Phân tích 11 dạng lỗi
    # -------------------------------------------------------------
    slide13 = prs.slides[12]
    slide13.notes_slide.notes_text_frame.text = (
        "Phân tích chuyên sâu 11 dạng lỗi trên tập phân tích cho thấy:\n"
        "Mô hình đề xuất cải thiện mạnh ở các lỗi đa chuỗi (giảm từ 12 xuống 4 lỗi) và ngữ cảnh gọi tên động vật (giảm từ 8 xuống 1 lỗi).\n"
        "Đồng thời, trên toàn bộ tập test, kết quả đánh giá phân tách cũng chứng minh PhoBERT-BiLSTM-CRF vượt trội ở cả dạng Đơn chuỗi (57,78% vs 53,41%) và Đa chuỗi phân tán (63,35% vs 60,74%)."
    )

    # -------------------------------------------------------------
    # SLIDE 15: Ma trận nhầm lẫn BIO trên toàn bộ Test Set
    # -------------------------------------------------------------
    slide15 = prs.slides[14]
    for s in slide15.shapes:
        if s.name == 'Text 15' and s.has_text_frame:
            s.text_frame.text = "Ma trận nhầm lẫn BIO trên toàn bộ 1.106 câu test"
            for p in s.text_frame.paragraphs:
                p.font.name = 'Times New Roman'
                p.font.bold = True
        elif s.name == 'Rectangle 28' and s.has_text_frame:
            s.text_frame.text = (
                "Hàng: Nhãn thực tế (Ground Truth)\n"
                "Cột: Nhãn dự đoán (Predicted)\n\n"
                "Toàn bộ Test: 1.106 câu (13.444 từ)\n\n"
                "• True O đoán đúng: 10.822 từ (97,3%)\n"
                "• True B-HOS đoán đúng: 867 từ (68,6%)\n"
                "• True I-HOS đoán đúng: 477 từ (46,1%)\n\n"
                "Phân tích nhầm lẫn:\n"
                "• Bỏ sót (B/I → O): 759 từ\n"
                "• Đánh dấu nhầm (O → B/I): 298 từ"
            )
            for p in s.text_frame.paragraphs:
                p.font.name = 'Times New Roman'
                p.font.size = Pt(13)
        elif s.shape_type == pptx.enum.shapes.MSO_SHAPE_TYPE.PICTURE:
            with open('reports/figures/03_confusion_matrix_bio.png', 'rb') as img_f:
                s.part.related_part(s._element.blip_rId)._blob = img_f.read()

    slide15.notes_slide.notes_text_frame.text = (
        "Ma trận nhầm lẫn này được tính toán chính xác trên toàn bộ 1.106 câu của tập test (13.444 từ).\n"
        "Mô hình đạt độ chính xác rất cao ở nhãn O (10.822 từ đúng). Điểm nghẽn chính nằm ở việc bỏ sót (759 từ B/I bị phân loại nhầm thành O) do các biến thể tiếng lóng và viết tắt.\n"
        "Trong khi đó, số từ bình thường bị đánh dấu nhầm thành xúc phạm chỉ có 298 từ, cho thấy mô hình có độ tin cậy cao, hạn chế báo động giả."
    )

    # -------------------------------------------------------------
    # SLIDE 18: Hạn chế và hướng cải thiện
    # -------------------------------------------------------------
    slide18 = prs.slides[17]
    for s in slide18.shapes:
        if s.name == 'Rectangle 42' and s.has_text_frame:
            s.text_frame.text = (
                "Ưu tiên hoàn thiện: Tinh chỉnh ngưỡng động của cổng Intent Gate, "
                "bổ sung dữ liệu tiền xử lý từ lóng teencode và mở rộng tập dữ liệu châm biếm."
            )
            for p in s.text_frame.paragraphs:
                p.font.name = 'Times New Roman'
                p.font.size = Pt(13)
    slide18.notes_slide.notes_text_frame.text = (
        "Từ kết quả thực nghiệm và ứng dụng, nhóm xác định ba điểm cần cải thiện:\n"
        "1. Xử lý từ lóng và biến âm viết tắt trên mạng xã hội.\n"
        "2. Tối ưu ngưỡng kích hoạt của cổng Intent Gate trên tập dev.\n"
        "3. Thu thập thêm các mẫu câu mang tính châm biếm, khen đểu."
    )

    # -------------------------------------------------------------
    # SLIDE 19: Kết luận và thứ tự ưu tiên
    # -------------------------------------------------------------
    slide19 = prs.slides[18]
    for s in slide19.shapes:
        if s.name == 'Text 15' and s.has_text_frame:
            s.text_frame.text = "Kết luận và Kết quả đạt được"
            for p in s.text_frame.paragraphs:
                p.font.name = 'Times New Roman'
                p.font.bold = True
        elif s.name == 'Text 19' and s.has_text_frame:
            s.text_frame.text = "KẾT QUẢ ĐÃ XÁC THỰC BẰNG LOG TEST"
        elif s.name == 'Text 22' and s.has_text_frame:
            s.text_frame.text = "Kết quả thực nghiệm 4 mô hình"
        elif s.name == 'Text 24' and s.has_text_frame:
            s.text_frame.text = (
                "PhoBERT–BiLSTM–CRF đạt F1 62,21% (+2,98% so với Baseline).\n"
                "Tầng CRF giảm lỗi cú pháp O → I-HOS từ 110 lần xuống 8 lần.\n"
                "Lỗi lệch ranh giới cụm thấp nhất đạt 22,93%."
            )
        elif s.name == 'Text 27' and s.has_text_frame:
            s.text_frame.text = "GIÁ TRỊ & HƯỚNG PHÁT TRIỂN"
        elif s.name == 'Text 30' and s.has_text_frame:
            s.text_frame.text = "Đóng góp và ứng dụng thực tiễn"
        elif s.name == 'Text 32' and s.has_text_frame:
            s.text_frame.text = (
                "Quy trình thử nghiệm Ablation Study bóc tách khoa học, minh bạch.\n"
                "Hệ thống Web demo tương tác thời gian thực với React & FastAPI.\n"
                "Đã đối chiếu 100% kết quả khớp chuẩn với log tập test."
            )
        elif s.name == 'Text 35' and s.has_text_frame:
            s.text_frame.text = "Hai định hướng mở rộng"
        elif s.name == 'Text 37' and s.has_text_frame:
            s.text_frame.text = "01 // Tối ưu ngưỡng động Intent Gate"
        elif s.name == 'Text 39' and s.has_text_frame:
            s.text_frame.text = "02 // Bổ sung từ lóng và châm biếm"
        elif s.name == 'Rectangle 46' and s.has_text_frame:
            s.text_frame.text = (
                "Mục tiêu: Nhận diện chính xác ranh giới chuỗi xúc phạm trong bình luận tiếng Việt, "
                "hỗ trợ đắc lực và giảm tải cho người kiểm duyệt nội dung số."
            )

    slide19.notes_slide.notes_text_frame.text = (
        "Kính thưa Thầy và các bạn, đồ án đã hoàn thành trọn vẹn mục tiêu đề ra:\n"
        "1. Đã triển khai và đánh giá đầy đủ 4 mô hình trên toàn bộ 1.106 câu test với log kiểm chứng 100%.\n"
        "2. Chứng minh rõ nét hiệu quả của tầng BiLSTM và CRF trong việc nâng Span-F1 lên 62,21% và triệt tiêu 92,7% lỗi vi phạm cú pháp BIO.\n"
        "3. Xây dựng hoàn chỉnh ứng dụng web demo hỗ trợ kiểm duyệt và che từ xúc phạm tự động."
    )

    # Save presentation
    prs.save(output_path)
    print(f"🎉 Successfully created new official presentation: {output_path}")

if __name__ == '__main__':
    build_presentation()
