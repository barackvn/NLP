# NỘI DUNG VÀ LỜI NÓI CHO 18 SLIDE

Đồng bộ hình ngày 16/09/2026. Bản trình chiếu: `../outputs/Bao_cao_ViHOS_18_slide_Cap_nhat_hinh.pptx`.

# THUYẾT MINH ViHOS — 18 SLIDE, CẬP NHẬT THEO HÌNH MỚI

**Dùng với:** `Bao_cao_ViHOS_18_slide_Cap_nhat_hinh.pptx`.  
**Cách nói:** ngắn, dễ hiểu; chỉ đọc phần trong ngoặc kép.  
**Thời gian:** các mốc cộng lại 12 phút 45 giây; dành thêm 1–2 phút cho chuyển người, chỉ hình và dừng nhịp. Tập một lượt với đồng hồ, kết thúc trước phút 15.

Bản này đồng bộ hình cập nhật ngày 16/09/2026. Bộ đối chứng có **ba mô hình: Linear, CRF và BiLSTM–CRF có DualHead**. Hình 4 trình bày 11 dạng lỗi; ma trận dùng 100 câu đầu test, không phải toàn bộ test. Hai nhóm 100 câu này không được mặc định là cùng một tập.

---

## Slide 1 — Giới thiệu | 25 giây

“Em xin chào Thầy và các bạn. Nhóm em trình bày bài toán tìm cụm xúc phạm trong bình luận tiếng Việt. Bản cập nhật so sánh Linear, CRF và mô hình BiLSTM–CRF có DualHead. Em sẽ giải thích từng tầng bằng lời đơn giản, sau đó trình bày các hình kết quả mới.”

## Slide 2 — Bài toán và lý do thực hiện | 45 giây

“Khi có nhiều bình luận, đọc từng câu để tìm nội dung xúc phạm sẽ mất thời gian. Tiếng Việt còn có từ lóng và cách viết biến dạng nên việc nhận ra chúng không đơn giản.

Nhóm muốn xây dựng hệ thống chỉ ra đúng từ hoặc cụm cần xem lại, đồng thời hạn chế bỏ sót và đánh dấu nhầm.

Kết quả giúp người kiểm duyệt biết phần nào cần chú ý. Nhóm so sánh các cách làm để chọn hướng phù hợp, sau đó đưa lên ứng dụng web.”

## Slide 3 — Đầu vào, đầu ra và ba nhãn BIO | 45 giây

“Đầu vào là một bình luận tiếng Việt. Đầu ra là các cụm xúc phạm được tìm thấy.

Để đánh dấu vị trí, nhóm dùng ba nhãn: B là bắt đầu cụm, I là tiếp tục cụm, O là ngoài cụm.

Ví dụ ‘Bạn nói thật ngu’, từ ‘ngu’ là cụm cần đánh dấu trong ví dụ này nên nhận B; những từ còn lại nhận O. Nếu câu có nhiều cụm riêng biệt thì mỗi cụm bắt đầu lại bằng B.”

**Nhớ:** Đây là ví dụ minh họa nhãn, không phải kết quả kiểm tra mô hình.

## Slide 4 — Dữ liệu ViHOS | 35 giây

“Dữ liệu gồm hơn 11 nghìn bình luận đã có nhãn. Nhóm chia thành phần để học, phần để chọn mô hình và phần để kiểm tra cuối cùng.

Phần lớn từ trong dữ liệu nằm ngoài cụm xúc phạm. Vì vậy, chỉ đếm số từ đoán đúng là chưa đủ. Nhóm cần đo xem mô hình tìm được bao nhiêu cụm và có đánh dấu nhầm hay không.”

## Slide 5 — Ba mô hình trong bộ so sánh mới | 60 giây

“Cả ba mô hình đều dùng PhoBERT để tạo thông tin về từ trong ngữ cảnh.

Linear chọn nhãn cho từng vị trí. CRF xét thêm quan hệ giữa các nhãn. Mô hình thứ ba dùng BiLSTM–CRF để tìm cụm, đồng thời có đầu kiểm tra cả câu và cổng lọc.

Bộ hình mới so sánh ba cấu hình này. Vì mô hình cuối có cả BiLSTM và DualHead, nhóm chưa thể chỉ dựa vào ba dòng kết quả để kết luận riêng từng phần đóng góp bao nhiêu. Trang tiếp theo giải thích vai trò của từng tầng.”

## Slide 6 — Mỗi tầng làm một việc gì? | 60 giây

“Có thể hiểu phương pháp của nhóm như một nhóm làm việc, mỗi phần phụ trách một việc.

PhoBERT giống người đọc câu: tạo thông tin về từ trong ngữ cảnh.

BiLSTM giống sổ tay đọc hai chiều: nối thông tin phía trước và phía sau.

CRF kiểm tra quan hệ giữa các nhãn rồi chọn cả chuỗi nhãn. Nó giúp giảm nhãn thiếu hợp lý, nhưng không bảo đảm hết mọi lỗi.

Ba phần này phục vụ việc tìm cụm. Đầu Intent làm thêm việc kiểm tra cả câu có chứa xúc phạm hay không.

Cuối cùng, cổng dùng kết quả kiểm tra câu để quyết định giữ hay bỏ các nhãn nghi vấn. Em sẽ giải thích cổng này ở trang tiếp theo.”

**Không cần đọc:** kích thước vector, công thức hoặc tên thuật toán Viterbi. Nếu được hỏi: vector PhoBERT có 768 chiều; Linear tạo điểm nhãn trước CRF.

## Slide 7 — Dual head hoạt động thế nào? | 75 giây

“Dual head có thể hiểu là hai đầu làm hai việc, cùng dùng chung PhoBERT.

Đầu thứ nhất tìm những từ hoặc cụm có thể xúc phạm. Đầu thứ hai đánh giá cả câu có chứa nội dung xúc phạm hay không.

Sau đó có một cổng lọc. Nếu điểm của đầu thứ hai thấp hơn 0,5, các nhãn xúc phạm được đổi về O. Nếu điểm từ 0,5 trở lên, hệ thống giữ kết quả của đầu tìm cụm.

Mục tiêu là tránh chỉ vì gặp một từ nhạy cảm mà đánh dấu nhầm. Tuy nhiên, nếu đầu kiểm tra câu đoán sai, cổng cũng có thể xóa mất cụm xúc phạm thật. Vì vậy, cần xem cả lợi ích và nguy cơ bỏ sót.”

**Nếu Thầy hỏi thêm:** Nhãn của đầu kiểm tra câu mặc định được suy từ việc câu có nhãn B hoặc I. Nó không trực tiếp đo ý định trong đầu người viết. Công thức huấn luyện là `L = L_CRF + 0,5 × L_BCE`.

## Slide 8 — Ví dụ hai vế câu | 60 giây

“Xét hai vế ‘Con chó này đẹp’ và ‘Mày đúng là con chó’. Cùng có từ ‘chó’, nhưng một vế nói về thú cưng, vế kia công kích người khác.

Để minh họa cổng, giả sử hệ thống xử lý riêng từng vế. Vế đầu có điểm 0,05, thấp hơn 0,5, nên cổng bỏ nhãn nghi vấn và giữ nguyên câu khen.

Vế sau có điểm 0,98, vượt ngưỡng, nên cổng giữ cụm đã tìm được để có thể che bằng dấu sao.

Hai điểm này dùng để giải thích cách hoạt động, chưa phải xác suất đầu Intent đo được cho từng vế. Bộ lọc trên web hiện còn dùng luật từ khóa. Điều nhóm hướng đến là giữ câu bình thường và tìm đúng cụm công kích.”

**Nhớ:** Muốn mô hình đánh giá riêng từng vế phải tách và chạy từng vế. Cổng không tự tìm thêm cụm mà đầu tìm cụm đã bỏ sót. Không khẳng định từ khóa nào cũng khiến mô hình cũ sai hoặc dual head sẽ loại bỏ hoàn toàn đánh dấu nhầm.

## Slide 9 — Bảng kết quả ba mô hình | 75 giây

“Precision cho biết trong các cụm đã đánh dấu có bao nhiêu cụm đúng. Recall cho biết trong các cụm cần tìm, mô hình tìm được bao nhiêu. F1 kết hợp cả hai.

Theo bảng cập nhật, F1 của Linear là 66,28%, CRF là 68,61% và mô hình có DualHead là 71,35%. Như vậy, mô hình cuối cao hơn Linear 5,07 điểm phần trăm và cao hơn CRF 2,74 điểm.

Các số này theo bảng dùng tạo hình. Nhóm cần đối chiếu log đánh giá trên toàn bộ test để xác nhận kết quả của checkpoint.”

## Slide 10 — Biểu đồ F1 mới | 20 giây

“Hình mới đã có DualHead ở cột thứ ba, đạt 71,35% theo báo cáo. Điều cần nhớ là mức tăng 5,07 điểm phần trăm so với Linear. Trục tung bắt đầu từ 60%, nên mình đọc giá trị số thay vì so tỷ lệ chiều cao các cột.”

## Slide 11 — Lỗi ranh giới từ ghép | 30 giây

“Hình này chỉ nói về lỗi xác định đầu cuối cụm. Tỷ lệ của Linear là 25,8%, CRF là 18,4%, còn DualHead là 13,8%.

Chênh lệch giữa Linear và DualHead là 12 điểm phần trăm. Đây là phép trừ hai tỷ lệ, không phải mức giảm tương đối 12%. Lỗi vẫn còn, nên cần tiếp tục kiểm tra ranh giới cụm.”

## Slide 12 — Mười một dạng lỗi trên 100 câu phân tích | 35 giây

“Hình số 4 đã đổi thành 11 dạng lỗi, không còn là biểu đồ đơn chuỗi và đa chuỗi.

Em chỉ nhấn mạnh hai nhóm: lỗi đa chuỗi giảm từ 12 xuống 4; đánh dấu nhầm từ động vật giảm từ 8 xuống 1. Tuy vậy, lỗi trích dẫn vẫn là 4 và lỗi nhãn gốc vẫn là 3.

Như vậy, có nhóm cải thiện rõ nhưng không phải mọi lỗi đều giảm hoặc biến mất.”

## Slide 13 — Ma trận BIO trên 100 câu đầu test | 35 giây

“Ma trận mới dùng 100 câu đầu của tập test, gồm 1.186 từ. Hàng là nhãn thật, cột là nhãn dự đoán.

Có 25 từ B và 34 từ I bị đoán thành O, tức 59 từ thuộc cụm bị bỏ sót. Ngược lại, có 28 từ O bị đánh dấu thành B hoặc I.

Đây là kết quả cấp từ trên một phần test. Không dùng nó để thay cho điểm F1 cấp cụm của toàn bộ test.”

## Slide 14 — Ca từ lóng ở phiên bản đơn đầu trước đây | 40 giây

“Ca này được giữ từ báo cáo trước: Linear và CRF tìm được cụm từ lóng, còn BiLSTM–CRF chưa có DualHead bỏ sót. Đây chưa phải kết quả chạy của DualHead mới.

Bài học là điểm tổng thể cao hơn không có nghĩa câu nào cũng đúng hơn. Nhóm cần thử lại các câu khó trên phiên bản hiện tại. Cổng chỉ giữ hoặc bỏ nhãn, không tự tìm thêm cụm mà đầu tìm cụm đã bỏ qua.”

## Slide 15 — Ứng dụng web | 40 giây

“Người dùng nhập bình luận, hệ thống trả về các cụm được đánh dấu và nội dung đã che.

Hiện cần phân biệt hai bước: cổng trong mô hình dual head và bộ lọc từng vế bằng luật trên web. Khi đánh giá, nhóm cần biết kết quả cải thiện đến từ bước nào.

Ứng dụng cũng cần hiển thị rõ mô hình đã tải. Nếu chỉ đang dùng cách xử lý dự phòng thì không nên coi đó là kết quả của mô hình đã huấn luyện.”

## Slide 16 — Hạn chế và hướng cải thiện | 45 giây

“Có ba việc nhóm cần làm tiếp.

Thứ nhất, bổ sung ví dụ từ lóng và viết tắt, kiểm tra để nhãn vẫn đúng.

Thứ hai, thử các mức ngưỡng của dual head và so sánh khi bật, tắt cổng. Cần đo cả đánh dấu nhầm và bỏ sót.

Thứ ba, bổ sung câu châm biếm và câu có nhiều vế khác nghĩa. Trước khi kết luận hiệu quả, nhóm cần lưu kết quả test và tách rõ phần đóng góp của mô hình với phần lọc bằng luật.”

## Slide 17 — Kết luận | 30 giây

“Bộ đối chứng mới có ba mô hình. Theo báo cáo, mô hình có DualHead đạt F1 71,35%, cao hơn Linear 5,07 điểm phần trăm. Lỗi ranh giới giảm từ 25,8 xuống 13,8%.

Hình 11 dạng lỗi vẫn cho thấy các trường hợp khó như châm biếm và trích dẫn. Việc tiếp theo là xác nhận kết quả trên toàn bộ test, cải thiện dữ liệu và tách rõ hiệu quả của mô hình với bộ lọc luật.”

## Slide 18 — Cảm ơn | 10 giây

“Nhóm em xin kết thúc phần trình bày. Cảm ơn Thầy và các bạn đã lắng nghe. Nhóm em xin nhận câu hỏi và góp ý.”

---

