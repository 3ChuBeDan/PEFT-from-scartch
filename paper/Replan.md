
---

### 1. Nguyên tắc định hướng lại

- **Giữ nguyên lõi báo cáo** – tái cài đặt thuật toán LoRA / DoRA, đánh giá trên PhoBERT cho bài toán phân loại tiếng Việt, trong điều kiện tài nguyên hạn chế (GPU RTX 3060 12GB, mô hình ≤ 2B tham số).
- **Thay thế tập ESG** bằng tập dữ liệu ngoài đã có sẵn và sạch (có thể là multi-label hoặc multi-class). Lý do thay đổi được ghi rõ trong báo cáo: *dữ liệu ESG chưa được gán nhãn đầy đủ, không thể hoàn thiện kịp tiến độ*.
- **Tập UIT-VSFC** vẫn giữ vai trò sanity-check (phân loại cảm xúc 3 lớp), đồng thời là nơi thực hiện ablation (target modules, merge test…) để tăng chiều sâu phân tích.
- **Tập dữ liệu ngoài** sẽ trở thành benchmark ứng dụng chính, chứng minh tính tổng quát của LoRA/DoRA trên một tác vụ phân loại tiếng Việt khác.

### 2. Điều chỉnh mục tiêu & phạm vi báo cáo

Cập nhật các mục trong file objective:

#### a. Định vị lại nhiệm vụ chính
- [x] ~~Định vị ESG multi-label là bài toán ứng dụng chính~~  
- [ ] **Mới:** Định vị tập **dữ liệu ngoài** (tạm gọi `EXT`) là bài toán ứng dụng chính. Mô tả rõ `EXT`: tên tập, số lớp/nhãn, kích thước, loại tác vụ (multi-label hay multi-class), nguồn gốc.  
- [ ] **Mới:** UIT-VSFC vẫn là benchmark phụ để kiểm tra tính ổn định và làm ablation.  

#### b. Cập nhật phần hạn chế
Thay các mục liên quan đến ESG bằng thực tế mới:
- [ ] **Thay:** “Chưa có kết quả benchmark đầy đủ cho dữ liệu ESG” → “Đã thay thế ESG bằng tập EXT do ESG chưa hoàn thiện nhãn.”  
- [ ] **Thay:** “Dataset chính trong bảng hiện tại là UIT-VSFC, chưa phải dữ liệu ESG mục tiêu của dự án” → “Hai dataset được dùng: UIT-VSFC (sentiment) và EXT (tác vụ chính).”  

#### c. Sửa nội dung báo cáo (paper.tex)
- [ ] Abstract: nêu rõ báo cáo đánh giá trên hai tập phân loại tiếng Việt (UIT-VSFC và EXT), thay vì ESG.  
- [ ] Introduction: giải thích ngắn gọn ESG không khả thi, EXT được chọn làm thực nghiệm chính.  
- [ ] Thêm bảng so sánh “Paper DoRA gốc vs báo cáo này” nhưng nay cập nhật dataset.  
- [ ] Discussion: xem việc không dùng ESG là quyết định do dữ liệu, không phải thiếu sót phương pháp.  

### 3. Kế hoạch thực nghiệm mới

Giữ nguyên định dạng đánh giá (FT, LoRA r8/r16, DoRA r8/r16, 3 seed), nhưng áp dụng cho EXT. Các ưu tiên sửa lại:

| Ưu tiên | Công việc |
|--------|-----------|
| 1 | Chạy đầy đủ FT, LoRA (r8, r16), DoRA (r8, r16) trên **EXT** với 3 seed. |
| 2 | Nếu FT trên EXT quá nặng, chạy 1 seed FT làm baseline, LoRA/DoRA vẫn 3 seed. |
| 3 | Báo cáo metric phù hợp với EXT: nếu multi-label → micro-F1, macro-F1, weighted-F1, samples-F1, hamming loss, per-label F1. Nếu multi-class → accuracy, macro-F1, weighted-F1, confusion matrix. |
| 4 | Phân tích per-label (nếu multi-label) để phát hiện nhãn hiếm, giống kế hoạch cũ cho S5, G9. |
| 5 | Ablation target modules: so sánh ít nhất `query,value` vs `query,key,value` trên UIT-VSFC (hoặc EXT nếu nhanh). |
| 6 | Kiểm tra merge() bằng logits trước/sau merge trên một batch bất kỳ. |
| 7 | Có thể thêm baseline frozen PhoBERT + classifier để tách biệt vai trò của adapter. |

### 4. Cập nhật danh sách “Việc cần làm” (todo) thay thế ESG bằng EXT

- [ ] Tạo thư mục dữ liệu cho EXT, viết script tiền xử lý, chia train/val/test nhất quán.  
- [ ] Cập nhật file CSV kết quả với cột `dataset` = `EXT`, ghi rõ task type.  
- [ ] Khi chạy xong, cập nhật `paper.tex`: thêm bảng kết quả EXT, phân tích, so sánh với UIT-VSFC.  
- [ ] Điều chỉnh Conclusion: hướng mở rộng là “thêm các tác vụ phân loại tiếng Việt khác (multi-label, đa miền)” thay vì nhắc đến ESG.  
- [ ] Loại bỏ hoặc gạch bỏ những mục trong file objective liên quan trực tiếp đến ESG (có thể giữ lại ghi chú “đã thay thế” để lịch sử).  

### 5. Tận dụng thế mạnh từ việc mất ESG

Việc không dùng được ESG thực ra là cơ hội để báo cáo tập trung hơn, minh bạch hơn.  
Bạn có thể:
- Làm sâu ablation (so sánh rank, alpha, target modules) trên UIT-VSFC, biến báo cáo thành một “nghiên cứu thực nghiệm về PEFT cho PhoBERT”.
- Nếu EXT là tập multi-label, báo cáo vẫn giữ được tính thời sự của bài toán phân loại đa nhãn tiếng Việt, vốn ít được khảo sát với PEFT.
- Nhấn mạnh bài học: dữ liệu thực tế thường không hoàn hảo, nhưng ta vẫn có thể đánh giá thuật toán trên các tập chuẩn thay thế – đây là một điểm cộng về tính thực tiễn.

### 6. Tóm tắt hành động cho session tiếp theo

1. Đọc lại `paper.tex` và xác định chính xác tên/cấu trúc tập dữ liệu ngoài sẽ dùng.  
2. Chuẩn hóa EXT vào pipeline hiện có (cùng định dạng tokenizer, dataloader).  
3. Chạy thực nghiệm theo bảng ưu tiên ở trên, ghi kết quả vào CSV.  
4. Sửa toàn bộ file objective (có thể dùng ngay bản trả lời này làm patch).  
5. Cập nhật báo cáo, biên dịch thử `latexmk -pdf -g paper.tex` để kiểm tra.  

Nếu bạn cho tôi biết thêm thông tin về tập dữ liệu ngoài (số mẫu, số lớp, có phải multi-label không), tôi có thể chi tiết hơn các metric và bảng biểu cần thêm.