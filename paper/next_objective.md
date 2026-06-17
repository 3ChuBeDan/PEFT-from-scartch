# Next Objective: Cải thiện báo cáo tái hiện DoRA

File này là ghi chú bàn giao cho các session tiếp theo. Mục tiêu là ghi lại trạng thái hiện tại của báo cáo, các hạn chế còn tồn tại, và thứ tự công việc nên làm để cải thiện chất lượng tái hiện thực nghiệm.

## 0. Việc cần làm trước roadmap chính

- [x] Viết rõ phần cài đặt LoRA/DoRA và cách tích hợp vào training trước khi chạy thêm thực nghiệm.
- [x] Dựa trên `paper/implementation_exposition.md` để mở rộng phần `Chi tiết cài đặt` trong `paper.tex`.
- [x] Khi cập nhật `paper.tex`, trình bày theo hướng implementation-first: `LoRALinear`, `DoRALinear`, `apply_peft()`, sau đó mới tới luồng `Trainer`.
- [x] Không trộn việc viết rõ implementation với roadmap benchmark ESG/PhoBERT; đây là bước cải thiện báo cáo cần làm trước.

## 1. Đánh giá nhanh báo cáo hiện tại

- [x] `paper/paper.tex` đã được chuyển sang dạng reproduction report, nhưng vẫn giữ layout ICML.
- [x] Báo cáo đã nêu rõ phạm vi tái hiện là một phần: cài đặt lõi LoRA/DoRA trên `torch.nn.Linear`, tích hợp với PhoBERT và benchmark trên UIT-VSFC.
- [x] Báo cáo đã có bảng kết quả chính cho FT, LoRA r8/r16 và DoRA r8/r16 trên ba seed 42, 43, 44.
- [x] Báo cáo đã ghi rõ DoRA chưa vượt LoRA một cách nhất quán trong thiết lập hiện tại.
- [x] Báo cáo đã loại ba run DoRA r8 đầu file CSV khỏi bảng chính vì có dấu hiệu smoke/debug runs.
- [ ] Chưa có kết quả benchmark đầy đủ cho dữ liệu ESG multi-label.
- [ ] Chưa có bảng đối chiếu trực tiếp với các số liệu trong paper DoRA gốc.

## 2. Hạn chế hiện tại

- [ ] Báo cáo chưa tái hiện benchmark gốc trên LLaMA, LLaVA hoặc VL-BART.
- [ ] Dataset chính hiện tại là UIT-VSFC, không phải dataset trong paper DoRA gốc.
- [ ] Kết quả hiện tại cho thấy DoRA gần FT và LoRA về accuracy, nhưng chưa chứng minh được ưu thế rõ ràng so với LoRA.
- [ ] Chưa có ablation theo target modules, ví dụ `query,value` so với `query,key,value,dense` hoặc các cấu hình attention/MLP mở rộng.
- [ ] Chưa có per-class F1, confusion matrix, loss curve hoặc phân tích lỗi mẫu dự đoán sai.
- [ ] Chưa mô tả đủ chi tiết optimizer/scheduler thực tế của Hugging Face `Trainer`.
- [ ] Chưa có citation riêng cho PhoBERT và UIT-VSFC trong `references.bib`.
- [ ] Chưa xác minh lại các cảnh báo LaTeX còn lại như duplicate anchor của bảng nếu cần bản nộp thật sạch.

## 3. Thông tin cần bổ sung vào báo cáo

- [ ] Bổ sung tên môn học và ngày nộp chính thức vào phần Giới thiệu hoặc thông tin chung.
- [ ] Bổ sung CPU, RAM hệ thống và hệ điều hành vào mục môi trường.
- [ ] Thêm citation cho PhoBERT.
- [ ] Thêm citation cho UIT-VSFC hoặc nguồn Hugging Face dataset nếu không tìm được paper gốc phù hợp.
- [ ] Ghi rõ optimizer/scheduler mặc định của `Trainer` đang dùng trong phiên bản Transformers hiện tại.
- [ ] Thêm bảng "Paper gốc vs báo cáo này" để làm rõ khác biệt về model, dataset, task, metric và tài nguyên.
- [ ] Nếu có kết quả ESG multi-label, thêm bảng kết quả phụ hoặc section riêng trong phần thực nghiệm.

## 4. Thực nghiệm cần chạy tiếp

- [ ] Ưu tiên 1: chạy benchmark ESG multi-label với FT, LoRA r8, LoRA r16, DoRA r8 và DoRA r16 trên ít nhất ba seed.
- [ ] Ưu tiên 2: chạy ablation target modules, tối thiểu so sánh `query,value` với một cấu hình mở rộng hơn.
- [ ] Ưu tiên 3: sinh per-class F1 và confusion matrix cho UIT-VSFC.
- [ ] Ưu tiên 4: sinh per-label F1, micro-F1, macro-F1, samples-F1 và hamming loss cho ESG multi-label.
- [ ] Ưu tiên 5: kiểm tra tính đúng của `merge()` bằng cách so sánh logits trước và sau merge cho LoRA/DoRA.
- [ ] Ưu tiên 6: nếu có tài nguyên, tái hiện thêm một thí nghiệm nhỏ gần paper DoRA gốc hơn thay vì chỉ dùng PhoBERT.

## 5. Tiêu chí cập nhật báo cáo sau khi có kết quả mới

- [ ] Mỗi bảng kết quả phải ghi rõ dataset, số seed, metric, rank, target modules và rule loại run nếu có.
- [ ] Không đưa run smoke/debug vào bảng chính.
- [ ] Nếu thêm ESG, phải nói rõ đây là benchmark mở rộng của repo, không phải benchmark gốc của DoRA paper.
- [ ] Nếu DoRA vẫn không vượt LoRA, giữ phân tích trung thực và không viết theo hướng quảng bá.
- [ ] Nếu thay target modules hoặc learning rate, phải cập nhật cả bảng siêu tham số và phần thảo luận.
- [ ] Sau mỗi lần sửa `paper.tex`, chạy `latexmk -pdf -g paper.tex` trong thư mục `paper`.

## 6. Ghi chú cho session sau

- [ ] Bắt đầu bằng cách đọc `paper/paper.tex`, `paper/objective.md`, file này và `results/benchmark_results.csv`.
- [ ] Kiểm tra xem có dòng kết quả mới nào trong CSV sau các run hiện tại không.
- [ ] Nếu chạy thêm benchmark, lưu kết quả vào cùng schema CSV hiện tại để dễ tổng hợp.
- [ ] Nếu tạo script tổng hợp kết quả, ưu tiên xuất ra bảng LaTeX để copy trực tiếp vào paper.
- [ ] Giữ nguyên tinh thần báo cáo kỹ thuật: trung thực, ghi rõ phạm vi tái hiện, không tuyên bố phát minh phương pháp mới.
