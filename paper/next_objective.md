# Next Objective: Cải thiện báo cáo DoRA với benchmark UIT-ViON

File này là ghi chú bàn giao cho các session tiếp theo. Định hướng mới là xem dự án như một báo cáo kỹ thuật về cài đặt và đánh giá LoRA/DoRA trong điều kiện tài nguyên hạn chế, tập trung vào PhoBERT và các bài toán phân loại văn bản tiếng Việt. Do máy hiện tại chỉ phù hợp với mô hình tối đa khoảng 2 tỷ tham số, roadmap không đặt mục tiêu chạy lại benchmark lớn của DoRA trên LLaMA, LLaVA hoặc VL-BART.

Thay đổi quan trọng: dữ liệu ESG nội bộ chưa được gán nhãn/làm sạch đầy đủ và mất cân bằng mạnh, nên không dùng làm benchmark chính ở giai đoạn này. Benchmark ứng dụng chính sẽ chuyển sang UIT-ViON, một tập phân loại chủ đề tin tức tiếng Việt có nhãn sẵn, quy mô lớn và phù hợp với PhoBERT.

## 0. Việc đã làm trước roadmap chính

- [x] Viết rõ phần cài đặt LoRA/DoRA và cách tích hợp vào training trước khi chạy thêm thực nghiệm.
- [x] Tạo `paper/implementation_exposition.md` để ghi cách trình bày implementation theo hướng kỹ thuật.
- [x] Dựa trên `paper/implementation_exposition.md` để mở rộng phần `Chi tiết cài đặt` trong `paper.tex`.
- [x] Trình bày theo hướng implementation-first: `LoRALinear`, `DoRALinear`, `apply_peft()`, sau đó tới luồng `Trainer`.
- [x] Tách việc mô tả implementation khỏi roadmap benchmark.
- [x] Sau PR #1, LoRA/DoRA đã có thêm save/load adapter, enable/disable adapter, merge/unmerge cho LoRA, `init_magnitude` và detached gradient cho DoRA.
- [x] `results/benchmark_results_2.csv` là schema v2/canonical cho các run mới; `results/benchmark_results.csv` chỉ giữ vai trò lịch sử vì đã lẫn schema cũ/mới.

## 1. Định hướng mới của báo cáo

- [x] Giữ layout ICML, nhưng giọng văn là báo cáo kỹ thuật thay vì bài báo đề xuất phương pháp mới.
- [x] Xem DoRA paper là nguồn phương pháp chính, còn báo cáo này là tái cài đặt thuật toán lõi và kiểm chứng trong bối cảnh nhỏ hơn.
- [x] Định vị UIT-VSFC là benchmark sanity-check cho phân loại cảm xúc tiếng Việt, đồng thời dùng cho ablation nhỏ.
- [x] Định vị UIT-ViON là benchmark ứng dụng chính thay cho ESG nội bộ.
- [ ] Tiêu chí thành công chính là: giảm tham số trainable/checkpoint/VRAM trong khi giữ chất lượng phân loại chấp nhận được, không phải tái hiện số liệu benchmark LLM/LVLM gốc.
- [ ] Khi sửa `paper.tex`, đổi các câu dễ gây hiểu nhầm từ "tái hiện benchmark gốc" sang "tái hiện thuật toán lõi trong điều kiện tài nguyên hạn chế".

## 2. Lý do thay ESG bằng UIT-ViON

- [x] ESG nội bộ chưa được gán nhãn và làm sạch hoàn chỉnh.
- [x] ESG nội bộ mất cân bằng mạnh; một số nhãn quá hiếm nên metric dễ dao động và khó kết luận công bằng về LoRA/DoRA.
- [x] UIT-ViON có nhãn sẵn, quy mô lớn, có nguồn công khai và có paper/dataset để trích dẫn.
- [x] UIT-ViON là bài toán phân loại chủ đề tin tức tiếng Việt, khác miền với UIT-VSFC nên giúp đánh giá tổng quát hơn.
- [ ] Cần kiểm tra trực tiếp file dữ liệu UIT-ViON để xác nhận format, số lớp, phân bố nhãn và split.
- [x] Kế hoạch chạy UIT-ViON dùng hai giai đoạn: stratified subset trước, sau đó full dataset hoặc subset lớn cho bảng chính.

## 3. Phạm vi không làm do giới hạn tài nguyên

- [x] Không chạy benchmark gốc trên LLaMA, LLaVA hoặc VL-BART vì không phù hợp tài nguyên phần cứng và không khớp trực tiếp với bài toán phân loại hiện tại.
- [x] Không xem việc thiếu benchmark LLaMA/LLaVA/VL-BART là lỗi cần sửa của báo cáo; đây là giới hạn phạm vi cần ghi rõ.
- [x] Không ưu tiên causal LLM lớn cho bài toán phân loại khi PhoBERT đã phù hợp hơn về ngôn ngữ, kích thước và tài nguyên.
- [ ] Chỉ cân nhắc mô hình lớn hơn PhoBERT nếu mô hình đó dưới khoảng 2B tham số, chạy được trên GPU hiện tại, và có lý do rõ ràng cho tác vụ phân loại.
- [ ] Nếu thêm mô hình dưới 2B, phải xem đó là benchmark phụ, không thay thế trọng tâm PhoBERT/UIT-ViON.

## 4. Đánh giá nhanh trạng thái hiện tại

- [x] `paper/paper.tex` đã chuyển sang dạng technical reproduction report và vẫn giữ layout ICML.
- [x] Báo cáo đã mô tả phạm vi tái hiện: cài đặt lõi LoRA/DoRA trên `torch.nn.Linear`, tích hợp với PhoBERT và benchmark trên UIT-VSFC.
- [x] Báo cáo đã có bảng kết quả FT, LoRA r8/r16 và DoRA r8/r16 trên UIT-VSFC với ba seed 42, 43, 44.
- [x] Báo cáo đã ghi rõ DoRA chưa vượt LoRA nhất quán trong thiết lập PhoBERT/UIT-VSFC.
- [x] Báo cáo đã loại ba run DoRA r8 đầu file CSV khỏi bảng chính vì có dấu hiệu smoke/debug runs.
- [x] Đã có script chuẩn hóa UIT-ViON hoặc dataset tương thích sang CSV `text,label,split`: `scripts/prepare_uit_vion.py`.
- [ ] Chưa có kết quả benchmark UIT-ViON.
- [x] Đã cập nhật `paper.tex` để thay ESG bằng UIT-ViON trong định hướng thực nghiệm tiếp theo.

## 5. Hạn chế hiện tại cần ghi rõ

- [x] Unit tests đã kiểm tra `merge()` giữ logits cho LoRA/DoRA trên layer nhỏ.
- [x] Đã có citation riêng cho PhoBERT, UIT-VSFC và UIT-ViON trong `references.bib`.


## 6. Việc cần sửa trong báo cáo chính

- [x] Sửa Abstract để nói rõ báo cáo đánh giá LoRA/DoRA trên PhoBERT với UIT-VSFC và UIT-ViON, không phải ESG.
- [x] Sửa Introduction để giải thích ESG nội bộ chưa đủ sạch nên được chuyển thành hướng ứng dụng dài hạn.
- [x] Sửa phần dữ liệu: UIT-VSFC là sanity-check; UIT-ViON là benchmark chính cho tác vụ phân loại chủ đề tin tức.
- [x] Cập nhật phần implementation: DoRA chuẩn hóa theo cột (`dim=0`), `init_magnitude=weight_norm`, `use_detached_gradient=True`, dropout hiệu dụng của DoRA là `0.0`, adapter save/load được kiểm thử.
- [x] Thêm bảng "Paper DoRA gốc vs báo cáo này" gồm: model, task, dataset, metric, tài nguyên, phạm vi tái hiện.
- [x] Sửa Discussion để trình bày thiếu LLaMA/LLaVA/VL-BART như một quyết định phạm vi, không phải mục tiêu còn dang dở.
- [x] Sửa Conclusion để bỏ hướng "nếu có tài nguyên thì tái hiện benchmark gốc"; thay bằng hướng "mở rộng sang các tác vụ phân loại tiếng Việt khác và dữ liệu ESG sau khi được làm sạch".
- [ ] Nếu có kết quả UIT-ViON, thêm bảng kết quả chính và phân tích nhầm lẫn giữa các chủ đề.

## 7. Thực nghiệm cần chạy tiếp

- [x] Ưu tiên 1: thêm script chuẩn hóa UIT-ViON vào pipeline hiện có.
- [ ] Ưu tiên 2: kiểm tra phân bố nhãn UIT-ViON và quyết định dùng full dataset hay stratified subset.
- [ ] Ưu tiên 3: chạy FT, LoRA r8, LoRA r16, DoRA r8 và DoRA r16 trên UIT-ViON với ba seed nếu thời gian cho phép.
- [ ] Nếu FT trên UIT-ViON quá tốn thời gian, chạy FT một seed làm mốc tham chiếu và ghi rõ giới hạn; LoRA/DoRA vẫn nên chạy ba seed.
- [ ] Ưu tiên 4: báo cáo metric UIT-ViON gồm accuracy, macro-F1, weighted-F1 và confusion matrix.
- [ ] Ưu tiên 5: chạy ablation target modules trên UIT-VSFC, tối thiểu so sánh `query,value` với `query,key,value`.
- [ ] Ưu tiên 6: thêm baseline classifier-only hoặc frozen PhoBERT nếu muốn tách lợi ích của adapter khỏi classifier head.
- [ ] Ưu tiên 7: kiểm tra tính đúng của `merge()` bằng cách so sánh logits trước và sau merge cho LoRA/DoRA trên cùng batch.

## 8. Tiêu chí cập nhật kết quả

- [ ] Mỗi bảng kết quả phải ghi rõ dataset, task type, số seed, rank, alpha, dropout, target modules và metric.
- [ ] Không đưa smoke/debug run vào bảng chính.
- [ ] Nếu dùng subset UIT-ViON, phải ghi rõ cách lấy mẫu, số mẫu mỗi lớp và seed chia dữ liệu.
- [ ] Nếu UIT-ViON có mất cân bằng nhãn, phải dùng macro-F1 và confusion matrix để phân tích thay vì chỉ nhìn accuracy.
- [ ] Nếu DoRA không vượt LoRA, giữ phân tích trung thực: DoRA là biến thể PEFT cạnh tranh, không phải kết luận thắng tuyệt đối.
- [ ] Nếu thay learning rate, target modules hoặc max length, phải cập nhật bảng siêu tham số.
- [ ] Sau mỗi lần sửa `paper.tex`, chạy `latexmk -pdf -g paper.tex` trong thư mục `paper`.

## 9. Ghi chú cho session sau

- [ ] Bắt đầu bằng cách đọc `paper/paper.tex`, `paper/Replan.md`, `paper/implementation_exposition.md`, file này và `results/benchmark_results_2.csv`; chỉ đọc `results/benchmark_results.csv` khi cần đối chiếu lịch sử.
- [x] Kiểm tra repo có sẵn script/dataset loader nào cho UIT-ViON chưa.
- [x] Nếu chưa có, tạo script tiền xử lý UIT-ViON sao cho output khớp pipeline hiện tại.
- [ ] Khi chạy thêm benchmark, giữ schema CSV v2 hiện tại trong `results/benchmark_results_2.csv` và thêm `dataset=uit-vion` để dễ tổng hợp mean/std.
- [ ] Nếu tạo script tổng hợp kết quả, ưu tiên xuất cả Markdown và LaTeX table.
- [ ] Giữ tinh thần báo cáo kỹ thuật: trung thực về phạm vi, rõ về tài nguyên, không tuyên bố phát minh phương pháp mới.
