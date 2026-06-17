# Implementation Exposition: LoRA/DoRA và luồng train

File này ghi lại cách nên trình bày phần cài đặt LoRA, DoRA và tích hợp vào quá trình huấn luyện trong báo cáo chính. Nội dung được viết theo hướng implementation-first: bắt đầu từ module PyTorch cụ thể, sau đó giải thích cách thay thế module trong PhoBERT và cách `train.py` điều phối huấn luyện.

## 1. Cách implement LoRA

Code liên quan: `src/peft/lora.py`.

Trong repo này, LoRA được cài đặt bằng lớp `LoRALinear`, dùng để bọc một lớp `torch.nn.Linear` đã tồn tại trong mô hình tiền huấn luyện. Khi khởi tạo, lớp này sao chép `linear.weight` và `linear.bias` từ lớp gốc sang các tham số frozen. Hai tensor này giữ nguyên giá trị tiền huấn luyện và không nhận gradient trong quá trình huấn luyện LoRA.

Thay vì cập nhật trực tiếp trọng số gốc, `LoRALinear` thêm hai ma trận trainable:

- `lora_A` có shape `[rank, in_features]`.
- `lora_B` có shape `[out_features, rank]`.

Ma trận `lora_A` được khởi tạo bằng Kaiming uniform, còn `lora_B` được khởi tạo bằng 0. Cách khởi tạo này khiến cập nhật LoRA ban đầu bằng 0, nên output ban đầu của wrapper trùng với output của lớp tuyến tính gốc. Hệ số scale được tính bằng:

```text
scaling = alpha / rank
```

Trong forward pass, module tính hai nhánh:

```text
base = F.linear(x, weight, bias)
adapter = F.linear(F.linear(dropout(x), lora_A), lora_B)
output = base + adapter * scaling
```

Như vậy, nhánh base giữ nguyên năng lực của mô hình tiền huấn luyện, còn nhánh adapter học phần cập nhật hạng thấp. Khi cần merge để suy luận hoặc kiểm tra, `LoRALinear.merge()` tạo một `nn.Linear` thông thường với trọng số:

```text
merged_weight = weight + (lora_B @ lora_A) * scaling
```

Điểm nên nhấn mạnh trong báo cáo: LoRA trong repo này không dùng thư viện PEFT có sẵn, mà tự thay thế trực tiếp `nn.Linear`; trọng số gốc bị đóng băng, chỉ `lora_A`, `lora_B` và classifier head được huấn luyện trong chế độ LoRA.

## 2. Cách implement DoRA

Code liên quan: `src/peft/dora.py`.

DoRA được cài đặt bằng lớp `DoRALinear`, cũng bọc một lớp `torch.nn.Linear` gốc. Khác với LoRA, DoRA không cộng trực tiếp cập nhật hạng thấp vào trọng số cuối cùng. Thay vào đó, module phân rã trọng số thành độ lớn và hướng.

Khi khởi tạo, trọng số gốc `W0` được lưu vào buffer `weight_base`, không phải parameter trainable. Bias của lớp gốc, nếu có, cũng được sao chép và đóng băng. Module sau đó tạo ba nhóm tham số học được:

- `magnitude`: vector độ lớn, được khởi tạo bằng norm của từng hàng trọng số gốc.
- `lora_A`: ma trận hạng thấp thứ nhất.
- `lora_B`: ma trận hạng thấp thứ hai.

Do trọng số của `torch.nn.Linear` trong PyTorch có shape `[out_features, in_features]`, triển khai hiện tại chuẩn hóa theo từng hàng đầu ra. Cập nhật hướng được tính như sau:

```text
direction_update = (lora_B @ lora_A) * scaling
direction = weight_base + direction_update
norm = vector_norm(direction, dim=1, keepdim=True)
merged_weight = magnitude * direction / norm
```

Trong forward pass, `DoRALinear` dựng lại `merged_weight` rồi gọi:

```text
output = F.linear(x, merged_weight, bias)
```

Điểm khác biệt cần viết rõ: LoRA có nhánh adapter cộng vào output của base layer, còn DoRA tái tạo lại trọng số hiệu dụng từ thành phần magnitude và direction ở mỗi forward. Về mặt tham số, DoRA học thêm `magnitude` ngoài hai ma trận `lora_A`, `lora_B`. Khi merge, `DoRALinear.merge()` tạo một `nn.Linear` mới mang trọng số đã phân rã và chuẩn hóa, phục vụ suy luận hoặc kiểm tra tương đương.

## 3. Cách tích hợp LoRA/DoRA vào PhoBERT

Code liên quan: `src/peft/apply.py` và `train.py`.

Việc tích hợp không sửa trực tiếp kiến trúc PhoBERT ở mức source code của Transformers. Thay vào đó, repo duyệt cây module sau khi model đã được tạo, tìm các lớp `nn.Linear` có tên khớp với `target_modules`, rồi thay chúng bằng wrapper LoRA hoặc DoRA.

Hàm chính là `apply_peft()`:

- Duyệt `model.named_modules()`.
- Với mỗi module, kiểm tra tên module bằng `_target_match()`.
- Chỉ chọn module là instance của `nn.Linear`.
- Lấy parent module bằng `model.get_submodule(parent_name)`.
- Thay child module bằng `LoRALinear.from_linear(...)` hoặc `DoRALinear.from_linear(...)`.
- Trả về danh sách tên đầy đủ của các module đã bị thay.

Target mặc định hiện tại là:

```text
query,value
```

Với PhoBERT-base có 12 encoder layers, mỗi layer có self-attention query projection và value projection. Vì vậy cấu hình mặc định thay 24 lớp tuyến tính:

```text
roberta.encoder.layer.{0..11}.attention.self.query
roberta.encoder.layer.{0..11}.attention.self.value
```

Trong báo cáo nên nhấn mạnh đây là PEFT trên attention projections, không phải thay toàn bộ mô hình. Classifier head vẫn được huấn luyện để phù hợp với số nhãn của từng dataset.

## 4. Luồng train trong `train.py`

Code liên quan: `train.py`.

Script `train.py` điều phối toàn bộ pipeline từ CLI đến huấn luyện và ghi kết quả. Các tham số quan trọng gồm:

- `--method ft|lora|dora`
- `--rank`
- `--alpha`
- `--dropout`
- `--target-modules`
- `--task-type single_label|multi_label`
- `--label-threshold`
- `--dataset`
- `--label-map`

Sau khi parse arguments, script đặt default tự động:

- Nếu `alpha` không được truyền và method là LoRA/DoRA, dùng `alpha = 2 * rank`.
- Nếu `learning_rate` không được truyền, FT dùng `2e-5`, LoRA/DoRA dùng `2e-4`.

Pipeline huấn luyện có thể trình bày theo các bước:

1. Load tokenizer của `vinai/phobert-base-v2`.
2. Load dataset UIT-VSFC hoặc file CSV/JSONL cục bộ.
3. Encode label:
   - single-label dùng nhãn nguyên.
   - multi-label dùng vector multi-hot.
4. Tokenize văn bản với `max_length=256`.
5. Build PhoBERT classifier bằng `build_phobert_classifier()`.
6. Gọi `configure_method()` để chọn chế độ train.
7. Đếm tổng tham số và tham số trainable.
8. Tạo `TrainingArguments` và `Trainer`.
9. Train, evaluate, đo thời gian và VRAM đỉnh.
10. Lưu checkpoint và append một dòng vào `results/benchmark_results.csv`.

Trong `configure_method()`:

- Với FT, mọi tham số của model được đặt `requires_grad=True`.
- Với LoRA/DoRA, trước hết gọi `freeze_all_but_classifier()` để đóng băng backbone và chỉ giữ classifier trainable.
- Sau đó gọi `apply_peft()` để thay các module mục tiêu bằng LoRA/DoRA wrapper. Các tham số mới trong wrapper mặc định là trainable.

Metrics phụ thuộc vào loại task:

- `single_label`: accuracy, macro-F1, weighted-F1.
- `multi_label`: sigmoid threshold, micro-F1, macro-F1, weighted-F1, samples-F1, hamming loss.

## 5. Checkpoint, tham số trainable và logging

Code liên quan: `train.py`.

Sau huấn luyện, cách lưu checkpoint khác nhau giữa FT và PEFT:

- FT gọi `trainer.save_model(...)`, lưu full model checkpoint.
- LoRA/DoRA gọi `save_trainable_checkpoint(...)`, chỉ lưu các tensor liên quan tới tham số trainable.

Điều này giải thích vì sao checkpoint FT trong benchmark lớn hơn rất nhiều so với LoRA/DoRA. Với benchmark UIT-VSFC hiện tại, FT có checkpoint khoảng 5.1 GB, trong khi LoRA/DoRA chỉ khoảng 3.4--4.6 MB.

Mỗi run ghi lại các thông tin sau vào CSV:

- method, model, dataset, rank, alpha, dropout, seed.
- metrics.
- số tham số trainable và tổng tham số.
- trainable percent.
- thời gian train.
- peak VRAM.
- checkpoint size.
- output directory.

Trong báo cáo, phần này nên được dùng để giải thích rằng so sánh không chỉ dựa trên chất lượng phân loại mà còn dựa trên chi phí tài nguyên.

## 6. Cách đưa nội dung này vào `paper.tex`

Không nên đưa toàn bộ nội dung file này vào paper chính vì sẽ quá dài. Thay vào đó, phần `Chi tiết cài đặt` trong `paper.tex` nên được mở rộng theo ba đoạn ngắn:

1. `LoRALinear`: frozen base weight/bias, hai ma trận A/B, forward và merge.
2. `DoRALinear`: `weight_base`, `magnitude`, cập nhật hướng hạng thấp, chuẩn hóa theo hàng.
3. `Tích hợp vào PhoBERT và Trainer`: `freeze_all_but_classifier()`, `apply_peft()`, target modules `query,value`, checkpoint FT vs PEFT.

Có thể thêm một bảng nhỏ trong paper chính:

| Chế độ | Tham số trainable | Module bị thay | Checkpoint |
| --- | --- | --- | --- |
| FT | Toàn bộ model | Không | Full model |
| LoRA | Classifier + `lora_A`, `lora_B` | `query,value` | Trainable tensors |
| DoRA | Classifier + `magnitude`, `lora_A`, `lora_B` | `query,value` | Trainable tensors |

Nếu cần thêm chi tiết code, đưa vào appendix hoặc tham chiếu file này như ghi chú nội bộ khi viết lại báo cáo.
