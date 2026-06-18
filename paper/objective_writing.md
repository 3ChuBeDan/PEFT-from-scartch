# Objective Writing: Bao cao PEFT from scratch tren UIT-ViON

File nay la brief viet lai bao cao ky thuat theo goc nhin doc lap. Bao cao can duoc trinh bay cho nguoi doc ben ngoai du an, co nen tang ML/NLP co ban, nhung khong biet boi canh repo hay cac muc tieu noi bo.

## 1. Muc tieu bao cao

Muc tieu chinh la trinh bay qua trinh xay dung PEFT from scratch va kiem nghiem thuc nghiem tren UIT-ViON. Bao cao can nhan manh ba cau hoi:

- LoRA va DoRA co the duoc cai dat truc tiep bang PyTorch nhu the nao?
- Cac module PEFT tu cai dat duoc tich hop vao PhoBERT cho bai toan phan loai van ban tieng Viet ra sao?
- Ket qua tren UIT-ViON cho thay trade-off giua chat luong phan loai va chi phi tai nguyen nhu tham so trainable, thoi gian train, VRAM va kich thuoc checkpoint nhu the nao?

Bao cao khong nen duoc viet nhu mot ghi chu noi bo. Hay dinh vi no nhu mot reproduction/implementation report doc lap ve LoRA/DoRA tren PhoBERT.

## 2. Doi tuong doc

Doi tuong doc la nguoi ngoai du an:

- Co the biet cac khai niem co ban ve Transformer, fine-tuning va NLP.
- Co the chua biet codebase, notebook, cau truc file, hay lich su phat trien.
- Can duoc giai thich du ro de hieu tai sao chon PhoBERT, UIT-ViON, LoRA, DoRA va official PEFT lam diem doi chieu.

Van phong nen la tieng Viet hoc thuat, ro rang, mach lac. Tranh viet theo kieu nhat ky lam viec, task list noi bo, hay ban giao session.

## 3. Goc nhin va pham vi

Goc nhin cua bao cao:

- Day la bao cao ky thuat ve PEFT from scratch.
- Trong tam thuc nghiem la UIT-ViON, mot tap phan loai chu de tin tuc tieng Viet.
- Mo hinh nen la `vinai/phobert-base-v2`.
- Cac phuong phap so sanh gom full fine-tuning, LoRA va DoRA.
- Ngoai implementation tu cai dat, bao cao co them doi chieu voi thu vien PEFT official de kiem tra tinh hop ly cua ket qua.

Nhung noi dung khong nam trong pham vi:

- Khong mo ta bao cao nhu mot phan cua du an khac.
- Khong nhac du lieu noi bo hoac muc tieu ngoai UIT-ViON.
- Khong nhac lich su quan ly ma nguon, thao tac phien ban, review code, nhanh lam viec, hay cac ghi chu phat trien noi bo.
- Khong tuyen bo day la mot phuong phap moi neu thuc chat la tai cai dat va kiem nghiem ky thuat.

## 4. Dinh dang bao cao

Bao cao cuoi cung van giu format ICML 2024 two-column trong `paper.tex`.

Yeu cau trinh bay:

- Dung layout ICML 2 cot, title, abstract, sections va references theo khung hien co.
- Viet bang tieng Viet hoc thuat, ngan gon, tranh lap y.
- Uu tien bang va hinh co kha nang doc nhanh trong layout 2 cot.
- Neu bang qua rong, dua bang phu vao appendix hoac rut gon cot trong than bai chinh.
- Cac cong thuc LoRA/DoRA nen viet bang LaTeX ro rang, khong dua qua nhieu code vao than bai.

## 5. Cau truc de xuat

### Abstract

Tom tat bai toan, pham vi cai dat, dataset UIT-ViON, cac phuong phap so sanh va ket qua chinh. Can noi ro day la bao cao cai dat va kiem nghiem PEFT from scratch, khong phai de xuat phuong phap moi.

### Introduction

Gioi thieu nhu cau fine-tuning hieu qua tham so cho mo hinh ngon ngu tieng Viet. Trinh bay ly do chon PhoBERT va UIT-ViON. Neu LoRA/DoRA duoc nhac den, can dinh vi chung la hai ky thuat PEFT can duoc tai cai dat va so sanh trong cung dieu kien.

### Background: LoRA, DoRA, PEFT

Giai thich ngan gon:

- LoRA dong bang trong so goc va hoc cap nhat hang thap.
- DoRA tach trong so thanh magnitude va direction.
- PEFT giam so tham so trainable va kich thuoc checkpoint so voi full fine-tuning.

Chi can dua cac cong thuc can thiet de doc hieu implementation.

### From-scratch implementation

Mo ta ro cac thanh phan:

- `LoRALinear`: boc `torch.nn.Linear`, giu frozen base weight/bias, hoc `lora_A`, `lora_B`, scale bang `alpha / rank`.
- `DoRALinear`: giu `weight_base`, hoc `magnitude`, `lora_A`, `lora_B`, dung cap nhat direction va chuan hoa theo cot.
- `apply_peft`: duyet module tree cua PhoBERT va thay cac target modules bang wrapper LoRA/DoRA.
- Target modules mac dinh: `query,value`.
- Classifier head van duoc train trong che do PEFT.

Can viet theo huong implementation-first: tu module PyTorch, den cach thay module trong PhoBERT, roi den luong train bang `Trainer`.

### Experimental setup on UIT-ViON

Mo ta dataset va cau hinh:

- Dataset: `data/uit_vion/subset.csv`.
- Task: single-label Vietnamese news topic classification.
- So lop: 13.
- Split: train, validation, test.
- Model: `vinai/phobert-base-v2`.
- Methods: FT, LoRA r8, LoRA r16, DoRA r8, DoRA r16.
- Seeds: 97, 98, 99.
- Rank/alpha: r8/alpha16 va r16/alpha32.
- Dropout: LoRA 0.05; DoRA effective dropout 0.0.
- Max length, batch size, epoch, learning rate va target modules can duoc ghi thanh bang.

### Results

Trinh bay ket qua chinh tu `results/benchmark_results_2.csv`.

Nen uu tien cac cot:

- accuracy.
- trainable params va trainable percent.
- train time.
- peak VRAM.
- checkpoint size.

Macro-F1 va weighted-F1 co the giu trong bang phu hoac audit, nhung khong can la graph chinh trong thiet lap single-label nay neu chung gan trung voi accuracy.

### Official PEFT comparison

Dung `results/offical_benchmark_result.csv`, `results/official_vs_custom_comparison.csv` va `results/official_vs_custom_summary.csv` de so sanh PEFT tu cai dat voi PEFT official.

Cach trinh bay:

- Ghep cac run theo `method + rank + seed`.
- Bao cao delta theo huong `official - custom`.
- Uu tien accuracy, train time, peak VRAM, checkpoint size.
- Neu co khac biet total params giua custom va official, giai thich day co the den tu cach PEFT official dong goi model/classifier/module saved, khong nen ket luan voi vang la loi.

Graph nen gom:

- Bar chart custom vs official cho accuracy, train time, peak VRAM.
- Delta chart official - custom.
- Scatter paired accuracy voi duong `x=y` va dai tham chieu +/-1%, +/-2%, +/-3%.

### Discussion

Phan tich trade-off:

- FT co chat luong manh nhung train toan bo tham so va checkpoint lon.
- LoRA/DoRA giam dang ke tham so trainable va checkpoint size.
- DoRA khong nhat thiet vuot LoRA trong moi cau hinh nho; can trinh bay trung thuc theo ket qua.
- Official PEFT la doi chieu thuc dung giup kiem tra xu huong va chi phi, khong phai muc tieu chinh cua implementation from scratch.

Neu co sai lech nho giua custom va official, can thao luan ve initialization, module wrapping, classifier head, total params, logging va cach save checkpoint.

### Conclusion

Tom tat nhung gi da lam duoc:

- Cai dat LoRA/DoRA from scratch tren `torch.nn.Linear`.
- Tich hop vao PhoBERT.
- Kiem nghiem tren UIT-ViON voi nhieu seed va rank.
- Doi chieu voi PEFT official.

Ket luan can trung thuc: bao cao chung minh pipeline PEFT tu cai dat co the dat chat luong phan loai canh tranh voi chi phi tai nguyen thap hon full fine-tuning, trong pham vi PhoBERT va UIT-ViON.

## 6. Yeu cau trinh bay chi tiet cai dat

Phan cai dat la phan quan trong nhat. Can viet du de nguoi doc co the lap lai y tuong:

- Mo ta tensor shape cua `lora_A`, `lora_B`, `magnitude`.
- Noi ro base weight va bias duoc dong bang.
- Noi ro output ban dau cua LoRA trung voi base layer vi `lora_B` khoi tao bang 0.
- Noi ro DoRA dung magnitude va direction, khac LoRA o cho tai tao effective weight.
- Noi ro target module trong PhoBERT la attention `query,value`.
- Noi ro checkpoint FT luu full model, con LoRA/DoRA luu adapter/trainable tensors.

Khong can dan code dai. Neu can, chi dua pseudo-code ngan hoac cong thuc.

## 7. Yeu cau trinh bay ket qua

Nguon so lieu chinh:

- `results/benchmark_results_2.csv`: ket qua PEFT tu cai dat.
- `results/offical_benchmark_result.csv`: ket qua PEFT official.
- `results/official_vs_custom_comparison.csv`: so sanh paired run.
- `results/official_vs_custom_summary.csv`: tong hop theo method/rank.

Metric nen uu tien:

- Accuracy cho chat luong phan loai.
- Trainable params va trainable percent cho muc do tiet kiem tham so.
- Train time cho chi phi huan luyen.
- Peak VRAM cho chi phi bo nho.
- Checkpoint size cho chi phi luu tru.

Macro-F1/weighted-F1:

- Co the bao cao trong bang phu.
- Khong can la hinh chinh neu single-label va gia tri gan voi accuracy.
- Neu nhac den, can noi chung duoc giu de audit va doi chieu voi cac thiet lap khac.

## 8. Nhung dieu khong duoc viet

Khi viet bao cao, tranh cac noi dung sau:

- Khong nhac tu khoa lien quan den du lieu hoac muc tieu ngoai UIT-ViON.
- Khong nhac lich su quan ly ma nguon, thao tac phien ban, review code, nhanh lam viec, hay lich su phat trien repo.
- Khong mo ta day la bao cao noi bo.
- Khong viet rang phuong phap nay la dong gop thuat toan moi.
- Khong dua cac smoke/debug run vao bang ket qua chinh.
- Khong noi qua ve ket qua neu LoRA/DoRA chi canh tranh chu khong vuot FT mot cach nhat quan.

## 9. Checklist truoc khi viet lai `paper.tex`

- [ ] Title phan anh dung goc nhin: PEFT from scratch tren UIT-ViON.
- [ ] Abstract khong co noi dung ngoai pham vi.
- [ ] Introduction giai thich PhoBERT, UIT-ViON va PEFT.
- [ ] Background co cong thuc LoRA/DoRA vua du.
- [ ] Implementation noi ro `LoRALinear`, `DoRALinear`, `apply_peft`.
- [ ] Experimental setup co bang hyperparameters.
- [ ] Results co bang chinh va graph chinh.
- [ ] Official comparison duoc trinh bay nhu mot doi chieu.
- [ ] Discussion trung thuc ve trade-off va han che.
- [ ] Ket luan khong tuyen bo qua muc.
- [ ] Van giu ICML 2024 two-column format.

## 10. Kiem tra noi dung sau khi viet

Sau khi viet lai bao cao hoac objective, can quet va loai bo moi dau vet cua thong tin ngoai pham vi, bao gom boi canh du an khac, du lieu noi bo, thao tac quan ly ma nguon, review code, nhanh lam viec, lich su phat trien repo, va cac ghi chu dieu phoi noi bo.

Can dong thoi dam bao cac tu khoa can co xuat hien:

```text
PEFT from scratch
UIT-ViON
PhoBERT
LoRA
DoRA
ICML 2024 two-column
```
