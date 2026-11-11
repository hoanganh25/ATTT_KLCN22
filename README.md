# Semantic Voice V0.1 — AWGN và Rayleigh

Repo triển khai mô hình truyền tiếng nói trên đặc trưng log-Mel. Encoder mã hóa từng patch log-Mel thành latent, kênh AWGN hoặc Rayleigh tác động lên latent, và Decoder khôi phục log-Mel. Bộ tổng hợp Power-Mel Joint chuyển log-Mel khôi phục thành dạng sóng để tính thêm waveform MSE, STOI và PESQ-WB.

Quy trình thực nghiệm gồm: chuẩn bị dữ liệu → huấn luyện bốn cấu hình → ước lượng và chọn ngưỡng energy-mask chỉ từ TRAIN/DEV → đánh giá bốn checkpoint trên TEST → vẽ biểu đồ Chương 4.

## Cấu trúc chính

| Đường dẫn | Nội dung |
|---|---|
| `Data/` | Quản lý dữ liệu đầu vào và các tập dữ liệu đã chuẩn bị cho mô hình |
| `prepared/` | Cache log-Mel, manifest và thống kê chuẩn hóa được tạo từ âm thanh |
| `channels/` | Mô phỏng các kênh truyền AWGN, Rayleigh và kênh không nhiễu |
| `mel_frontend/` | Tiền xử lý âm thanh và chuyển đổi sang đặc trưng log-Mel |
| `mel_inverse/` | Khôi phục dạng sóng từ đặc trưng Mel bằng Power-Mel Joint |
| `model/CNN_Baseline/` | Encoder và Decoder của mô hình CNN Baseline |
| `model/SE-ResNet/` | Encoder và Decoder của mô hình SE-ResNet |
| `training/` | Các thành phần dùng chung trong quá trình huấn luyện và lấy mẫu SNR |
| `evaluation/` | Tính toán chỉ số và tổng hợp kết quả đánh giá |
| `runs/` | Cấu hình, lịch sử huấn luyện và checkpoint của các mô hình |
| `reports/` | Kết quả đánh giá, thống kê dữ liệu và ngưỡng energy-mask |
| `chart/` | Các công cụ vẽ biểu đồ Chương 4 và biểu đồ loss huấn luyện |
| `chart/ve_bieu_do_4.5_va_4.6/` | Biểu đồ loss TRAIN/DEV và DEV MSE theo SNR |
| `chart/ve_bieu_do_4.7/`–`chart/ve_bieu_do_4.14/` | Các biểu đồ kết quả đánh giá trên TEST |
| `chart/chart_train_dev_loss/` | Biểu đồ loss TRAIN/DEV độc lập, không thuộc bộ biểu đồ Chương 4 dùng chung |

Dataset, cache, checkpoint và kết quả báo cáo không được đóng gói trong repo. Các thư mục tương ứng bị `.gitignore` bỏ qua; cần chuẩn bị hoặc sinh chúng trên máy chạy.

## Cài đặt

Yêu cầu Python 3.10 trở lên và PyTorch phù hợp với CPU/GPU của máy. Các thư viện còn lại nằm trong `requirements.txt`.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu dùng GPU NVIDIA, cài bản PyTorch/CUDA tương ứng theo hướng dẫn cài PyTorch trước hoặc sau khi cài `requirements.txt`.

## Chuẩn bị dữ liệu

Đặt WAV/FLAC trong một thư mục dataset. Ví dụ với LibriSpeech:

```powershell
python prepare_dataset.py `
  --dataset "D:\Datasets\LibriSpeech" `
  --cache prepared `
  --out Data/splits `
  --split-unit speaker `
  --seed 42
```

Frontend chuyển âm thanh về mono 16 kHz, tạo log-Mel 96 dải với STFT `n_fft=1024`, `hop_length=256`, rồi chia thành patch 96×96. Thống kê chuẩn hóa được tính từ TRAIN. Nếu dataset có đủ thư mục split LibriSpeech chính thức, chương trình dùng các split đó; nếu không, chương trình chia theo speaker. Cần ít nhất ba nhóm speaker độc lập.

Lệnh tạo `prepared/` và `Data/splits/`. Nếu các thư mục đã tồn tại với cấu hình khác, hãy chọn đường dẫn cache/output mới; không dùng `--max-files` cho kết quả báo cáo.

## Huấn luyện bốn cấu hình

Cấu hình dưới đây khớp với `config.json` trong các thư mục `runs/`: 20 epoch, seed 42, SNR huấn luyện lấy mẫu đều trong [0, 20] dB và validation tại 0, 5, 10, 15, 20 dB. CNN dùng batch 16; SE-ResNet dùng profile `repo`, latent 8 kênh, batch 64.

Mỗi lệnh cần thư mục output mới hoặc rỗng. Script không ghi đè run đã tồn tại.

### 1. CNN Baseline — AWGN

```powershell
python train_CNN_baseline.py `
  --splits Data/splits `
  --channel awgn `
  --run runs/cnn_baseline/awgn `
  --epochs 20 --batch-size 16 --lr 0.001 `
  --train-strategy multi --snr-min 0 --snr-max 20 --snr-sampling uniform `
  --val-snrs 0,5,10,15,20 --seed 42
```

### 2. CNN Baseline — Rayleigh

```powershell
python train_CNN_baseline.py `
  --splits Data/splits `
  --channel rayleigh `
  --run runs/cnn_baseline/rayleigh `
  --epochs 20 --batch-size 16 --lr 0.001 `
  --train-strategy multi --snr-min 0 --snr-max 20 --snr-sampling uniform `
  --val-snrs 0,5,10,15,20 --seed 42
```

### 3. SE-ResNet — AWGN

```powershell
python train_se_resnet.py `
  --splits Data/splits --channel awgn --profile repo `
  --latent-channels 8 --blocks 0 `
  --epochs 20 --batch-size 64 --accum-steps 1 --lr 0.001 `
  --snr-min 0 --snr-max 20 --snr-sampling uniform `
  --snr-grid 0,5,10,15,20 --val-snrs 0,5,10,15,20 `
  --seed 42 --num-workers 0 --threads 2 `
  --out runs/se_resnet_deepscs/awgn_repo8_b64
```

### 4. SE-ResNet — Rayleigh

```powershell
python train_se_resnet.py `
  --splits Data/splits --channel rayleigh --profile repo `
  --latent-channels 8 --blocks 0 `
  --epochs 20 --batch-size 64 --accum-steps 1 --lr 0.001 `
  --snr-min 0 --snr-max 20 --snr-sampling uniform `
  --snr-grid 0,5,10,15,20 --val-snrs 0,5,10,15,20 `
  --seed 42 --num-workers 0 --threads 2 `
  --out runs/se_resnet_deepscs/rayleigh_repo8_b64
```

Mỗi run lưu `config.json`, `history.csv` và `best_model.pth`. Best checkpoint được chọn theo DEV MSE; TEST không được dùng trong huấn luyện. Với Rayleigh, mô hình giả định ghép latent thành cặp I/Q, block flat fading, CSI hoàn hảo và cân bằng MMSE.

## Chọn threshold energy-mask trước khi đánh giá

Phải chạy bước này trước bốn lệnh đánh giá. Script lấy mẫu log-Mel từ TRAIN để tạo các ngưỡng ứng viên, rồi kiểm tra tỷ lệ ô dương trên DEV. Script không đọc TEST và không tối ưu F1 trên TEST.

```powershell
python estimate_energy_mask_threshold.py `
  --splits Data/splits `
  --cache prepared `
  --out reports/energy_mask_threshold_v1 `
  --samples-per-utterance 128 --seed 42 `
  --percentiles 65,70,75,80,85,90 --choose-percentile 75
```

Mở `reports/energy_mask_threshold_v1/selected_threshold.json` và dùng giá trị `energy_mask_threshold_raw_logmel` giống nhau cho cả bốn lần đánh giá. Ngưỡng được chọn mặc định là percentile 75 của mẫu TRAIN; đây là ngưỡng ứng viên cho energy-mask, không phải threshold được tối ưu F1. Để nạp giá trị trực tiếp trong PowerShell:

```powershell
$maskThreshold = (Get-Content reports/energy_mask_threshold_v1/selected_threshold.json -Raw | ConvertFrom-Json).energy_mask_threshold_raw_logmel
```

Nếu thư mục output đã có kết quả, chọn một đường dẫn `--out` mới rồi đọc threshold từ `selected_threshold.json` trong đường dẫn đó. Không chọn threshold dựa trên kết quả TEST.

## Đánh giá bốn checkpoint

Các tham số khớp với `evaluation_config.json` của bốn báo cáo: `mode=both`, toàn bộ 2.620 utterance TEST (`--max-utterances 0`), một lần lặp, các SNR 0–20 dB, Power-Mel Joint 64 iteration, float32, batch 16, seed 42. Mỗi đánh giá gồm đối chứng `no_noise` và kênh huấn luyện tương ứng. Tất cả lệnh đều truyền cùng `$maskThreshold` đã chọn ở bước trước.

Mỗi thư mục `--out` phải mới hoặc rỗng vì evaluator không ghi đè báo cáo cũ.

Lưu ý về các cấu hình CNN hiện có: `evaluation_config.json` trong các report cũ ghi đường dẫn checkpoint cũ `runs/multi_snr/...`, trong khi các checkpoint hiện tại nằm dưới `runs/cnn_baseline/...`. Các lệnh dưới đây dùng checkpoint hiện có trong `runs/`; khi chạy lại, evaluator sẽ ghi đường dẫn thực tế vào `evaluation_config.json` mới. Các tham số đánh giá còn lại lấy theo cấu hình report đã lưu.

### 1. CNN Baseline — AWGN

```powershell
python evaluate.py `
  --mode both --splits Data/splits `
  --checkpoint cnn=runs/cnn_baseline/awgn/best_model.pth `
  --architecture cnn=cnn --channels no_noise,awgn --snrs 0,5,10,15,20 `
  --repeats 1 --max-utterances 0 --iterations 64 `
  --joint-device cuda --precision float32 `
  --lambda-mel 1 --rho 1 --inner-steps 5 --init-iterations 40 --init multiplicative `
  --batch-size 16 --seed 42 --energy-mask-threshold $maskThreshold `
  --out reports/cnn_baseline_awgn
```

### 2. CNN Baseline — Rayleigh

```powershell
python evaluate.py `
  --mode both --splits Data/splits `
  --checkpoint cnn=runs/cnn_baseline/rayleigh/best_model.pth `
  --architecture cnn=cnn --channels no_noise,rayleigh --snrs 0,5,10,15,20 `
  --repeats 1 --max-utterances 0 --iterations 64 `
  --joint-device cuda --precision float32 `
  --lambda-mel 1 --rho 1 --inner-steps 5 --init-iterations 40 --init multiplicative `
  --batch-size 16 --seed 42 --energy-mask-threshold $maskThreshold `
  --out reports/cnn_baseline_rayleigh
```

### 3. SE-ResNet — AWGN

```powershell
python evaluate.py `
  --mode both --splits Data/splits `
  --checkpoint se=runs/se_resnet_deepscs/awgn_repo8_b64/best_model.pth `
  --architecture se=se_resnet --channels no_noise,awgn --snrs 0,5,10,15,20 `
  --repeats 1 --max-utterances 0 --iterations 64 `
  --joint-device cuda --precision float32 `
  --lambda-mel 1 --rho 1 --inner-steps 5 --init-iterations 40 --init multiplicative `
  --batch-size 16 --seed 42 --energy-mask-threshold $maskThreshold `
  --out reports/se_resnet_awgn
```

### 4. SE-ResNet — Rayleigh

```powershell
python evaluate.py `
  --mode both --splits Data/splits `
  --checkpoint se=runs/se_resnet_deepscs/rayleigh_repo8_b64/best_model.pth `
  --architecture se=se_resnet --channels no_noise,rayleigh --snrs 0,5,10,15,20 `
  --repeats 1 --max-utterances 0 --iterations 64 `
  --joint-device cuda --precision float32 `
  --lambda-mel 1 --rho 1 --inner-steps 5 --init-iterations 40 --init multiplicative `
  --batch-size 16 --seed 42 --energy-mask-threshold $maskThreshold `
  --out reports/se_resnet_rayleigh
```

Mỗi thư mục báo cáo lưu `evaluation_config.json`, danh sách utterance TEST, CSV chi tiết và CSV tổng hợp. Kết quả gồm normalized/raw log-Mel MSE, energy-mask Precision/Recall/F1, waveform MSE, STOI và PESQ-WB. Đây là đánh giá tái tạo tiếng nói; repo không báo cáo WER/CER/ASR.

> Các run/report có sẵn trong workspace sẽ làm lệnh tương ứng báo lỗi nếu output không rỗng. Hãy giữ lại kết quả cũ và dùng một thư mục output mới khi chạy lại; để bộ chart đọc lần đánh giá mới, cập nhật tên thư mục report trong `MODELS` ở `chart/bieu_do_chuong_4.py` cho khớp.

## Vẽ biểu đồ Chương 4

Các wrapper biểu đồ đều gọi logic chung ở `chart/bieu_do_chuong_4.py`. Chạy từ thư mục gốc dự án sau khi đã có bốn `history.csv` và bốn thư mục report chuẩn ở trên. Dùng `--root .` để chỉ rõ thư mục dự án và `--out` để gom kết quả xuất vào một nơi.

### Hình 4.5 và 4.6 — dữ liệu TRAIN/DEV

```powershell
python chart/ve_bieu_do_4.5_va_4.6/ve_bieu_do_4_5_4_6.py `
  --root . --out reports/charts/chapter4/train_dev
```

- Hình 4.5: đường loss TRAIN/DEV của bốn cấu hình, đánh dấu epoch được chọn theo DEV.
- Hình 4.6: DEV MSE theo SNR tại epoch được chọn.

### Hình 4.7–4.14 — kết quả TEST

```powershell
python chart/ve_bieu_do_4.7/ve_bieu_do_4_7.py --root . --out reports/charts/chapter4/test
python chart/ve_bieu_do_4.8/ve_bieu_do_4_8.py --root . --out reports/charts/chapter4/test
python chart/ve_bieu_do_4.9/ve_bieu_do_4_9.py --root . --out reports/charts/chapter4/test
python chart/ve_bieu_do_4.10/ve_bieu_do_4_10.py --root . --out reports/charts/chapter4/test
python chart/ve_bieu_do_4.11/ve_bieu_do_4_11.py --root . --out reports/charts/chapter4/test
python chart/ve_bieu_do_4.12/ve_bieu_do_4_12.py --root . --out reports/charts/chapter4/test
python chart/ve_bieu_do_4.13/ve_bieu_do_4_13.py --root . --out reports/charts/chapter4/test
python chart/ve_bieu_do_4.14/ve_bieu_do_4_14.py --root . --out reports/charts/chapter4/test
```

Các hình lần lượt là: normalized log-Mel MSE, raw log-Mel MSE, energy-mask F1, đối chứng no-noise, PESQ-WB, STOI, waveform MSE và biểu đồ tổng hợp đối sánh. Bộ vẽ kiểm tra checkpoint, history, cấu hình đánh giá và độ phủ SNR trước khi xuất; thiếu/sai đầu vào sẽ báo lỗi thay vì tự điền dữ liệu. Mỗi hình được lưu thành PNG, PDF và CSV giá trị.

## Lưu ý về kết quả

- Ngưỡng energy-mask phải được xác định trước từ TRAIN/DEV và giữ cố định cho cả bốn mô hình.
- No-noise là đối chứng riêng, không gán SNR.
- SNR của Rayleigh là SNR danh định trước fading; SNR tức thời thay đổi theo fading.
- Hình 4.5–4.6 chỉ dùng TRAIN/DEV để theo dõi huấn luyện và chọn epoch. Các kết luận đánh giá cuối cùng dùng TEST.
- Repo hiện chưa kèm tệp LICENSE. Dataset âm thanh không được đóng gói; hãy tuân theo điều khoản của bộ dữ liệu nguồn.
