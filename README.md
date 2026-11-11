# Semantic Voice — V0.1 (AWGN & Rayleigh)

Mã nguồn cho hệ thống truyền tiếng nói dựa trên biểu diễn log-Mel và mạng neural. Encoder nén đặc trưng tiếng nói thành latent, kênh AWGN hoặc Rayleigh làm nhiễu/suy hao latent, rồi Decoder khôi phục log-Mel. Khi đánh giá dạng sóng, log-Mel khôi phục được đưa qua bộ tổng hợp Power-Mel Joint.

Phiên bản này giữ kiến trúc CNN Baseline V0.1 và bổ sung SE-ResNet thích nghi cho đầu vào log-Mel. Mô hình được huấn luyện từ đầu; checkpoint COALA cũ không tương thích.

## Tính năng

- Đọc âm thanh WAV/FLAC, chuyển về mono 16 kHz và tạo log-Mel 96 dải.
- Chia utterance thành các patch 96×96; chuẩn hóa theo từng dải Mel với thống kê chỉ tính từ train.
- Kênh AWGN và Rayleigh phẳng quasi-static. Rayleigh ghép các latent channel thành cặp I/Q và giả định CSI hoàn hảo cùng bộ cân bằng MMSE.
- Huấn luyện CNN Baseline hoặc SE-ResNet với SNR cố định hoặc lấy mẫu theo batch.
- Đánh giá MSE trên log-Mel và, tùy chọn, MSE dạng sóng, STOI, PESQ-WB.

> Mô hình Rayleigh hiện tại dùng giả định CSI hoàn hảo; đây là mô phỏng kênh thuận lợi, không phải ước lượng kênh thực tế. Các chỉ số WER/CER/ASR không thuộc quy trình đánh giá chính thức của repo.

## Cấu trúc repo

| Đường dẫn | Nội dung |
|---|---|
| `channels/` | Mô hình AWGN, Rayleigh và factory tạo kênh |
| `mel_frontend/` | Đọc âm thanh, STFT, Mel filterbank và log-Mel |
| `mel_inverse/` | Khôi phục/tổng hợp dạng sóng từ Mel |
| `model/CNN_Baseline/` | Encoder và Decoder CNN cơ sở |
| `model/SE-ResNet/` | Encoder và Decoder SE-ResNet |
| `training/` | Vòng lặp huấn luyện và lấy mẫu SNR |
| `evaluation/` | Chỉ số và tiện ích báo cáo |
| `prepare_dataset.py` | Tạo cache log-Mel và các tập train/dev/test |
| `train_CNN_baseline.py` | Huấn luyện CNN Baseline |
| `train_se_resnet.py` | Huấn luyện SE-ResNet |
| `evaluate.py` | Đánh giá checkpoint trên test set |

Các thư mục `Data/splits/`, `prepared/`, `runs/`, `reports/` và tệp âm thanh/checkpoint bị loại khỏi Git theo `.gitignore`. Cần tự chuẩn bị dataset và sinh lại dữ liệu/kết quả cục bộ.

## Yêu cầu

- Python 3.10 trở lên.
- PyTorch 2.2 trở lên; cài đúng bản CPU hoặc CUDA phù hợp với máy từ trang cài đặt PyTorch.
- Các thư viện Python còn lại được liệt kê trong `requirements.txt`.
- Dataset chứa tệp `.wav` hoặc `.flac`. Có thể dùng LibriSpeech; script nhận diện các thư mục split chuẩn như `train-clean-100`, `dev-clean`, `test-clean`.

## Cài đặt

Ví dụ trong PowerShell từ thư mục gốc repo:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu cần GPU NVIDIA, cài PyTorch theo lệnh được PyTorch selector đề xuất cho phiên bản CUDA của máy, sau đó cài các gói còn lại từ `requirements.txt`.

## Chuẩn bị dữ liệu

Trỏ `--dataset` tới thư mục gốc chứa audio. Ví dụ:

```powershell
python prepare_dataset.py `
  --dataset "D:\Datasets\LibriSpeech" `
  --cache prepared `
  --out Data/splits `
  --split-unit speaker `
  --seed 42
```

Lệnh tạo cache log-Mel trong `prepared/` và các tệp `train.npy`, `val.npy`, `test.npy` cùng manifest/normalization trong `Data/splits/`. Nếu dataset có đủ split LibriSpeech chuẩn, chương trình sử dụng các split chính thức; nếu không, chương trình chia theo speaker mặc định để tránh speaker leakage. Cần ít nhất ba nhóm speaker độc lập. Tham số `--max-files` chỉ phù hợp chạy thử, không dùng cho kết quả báo cáo.

Đường dẫn đầu vào và các thiết lập được ghi trong manifest. Nếu muốn chạy với dataset hoặc cấu hình chia tập khác, hãy chọn thư mục `--cache` và `--out` mới để tránh trộn dữ liệu của các lần chạy.

## Huấn luyện

CNN Baseline với AWGN, huấn luyện SNR cố định 10 dB:

```powershell
python train_CNN_baseline.py --splits Data/splits --channel awgn --snr 10 --run runs/cnn_baseline/awgn
```

CNN Baseline với Rayleigh:

```powershell
python train_CNN_baseline.py --splits Data/splits --channel rayleigh --snr 10 --run runs/cnn_baseline/rayleigh
```

Để lấy mẫu SNR mỗi batch trong khoảng 0–20 dB, thêm `--train-strategy multi`. Script SE-ResNet dùng chiến lược này mặc định:

```powershell
python train_se_resnet.py --splits Data/splits --channel awgn --profile repo --out runs/se_resnet/awgn
```

Đổi `--channel awgn` thành `rayleigh` để huấn luyện trên Rayleigh. Có thể điều chỉnh số epoch, batch size, dải/grid SNR, seed và thiết bị qua các tham số CLI. Chạy `python train_CNN_baseline.py --help` hoặc `python train_se_resnet.py --help` để xem toàn bộ tùy chọn.

Checkpoint tốt nhất và cấu hình/lịch sử huấn luyện được ghi dưới thư mục output của từng run. Mỗi thư mục output cần mới hoặc rỗng; dùng tùy chọn resume của SE-ResNet để tiếp tục run đã lưu.

## Đánh giá

Ví dụ đánh giá checkpoint CNN trên AWGN, tại các SNR 0, 5, 10, 15 và 20 dB:

```powershell
python evaluate.py `
  --mode both `
  --splits Data/splits `
  --checkpoint cnn=runs/cnn_baseline/awgn/best_model.pth `
  --channels awgn `
  --snrs 0,5,10,15,20 `
  --out reports/cnn_awgn
```

`--mode mel` chỉ tính chỉ số log-Mel, `--mode wave` chỉ đánh giá dạng sóng, và `--mode both` chạy cả hai. Đánh giá mặc định chọn tối đa 30 utterance test; đặt `--max-utterances 0` để dùng toàn bộ test set. Có thể thêm nhiều `--checkpoint NAME=PATH`, chỉ định kiến trúc bằng `--architecture NAME=cnn|se_resnet`, hoặc dùng `--channels awgn,rayleigh`.

Kết quả được ghi thành CSV chi tiết/tổng hợp cùng cấu hình và danh sách utterance đã chọn trong thư mục `--out`. Đánh giá dạng sóng dùng Power-Mel Joint (mặc định 64 vòng lặp) và báo cáo waveform MSE, STOI, PESQ-WB; vì đây là bước tối ưu số, thời gian chạy có thể đáng kể.

Một số tiện ích khác:

```powershell
python dataset_statistics.py --splits Data/splits --out reports/dataset_stats
python estimate_energy_mask_threshold.py --splits Data/splits --out reports/energy_mask_threshold_v1
```

Ngưỡng energy-mask và Precision/Recall/F1 là phân tích phụ trên mặt nạ năng lượng time-frequency, không phải chỉ số nhận dạng tiếng nói.

## Tiền xử lý âm thanh

Cấu hình frontend mặc định:

- Tần số lấy mẫu: 16.000 Hz, mono.
- STFT: `n_fft=1024`, `hop_length=256`.
- Mel: 96 dải từ 0 Hz đến Nyquist.
- Đầu vào mạng: `[B, 1, 96, 96]`.
- Chuẩn hóa: z-score riêng mỗi dải Mel, thống kê ước lượng từ train.

Không đưa cùng speaker vào nhiều split nếu cần đánh giá tổng quát hóa theo speaker. Hãy giữ lại manifest, cấu hình run và cấu hình đánh giá khi lưu trữ kết quả để đảm bảo có thể truy vết thí nghiệm.

## Trích dẫn và giấy phép

Repo chưa kèm tệp LICENSE. Hãy xác nhận giấy phép với chủ sở hữu trước khi phân phối lại hoặc dùng cho mục đích thương mại. Nếu sử dụng LibriSpeech, tải và tuân theo điều khoản của bộ dữ liệu từ nguồn phát hành chính thức; dữ liệu âm thanh không được đóng gói trong repo này.
