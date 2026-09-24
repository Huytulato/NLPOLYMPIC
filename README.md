# NLPOLYMPIC: Dịch máy Trung → Việt

## 1. Bối cảnh thực tế & Mục tiêu

Trong thời đại toàn cầu hóa, việc giao lưu kinh tế và văn hóa giữa Việt Nam và Trung Quốc diễn ra mạnh mẽ. Dịch máy (Machine Translation - MT) là công cụ then chốt để xử lý hàng triệu tài liệu và hội thoại nhanh chóng.

Trong đề thi này, các đội thi sẽ:

- Huấn luyện mô hình dịch từ tiếng Trung giản thể (中文) sang tiếng Việt.
- Đánh giá kết quả bằng thước đo chuẩn quốc tế: **SacreBLEU**.

## 2. Nhiệm vụ

Xây dựng một mô hình dịch máy tự động có khả năng dịch câu tiếng Trung giản thể sang tiếng Việt chính xác và tự nhiên.

Các đội thi có thể lựa chọn các hướng triển khai như: Rule-based, Statistical Machine Translation (SMT), hoặc Neural Machine Translation (NMT - Transformer/Seq2Seq).

> **Lưu ý:** Không được sử dụng các mô hình dịch Hoa-Việt đã được huấn luyện sẵn (pretrained models).

## 3. Cấu trúc dữ liệu

Bộ dữ liệu được chia thành ba phần, mỗi tệp chỉ chứa văn bản thuần (một câu trên một dòng), mã hóa UTF-8:

| Tệp dữ liệu | Nội dung | Ngôn ngữ |
|---|---|---|
| `train.zh` / `train.vi` | Tập huấn luyện song ngữ | Trung / Việt |
| `public_test.zh` | Tập kiểm tra công khai | Tiếng Trung |
| `private_test.zh` | Tập kiểm tra bí mật | Tiếng Trung |

## 4. Hướng dẫn nộp kết quả

Các đội thi cần nộp một file **CSV** gồm 2 cột: `tieng_trung` và `tieng_viet`.

## 5. Tiêu chí đánh giá

Vòng này chỉ đánh giá trên **Private Test**. Điểm số cuối cùng của đội thi được tính theo công thức:

- Nếu `score > min_score`:

  ```
  Final Score = 100 × (score − min_score) / (max_score − min_score)
  ```

- Nếu `score <= min_score`:

  ```
  Final Score = 0
  ```

Trong đó:

- `score`: chỉ số SacreBLEU của đội thi trên tập Private Test.
- `max_score`: chỉ số SacreBLEU lớn nhất của các đội thi trên hệ thống.
- `min_score`: chỉ số SacreBLEU của mô hình baseline từ Ban tổ chức.

### Thước đo SacreBLEU

SacreBLEU đo lường độ tương đồng giữa bản dịch máy và bản dịch tham chiếu:

```
SacreBLEU = BP · exp( Σ_{n=1..N} w_n · log p_n )
```

Giải thích thông số:

- `p_n`: độ chính xác n-gram bậc `n`.
- `w_n`: trọng số (thường là 1/4 cho n = 1, 2, 3, 4).
- `BP`: hệ số phạt độ ngắn (Brevity Penalty), tính bằng:

  ```
  BP = 1               nếu c > r
  BP = exp(1 − r/c)    nếu c <= r
  ```

  *(Trong đó `c` là độ dài bản dịch máy, `r` là độ dài bản dịch tham chiếu.)*

---

## Phương pháp (notebook `transformer_nmt.ipynb`)

Hướng làm là **Neural Machine Translation với Transformer encoder–decoder, train từ trọng số ngẫu nhiên**. Không dùng mô hình pretrained nào, đúng lưu ý của đề.

### Tổng quan pipeline

```
train.zh/vi ─► chia train/valid ─► SentencePiece ─► Transformer ─► top-K checkpoint
                                                                        │
private_test.zh ─► encode ─► beam search (model trung bình / ensemble) ─► hậu xử lý ─► CSV
```

### Vì sao chọn Transformer

Baseline của BTC là GRU Seq2Seq **không có attention**: encoder nén cả câu vào một vector 128 chiều, decoder chỉ nhìn vector đó. Cách này mất thông tin với câu dài, và output bị hỏng (ví dụ `Tôi là___________________ .` trong `public_submission.csv`). Transformer có **attention**: mỗi từ sinh ra được "nhìn" lại toàn bộ câu nguồn. Đây là kiến trúc chuẩn hiện nay cho dịch máy. SMT/rule-based cũng được phép, nhưng với 32k cặp câu thì NMT nhỏ có regularization tốt thường cho BLEU cao hơn.

### 1. Chuẩn bị dữ liệu

| Việc làm | Lý do |
|---|---|
| Chia **5% valid ngẫu nhiên** (seed cố định) | Baseline lấy 10% cuối file nên tập valid không đại diện. Valid ngẫu nhiên phản ánh đúng hơn điểm trên private test. |
| Bỏ cặp câu rỗng hoặc độ dài lệch > 3 lần | Thường là cặp dịch sai, làm nhiễu model. Chỉ bỏ khoảng 0,1% dữ liệu. |
| Đọc **mọi dòng** của file test, không lọc | File nộp phải đúng số dòng (1.781) và đúng thứ tự. |

### 2. Tokenizer: SentencePiece unigram

- Train riêng cho tiếng Trung và tiếng Việt, vocab khoảng **8.000** mỗi bên, chỉ học trên tập train.
- Tách thành **subword** để xử lý được từ hiếm và từ chưa gặp: từ lạ được ghép từ các mảnh nhỏ hơn thay vì thành `<unk>`.
- **Giữ nguyên định dạng tách từ `_` của tiếng Việt** (`thay_đổi`, `có_thể`). Bản dịch tham chiếu nhiều khả năng cùng định dạng với `train.vi`. Nếu model viết `thay đổi` trong khi tham chiếu là `thay_đổi`, n-gram không khớp và BLEU giảm mạnh.
- Tiếng Việt dùng chuẩn hóa `identity` để `decode(encode(câu)) == câu`, không làm biến đổi dấu thanh. Notebook kiểm tra điều này trên tập valid.

### 3. Mô hình

| Thông số | Giá trị | Ghi chú |
|---|---|---|
| Kiến trúc | `nn.Transformer` encoder–decoder | 4 lớp encoder + 4 lớp decoder |
| `d_model` / FFN / heads | 256 / 1024 / 4 | Khoảng 10M tham số, vừa với 32k câu |
| Chuẩn hóa | Pre-norm (`norm_first=True`) | Train ổn định hơn post-norm khi model nhỏ, dữ liệu ít |
| Vị trí | Positional encoding sin/cos | |
| Chia sẻ trọng số | Embedding decoder = lớp output | Bớt tham số, đỡ overfit |
| Dropout | 0.3 | Dữ liệu ít nên dropout cao |

Model **to hơn không tốt hơn** với 32k câu: overfit nhanh. Cấu hình trên là mức phổ biến cho các bộ dữ liệu cỡ nhỏ (kiểu IWSLT low-resource).

### 4. Huấn luyện

- **Batch theo số token** (khoảng 2.048 token/batch): gom các câu dài gần bằng nhau để giảm padding.
- **AdamW** (β = 0.9/0.98), lr đỉnh 5e-4, **warmup 1.000 bước** rồi giảm theo `1/√step`. Đây là lịch chuẩn của Transformer, giúp tránh phân kỳ lúc đầu.
- **Label smoothing 0.1**: không ép model tự tin 100% vào một token, giúp tổng quát tốt hơn và tăng BLEU.
- **Gradient clipping 1.0**, **mixed precision (fp16)** trên GPU.
- Sau mỗi epoch đo **SacreBLEU trên valid** (chính thước đo của đề, không phải loss). Dừng sớm nếu BLEU không tăng sau 10 epoch.
- **Tùy chọn R-Drop** (`RDROP_ALPHA`): mỗi batch chạy 2 lần với dropout khác nhau, rồi phạt độ lệch KL giữa 2 dự đoán. Regularization mạnh, hữu ích khi dữ liệu ít.

### 5. Suy luận (decode)

- **Checkpoint averaging:** lấy trung bình trọng số của **5 checkpoint có BLEU valid cao nhất**. Model thu được "mượt" hơn, thường thêm khoảng +0.5–1 BLEU mà không tốn thêm thời gian train.
- **Beam search** (beam 5) thay cho greedy: giữ 5 giả thuyết tốt nhất ở mỗi bước. Điểm cuối chia cho *length penalty* `((5+len)/6)^α` để không thiên vị câu ngắn.
- **Tinh chỉnh `α`** trên valid (0.6 / 1.0 / 1.4). BLEU có Brevity Penalty phạt bản dịch ngắn, nên cần chọn độ dài phù hợp.
- **Tùy chọn ensemble** (`SEEDS = [42, 43, 44]`): train nhiều model với seed khác nhau, rồi lấy trung bình xác suất ở mỗi bước decode.
- Notebook tự so sánh các cấu hình trên valid (greedy / beam / trung bình / ensemble) và **chọn cấu hình có BLEU cao nhất** để dịch test.

### 6. Hậu xử lý và nộp bài

- Dọn `_` thừa (`__` → `_`, bỏ `_` ở đầu/cuối từ) và gộp từ lặp ≥ 3 lần liên tiếp, là hai lỗi hay gặp của NMT.
- Ghi `outputs/private_submission.csv`: 2 cột `tieng_trung`, `tieng_viet`, mã hóa `utf-8-sig` giống file mẫu. Notebook tự kiểm tra số dòng, tên cột và không có ô rỗng, rồi nén thêm file `.zip`.
- Tạo luôn `public_submission.csv` để thử trên public leaderboard (nếu có).

### Cách chạy

1. Mở `transformer_nmt.ipynb` trên **Kaggle hoặc Colab**, bật GPU T4.
2. Thêm thư mục `dataset/`. Notebook tự dò đường dẫn, nếu không thấy thì đặt `DATA_DIR`.
3. Chạy thử với `QUICK_RUN = True` (1–2 phút) để kiểm tra pipeline.
4. Đặt `QUICK_RUN = False` rồi *Run All*. Nộp file `outputs/private_submission.csv`.

### Kết quả chạy kiểm thử (CPU, 6 epoch)

Chạy cấu hình thật (11M tham số, 196 batch/epoch), giới hạn 6 epoch trên CPU chỉ để xác nhận model học được. Bản chạy đầy đủ trên GPU sẽ train tới khi BLEU valid chững lại.

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| BLEU valid (greedy) | 0.4 | 4.0 | 6.8 | 9.2 | 11.4 | 13.7 |

- Beam 5: 14.2. Beam 5 + `lenpen = 1.4`: **14.9**.
- Trung bình top-5 checkpoint cho kết quả **kém hơn** (8.1) trong lần chạy này. Lý do: khi mới train 6 epoch model còn cải thiện nhanh qua từng epoch, nên trung bình với các epoch đầu làm model yếu đi. Khi train đủ lâu, các checkpoint tốt nhất nằm gần nhau và averaging mới có lợi. Notebook tự so sánh trên valid và chọn cấu hình tốt nhất, nên không cần chỉnh tay.
- Ví dụ output: `我 现在 在 机场 。` → `Bây_giờ tôi ở sân_bay .`

### Hướng cải thiện tiếp

- Ensemble 3 seed, bật R-Drop, thử `d_model = 512` nếu BLEU valid vẫn còn tăng.
- Tokenizer dùng chung (joint vocab) cho cả hai ngôn ngữ, hoặc tokenizer mức ký tự cho tiếng Trung.
- Train trên toàn bộ dữ liệu (cả phần valid) với đúng số epoch đã chọn trước khi nộp bản cuối.
