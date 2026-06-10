# ⚙️ Hướng dẫn Cài đặt & Thiết lập Môi trường (SETUP.md)

Tài liệu này hướng dẫn chi tiết cách thiết lập môi trường ảo, cài đặt các thư viện lõi độc quyền (như thư viện tính toán GFLOPS, quản lý tài nguyên hệ thống) và cấu trúc thư mục chuẩn để chạy chuỗi thực nghiệm Mixture of Experts (MoE) cho bài toán Suy luận Ngôn ngữ Tự nhiên tiếng Việt (Vietnamese NLI).

---

## 1. Yêu cầu Hệ thống & Phần cứng (System Requirements)
Do dự án kiểm thử các mô hình ngôn ngữ lớn (Pre-trained Language Models - PLMs) phiên bản **Large** kết hợp với cấu trúc đa chuyên gia (đa nhánh mạng), yêu cầu phần cứng được khuyến nghị như sau:
* **Hệ điều hành:** Linux (Ubuntu 20.04/22.04 LTS) hoặc Windows 10/11 (Khuyến nghị sử dụng WSL2 để tối ưu hóa hiệu năng tính toán của PyTorch).
* **Phần cứng:** * **GPU:** NVIDIA GPU hỗ trợ CUDA (Khuyến nghị dòng RTX 3090, RTX 4090, A100 hoặc ít nhất là RTX 3060/4060 Ti phiên bản 16GB VRAM). 
  * **VRAM:** Tối thiểu 16GB cho `PhoBERT-large` (8 Experts). Tối thiểu 24GB cho `XLM-RoBERTa-large` (32 Experts).
  * *Ghi chú:* Đối với các máy cấu hình thấp hơn, hệ thống đã được tích hợp cơ chế **Gradient Accumulation** (Tích lũy gradient qua micro-batch) tại file `XLM.ipynb` để tránh lỗi tràn bộ nhớ đồ họa (**Out-Of-Memory - OOM**).
* **Python:** Phiên bản **3.9** hoặc **3.10** (Đảm bảo tính tương thích tốt nhất giữa mã nguồn `transformers` và thư viện tính toán toán học `thop`).

---

## 2. Khởi tạo Môi trường ảo (Virtual Environment Setup)
Để tránh xung đột phiên bản của các thư viện hệ thống, tất cả thành viên trong nhóm nghiên cứu bắt buộc phải khởi tạo môi trường ảo độc lập theo các lệnh dưới đây:
```text

# 1. Di chuyển vào thư mục gốc chứa mã nguồn của dự án
cd ".\\MoE_Vietnamese_NLI"

# 2. Khởi tạo thư mục môi trường ảo .venv
python -m venv .venv

# 3. Kích hoạt môi trường ảo (Dành cho Windows Powershell)
.\\.venv\\Scripts\\Activate.ps1

# 3b. Kích hoạt môi trường ảo (Dành cho Linux / MacOS hoặc Git Bash)
source .venv/bin/activate

# 4. Nâng cấp bộ quản lý gói pip lên phiên bản mới nhất để tránh lỗi wheel
python -m pip install --upgrade pip

```

---

## 3. Cài đặt các Thư viện Tiền đề (Dependencies Installation)

### Bước 3.1: Cài đặt PyTorch Deep Learning Framework (Có hỗ trợ CUDA)

Truy cập trang chủ PyTorch hoặc chạy lệnh chuẩn dưới đây để cài đặt phiên bản PyTorch hỗ trợ tăng tốc phần cứng qua GPU (Thay thế `cu118` bằng phiên bản CUDA Toolkit tương ứng trên máy của bạn, ví dụ `cu121` hoặc `cu117`):


pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)



### Bước 3.2: Cài đặt các thư viện Xử lý Ngôn ngữ & Quản lý Thí nghiệm

Cài đặt tổ hợp gói phân tích dữ liệu, xử lý mô hình Transformer, và giao diện lập trình Jupyter Lab:

pip install transformers datasets pandas numpy scikit-learn jupyterlab tqdm accelerate sentencepiece



### Bước 3.3: Cài đặt thư viện Đo lường & Giám sát Phần cứng (Bắt buộc)

Để trích xuất được các thông số hệ thống phục vụ cho việc viết bài báo khoa học (Paper) và làm báo cáo, thành viên cần cài đặt chính xác 2 thư viện sau:

* `thop` (PyTorch-OpCounter): Dùng để tính toán toán học số phép toán dấu phẩy động (**GFLOPS**) của mô hình MoE khi thay đổi các cơ chế Routing.
* `psutil`: Dùng để đo đạc và ghi nhận lượng RAM/CPU tiêu thụ thực tế của hệ thống.


pip install thop psutil



---

## 4. Cấu trúc Thư mục & Định dạng Dữ liệu Đầu vào

Để các file Jupyter Notebook chạy tuần tự mà không bị lỗi đường dẫn (`FileNotFoundError`), toàn bộ dữ liệu và tệp mã nguồn phải được sắp xếp nghiêm ngặt theo cấu trúc cây thư mục dưới đây:

```text
MoE_Vietnamese_NLI/
├── train.csv                # Bộ dữ liệu Huấn luyện (80%)
├── validation.csv           # Bộ dữ liệu Kiểm định (10%)
├── test.csv                 # Bộ dữ liệu Đánh giá độc lập (10%)
├── README.md                # Tài liệu mặt tiền của repo
├── TONG_QUAN.md             # Tài liệu mô tả kỹ thuật chi tiết
├── SETUP.md                 # Chính là tài liệu này
└── Code/
    ├── EDA.ipynb            # Phân tích khám phá dữ liệu ban đầu
    └── experiment/
        ├── routing_adaptive.py      # Module mạng Adaptive Dynamic Routing
        ├── routing_deepseek.py      # Module mạng DeepSeek Routing
        ├── routing_expert_choice.py # Module mạng Expert Choice Routing
        ├── routing_micro.py         # Module mạng MICRO Routing
        ├── routing_smoe.py          # Module mạng Soft MoE Routing
        ├── PhoBERT.ipynb            # Pipeline thử nghiệm PhoBERT Base
        ├── PhoBERT_large.ipynb      # Pipeline nâng cao cho PhoBERT Large (8 Experts)
        └── XLM.ipynb                # Pipeline tối ưu hóa cho XLM-R Large (32 Experts)

```

### 📋 Quy chuẩn dữ liệu đầu vào (Data Constraints):

Bộ dữ liệu NLI tiếng Việt được chuẩn hóa và chia tách theo **Tỷ lệ Phân bổ nghiêm ngặt là 9:1:1** (90% Train, 10% Validation, 10% Test).
Mỗi tệp `.csv` bắt buộc phải chứa cấu trúc bảng với các tiêu đề cột (headers) sau:

1. `premise`: Chuỗi văn bản chứa câu tiền đề.
2. `hypothesis`: Chuỗi văn bản chứa câu giả thuyết cần suy luận.
3. `label`: Nhãn kết quả suy luận logic, chỉ nhận 1 trong 3 giá trị chuỗi văn bản: `entailment` (Kéo theo), `neutral` (Trung tính), hoặc `contradiction` (Mâu thuẫn).

---

## 5. Hướng dẫn Thực thi Luồng Huấn luyện (Pipeline Execution)

1. Đảm bảo bạn đã đặt đúng 3 file dữ liệu (`train.csv`, `validation.csv`, `test.csv`) vào thư mục gốc dự án.
2. Khởi chạy giao diện quản lý thí nghiệm:
jupyter lab




3. Truy cập vào thư mục `Code/experiment/`, mở tệp `PhoBERT_large.ipynb` hoặc `XLM.ipynb`.
4. Tìm đến **Cell số 2 (Class Config)** và tiến hành kiểm tra/sửa đổi các đường dẫn biến cố định sau sao cho trỏ chính xác về thư mục thực tế trên máy của bạn:
* `Config.TRAIN_CSV`
* `Config.VAL_CSV`
* `Config.TEST_CSV`
* `Config.CHECKPOINT_DIR` (Nơi hệ thống sẽ tự sinh ra để lưu trữ các file trọng số `.pth`)
* `Config.RESULTS_CSV` (Nơi hệ thống tự động append kết quả đánh giá chi tiết của từng Epoch)


5. Chọn `Run -> Run All Cells` để kích hoạt luồng tự động chạy đa Seed và đa Routing.

---

## 6. Kết quả ghi nhận sau khi Setup thành công

Sau khi kết thúc quá trình huấn luyện tự động, tại đường dẫn bạn khai báo ở biến `Config.RESULTS_CSV`, hệ thống sẽ tự động xuất ra bảng log CSV chứa đầy đủ các thông số thực nghiệm để phục vụ phân tích:

* **Val_Acc & Val_F1**: Chỉ số độ chính xác và Macro F1-Score để đánh giá chất lượng hội tụ của mô hình.
* **GFlops**: Độ nặng chi phí tính toán lý thuyết được trích xuất từ lớp `thop.profile`.
* **Runtime_ms**: Tốc độ xử lý thực tế tính trên từng batch dữ liệu (`ms/b`).
* **VRAM_MB**: Đỉnh bộ nhớ đồ họa tiêu thụ tối đa của thuật toán định tuyến đó.
* **Expert_Usage**: Mảng phân phối xác suất sử dụng của các chuyên gia (Dùng để import vào tập script tiếp theo để vẽ biểu đồ Pareto).
"""

