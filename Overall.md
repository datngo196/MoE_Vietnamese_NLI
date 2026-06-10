### 3. File `TONG_QUAN.md`


# 📊 Tổng quan Kiến trúc & Chức năng Mã nguồn

Tài liệu này giải thích chi tiết vai trò của từng thành phần mã nguồn, giúp các thành viên trong nhóm hiểu rõ luồng xử lý (pipeline) của mô hình và ý nghĩa của các kết quả trả về.

---

## 1. Các Module Định tuyến (Routing Scripts)
Đây là các file chứa kiến trúc mạng Neural lõi của lớp Mixture of Experts (MoE). Chúng được thiết kế dạng module độc lập (plug-and-play), nhận đầu vào là các `hidden_states` từ Transformer và trả về đặc trưng đã được tính toán qua các chuyên gia.

* **`routing_adaptive.py` (Adaptive Dynamic MoE):** Không ép buộc số lượng token cho mỗi chuyên gia. Thay vào đó, nó tính toán điểm qua hàm Sigmoid và dùng một `threshold` (ngưỡng). Chuyên gia nào có điểm cao hơn ngưỡng sẽ được kích hoạt. Đi kèm là cơ chế `dynamic_bias` tự động cập nhật trong lúc train để chống lại hiện tượng mất cân bằng tải (load imbalance).
* **`routing_deepseek.py` (DeepSeek MoE):** Tách bộ chuyên gia làm 2 nhóm: *Shared Experts* (chuyên gia dùng chung, luôn xử lý mọi token để lấy bối cảnh) và *Routed Experts* (chỉ kích hoạt Top-K để xử lý các đặc trưng cụ thể).
* **`routing_expert_choice.py` (Expert Choice MoE):** Đảo ngược tư duy thông thường. Thay vì Token đi tìm Expert, thì Expert sẽ chủ động chọn ra các Token phù hợp nhất với nó dựa trên chỉ số `capacity_factor` (sức chứa). Đảm bảo không chuyên gia nào bị "chết" (dead experts) hoặc quá tải.
* **`routing_micro.py` (MICRO MoE):** Sử dụng bộ phân loại nhận thức (Cognitive Classifier) thay vì Router tuyến tính thông thường, kết hợp với định tuyến mềm (Soft routing) để tối ưu hóa luồng lan truyền ngược (gradient flow).
* **`routing_smoe.py` (Soft MoE):** Sử dụng ma trận Attention để phân phối trọng số của toàn bộ chuyên gia cho từng token, làm mượt hóa quá trình quyết định.

---

## 2. Các Jupyter Notebooks (Training Pipelines)
Các file `.ipynb` là nơi lắp ráp PLM (PhoBERT/XLM) với các mô-đun Routing ở trên để tạo thành mạng `UnifiedMoENLI`, sau đó tiến hành huấn luyện.

* **`EDA.ipynb`:** Dùng để phân tích độ dài câu, tỷ lệ nhãn, giúp quyết định chỉ số `MAX_LEN` chuẩn xác nhất cho bộ dữ liệu (hiện đang set là 256).
* **`PhoBERT_large.ipynb`:** Pipeline thực nghiệm hoàn chỉnh trên kiến trúc PhoBERT-large (8 Experts). Đã tích hợp vòng lặp tự động chạy qua 3 Seeds x 5 Routing, cơ chế `CheckpointManager` tự lưu file weight tốt nhất và tự động nối (append) kết quả vào file CSV.
* **`XLM.ipynb`:** Pipeline dành riêng cho mô hình đa ngữ XLM-RoBERTa-large (scale lên 32 Experts). Tích hợp cơ chế **Gradient Accumulation** ở cấp độ micro-batch để giải quyết bài toán thiếu VRAM khi chạy mô hình khổng lồ. Có hệ thống Resume tự động khôi phục epoch nếu máy bị sập nguồn giữa chừng.

---

## 3. Hiểu các kết quả trả về (Interpreting Results)
Sau khi huấn luyện, hệ thống sẽ sinh ra file `results_...csv`. Đây là các thông số chính nhóm cần quan tâm để phân tích trong báo cáo:

* **Val_Acc & Val_F1:** Độ chính xác trên tập Validation. Trọng tâm đánh giá là **Macro F1-Score** để đảm bảo mô hình không bị thiên lệch nhãn. Điểm F1 cũng là cơ sở để kích hoạt Dừng sớm (Early Stopping).
* **GFlops:** Đo lường số phép toán dấu phẩy động. Chỉ số này đại diện cho "độ nặng toán học" của từng cơ chế định tuyến. GFlops càng thấp mô hình càng nhẹ.
* **Runtime_ms:** Tốc độ suy luận thực tế (Inference speed) tính bằng mili-giây trên mỗi batch.
* **VRAM_MB:** Bộ nhớ GPU đỉnh đã sử dụng. GFlops có thể thấp nhưng nếu cơ chế code không tốt, VRAM vẫn có thể tràn.
* **Entropy & Expert_Usage:** `Entropy` đo lường mức độ phân tán dữ liệu qua các chuyên gia (càng cao càng cân bằng). `Expert_Usage` xuất ra một mảng tần suất (VD: `[0.2, 0.1, 0.5,...]`). **Mục đích:** Dùng mảng này để vẽ biểu đồ Pareto (Quy luật 80/20) đánh giá xem có chuyên gia nào đang phải "gánh" toàn bộ hệ thống hay không.
