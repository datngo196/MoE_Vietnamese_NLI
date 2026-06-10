# 🚀 Vietnamese Natural Language Inference with Mixture of Experts (MoE)

## 📌 Giới thiệu Dự án
Dự án này tập trung nghiên cứu và đánh giá hiệu năng của kiến trúc **Mixture of Experts (MoE)** khi được tích hợp vào các Mô hình Ngôn ngữ Lớn (PLMs) như **PhoBERT-large** và **XLM-RoBERTa-large**. Bài toán mục tiêu là **Vietnamese Natural Language Inference (NLI)** – Suy luận ngôn ngữ tự nhiên tiếng Việt (phân loại các cặp câu thành Entailment, Neutral, hoặc Contradiction).

Thay vì sử dụng các mô hình Dense khổng lồ tiêu tốn nhiều tài nguyên, dự án áp dụng cơ chế MoE để chỉ kích hoạt một lượng tham số (chuyên gia) nhất định cho mỗi token. Mục tiêu là tìm ra điểm cân bằng tối ưu (Pareto frontier) giữa **Độ chính xác (F1-Score)** và **Chi phí tính toán (GFlops, VRAM)**.

## 🧩 Kiến trúc & Cơ chế Định tuyến (Routing)
Dự án đóng băng (freeze) 12 layer đầu của PLM để trích xuất đặc trưng nền tảng, sau đó thay thế các layer cuối bằng lớp `UnifiedMoENLI` tự xây dựng. Chúng ta so sánh 5 cơ chế định tuyến tiên tiến nhất:

1. **SMoE (Soft MoE):** Định tuyến mềm dựa trên Attention Matrix.
2. **MICRO MoE:** Sử dụng bộ phân loại nhận thức (Cognitive Classifier) để điều hướng.
3. **Expert Choice:** Các chuyên gia tự chọn Token dựa trên sức chứa (Capacity Factor), đảm bảo cân bằng tải tuyệt đối.
4. **Adaptive Dynamic:** Định tuyến động với ngưỡng kích hoạt (threshold) và độ lệch tự thích ứng (dynamic bias).
5. **DeepSeek MoE:** Tách biệt chuyên gia dùng chung (Shared) để học kiến thức nền và chuyên gia định tuyến (Routed) để học chuyên sâu.

## 📊 Kết quả kỳ vọng (Expected Outcomes)
Sau khi chạy toàn bộ pipeline, hệ thống sẽ tự động xuất ra file tổng hợp `results.csv` chứa:
* **Hiệu năng:** Accuracy và Macro F1-Score trên các tập Validation/Test.
* **Tài nguyên:** GFlops (đo lường độ nặng toán học), VRAM (MB) tiêu thụ đỉnh, và Tốc độ suy luận (Runtime_ms/batch).
* **Đặc tính MoE:** Routing Entropy và phân phối sử dụng của từng chuyên gia (Expert Usage) để vẽ biểu đồ Pareto phân tích tính chuyên môn hóa.

## 📂 Cấu trúc Tài liệu Hướng dẫn
Để bắt đầu làm việc với code, vui lòng đọc các tài liệu sau theo thứ tự:
1. ⚙️ [**SETUP.md**](SETUP.md): Hướng dẫn cài đặt môi trường, thư viện và chuẩn bị dữ liệu.
2. 📖 [**Overall.md**](Overall.md): Giải thích chi tiết chức năng của từng file code (scripts, notebooks) và luồng chạy của hệ thống.