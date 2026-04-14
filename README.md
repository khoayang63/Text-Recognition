# Text-Recognition

Dự án xây dựng hệ thống OCR (Optical Character Recognition) nhằm phát hiện và nhận dạng văn bản từ ảnh.

Text Detection: phát hiện vùng chứa văn bản trong ảnh
Text Recognition: nhận dạng nội dung văn bản từ các vùng đã phát hiện

Frameworks\libraries: OpenCV, NumPy, PyTorch, Ultralytics yolov11

Dự án này sử dụng bộ dữ liệu ICDAR 2003, một bộ dữ liệu tiêu chuẩn được sử dụng rộng rãi trong các bài toán phát hiện và nhận dạng văn bản trong ảnh (OCR).


## ▶️ Hướng dẫn chạy

1. Clone repository
git clone https://github.com/khoayang63/Text-Recognition.git


2. Tạo môi trường ảo (khuyến khích)
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```
3. Cài đặt thư viện
```bash
pip install -r requirements.txt
```
Nếu có gpu thì tải torch và torchvision gpu: 
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
4. Run
```bash 
    python ocr.py --image "your_image_path"
```
<img width="420" height="281" alt="image" src="https://github.com/user-attachments/assets/1ac953df-299b-49c8-ac97-4c2c5e849c7b" />
<img width="411" height="274" alt="image" src="https://github.com/user-attachments/assets/27613634-c6d8-4502-94a2-8dcc5b0b0278" />

