import os
import shutil
from ocr_recognition import parse_icdar
from sklearn.model_selection import train_test_split
from collections import defaultdict
from ultralytics import YOLO

def convert_to_yolo_format(data, image_root):
    """
    return:
        list of (image_path, yolo_labels)
        
        yolo_labels: list of [class_id, x_center, y_center, w, h]
    """

    yolo_dict = defaultdict(list)

    for item in data:
        img_path = os.path.join(image_root, item["image"])

        # size từ XML
        w, h = item["img_size"]
        if w is None or h is None:
            print(f"Missing size for {img_path}")
            continue

        x, y, bw, bh = item["bbox"]

        # normalize YOLO
        x_center = (x + bw / 2) / w
        y_center = (y + bh / 2) / h
        bw /= w
        bh /= h

        # clamp 
        x_center = min(max(x_center, 0), 1)
        y_center = min(max(y_center, 0), 1)
        bw = min(max(bw, 0), 1)
        bh = min(max(bh, 0), 1)

        # append bbox vào image tương ứng
        yolo_dict[img_path].append([0, x_center, y_center, bw, bh])

    # convert dict → list
    yolo_data = []
    for img_path, labels in yolo_dict.items():
        yolo_data.append((img_path, labels))

    return yolo_data

def save_yolo_split(data, output_root, split_name):
    """
    data: list[(image_path, yolo_labels)]
    split_name: "train" | "val" | "test"
    """

    img_dir = os.path.join(output_root, split_name, "images")
    label_dir = os.path.join(output_root, split_name, "labels")

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    for img_path, labels in data:
        if not os.path.exists(img_path):
            print(f"Missing image: {img_path}")
            continue

        img_name = os.path.basename(img_path)
        txt_name = os.path.splitext(img_name)[0] + ".txt"

        save_img_path = os.path.join(img_dir, img_name)
        shutil.copy(img_path, save_img_path)

        label_path = os.path.join(label_dir, txt_name)

        with open(label_path, "w") as f:
            for cls, x, y, w, h in labels:
                f.write(f"{cls} {x} {y} {w} {h}\n")

def save_yolo_dataset(train, val, test, output_root="yolo_data"):
    save_yolo_split(train, output_root, "train")
    save_yolo_split(val, output_root, "val")
    save_yolo_split(test, output_root, "test")

def main():
    # model = YOLO("yolo11m.pt")

    # results = model.train(
    #     data = 'yolo_data/data.yaml',
    #     epochs = 100,
    #     imgsz=640,
    #     patience=20,
    #     batch=8,
    #     plots=True,
    #     workers=2
    # )
    best_path = "runs/detect/train4/weights/best.pt"
    model = YOLO(best_path)

    metrics = model.val(split='test')
    print(f"""
    Evaluation Results:
    - Precision : {metrics.box.p[0]:.3f}
    - Recall    : {metrics.box.r[0]:.3f}
    - mAP50     : {metrics.box.map50:.3f}
    - mAP50-95  : {metrics.box.map:.3f}
    """)

    
if __name__ == "__main__":
    main()
