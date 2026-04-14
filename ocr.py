from ocr_recognition import idx2char, CRNN, decode, data_transforms
import torch
import cv2
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from ultralytics import YOLO


def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = Image.fromarray(img)

    transform = transforms.Compose([
        transforms.Resize((100, 420)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    img = transform(img)
    img = img.unsqueeze(0)

    return img

def infer(model, img_tensor):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_tensor = img_tensor.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)  # (T, B, vocab)

    # greedy decode
    preds = outputs.argmax(2)  # (T, B)
    preds = preds.permute(1, 0)  # (B, T)

    return preds

def show_predictions(model, folder, idx2char, num_images=30):
    image_files = os.listdir(folder)

    image_files = random.sample(image_files, min(num_images, len(image_files)))

    plt.figure(figsize=(15, 20))

    for i, img_name in enumerate(image_files):
        img_path = os.path.join(folder, img_name)

        # đọc ảnh gốc (để hiển thị)
        img_display = cv2.imread(img_path)
        img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)

        # preprocess + infer
        img = preprocess_image(img_path)
        preds = infer(model, img)
        text = decode(preds, idx2char)[0]

        # plot
        plt.subplot(10, 5, i + 1)
        plt.imshow(img_display)
        plt.title(text, fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# ===================== DETECTION =====================
def text_detection(img, model, conf_thres=0.3):
    results = model(img, verbose=False)[0]

    bboxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)
    confs = results.boxes.conf.cpu().numpy()

    keep = confs >= conf_thres

    return bboxes[keep], classes[keep], confs[keep], results.names


# ===================== RECOGNITION =====================
def text_recognition(crop, transform, model, idx2char, device):
    if isinstance(crop, np.ndarray):
        crop = Image.fromarray(crop)
    img = transform(crop).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(img)

    preds = logits.permute(1, 0, 2).argmax(2)
    text = decode(preds, idx2char)[0]

    return text


# ===================== VISUALIZE =====================
def visualize(img, detections, names):
    img = img.copy()

    for bbox, cls, conf, text in detections:
        x1, y1, x2, y2 = map(int, bbox)

        label = f"({conf:.2f}):{text}"

        # box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # text
        cv2.putText(
            img,
            label,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    return img


# ===================== PIPELINE =====================
def run_ocr_pipeline(
    img_path,
    detection_model,
    recognition_model,
    transform,
    idx2char,
    device
):
    if isinstance(img_path, str):
        img = cv2.imread(img_path)
    else:
        img = img_path  # đã là frame rồi

    # YOLO detect
    bboxes, classes, confs, names = text_detection(img, detection_model)

    detections = []

    for bbox, cls, conf in zip(bboxes, classes, confs):
        x1, y1, x2, y2 = map(int, bbox)

        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        # BGR → RGB → PIL (nếu transform cần)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        text = text_recognition(
            crop_rgb,
            transform,
            recognition_model,
            idx2char,
            device
        )

        detections.append((bbox, cls, conf, text))

    result_img = visualize(img, detections, names)

    return result_img, detections


import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="OCR Pipeline")

    parser.add_argument("--image", type=str, help="Path to image")
    parser.add_argument("--video", type=str, help="Path to video")
    parser.add_argument("--realtime", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load models
    detection_model = YOLO("runs/detect/train4/weights/best.pt")

    recognition_model = CRNN(
        vocab_size=len(idx2char),
        hidden_size=256,
        n_layers=3
    ).to(device)

    recognition_model.load_state_dict(
        torch.load("checkpoints/best.pth", map_location=device)["model_state_dict"]
    )

    recognition_model.eval()

    if args.image:
        result_img, detections = run_ocr_pipeline(
            args.image,
            detection_model,
            recognition_model,
            data_transforms["test"],
            idx2char,
            device
        )

        cv2.imshow("OCR Result", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        for d in detections:
            print(d)

    elif args.video:
        cap = cv2.VideoCapture(args.video)

    elif args.realtime:
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result_img, _ = run_ocr_pipeline(
                frame,   # truyền trực tiếp frame
                detection_model,
                recognition_model,
                data_transforms["test"],
                idx2char,
                device
            )

            cv2.imshow("OCR Video", result_img)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC để thoát
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        print("Please provide --image or --video")


if __name__ == "__main__":
    main()