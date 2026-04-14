import os
import random
import time
import xml.etree.ElementTree as ET

import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm


from dotenv import load_dotenv
load_dotenv()

token = os.getenv("hf_token")


def parse_icdar(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    data = []

    for image in root.findall('image'):
        image_name = image.find('imageName').text

        resolution = image.find('resolution')
        if resolution is not None:
            img_w = int(resolution.attrib.get('x', 0))
            img_h = int(resolution.attrib.get('y', 0))
        else:
            img_w, img_h = None, None

        rectangles = image.find('taggedRectangles')
        if rectangles is None:
            continue

        for rect in rectangles.findall('taggedRectangle'):
            x = float(rect.attrib['x'])
            y = float(rect.attrib['y'])
            w = float(rect.attrib['width'])
            h = float(rect.attrib['height'])

            tag = rect.find('tag')
            text = tag.text if tag is not None else ""

            data.append({
                "image": image_name,
                "bbox": [x, y, w, h],
                "text": text,
                "img_size": (img_w, img_h) 
            })

    return data


def crop_data(data, root_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    results = []
    count = 0

    for item in data:
        img_path = os.path.join(root_dir, item["image"])
        text = item["text"]
        x, y, w, h = map(int, item["bbox"])

        # đọc ảnh
        img = cv2.imread(img_path)
        if img is None:
            continue

        # crop
        crop = img[y:y+h, x:x+w]

        # filter 1: ảnh rỗng
        if crop.size == 0:
            continue

        # filter 2: quá nhỏ
        if crop.shape[0] < 10 or crop.shape[1] < 10:
            continue

        # filter 3: quá trắng / đen
        mean_val = crop.mean()
        if mean_val < 35 or mean_val > 220:
            continue

        # lưu ảnh
        filename = f"{count:06d}.jpg"
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, crop)

        # lưu label
        results.append(f"{save_path}\t{text}")

        count += 1
    
        with open(os.path.join(save_dir, 'labels.txt'), "w", encoding="utf-8") as f:
            for r in results:
                f.write(f'{r}\n')

    return results


xml_path = 'SceneTrialTrain/words.xml'
save_dir = 'cropped'

# data = parse_icdar(xml_path)
# results = crop_data(
#     data,
#     root_dir=".\SceneTrialTrain",
#     save_dir="cropped"
# )


# for i in range(10):
#     print(results[i])

img_path, labels = [], []

with open(os.path.join(save_dir, 'labels.txt'), 'r') as f:
    for r in f:
        pth, label = r.strip().split("\t")
        img_path.append(pth)
        labels.append(label.lower())

# print(f"Image found: {len(img_path)}, label found: {len(labels)}")

def build_vocab(labels):
    charset = set()

    for label in labels:
        charset.update(list(label.lower()))

    vocab = sorted(list(charset))
    char2idx = {c: i+1 for i, c in enumerate(vocab)}  # 0 để dành cho blank
    idx2char = {i: c for c, i in char2idx.items()}

    return vocab, char2idx, idx2char

vocab, char2idx, idx2char = build_vocab(labels)
vocab_size= len(vocab)
# print(f"Vocab size = {len(vocab)}")

data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((100, 420)),
        transforms.ColorJitter(
            brightness=0.5,
            contrast=0.5,
            saturation=0.5
        ),
        transforms.Grayscale(num_output_channels=1),
        transforms.GaussianBlur(3),
        transforms.RandomAffine(
            degrees=1,
            shear=1
        ),
        transforms.RandomPerspective(
            distortion_scale=0.3,
            p=0.5,
            interpolation=3
        ),
        transforms.RandomRotation(degrees=2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]),

    "test": transforms.Compose([
        transforms.Resize((100, 420)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
}

class OCRDataset(Dataset):
    def __init__(self, images, labels, char_to_idx, transform=None):
        self.images = images
        self.labels = labels
        self.char_to_idx = char_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        text = self.labels[idx]

        # đọc ảnh
        img = cv2.imread(img_path)

        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")

        # BGR → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # → PIL
        img = Image.fromarray(img)

        # transform
        if self.transform:
            img = self.transform(img)

        # encode label
        label = torch.tensor(
            [self.char_to_idx[c] for c in text],
            dtype=torch.long
        )

        return img, label

def collate_fn(batch):
    images, labels = zip(*batch)

    # stack images
    images = torch.stack(images)

    # label lengths
    label_lengths = torch.tensor(
        [len(l) for l in labels],
        dtype=torch.long
    )

    # concat labels (CTC yêu cầu)
    labels_concat = torch.cat(labels)

    return images, labels_concat, label_lengths


def get_loaders():
    X_train, X_test, y_train, y_test = \
    train_test_split(
        img_path,
        labels,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    train_dataset = OCRDataset(X_train, y_train, char2idx, data_transforms['train'])
    test_dataset = OCRDataset(X_test, y_test, char2idx, data_transforms['test'])

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4, 
        pin_memory=True, # nếu dùng GPU
        collate_fn=collate_fn,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    return train_loader, test_loader, train_dataset, test_dataset

class CRNN(nn.Module):
    def __init__(self,      
        hidden_size,
        n_layers,
        vocab_size=vocab_size,
        dropout=0.3,
        unfreeze_layers=3
    ):
        super().__init__()
        backbone = timm.create_model(
            "resnet152",  
            in_chans=1,      
            pretrained=True
        )

        modules = list(backbone.children())[:-2]
        modules.append(nn.AdaptiveAvgPool2d((1, None)))

        self.backbone = nn.Sequential(*modules)

        # freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # unfreeze last layers
        for param in self.backbone[-unfreeze_layers:].parameters():
            param.requires_grad = True
        
        self.proj = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(dropout)
        )

        self.gru = nn.GRU(
            512,
            hidden_size,
            n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if n_layers>1 else 0
        )

        self.layer_norm = nn.LayerNorm(
            hidden_size*2
        )
        
        self.head = nn.Sequential(
            nn.Linear(hidden_size*2, vocab_size),
            nn.LogSoftmax(dim=2)
        )

    def forward(self, x):
        # x(B, C, H, W)
        x = self.backbone(x) # (B, 2048, 1, W)
        x = x.permute(0, 3, 1, 2) #(B, W, 2048, 1)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.proj(x)
        x, _ = self.gru(x) # (B, W, 2*hidden)
        x = self.layer_norm(x)
        x = self.head(x) # (B, W, vocab_size)
        x = x.permute(1, 0, 2) # (W, B, vocab_size)
        return x


def decode(encoded_sequences, idx_to_char, blank=0):
    decoded_sequences = []

    for seq in encoded_sequences:
        decoded_label = []
        prev_token = None

        for token in seq:
            token = token.item()

            # bỏ blank
            if token == blank:
                prev_token = None
                continue

            # bỏ lặp
            if token == prev_token:
                continue

            decoded_label.append(idx_to_char[token])
            prev_token = token

        decoded_sequences.append("".join(decoded_label))

    return decoded_sequences

def show_batch(imgs, labels, lengths):
    decoded = []

    start = 0
    for l in lengths:
        seq = labels[start:start+l]
        decoded.append(''.join(idx2char[x.item()] for x in seq))
        start += l
    print(decoded)
    grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=True)
    plt.figure(figsize=(10, 20))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.axis("off")

    print(decoded)
    plt.show()



def main():
    train_loader, test_loader, train_dataset, test_dataset = get_loaders()

    train_features, train_labels, train_lengths = next(iter(train_loader))
    print(train_features.shape)
    show_batch(train_features, train_labels, train_lengths)

if __name__ == "__main__":
    main()