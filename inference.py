import torch
import numpy as np
import os
import cv2
import torch.nn.functional as F
import argparse
from pathlib import Path
from PIL import Image
from timm.models import create_model
import torchvision.transforms as transforms
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels
import models123
from tool.imutils import crf_inference_label
import streamlit as st
from mmseg.apis import inference_segmentor, init_segmentor
# 🟢 Danh sách 21 lớp (Thêm background ở ID=0)
categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
              'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
              'tvmonitor']


def load_model(checkpoint_path, model_name, device):
    """Load model từ checkpoint."""
    print(f"🔄 Loading model: {model_name} from checkpoint: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ Checkpoint file not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    checkpoint_model = checkpoint.get("model", checkpoint)

    num_classes = checkpoint_model["head.weight"].shape[0]
    print(f"📌 Detected num_classes in checkpoint: {num_classes}")

    model_params = dict(
        model_name=model_name,
        pretrained=False,
        num_classes=num_classes,
        drop_rate=0.0,
        drop_path_rate=0.1,
        drop_block_rate=None,
        reduction=8
    )

    model = create_model(**model_params)
    model.load_state_dict(checkpoint_model, strict=False)
    model.to(device)
    model.eval()

    print("✅ Model loaded successfully!")
    return model


def preprocess_image(image_path, input_size=224):
    """Xử lý ảnh đầu vào."""
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def generate_cams(model, image_tensor, device, image_path, image_name, output_cam_dir, classnames):
    """Tạo CAMs cho các lớp chỉ định và lưu `{image_name}_{class_idx}.npy`."""
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output, cams, patch_attn = model(image_tensor, return_att=True, attention_type='fused')
    probs = torch.sigmoid(output)
    # Binarize by threshold
    pred_labels = (probs > 0.5).float()
    patch_attn = patch_attn.squeeze(0)
    fine_cam = torch.matmul(patch_attn.unsqueeze(1), cams.view(cams.shape[0], cams.shape[1], -1, 1))
    fine_cam = fine_cam.reshape(cams.shape[0], cams.shape[1], cams.shape[2], cams.shape[3])

    original_img = cv2.imread(image_path)
    h, w, _ = original_img.shape  # Lấy kích thước gốc của ảnh

    output_cam_dir = Path(output_cam_dir)
    output_cam_dir.mkdir(parents=True, exist_ok=True)

    cam_dict = {}

    if classnames == ["-1"]:  # Nếu `-1`, lưu tất cả lớp
        classnames = categories

    # 🟢 Lưu chỉ các CAM của `classnames` được chỉ định
    for class_name in classnames:
        if class_name in categories:
            class_idx = categories.index(class_name)
            cam = fine_cam[0, class_idx - 1].cpu().numpy()

            # Normalize và resize về kích thước ảnh gốc
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            cam_resized = cv2.resize(cam, (w, h))

            cam_dict[class_name] = cam_resized
            np.save(output_cam_dir / f"{image_name}_{class_idx}.npy", cam_resized)

            # 🟢 Lưu ảnh CAMs
            heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(original_img, 0.5, heatmap, 0.5, 0)
            cam_img_path = output_cam_dir / f"{image_name}_{class_name}.png"
            cv2.imwrite(str(cam_img_path), overlay)
            print(f"✅ Saved CAM image for class {class_name} to {cam_img_path}")

    return cam_dict


def generate_pseudo_mask(image_path, cam_dict, output_path, threshold=1.0, use_crf=True):
    """Tạo và lưu pseudo-mask từ CAM theo thuật toán chuẩn."""
    original_img = cv2.imread(image_path)
    h, w, _ = original_img.shape

    # 🟢 Load tất cả CAMs (bao gồm cả background)
    cams = np.array([cam_dict[key] for key in cam_dict.keys()])
    label_key = np.array([categories.index(key) for key in cam_dict.keys()]).astype(np.uint8)

    # 🟢 Thêm background nếu chưa có
    if 'background' not in cam_dict:
        background_cam = 1 - np.max(cams, axis=0)
        background_cam = np.clip(background_cam, 0, 1)  # Đảm bảo giá trị hợp lệ
        cams = np.vstack([background_cam[None, :, :], cams])
        label_key = np.insert(label_key, 0, 0)  # Nền luôn có index=0

    # 🟢 Áp dụng Softmax
    cams = F.softmax(torch.tensor(cams).float(), dim=0).numpy()

    # 🟢 Lấy lớp có xác suất cao nhất
    predict = np.argmax(cams, axis=0).astype(np.uint8)

    # 🟢 Áp dụng CRF để tinh chỉnh
    if use_crf:
        predict = crf_inference_label(original_img, predict, n_labels=cams.shape[0])

    # 🟢 Lưu pseudo-mask
    cv2.imwrite(output_path, predict * (255 // len(categories)))
    print(f"✅ Saved Pseudo Mask to {output_path}")

    return predict

def compute_iou(predict, result, num_classes=21):
    """Tính IoU giữa `predict` và `result`."""
    iou_list = []
    for cls in range(num_classes):
        intersection = np.logical_and(predict == cls, result == cls).sum()
        union = np.logical_or(predict == cls, result == cls).sum()

        if union == 0:
            iou = float("nan")  # Tránh chia cho 0
        else:
            iou = intersection / union

        iou_list.append(iou)

    mean_iou = np.nanmean(iou_list)  # Tính trung bình IoU, bỏ qua NaN
    return iou_list, mean_iou

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_name = Path(args.image).stem
    inference_dir = Path("inference")
    output_cam_dir = inference_dir / "output-cam"

    inference_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, args.model, device)
    image_tensor = preprocess_image(args.image)

    # 🟢 Tạo CAMs và lưu chỉ các lớp được chỉ định
    cam_dict = generate_cams(model, image_tensor, device, args.image, image_name, output_cam_dir, args.classnames)

    # 🟢 Tạo Pseudo Mask
    pseudo_mask_path = inference_dir / "pseudo-mask.png"
    predict = generate_pseudo_mask(args.image, cam_dict, pseudo_mask_path)
    '''
    config_path = '/root/SemPLeS/inference/config/m2f-sl22-bt4-80k-512x-VOC.py'
    checkpoint_path = '/root/SemPLeS/inference/checkpoint/best_mIoU_iter_30000.pth'
    model_segment = init_segmentor(config_path, checkpoint_path, device='cuda:0')
    result = inference_segmentor(model_segment, args.image)
    result = np.array(result.pred_sem_seg.cpu().numpy(), dtype=np.uint8)  # Chuyển về numpy
    iou_list, mean_iou = compute_iou(predict, result, num_classes=len(categories))
    print(iou_list, mean_iou)
    '''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WSSS Inference")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default="deit_small_WeakTr_patch16_224")
    parser.add_argument("--classnames", nargs="+", default=["-1"], help="List of classnames or '-1' for all")

    args = parser.parse_args()
    main(args)

