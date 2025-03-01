import streamlit as st
import torch
import numpy as np
import os
import cv2
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from timm.models import create_model
import torchvision.transforms as transforms
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels
import models123
from tool.imutils import crf_inference_label

# 🟢 Danh sách 21 lớp (Thêm background ở ID=0)
categories = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 
    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# 🟢 Cấu hình Streamlit
st.set_page_config(layout="wide", page_title="WSSS Segmentation", page_icon="🚀")
st.title("🚀 Weakly Supervised Semantic Segmentation")

@st.cache_resource
def load_model(checkpoint_path, model_name, device):
    """Load segmentation model từ checkpoint."""
    if not os.path.exists(checkpoint_path):
        st.error(f"❌ Checkpoint file not found: {checkpoint_path}")
        return None

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    checkpoint_model = checkpoint.get("model", checkpoint)
    num_classes = checkpoint_model["head.weight"].shape[0]

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
    return model

def preprocess_image(image, input_size=224):
    """Xử lý ảnh đầu vào."""
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def get_class_probs(model, image_tensor):
    """
    Run the model once in forward pass (with return_att=False),
    just to get the classification (logits) for all classes.
    Returns:
      probs (tensor): shape (1, num_classes), sigmoid-applied
    """
    with torch.no_grad():
        output = model(image_tensor, return_att=False)[0]  
        # Model returns (cls_token_pred, coarse_cam_pred, fine_cam_pred),
        # so we take [0] => cls_token_pred => shape [B, num_classes]
    probs = torch.sigmoid(output)  # shape: (1, num_classes)
    return probs

def forward_with_attention(model, image_tensor):
    """
    Forward pass to get (output, cams, patch_attn).
    We'll use this to generate class activation maps.
    """
    with torch.no_grad():
        output, cams, patch_attn = model(image_tensor, return_att=True, attention_type='fused')
    return output, cams, patch_attn

def generate_cams(model, image_tensor, device, image, class_indices):
    """
    Generate CAM dictionary for given class indices
    (where class_indices is a list of integer indices in [1..20]).
    """
    output, cams, patch_attn = forward_with_attention(model, image_tensor)
    patch_attn = patch_attn.squeeze(0)
    fine_cam = torch.matmul(
        patch_attn.unsqueeze(1), cams.view(cams.shape[0], cams.shape[1], -1, 1)
    )
    fine_cam = fine_cam.reshape(cams.shape[0], cams.shape[1], cams.shape[2], cams.shape[3])

    original_img = np.array(image)
    h, w, _ = original_img.shape
    cam_dict = {}

    # For each chosen class index (1..20 => skipping background=0)
    for class_idx in class_indices:
        if class_idx <= 0 or class_idx >= len(categories):
            continue
        class_name = categories[class_idx]
        # Model's 1st channel => class_idx=1 => fine_cam[0,0]
        # So we do (class_idx - 1) because the model output doesn't have background.
        cam = fine_cam[0, class_idx - 1].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam_resized = cv2.resize(cam, (w, h))
        cam_dict[class_name] = cam_resized

    return cam_dict

def generate_pseudo_mask(image, cam_dict, use_crf=True):
    """Sinh Pseudo Mask từ CAMs."""
    original_img = np.array(image)
    h, w, _ = original_img.shape
    # Sort the keys by index in categories
    sorted_keys = sorted(cam_dict.keys(), key=lambda x: categories.index(x))
    # stack the CAMs
    cams = np.array([cam_dict[key] for key in sorted_keys])  
    label_key = np.array([categories.index(key) for key in sorted_keys]).astype(np.uint8)

    # Thêm background nếu chưa có
    if 'background' not in cam_dict:
        background_cam = 1 - np.max(cams, axis=0)
        background_cam = np.clip(background_cam, 0, 1)
        cams = np.vstack([background_cam[None, :, :], cams])
        label_key = np.insert(label_key, 0, 0)

    # Softmax across class dimension => shape: (num_classes, H, W)
    cams = F.softmax(torch.tensor(cams).float(), dim=0).numpy()
    # Argmax => pixel-level classification
    predict = np.argmax(cams, axis=0).astype(np.uint8)

    # CRF refinement
    if use_crf:
        predict = crf_inference_label(original_img, predict, n_labels=cams.shape[0])
    return predict

def overlay_mask_on_image(image, mask):
    """Vẽ mask màu lên ảnh gốc."""
    color_map = np.random.randint(0, 255, (len(categories), 3), dtype=np.uint8)
    mask_color = color_map[mask]
    overlay = cv2.addWeighted(np.array(image), 0.6, mask_color, 0.4, 0)
    return overlay


# ============ Streamlit App =================
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])
checkpoint_path = st.text_input("📌 Checkpoint Path", "/root/SemPLeS/semples3/voc/checkpoint_best_mIoU.pth")
model_name = st.text_input("📌 Model Name", "deit_small_WeakTr_patch16_224")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = load_model(checkpoint_path, model_name, device) if checkpoint_path else None

if uploaded_file and model:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📸 Uploaded Image", use_column_width=True)

    # ======================== MODE SELECTION ========================
    mode = st.radio("Choose Mode", ["Automatic Multi-Label Classification", "Manual Class Selection"])
    use_crf = st.checkbox("🛠 Apply CRF Refinement", value=True)
    if mode == "Automatic Multi-Label Classification":
        st.write(
            "The model will predict classes for this image, then generate CAMs & pseudo-mask "
            "based on the predicted classes."
        )
    else:
        st.write(
            "Select specific classes below to visualize CAMs & produce the pseudo-mask for those classes."
        )
        selected_classes = st.multiselect(
            "🎯 Select Classes (1..20, skipping background=0)",
            options=categories[1:],  # skip 'background'
            default=["person"]
        )

    if st.button("🔥 Generate"):
        image_tensor = preprocess_image(image).to(device)

        # (1) If Automatic => do classification
        if mode == "Automatic Multi-Label Classification":
            # -- Classification only pass
            probs, cams, patch_attn = forward_with_attention(model, image_tensor)
            pred_labels = (probs > 0.5).float()  # shape: (1, num_classes)

            predicted_classnames = []
            # We have 20 classes in the model output, they align to categories[1..20]
            for i, val in enumerate(pred_labels[0]):
                if val == 1:
                    predicted_classnames.append(categories[i + 1])  # shift by +1

            st.write("**Predicted Classes (image-level)**:", ", ".join(predicted_classnames))

            # Convert those predicted classes to indices
            class_indices = []
            for cname in predicted_classnames:
                class_idx = categories.index(cname)
                class_indices.append(class_idx)

        # (2) If Manual => user-chosen classes => get their indices
        else:
            predicted_classnames = selected_classes
            # Convert to indices
            class_indices = []
            for cname in selected_classes:
                class_idx = categories.index(cname)
                class_indices.append(class_idx)

        # If no classes chosen or predicted, skip
        if not class_indices:
            st.warning("No valid classes chosen/predicted. No CAMs to display.")
        else:
            # Generate CAM dict
            cam_dict = generate_cams(model, image_tensor, device, image, class_indices)

            # Generate pseudo-mask
            pseudo_mask = generate_pseudo_mask(image, cam_dict, use_crf)
            pseudo_mask_path = "pseudo_mask.png"
            cv2.imwrite(pseudo_mask_path, pseudo_mask * (255 // len(categories)))
            pseudo_mask_display = cv2.imread(pseudo_mask_path, cv2.IMREAD_GRAYSCALE)

            # Overlay
            overlay_image = overlay_mask_on_image(image, pseudo_mask)
            overlay_path = "overlay.png"
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))

            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("🎨 CAMs")
                for cls_name, cam_data in cam_dict.items():
                    heatmap = cv2.applyColorMap((cam_data * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    overlay_cam = cv2.addWeighted(np.array(image), 0.5, heatmap, 0.5, 0)
                    st.image(overlay_cam, caption=f"🔥 CAM - {cls_name}", use_column_width=True)

            with col2:
                st.subheader("🟢 Pseudo Mask")
                st.image(pseudo_mask_display, caption="🟢 Mask", use_column_width=True)

            with col3:
                st.subheader("📸 Overlay Mask")
                st.image(overlay_image, caption="🎭 Overlay on Image", use_column_width=True)

            # Download buttons
            with open(pseudo_mask_path, "rb") as file:
                st.download_button("💾 Download Pseudo Mask", file, file_name="pseudo_mask.png", mime="image/png")

            with open(overlay_path, "rb") as file:
                st.download_button("📥 Download Overlay Image", file, file_name="overlay.png", mime="image/png")
