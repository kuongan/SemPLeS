import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import tempfile
from mmseg.apis import init_model, inference_model

# Định nghĩa đường dẫn config và checkpoint của mô hình
CONFIG_PATH = "/root/SemPLeS/inference/config/m2f-sl22-bt4-80k-512x-VOC.py"
CHECKPOINT_PATH = "/root/SemPLeS/inference/checkpoint/best_mIoU_iter_30000.pth"

categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
              'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
              'tvmonitor']

# Khởi tạo mô hình segmentation (cache để tránh load lại mỗi lần chạy)
@st.cache_resource
def load_model():
    return init_model(CONFIG_PATH, CHECKPOINT_PATH, device="cuda" if torch.cuda.is_available() else "cpu")

model = load_model()

# Giao diện Streamlit
st.title("Semantic Segmentation with MMSegmentation")
st.write("Upload an image and run segmentation.")

# Upload ảnh
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Đọc ảnh
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Lưu ảnh tạm thời
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img:
        image.save(temp_img.name)
        img_path = temp_img.name

    # Chạy inference
    st.write("Running inference...")
    result = inference_model(model, img_path)

    # Trích xuất mặt nạ segmentation từ kết quả
    segmentation_mask = result.pred_sem_seg.data[0].cpu().numpy()

    # Xác định số lượng lớp (classes) từ segmentation_mask
    unique_labels = np.unique(segmentation_mask)
    num_classes = len(categories)  # Đảm bảo có đủ màu cho tất cả các lớp

    # Tạo bảng màu cho tất cả các lớp
    np.random.seed(42)  # Đảm bảo màu sắc nhất quán
    color_map = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)

    # Tạo ảnh màu từ segmentation mask
    colored_mask = np.zeros((*segmentation_mask.shape, 3), dtype=np.uint8)
    for label in unique_labels:
        if label == 0:
            continue  # Bỏ qua nền (background)
        if label >= num_classes:
            print(f"Warning: Class {label} exceeds color_map size ({num_classes}). Skipping...")
            continue
        colored_mask[segmentation_mask == label] = color_map[label]

    # Chuyển ảnh gốc sang numpy
    img_np = np.array(image)

    # Overlay ảnh segmentation lên ảnh gốc
    overlay = cv2.addWeighted(img_np, 0.6, colored_mask, 0.4, 0)

    # Hiển thị kết quả bằng Streamlit
    st.image(overlay, caption="Segmented Image", use_container_width=True)

    # Tạo chú thích màu cho các category
    legend_height = 25 * len(unique_labels)
    legend_width = 200
    legend_img = Image.new("RGB", (legend_width, legend_height), "white")
    draw = ImageDraw.Draw(legend_img)

    # Font mặc định cho chú thích
    try:
        font = ImageFont.truetype("arial.ttf", 18)  # Font Arial
    except IOError:
        font = ImageFont.load_default()  # Nếu không tìm thấy Arial, dùng font mặc định

    for i, label in enumerate(unique_labels):
        if label == 0:
            continue  # Bỏ qua background
        color = tuple(map(int, color_map[label]))  # Lấy màu tương ứng
        draw.rectangle([10, 10 + i * 25, 40, 35 + i * 25], fill=color, outline="black")
        draw.text((50, 10 + i * 25), categories[label], fill="black", font=font)

    # Hiển thị chú thích
    st.image(legend_img, caption="Category Legend", use_container_width=False)

    # Lưu ảnh kết quả segmentation
    output_file = "segmented_output.jpg"
    cv2.imwrite(output_file, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # Nút tải kết quả segmentation
    with open(output_file, "rb") as file:
        st.download_button(
            label="Download Segmentation Result",
            data=file,
            file_name="segmented_output.jpg",
            mime="image/jpeg"
        )
